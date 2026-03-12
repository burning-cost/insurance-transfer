"""Tests for TransferPipeline."""

import numpy as np
import pytest
import warnings

from insurance_transfer.pipeline import TransferPipeline, PipelineResult
from insurance_transfer.glm_transfer import GLMTransfer


def _make_data(rng, n_src=500, n_tgt=80, p=5, shift=0.2):
    true_beta = np.array([0.3, -0.2, 0.1, 0.0, 0.1])
    X_src = rng.standard_normal((n_src, p))
    y_src = rng.poisson(np.exp(X_src @ true_beta)).astype(float)
    X_tgt = rng.standard_normal((n_tgt, p)) + shift
    y_tgt = rng.poisson(np.exp(X_tgt @ true_beta)).astype(float)
    exp_src = np.ones(n_src)
    exp_tgt = np.ones(n_tgt)
    return X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt


class TestTransferPipeline:
    def test_basic_run(self):
        rng = np.random.default_rng(80)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="glm",
            shift_test=False,
            run_diagnostic=False,
            random_state=0,
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert isinstance(result, PipelineResult)
        assert result.model is not None

    def test_with_shift_test(self):
        rng = np.random.default_rng(81)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng, shift=0.1)
        pipeline = TransferPipeline(
            method="glm",
            shift_test=True,
            shift_n_permutations=50,
            run_diagnostic=False,
            random_state=0,
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert result.shift_result is not None
        assert result.shift_p_value is not None
        assert 0.0 <= result.shift_p_value <= 1.0

    def test_with_diagnostic(self):
        rng = np.random.default_rng(82)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="glm",
            shift_test=False,
            run_diagnostic=True,
            diagnostic_test_size=0.25,
            glm_params={"family": "poisson"},
            random_state=0,
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert result.diagnostic_result is not None
        assert result.transfer_is_beneficial is not None

    def test_no_source_data(self):
        """Pipeline should work without source data (target-only GLM)."""
        rng = np.random.default_rng(83)
        X_tgt = rng.standard_normal((80, 4))
        y_tgt = rng.poisson(1.0, size=80).astype(float)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt)
        assert isinstance(result, PipelineResult)

    def test_method_used_attribute(self):
        rng = np.random.default_rng(84)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt)
        assert result.method_used == "glm"

    def test_auto_method(self):
        rng = np.random.default_rng(85)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(method="auto", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt, X_source=X_src, y_source=y_src)
        assert isinstance(result.model, GLMTransfer)

    def test_invalid_method_raises(self):
        rng = np.random.default_rng(86)
        _, _, _, X_tgt, y_tgt, _ = _make_data(rng)
        pipeline = TransferPipeline(method="invalid_method", shift_test=False, run_diagnostic=False)
        with pytest.raises(ValueError, match="method"):
            pipeline.run(X_tgt, y_tgt)

    def test_gbm_without_source_model_raises(self):
        rng = np.random.default_rng(87)
        _, _, _, X_tgt, y_tgt, _ = _make_data(rng)
        pipeline = TransferPipeline(method="gbm", shift_test=False, run_diagnostic=False)
        with pytest.raises(ValueError, match="source_model"):
            pipeline.run(X_tgt, y_tgt)

    def test_shift_warning_on_severe_shift(self):
        rng = np.random.default_rng(88)
        X_src = rng.standard_normal((300, 3))
        X_tgt = rng.standard_normal((60, 3)) + 5.0  # Very large shift
        y_src = rng.poisson(1.0, size=300).astype(float)
        y_tgt = rng.poisson(1.0, size=60).astype(float)

        pipeline = TransferPipeline(
            method="glm", shift_test=True, shift_n_permutations=100,
            run_diagnostic=False, random_state=0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline.run(X_tgt, y_tgt, X_source=X_src, y_source=y_src)
            # Should have issued a covariate shift warning
            shift_warnings = [x for x in w if "shift" in str(x.message).lower()]
            assert len(shift_warnings) >= 1

    def test_result_repr(self):
        rng = np.random.default_rng(89)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="glm", shift_test=True, shift_n_permutations=50,
            run_diagnostic=True, random_state=0,
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        r = repr(result)
        assert "PipelineResult" in r
        assert "method='glm'" in r

    def test_no_diagnostic_when_small_sample(self):
        rng = np.random.default_rng(90)
        X_tgt = rng.standard_normal((4, 3))  # Very small
        y_tgt = rng.poisson(1.0, size=4).astype(float)
        pipeline = TransferPipeline(
            method="glm", shift_test=False,
            run_diagnostic=True, diagnostic_test_size=0.2,
        )
        result = pipeline.run(X_tgt, y_tgt)
        # Should not crash; diagnostic may or may not run depending on n
        assert isinstance(result, PipelineResult)

    def test_glm_params_passed(self):
        rng = np.random.default_rng(91)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="glm",
            glm_params={"family": "poisson", "lambda_pool": 0.05, "lambda_debias": 0.1},
            shift_test=False,
            run_diagnostic=False,
        )
        result = pipeline.run(X_tgt, y_tgt, X_source=X_src, y_source=y_src)
        model = result.model
        assert isinstance(model, GLMTransfer)
        assert model.lambda_pool == 0.05

    def test_shift_p_value_none_without_shift_test(self):
        rng = np.random.default_rng(92)
        _, _, _, X_tgt, y_tgt, _ = _make_data(rng)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt)
        assert result.shift_p_value is None
        assert result.shift_result is None

    def test_transfer_beneficial_none_without_diagnostic(self):
        rng = np.random.default_rng(93)
        _, _, _, X_tgt, y_tgt, _ = _make_data(rng)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt)
        assert result.transfer_is_beneficial is None

    def test_cann_pipeline(self):
        """Pipeline with CANN method should work if torch available."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        rng = np.random.default_rng(94)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="cann",
            cann_params={"hidden_sizes": [8], "pretrain_epochs": 3, "finetune_epochs": 3},
            shift_test=False,
            run_diagnostic=False,
        )
        result = pipeline.run(
            X_tgt, y_tgt, exp_tgt,
            X_source=X_src, y_source=y_src, exposure_source=exp_src,
        )
        preds = result.model.predict(X_tgt)
        assert preds.shape == (80,)
