"""Tests for NegativeTransferDiagnostic and metric functions."""

import numpy as np
import pytest

from insurance_transfer.diagnostic import (
    NegativeTransferDiagnostic,
    TransferDiagnosticResult,
    poisson_deviance,
    gamma_deviance,
)


class ConstModel:
    def __init__(self, val: float):
        self.val = val

    def predict(self, X, exposure=None):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.val)


class ScaledModel:
    """Predicts y * scale."""
    def __init__(self, scale: float, y_ref: np.ndarray):
        self.scale = scale
        self.y_ref = y_ref

    def predict(self, X, exposure=None):
        return self.y_ref * self.scale


class TestPoissonDeviance:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        assert poisson_deviance(y, mu) == pytest.approx(0.0, abs=1e-10)

    def test_zero_response(self):
        y = np.array([0.0, 0.0, 0.0])
        mu = np.array([1.0, 1.0, 1.0])
        # D = 2 * mean(0 - (0 - 1)) = 2
        result = poisson_deviance(y, mu)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_positive_deviance(self):
        y = np.array([2.0, 3.0, 1.0])
        mu = np.array([1.0, 1.0, 1.0])
        result = poisson_deviance(y, mu)
        assert result > 0

    def test_overestimate_vs_underestimate(self):
        y = np.array([2.0] * 100)
        mu_over = np.array([3.0] * 100)
        mu_under = np.array([1.0] * 100)
        # Both should give positive deviance
        assert poisson_deviance(y, mu_over) > 0
        assert poisson_deviance(y, mu_under) > 0

    def test_numeric_stability_near_zero(self):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1e-5, 1.0, 2.0])
        result = poisson_deviance(y, mu)
        assert np.isfinite(result)

    def test_scalar_inputs(self):
        result = poisson_deviance(np.array([1.0]), np.array([1.0]))
        assert result == pytest.approx(0.0, abs=1e-10)


class TestGammaDeviance:
    def test_perfect_prediction(self):
        y = np.array([1.5, 2.0, 3.0])
        result = gamma_deviance(y, y)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_positive_deviance(self):
        y = np.array([2.0, 3.0])
        mu = np.array([1.0, 1.0])
        result = gamma_deviance(y, mu)
        assert result > 0

    def test_numeric_stability(self):
        y = np.array([0.001, 1.0, 100.0])
        mu = np.array([1.0, 1.0, 1.0])
        result = gamma_deviance(y, mu)
        assert np.isfinite(result)


class TestNegativeTransferDiagnostic:
    def _make_data(self, rng, n=100, p=4):
        X = rng.standard_normal((n, p))
        y = rng.poisson(np.exp(0.3 * X[:, 0])).astype(float)
        exposure = np.ones(n)
        return X, y, exposure

    def test_basic_evaluation(self):
        rng = np.random.default_rng(60)
        X, y, exposure = self._make_data(rng)
        transfer = ConstModel(1.2)
        target_only = ConstModel(1.0)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, transfer, target_only)
        assert isinstance(result, TransferDiagnosticResult)
        assert isinstance(result.ntg, float)
        assert isinstance(result.ntg_relative, float)

    def test_ntg_positive_when_transfer_worse(self):
        rng = np.random.default_rng(61)
        X, y, exposure = self._make_data(rng)
        # Transfer model is very bad (constant = 5, true mean ~1)
        transfer = ConstModel(5.0)
        # Target-only is better
        target_only = ConstModel(1.0)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, transfer, target_only)
        assert result.ntg > 0
        assert not result.transfer_is_beneficial

    def test_ntg_negative_when_transfer_better(self):
        rng = np.random.default_rng(62)
        X, y, exposure = self._make_data(rng)
        # Transfer model is perfect, target-only is bad
        transfer = ConstModel(1.0)
        target_only = ConstModel(5.0)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, transfer, target_only)
        assert result.ntg < 0
        assert result.transfer_is_beneficial

    def test_source_only_model(self):
        rng = np.random.default_rng(63)
        X, y, exposure = self._make_data(rng)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(
            X, y, exposure,
            transfer_model=ConstModel(1.0),
            target_only_model=ConstModel(1.0),
            source_only_model=ConstModel(2.0),
        )
        assert result.poisson_deviance_source_only is not None
        assert isinstance(result.poisson_deviance_source_only, float)

    def test_no_source_only_model(self):
        rng = np.random.default_rng(64)
        X, y, exposure = self._make_data(rng)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(
            X, y, exposure,
            transfer_model=ConstModel(1.0),
            target_only_model=ConstModel(1.0),
        )
        assert result.poisson_deviance_source_only is None

    def test_per_feature_analysis_keys(self):
        rng = np.random.default_rng(65)
        X, y, exposure = self._make_data(rng, p=5)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.0))
        assert len(result.per_feature_analysis) == 5

    def test_feature_names(self):
        rng = np.random.default_rng(66)
        X, y, exposure = self._make_data(rng, p=3)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(
            X, y, exposure,
            transfer_model=ConstModel(1.0),
            target_only_model=ConstModel(1.0),
            feature_names=["age", "ncb", "region"],
        )
        assert "age" in result.per_feature_analysis
        assert "ncb" in result.per_feature_analysis

    def test_n_test_attribute(self):
        rng = np.random.default_rng(67)
        X, y, exposure = self._make_data(rng, n=50)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.0))
        assert result.n_test == 50

    def test_default_exposure_none(self):
        rng = np.random.default_rng(68)
        X, y, _ = self._make_data(rng)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, None, ConstModel(1.0), ConstModel(1.0))
        assert isinstance(result, TransferDiagnosticResult)

    def test_gamma_deviance_metric(self):
        rng = np.random.default_rng(69)
        X, y, exposure = self._make_data(rng)
        y = y + 0.1  # Make all positive for gamma
        diag = NegativeTransferDiagnostic(metric="gamma_deviance")
        result = diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.0))
        assert isinstance(result, TransferDiagnosticResult)

    def test_custom_metric(self):
        rng = np.random.default_rng(70)
        X, y, exposure = self._make_data(rng)

        def mse(y_true, mu):
            return float(np.mean((y_true - mu) ** 2))

        diag = NegativeTransferDiagnostic(metric=mse)
        result = diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.0))
        assert isinstance(result, TransferDiagnosticResult)

    def test_invalid_metric_raises(self):
        diag = NegativeTransferDiagnostic(metric="not_a_metric")
        rng = np.random.default_rng(71)
        X, y, exposure = self._make_data(rng)
        with pytest.raises(ValueError, match="metric"):
            diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.0))

    def test_repr(self):
        rng = np.random.default_rng(72)
        X, y, exposure = self._make_data(rng)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.0))
        r = repr(result)
        assert "TransferDiagnosticResult" in r
        assert "NTG" in r

    def test_summary_table(self):
        rng = np.random.default_rng(73)
        X, y, exposure = self._make_data(rng)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.2))
        table = diag.summary_table(result)
        assert "Transfer model" in table
        assert "NTG" in table
        assert "Target-only baseline" in table

    def test_ntg_relative_percentage(self):
        rng = np.random.default_rng(74)
        X, y, exposure = self._make_data(rng)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exposure, ConstModel(1.0), ConstModel(1.0))
        # Same models -> NTG = 0
        assert result.ntg == pytest.approx(0.0, abs=1e-10)
        assert result.ntg_relative == pytest.approx(0.0, abs=1e-6)
