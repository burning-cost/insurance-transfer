"""Tests for CANNTransfer (PyTorch CANN pre-train/fine-tune).

All tests skip if torch is not installed.
"""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)


class TestCANNTransfer:
    def _make_data(self, rng, n_src=200, n_tgt=50, p=5):
        true_beta = np.array([0.4, -0.2, 0.1, 0.0, 0.15])
        X_src = rng.standard_normal((n_src, p))
        y_src = rng.poisson(np.exp(X_src @ true_beta)).astype(float)
        X_tgt = rng.standard_normal((n_tgt, p)) + 0.3
        y_tgt = rng.poisson(np.exp(X_tgt @ true_beta)).astype(float)
        return X_src, y_src, X_tgt, y_tgt

    def test_fit_source_then_fit(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(40)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(
            hidden_sizes=[16, 8],
            pretrain_epochs=3,
            finetune_epochs=3,
            random_state=42,
        )
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (50,)
        assert np.all(preds > 0)

    def test_fit_without_pretrain(self):
        """fit() without fit_source() should train from scratch."""
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(41)
        _, _, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(hidden_sizes=[8], pretrain_epochs=3, finetune_epochs=3)
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (50,)

    def test_head_only_strategy(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(42)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(
            hidden_sizes=[16], finetune_strategy="head_only",
            pretrain_epochs=3, finetune_epochs=3, random_state=0,
        )
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (50,)

    def test_all_strategy(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(43)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(
            hidden_sizes=[16], finetune_strategy="all",
            pretrain_epochs=3, finetune_epochs=3, random_state=0,
        )
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (50,)

    def test_progressive_strategy(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(44)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(
            hidden_sizes=[16], finetune_strategy="progressive",
            pretrain_epochs=4, finetune_epochs=4, random_state=0,
        )
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (50,)

    def test_invalid_strategy_raises(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(45)
        _, _, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(finetune_strategy="bad_strategy", pretrain_epochs=2, finetune_epochs=2)
        with pytest.raises(ValueError, match="finetune_strategy"):
            model.fit(X_tgt, y_tgt)

    def test_predict_requires_fit(self):
        from sklearn.exceptions import NotFittedError
        from insurance_transfer.cann_transfer import CANNTransfer
        model = CANNTransfer()
        with pytest.raises(NotFittedError):
            model.predict(np.random.randn(10, 5))

    def test_loss_attributes(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(46)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(
            hidden_sizes=[8], pretrain_epochs=3, finetune_epochs=3, random_state=0
        )
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        assert len(model.pretrain_losses_) == 3
        assert len(model.finetune_losses_) == 3

    def test_exposure_affects_predictions(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(47)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(hidden_sizes=[8], pretrain_epochs=3, finetune_epochs=3, random_state=1)
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        exp1 = np.ones(50)
        exp2 = np.ones(50) * 2.0
        preds1 = model.predict(X_tgt, exp1)
        preds2 = model.predict(X_tgt, exp2)
        # Doubling exposure should roughly double predictions
        ratio = preds2 / preds1
        np.testing.assert_allclose(ratio, 2.0, rtol=0.05)

    def test_scale_features_false(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(48)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(
            hidden_sizes=[8], scale_features=False,
            pretrain_epochs=3, finetune_epochs=3,
        )
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        assert model.scaler_ is None
        preds = model.predict(X_tgt)
        assert preds.shape == (50,)

    def test_predictions_all_positive(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(49)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng)
        model = CANNTransfer(hidden_sizes=[16, 8], pretrain_epochs=5, finetune_epochs=5)
        model.fit_source(X_src, y_src)
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert np.all(preds > 0)
        assert np.all(np.isfinite(preds))

    def test_with_exposure_source(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        rng = np.random.default_rng(50)
        X_src, y_src, X_tgt, y_tgt = self._make_data(rng, n_src=200, n_tgt=50)
        exp_src = rng.uniform(0.5, 2.0, 200)
        model = CANNTransfer(hidden_sizes=[8], pretrain_epochs=3, finetune_epochs=3)
        model.fit_source(X_src, y_src, exposure_source=exp_src)
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (50,)

    def test_get_params(self):
        from insurance_transfer.cann_transfer import CANNTransfer
        model = CANNTransfer(hidden_sizes=[32, 16], finetune_strategy="all")
        params = model.get_params()
        assert params["hidden_sizes"] == [32, 16]
        assert params["finetune_strategy"] == "all"
