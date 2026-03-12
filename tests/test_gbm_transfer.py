"""Tests for GBMTransfer (CatBoost source-as-offset).

All tests skip if catboost is not installed.
"""

import numpy as np
import pytest

try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CATBOOST_AVAILABLE, reason="catboost not installed"
)


def _make_source_model(rng, n=300, p=4, iterations=30):
    from catboost import CatBoostRegressor
    X = rng.standard_normal((n, p))
    y = rng.poisson(np.exp(0.4 * X[:, 0] - 0.2 * X[:, 1])).astype(float)
    model = CatBoostRegressor(loss_function="Poisson", iterations=iterations, verbose=0)
    model.fit(X, y)
    return model, X, y


class TestGBMTransferImport:
    def test_import_without_catboost_raises(self, monkeypatch):
        """Import should work; error only on fit if catboost missing."""
        from insurance_transfer.gbm_transfer import GBMTransfer
        assert GBMTransfer is not None


class TestGBMTransferFit:
    def test_offset_mode_fit_predict(self):
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(30)
        source, X_src, y_src = _make_source_model(rng)

        X_tgt = rng.standard_normal((80, 4)) + 0.3
        y_tgt = rng.poisson(np.exp(0.4 * X_tgt[:, 0])).astype(float)

        model = GBMTransfer(
            source_model=source,
            mode="offset",
            loss_function="Poisson",
            catboost_params={"iterations": 20, "verbose": 0},
        )
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (80,)
        assert np.all(preds > 0)

    def test_offset_mode_with_exposure(self):
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(31)
        source, _, _ = _make_source_model(rng)

        X_tgt = rng.standard_normal((60, 4))
        exp_tgt = rng.uniform(0.5, 3.0, 60)
        y_tgt = rng.poisson(np.exp(0.3 * X_tgt[:, 0]) * exp_tgt).astype(float)

        model = GBMTransfer(
            source_model=source,
            mode="offset",
            catboost_params={"iterations": 20, "verbose": 0},
        )
        model.fit(X_tgt, y_tgt, exposure=exp_tgt)
        preds = model.predict(X_tgt, exposure=exp_tgt)
        assert preds.shape == (60,)

    def test_source_log_offset_stored(self):
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(32)
        source, _, _ = _make_source_model(rng)

        X_tgt = rng.standard_normal((50, 4))
        y_tgt = rng.poisson(1.0, size=50).astype(float)

        model = GBMTransfer(
            source_model=source,
            mode="offset",
            catboost_params={"iterations": 10, "verbose": 0},
        )
        model.fit(X_tgt, y_tgt)
        assert hasattr(model, "source_log_offset_train_")
        assert model.source_log_offset_train_.shape == (50,)

    def test_target_model_fitted(self):
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(33)
        source, _, _ = _make_source_model(rng)

        X_tgt = rng.standard_normal((50, 4))
        y_tgt = rng.poisson(1.0, size=50).astype(float)

        model = GBMTransfer(
            source_model=source,
            mode="offset",
            catboost_params={"iterations": 10, "verbose": 0},
        )
        model.fit(X_tgt, y_tgt)
        assert hasattr(model, "target_model_")

    def test_predict_requires_fit(self):
        from catboost import CatBoostRegressor
        from insurance_transfer.gbm_transfer import GBMTransfer
        from sklearn.exceptions import NotFittedError
        rng = np.random.default_rng(34)
        source, _, _ = _make_source_model(rng)
        model = GBMTransfer(source_model=source)
        with pytest.raises(NotFittedError):
            model.predict(np.random.randn(10, 4))

    def test_invalid_mode_raises(self):
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(35)
        source, _, _ = _make_source_model(rng)
        X_tgt = rng.standard_normal((30, 4))
        y_tgt = rng.poisson(1.0, size=30).astype(float)
        model = GBMTransfer(source_model=source, mode="invalid_mode")
        with pytest.raises(ValueError, match="mode"):
            model.fit(X_tgt, y_tgt)

    def test_get_params(self):
        from catboost import CatBoostRegressor
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(36)
        source, _, _ = _make_source_model(rng)
        model = GBMTransfer(source_model=source, mode="offset", loss_function="Poisson")
        params = model.get_params()
        assert params["mode"] == "offset"
        assert params["loss_function"] == "Poisson"

    def test_init_model_mode(self):
        """init_model mode: continue training from source CatBoost model."""
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(37)
        source, _, _ = _make_source_model(rng)

        X_tgt = rng.standard_normal((60, 4))
        y_tgt = rng.poisson(1.0, size=60).astype(float)

        model = GBMTransfer(
            source_model=source,
            mode="init_model",
            loss_function="Poisson",
            catboost_params={"iterations": 10, "verbose": 0},
        )
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert preds.shape == (60,)

    def test_predictions_positive(self):
        from insurance_transfer.gbm_transfer import GBMTransfer
        rng = np.random.default_rng(38)
        source, _, _ = _make_source_model(rng)
        X_tgt = rng.standard_normal((40, 4))
        y_tgt = rng.poisson(1.0, size=40).astype(float)
        model = GBMTransfer(
            source_model=source,
            catboost_params={"iterations": 10, "verbose": 0},
        )
        model.fit(X_tgt, y_tgt)
        preds = model.predict(X_tgt)
        assert np.all(preds > 0)
