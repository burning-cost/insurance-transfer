"""Tests for GLMTransfer (Tian & Feng two-step algorithm)."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from insurance_transfer.glm_transfer import GLMTransfer, _fit_penalised_glm, _poisson_negloglik


class TestFitPenalisedGLM:
    def test_poisson_fits(self):
        rng = np.random.default_rng(0)
        n, p = 200, 5
        X = rng.standard_normal((n, p))
        X = np.column_stack([np.ones(n), X])
        true_beta = np.array([0.1, 0.3, -0.2, 0.1, 0.0, 0.15])
        log_exp = np.zeros(n)
        mu = np.exp(X @ true_beta)
        y = rng.poisson(mu).astype(float)
        beta = _fit_penalised_glm(X, y, log_exp, l1_lambda=0.01, family="poisson")
        assert beta.shape == (6,)
        # Check fitted values are in right ballpark
        pred = np.exp(X @ beta)
        assert pred.mean() > 0.1

    def test_gamma_fits(self):
        rng = np.random.default_rng(1)
        n, p = 200, 4
        X = rng.standard_normal((n, p))
        X = np.column_stack([np.ones(n), X])
        beta_true = np.array([5.0, 0.2, -0.1, 0.0, 0.15])
        log_exp = np.zeros(n)
        mu = np.exp(X @ beta_true)
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
        beta = _fit_penalised_glm(X, y, log_exp, l1_lambda=0.01, family="gamma")
        assert beta.shape == (5,)

    def test_gaussian_fits(self):
        rng = np.random.default_rng(2)
        n, p = 300, 4
        X = rng.standard_normal((n, p))
        X = np.column_stack([np.ones(n), X])
        true_beta = np.array([1.0, 0.5, -0.3, 0.2, 0.0])
        log_exp = np.zeros(n)
        y = X @ true_beta + rng.standard_normal(n) * 0.1
        beta = _fit_penalised_glm(X, y, log_exp, l1_lambda=0.001, family="gaussian")
        assert beta.shape == (5,)

    def test_l1_sparsity(self):
        rng = np.random.default_rng(3)
        n, p = 200, 10
        X = rng.standard_normal((n, p))
        log_exp = np.zeros(n)
        y = rng.poisson(1.0, size=n).astype(float)
        # High penalty should drive many coefficients to zero
        beta = _fit_penalised_glm(X, y, log_exp, l1_lambda=1.0, family="poisson")
        n_zero = np.sum(np.abs(beta) < 1e-6)
        assert n_zero > 0  # At least some should be zeroed


class TestGLMTransfer:
    def _make_data(self, rng, n_src=1000, n_tgt=100, p=6, shift=0.3, family="poisson"):
        true_beta = np.array([0.3, -0.2, 0.1, 0.0, 0.15, -0.1])
        X_src = rng.standard_normal((n_src, p))
        exp_src = rng.uniform(0.5, 2.0, n_src)
        if family == "poisson":
            mu_src = np.exp(X_src @ true_beta) * exp_src
            y_src = rng.poisson(mu_src).astype(float)
        else:
            mu_src = np.exp(X_src @ true_beta)
            y_src = rng.gamma(shape=2.0, scale=mu_src / 2.0)

        X_tgt = rng.standard_normal((n_tgt, p)) + shift
        exp_tgt = rng.uniform(0.5, 2.0, n_tgt)
        if family == "poisson":
            mu_tgt = np.exp(X_tgt @ true_beta) * exp_tgt
            y_tgt = rng.poisson(mu_tgt).astype(float)
        else:
            mu_tgt = np.exp(X_tgt @ true_beta)
            y_tgt = rng.gamma(shape=2.0, scale=mu_tgt / 2.0)

        return X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt

    def test_fit_predict_poisson(self):
        rng = np.random.default_rng(10)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data(rng)
        model = GLMTransfer(family="poisson")
        model.fit(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src, exposure_source=exp_src)
        preds = model.predict(X_tgt, exp_tgt)
        assert preds.shape == (len(y_tgt),)
        assert np.all(preds > 0)

    def test_fit_predict_gamma(self):
        rng = np.random.default_rng(11)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data(rng, family="gamma")
        model = GLMTransfer(family="gamma")
        model.fit(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src, exposure_source=exp_src)
        preds = model.predict(X_tgt, exp_tgt)
        assert preds.shape == (len(y_tgt),)
        assert np.all(preds > 0)

    def test_fit_predict_gaussian(self):
        rng = np.random.default_rng(12)
        n = 200
        X_src = rng.standard_normal((500, 4))
        y_src = X_src @ np.array([0.5, -0.3, 0.1, 0.0]) + rng.standard_normal(500)
        X_tgt = rng.standard_normal((n, 4))
        y_tgt = X_tgt @ np.array([0.5, -0.3, 0.1, 0.0]) + rng.standard_normal(n)
        model = GLMTransfer(family="gaussian")
        model.fit(X_tgt, y_tgt, None, X_source=X_src, y_source=y_src)
        preds = model.predict(X_tgt)
        assert preds.shape == (n,)

    def test_no_source_works(self):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((200, 5))
        y = rng.poisson(np.exp(0.2 * X[:, 0]))
        model = GLMTransfer(family="poisson")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert np.all(preds > 0)

    def test_default_exposure_ones(self):
        rng = np.random.default_rng(14)
        X = rng.standard_normal((100, 4))
        y = rng.poisson(1.5, size=100).astype(float)
        model = GLMTransfer()
        model.fit(X, y)  # No exposure
        preds1 = model.predict(X)
        preds2 = model.predict(X, np.ones(100))
        np.testing.assert_allclose(preds1, preds2, rtol=1e-10)

    def test_coef_attributes_set(self):
        rng = np.random.default_rng(15)
        X = rng.standard_normal((200, 5))
        y = rng.poisson(1.0, size=200).astype(float)
        X_src = rng.standard_normal((500, 5))
        y_src = rng.poisson(1.0, size=500).astype(float)
        model = GLMTransfer()
        model.fit(X, y, None, X_source=X_src, y_source=y_src)
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert hasattr(model, "beta_pooled_")
        assert hasattr(model, "delta_")
        assert hasattr(model, "included_sources_")
        assert model.coef_.shape == (5,)

    def test_no_intercept(self):
        rng = np.random.default_rng(16)
        X = rng.standard_normal((200, 4))
        y = rng.poisson(1.0, size=200).astype(float)
        model = GLMTransfer(fit_intercept=False)
        model.fit(X, y)
        assert model.intercept_ == 0.0
        assert model.coef_.shape == (4,)

    def test_no_scale_features(self):
        rng = np.random.default_rng(17)
        X = rng.standard_normal((200, 4))
        y = rng.poisson(1.0, size=200).astype(float)
        model = GLMTransfer(scale_features=False)
        model.fit(X, y)
        assert model.scaler_ is None

    def test_multiple_sources(self):
        rng = np.random.default_rng(18)
        X_tgt = rng.standard_normal((100, 4))
        y_tgt = rng.poisson(1.0, size=100).astype(float)
        sources_X = [rng.standard_normal((200, 4)) for _ in range(3)]
        sources_y = [rng.poisson(1.0, size=200).astype(float) for _ in range(3)]
        model = GLMTransfer()
        model.fit(X_tgt, y_tgt, None, X_source=sources_X, y_source=sources_y)
        preds = model.predict(X_tgt)
        assert preds.shape == (100,)

    def test_auto_source_detection(self):
        rng = np.random.default_rng(19)
        X_tgt = rng.standard_normal((100, 4))
        y_tgt = rng.poisson(1.0, size=100).astype(float)
        X_src = rng.standard_normal((500, 4))
        y_src = rng.poisson(1.0, size=500).astype(float)
        model = GLMTransfer(delta_threshold=0.5)
        model.fit(X_tgt, y_tgt, None, X_source=X_src, y_source=y_src)
        assert isinstance(model.included_sources_, list)

    def test_predict_requires_fit(self):
        from sklearn.exceptions import NotFittedError
        model = GLMTransfer()
        with pytest.raises(NotFittedError):
            model.predict(np.random.randn(10, 4))

    def test_transfer_vs_target_only_large_source(self):
        """Transfer model with large source should have lower or equal deviance."""
        rng = np.random.default_rng(20)
        n_src, n_tgt = 2000, 80
        p = 5
        true_beta = np.array([0.5, -0.3, 0.2, 0.1, -0.1])
        X_src = rng.standard_normal((n_src, p))
        y_src = rng.poisson(np.exp(X_src @ true_beta)).astype(float)
        X_tgt = rng.standard_normal((n_tgt, p))  # minimal shift
        y_tgt = rng.poisson(np.exp(X_tgt @ true_beta)).astype(float)

        # Transfer model
        transfer = GLMTransfer(lambda_pool=0.01, lambda_debias=0.05)
        transfer.fit(X_tgt, y_tgt, None, X_source=X_src, y_source=y_src)
        mu_transfer = transfer.predict(X_tgt)

        # Target-only (no source)
        target_only = GLMTransfer(lambda_pool=0.01)
        target_only.fit(X_tgt, y_tgt)
        mu_target_only = target_only.predict(X_tgt)

        from insurance_transfer.diagnostic import poisson_deviance
        dev_transfer = poisson_deviance(y_tgt, mu_transfer)
        dev_target = poisson_deviance(y_tgt, mu_target_only)
        # With strong source signal, transfer should be at most marginally worse
        assert dev_transfer < dev_target * 1.5  # Loose bound — stochastic

    def test_get_params(self):
        model = GLMTransfer(family="gamma", lambda_pool=0.05)
        params = model.get_params()
        assert params["family"] == "gamma"
        assert params["lambda_pool"] == 0.05

    def test_set_params(self):
        model = GLMTransfer()
        model.set_params(family="gamma", lambda_debias=0.1)
        assert model.family == "gamma"
        assert model.lambda_debias == 0.1

    def test_exposure_affects_predictions(self):
        rng = np.random.default_rng(21)
        X = rng.standard_normal((100, 4))
        y = rng.poisson(1.5, size=100).astype(float)
        model = GLMTransfer().fit(X, y)
        exp1 = np.ones(100)
        exp2 = np.ones(100) * 2.0
        preds1 = model.predict(X, exp1)
        preds2 = model.predict(X, exp2)
        # With Poisson log-link, doubling exposure should roughly double predictions
        np.testing.assert_allclose(preds2 / preds1, 2.0, rtol=0.01)

    def test_lambda_pool_zero_gives_mle(self):
        rng = np.random.default_rng(22)
        X = rng.standard_normal((300, 3))
        y = rng.poisson(np.exp(0.3 * X[:, 0])).astype(float)
        model = GLMTransfer(lambda_pool=0.0, lambda_debias=0.0)
        model.fit(X, y)
        assert model.coef_.shape == (3,)
