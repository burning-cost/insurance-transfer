"""
Penalised GLM transfer learning based on Tian & Feng (JASA 2023).

The two-step algorithm:
  1. Pool target and source data, fit an l1-penalised GLM (the "pooled" estimator).
  2. Fine-tune on target only: estimate delta = beta_target - beta_pooled,
     with an l1 penalty on delta so only meaningful adjustments are made.

The pooled step borrows statistical strength from source data. The debiasing
step corrects for any distribution mismatch. Auto-detection identifies and
excludes sources where the transfer direction is harmful (||delta||_1 > threshold).

Reference:
    Tian, Y. and Feng, Y. (2023). Transfer Learning under High-Dimensional
    Generalized Linear Models. Journal of the American Statistical Association,
    118(544), 2684-2697. arXiv: 2105.14328.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


Family = Literal["poisson", "gamma", "gaussian"]


# ---------------------------------------------------------------------------
# Log-likelihood and gradient helpers
# ---------------------------------------------------------------------------


def _poisson_negloglik(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_exposure: NDArray,
    l1_lambda: float,
) -> float:
    eta = X @ beta + log_exposure
    mu = np.exp(np.clip(eta, -30, 30))
    nll = np.mean(mu - y * eta)
    penalty = l1_lambda * np.sum(np.abs(beta))
    return float(nll + penalty)


def _poisson_grad(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_exposure: NDArray,
    l1_lambda: float,
) -> NDArray:
    eta = X @ beta + log_exposure
    mu = np.exp(np.clip(eta, -30, 30))
    residual = mu - y
    grad = X.T @ residual / len(y)
    # Subgradient of l1
    grad += l1_lambda * np.sign(beta)
    return grad


def _gamma_negloglik(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_exposure: NDArray,
    l1_lambda: float,
) -> float:
    eta = X @ beta + log_exposure
    mu = np.exp(np.clip(eta, -30, 30))
    # Gamma log-likelihood with dispersion = 1 (shape=1), up to constants
    nll = np.mean(y / mu + np.log(mu))
    penalty = l1_lambda * np.sum(np.abs(beta))
    return float(nll + penalty)


def _gamma_grad(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_exposure: NDArray,
    l1_lambda: float,
) -> NDArray:
    eta = X @ beta + log_exposure
    mu = np.exp(np.clip(eta, -30, 30))
    residual = 1.0 - y / mu
    grad = X.T @ residual / len(y)
    grad += l1_lambda * np.sign(beta)
    return grad


def _gaussian_negloglik(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_exposure: NDArray,
    l1_lambda: float,
) -> float:
    eta = X @ beta
    nll = 0.5 * np.mean((y - eta) ** 2)
    penalty = l1_lambda * np.sum(np.abs(beta))
    return float(nll + penalty)


def _gaussian_grad(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_exposure: NDArray,
    l1_lambda: float,
) -> NDArray:
    eta = X @ beta
    grad = -X.T @ (y - eta) / len(y)
    grad += l1_lambda * np.sign(beta)
    return grad


_NLL_FNS = {
    "poisson": (_poisson_negloglik, _poisson_grad),
    "gamma": (_gamma_negloglik, _gamma_grad),
    "gaussian": (_gaussian_negloglik, _gaussian_grad),
}


def _fit_penalised_glm(
    X: NDArray,
    y: NDArray,
    log_exposure: NDArray,
    l1_lambda: float,
    family: Family,
    beta_init: Optional[NDArray] = None,
) -> NDArray:
    """Fit a penalised GLM using L-BFGS-B with smoothed l1 (approximated via bounds)."""
    p = X.shape[1]
    if beta_init is None:
        beta_init = np.zeros(p)

    nll_fn, grad_fn = _NLL_FNS[family]

    # Smooth approximation: split beta = beta_pos - beta_neg, both >= 0
    # This converts l1 penalty to a bound-constrained problem
    # Augmented variable: theta = [beta_pos, beta_neg], size 2p
    def augmented_obj(theta: NDArray) -> float:
        beta_pos = theta[:p]
        beta_neg = theta[p:]
        beta = beta_pos - beta_neg
        return nll_fn(beta, X, y, log_exposure, l1_lambda)

    def augmented_grad(theta: NDArray) -> NDArray:
        beta_pos = theta[:p]
        beta_neg = theta[p:]
        beta = beta_pos - beta_neg
        g = grad_fn(beta, X, y, log_exposure, l1_lambda)
        # Gradient wrt beta_pos = g; wrt beta_neg = -g; but penalty already in grad
        # Recompute without penalty for clean split
        nll_only_pos, nll_only_neg = theta[:p] * 0, theta[p:] * 0
        # Actually: d/d(beta_pos) obj = grad_beta * 1 = g, d/d(beta_neg) = g * (-1)
        return np.concatenate([g, -g])

    # Initialise
    beta_pos0 = np.maximum(beta_init, 0.0)
    beta_neg0 = np.maximum(-beta_init, 0.0)
    theta0 = np.concatenate([beta_pos0, beta_neg0])
    bounds = [(0.0, None)] * (2 * p)

    result = minimize(
        augmented_obj,
        theta0,
        jac=augmented_grad,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    beta_pos = result.x[:p]
    beta_neg = result.x[p:]
    return beta_pos - beta_neg


class GLMTransfer(BaseEstimator, RegressorMixin):
    """Two-step penalised GLM transfer (Tian & Feng, JASA 2023).

    Borrows strength from one or more source datasets and corrects for
    covariate shift using a debiasing step on target data only.

    Step 1 (pooling): Pool source + target data. Fit an l1-penalised GLM
    with penalty ``lambda_pool``. This produces a pooled estimator that
    benefits from larger sample size.

    Step 2 (debiasing): Estimate the shift delta = beta_target - beta_pooled
    using target data only, with penalty ``lambda_debias`` on delta. The final
    coefficient is beta_pooled + delta_hat.

    Auto source detection: if ||delta_hat||_1 > ``delta_threshold`` for a given
    source, that source is excluded and the pooling step is re-run without it.

    Parameters
    ----------
    family:
        GLM family. ``'poisson'`` for claim counts (most common in pricing),
        ``'gamma'`` for severity, ``'gaussian'`` for continuous targets.
    lambda_pool:
        L1 penalty in the pooling step. Controls how much regularisation
        is applied when fitting on pooled source+target data.
    lambda_debias:
        L1 penalty in the debiasing step. Controls how aggressively the
        estimator corrects for source-target mismatch.
    delta_threshold:
        Auto-detection threshold. Sources with ||delta||_1 > this value
        are excluded. ``None`` disables auto-detection (all sources kept).
    scale_features:
        Whether to standardise continuous features before fitting.
        Recommended when features are on very different scales.
    fit_intercept:
        Whether to fit an intercept term.

    Attributes
    ----------
    coef_:
        Final coefficient vector (after debiasing).
    intercept_:
        Intercept value (zero if fit_intercept=False).
    beta_pooled_:
        Pooled estimator from step 1.
    delta_:
        Debiasing correction from step 2.
    included_sources_:
        Indices of source datasets used (after auto-detection filtering).
    scaler_:
        Fitted StandardScaler (or None if scale_features=False).

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_transfer import GLMTransfer
    >>> rng = np.random.default_rng(0)
    >>> n_src, n_tgt, p = 2000, 200, 8
    >>> X_src = rng.standard_normal((n_src, p))
    >>> exposure_src = np.ones(n_src)
    >>> true_beta = np.array([0.3, -0.2, 0.1, 0.0, 0.15, -0.1, 0.05, 0.0])
    >>> y_src = rng.poisson(np.exp(X_src @ true_beta))
    >>> X_tgt = rng.standard_normal((n_tgt, p)) + 0.3
    >>> exposure_tgt = np.ones(n_tgt)
    >>> y_tgt = rng.poisson(np.exp(X_tgt @ true_beta))
    >>> model = GLMTransfer(family='poisson', lambda_pool=0.01, lambda_debias=0.05)
    >>> model.fit(X_tgt, y_tgt, exposure_tgt, X_source=X_src,
    ...           y_source=y_src, exposure_source=exposure_src)
    GLMTransfer(...)
    >>> preds = model.predict(X_tgt, exposure_tgt)
    >>> preds.shape
    (200,)
    """

    def __init__(
        self,
        family: Family = "poisson",
        lambda_pool: float = 0.01,
        lambda_debias: float = 0.05,
        delta_threshold: Optional[float] = None,
        scale_features: bool = True,
        fit_intercept: bool = True,
    ) -> None:
        self.family = family
        self.lambda_pool = lambda_pool
        self.lambda_debias = lambda_debias
        self.delta_threshold = delta_threshold
        self.scale_features = scale_features
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X: NDArray) -> NDArray:
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        exposure: Optional[NDArray] = None,
        X_source: Optional[Union[NDArray, List[NDArray]]] = None,
        y_source: Optional[Union[NDArray, List[NDArray]]] = None,
        exposure_source: Optional[Union[NDArray, List[NDArray]]] = None,
    ) -> "GLMTransfer":
        """Fit the transfer GLM.

        Parameters
        ----------
        X:
            Target feature matrix, shape (n_target, p).
        y:
            Target response vector, shape (n_target,).
        exposure:
            Target exposure vector, shape (n_target,). Defaults to ones.
            Used as offset on log scale for Poisson/Gamma.
        X_source:
            Source feature matrix or list of matrices. If None, falls back
            to a standard penalised GLM on target data only.
        y_source:
            Source response vector or list matching X_source.
        exposure_source:
            Source exposure or list matching X_source. Defaults to ones.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_tgt = X.shape[0]

        if exposure is None:
            exposure = np.ones(n_tgt)
        exposure = np.asarray(exposure, dtype=np.float64)
        log_exp_tgt = np.log(np.maximum(exposure, 1e-10))

        # Normalise source inputs to lists
        if X_source is None:
            sources_X: List[NDArray] = []
            sources_y: List[NDArray] = []
            sources_exp: List[NDArray] = []
        elif isinstance(X_source, np.ndarray) and X_source.ndim == 2:
            sources_X = [np.asarray(X_source, dtype=np.float64)]
            sources_y = [np.asarray(y_source, dtype=np.float64)]
            exp_src = exposure_source if exposure_source is not None else np.ones(X_source.shape[0])
            sources_exp = [np.asarray(exp_src, dtype=np.float64)]
        else:
            sources_X = [np.asarray(xs, dtype=np.float64) for xs in X_source]
            sources_y = [np.asarray(ys, dtype=np.float64) for ys in y_source]
            if exposure_source is None:
                sources_exp = [np.ones(xs.shape[0]) for xs in sources_X]
            else:
                sources_exp = [np.asarray(es, dtype=np.float64) for es in exposure_source]

        # Fit scaler on target (we scale consistently across source/target)
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            sources_X_scaled = [self.scaler_.transform(xs) for xs in sources_X]
        else:
            self.scaler_ = None
            X_scaled = X
            sources_X_scaled = sources_X

        X_aug = self._add_intercept(X_scaled)
        sources_X_aug = [self._add_intercept(xs) for xs in sources_X_scaled]

        # Step 1: Pool and fit
        included = list(range(len(sources_X)))
        beta_pooled = self._pooled_fit(
            X_aug, y, log_exp_tgt,
            [sources_X_aug[i] for i in included],
            [sources_y[i] for i in included],
            [sources_exp[i] for i in included],
        )

        # Step 2: Debias on target
        delta = self._debias_fit(X_aug, y, log_exp_tgt, beta_pooled)

        # Auto source detection if threshold set
        if self.delta_threshold is not None and sources_X:
            delta_norm = np.sum(np.abs(delta))
            if delta_norm > self.delta_threshold:
                # Re-run excluding sources one at a time (greedy approach)
                included = self._auto_detect_sources(
                    X_aug, y, log_exp_tgt,
                    sources_X_aug, sources_y, sources_exp,
                )
                beta_pooled = self._pooled_fit(
                    X_aug, y, log_exp_tgt,
                    [sources_X_aug[i] for i in included],
                    [sources_y[i] for i in included],
                    [sources_exp[i] for i in included],
                )
                delta = self._debias_fit(X_aug, y, log_exp_tgt, beta_pooled)

        self.beta_pooled_ = beta_pooled
        self.delta_ = delta
        self.included_sources_ = included

        final_beta = beta_pooled + delta
        if self.fit_intercept:
            self.intercept_ = float(final_beta[0])
            self.coef_ = final_beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = final_beta

        return self

    def _pooled_fit(
        self,
        X_tgt_aug: NDArray,
        y_tgt: NDArray,
        log_exp_tgt: NDArray,
        sources_X_aug: List[NDArray],
        sources_y: List[NDArray],
        sources_exp: List[NDArray],
    ) -> NDArray:
        """Fit l1-penalised GLM on pooled source + target data."""
        if sources_X_aug:
            log_exp_sources = [np.log(np.maximum(e, 1e-10)) for e in sources_exp]
            X_pool = np.vstack([X_tgt_aug] + sources_X_aug)
            y_pool = np.concatenate([y_tgt] + sources_y)
            log_exp_pool = np.concatenate([log_exp_tgt] + log_exp_sources)
        else:
            X_pool = X_tgt_aug
            y_pool = y_tgt
            log_exp_pool = log_exp_tgt

        return _fit_penalised_glm(X_pool, y_pool, log_exp_pool, self.lambda_pool, self.family)

    def _debias_fit(
        self,
        X_aug: NDArray,
        y: NDArray,
        log_exp: NDArray,
        beta_pooled: NDArray,
    ) -> NDArray:
        """Estimate delta = beta_target - beta_pooled on target data only."""
        # Offset: absorb beta_pooled contribution into the linear predictor
        # For Poisson: eta = X @ (beta_pooled + delta) + log_exp
        #            = X @ delta + (X @ beta_pooled + log_exp)
        offset = X_aug @ beta_pooled + log_exp
        # We fit delta with a zero initialisation and the offset as exposure proxy
        # For Gaussian family the offset is different (additive in eta directly)
        p = X_aug.shape[1]

        if self.family == "gaussian":
            # y_adjusted = y - X @ beta_pooled; fit l1 regression on residual
            y_adj = y - X_aug @ beta_pooled

            def obj(delta: NDArray) -> float:
                nll = 0.5 * np.mean((y_adj - X_aug @ delta) ** 2)
                return nll + self.lambda_debias * np.sum(np.abs(delta))

            def grad(delta: NDArray) -> NDArray:
                g = -X_aug.T @ (y_adj - X_aug @ delta) / len(y_adj)
                return g + self.lambda_debias * np.sign(delta)

        else:
            # Use offset = X_aug @ beta_pooled + log_exp as the fixed term
            nll_fn, grad_fn = _NLL_FNS[self.family]

            def obj(delta: NDArray) -> float:
                return nll_fn(delta, X_aug, y, offset - log_exp + log_exp, self.lambda_debias)
                # Simplified: the offset replaces log_exposure in the original problem

            def obj(delta: NDArray) -> float:  # type: ignore[no-redef]
                eta = X_aug @ delta + offset
                if self.family == "poisson":
                    mu = np.exp(np.clip(eta, -30, 30))
                    nll = np.mean(mu - y * eta)
                else:  # gamma
                    mu = np.exp(np.clip(eta, -30, 30))
                    nll = np.mean(y / mu + np.log(mu))
                return float(nll + self.lambda_debias * np.sum(np.abs(delta)))

            def grad(delta: NDArray) -> NDArray:  # type: ignore[no-redef]
                eta = X_aug @ delta + offset
                if self.family == "poisson":
                    mu = np.exp(np.clip(eta, -30, 30))
                    r = mu - y
                else:  # gamma
                    mu = np.exp(np.clip(eta, -30, 30))
                    r = 1.0 - y / mu
                return X_aug.T @ r / len(y) + self.lambda_debias * np.sign(delta)

        # Augmented variable approach for l1
        theta0 = np.zeros(2 * p)

        def aug_obj(theta: NDArray) -> float:
            d = theta[:p] - theta[p:]
            return obj(d)

        def aug_grad(theta: NDArray) -> NDArray:
            d = theta[:p] - theta[p:]
            g = grad(d)
            return np.concatenate([g, -g])

        bounds = [(0.0, None)] * (2 * p)
        result = minimize(
            aug_obj, theta0, jac=aug_grad, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 500, "ftol": 1e-9}
        )
        delta_pos = result.x[:p]
        delta_neg = result.x[p:]
        return delta_pos - delta_neg

    def _auto_detect_sources(
        self,
        X_aug: NDArray,
        y: NDArray,
        log_exp: NDArray,
        sources_X_aug: List[NDArray],
        sources_y: List[NDArray],
        sources_exp: List[NDArray],
    ) -> List[int]:
        """Identify beneficial sources via greedy delta-norm screening."""
        n_sources = len(sources_X_aug)
        if n_sources == 0:
            return []

        # Try each source individually and check delta norm
        good_sources = []
        for i in range(n_sources):
            bp = self._pooled_fit(
                X_aug, y, log_exp,
                [sources_X_aug[i]], [sources_y[i]], [sources_exp[i]],
            )
            delta = self._debias_fit(X_aug, y, log_exp, bp)
            delta_norm = float(np.sum(np.abs(delta)))
            if delta_norm <= self.delta_threshold:  # type: ignore[operator]
                good_sources.append(i)

        return good_sources if good_sources else []

    def predict(
        self,
        X: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> NDArray:
        """Predict expected values.

        Parameters
        ----------
        X:
            Feature matrix, shape (n, p).
        exposure:
            Exposure vector. Defaults to ones. For Gaussian family,
            exposure has no effect.

        Returns
        -------
        Array of predictions, shape (n,).
        """
        check_is_fitted(self, ["coef_"])
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        if exposure is None:
            exposure = np.ones(n)
        exposure = np.asarray(exposure, dtype=np.float64)
        log_exp = np.log(np.maximum(exposure, 1e-10))

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        X_aug = self._add_intercept(X)
        # Reconstruct full beta
        if self.fit_intercept:
            beta = np.concatenate([[self.intercept_], self.coef_])
        else:
            beta = self.coef_

        eta = X_aug @ beta
        if self.family in ("poisson", "gamma"):
            return np.exp(eta + log_exp)
        else:
            return eta
