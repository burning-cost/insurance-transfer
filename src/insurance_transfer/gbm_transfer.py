"""
GBM transfer learning using CatBoost source-as-offset.

The source-as-offset pattern: train a CatBoost model on source data, generate
log-predictions on the target dataset, then use those as a fixed offset (baseline)
when fitting a residual CatBoost model on target data.

This is the practical GBM equivalent of the GLM debiasing step. The source model
contributes the "prior" and the target model learns the residual correction.

Two approaches are supported:
  1. ``init_model``: CatBoost's built-in warm-start — appends new trees to the
     source model forest. Fast but tightly coupled to CatBoost internals.
  2. Manual offset: Compute log-predictions from source model, pass as
     ``baseline`` parameter to CatBoost fit. More interpretable and the default.

CatBoost only. No XGBoost or LightGBM — per project standards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


TransferMode = Literal["offset", "init_model"]


class GBMTransfer(BaseEstimator, RegressorMixin):
    """CatBoost transfer via source-as-offset or warm-start.

    Takes a fitted CatBoost model trained on source data (or source data
    directly) and transfers knowledge to a thin target segment.

    Parameters
    ----------
    source_model:
        A fitted CatBoost model (CatBoostRegressor or CatBoostClassifier).
        Must have a ``predict()`` method that returns predictions in the
        native space (not log-space). The model is used to generate a
        log-offset for target training.
    mode:
        ``'offset'``: generate log-predictions from source model and pass
        as a fixed baseline offset to target CatBoost. The target model
        learns the residual.
        ``'init_model'``: use CatBoost's built-in ``init_model`` parameter
        to continue training from the source model. Faster but less
        interpretable.
    loss_function:
        CatBoost loss function. ``'Poisson'`` for claim counts,
        ``'Tweedie:variance_power=1.5'`` for pure premium.
    catboost_params:
        Additional parameters passed to the target CatBoost model.
        See CatBoost documentation. ``iterations``, ``learning_rate``,
        ``depth`` are common choices.
    cat_features:
        List of categorical feature names or indices. Passed directly
        to CatBoost.
    log_scale_source:
        If True (default for Poisson/Tweedie), the source predictions are
        already on the natural scale and will be log-transformed for offset.
        Set to False if source model already returns log-scale predictions.

    Attributes
    ----------
    target_model_:
        Fitted CatBoost model on target data.
    source_log_offset_train_:
        Log-predictions from source model on target training data.

    Examples
    --------
    >>> import numpy as np
    >>> # (Requires catboost to be installed)
    >>> try:
    ...     from catboost import CatBoostRegressor
    ...     from insurance_transfer import GBMTransfer
    ...     rng = np.random.default_rng(42)
    ...     X_src = rng.standard_normal((500, 4))
    ...     y_src = rng.poisson(np.exp(0.5 * X_src[:, 0] - 0.3 * X_src[:, 1]))
    ...     source = CatBoostRegressor(loss_function='Poisson', iterations=50, verbose=0)
    ...     source.fit(X_src, y_src)
    ...     X_tgt = rng.standard_normal((80, 4)) + 0.3
    ...     y_tgt = rng.poisson(np.exp(0.5 * X_tgt[:, 0] - 0.3 * X_tgt[:, 1]))
    ...     model = GBMTransfer(source_model=source, loss_function='Poisson',
    ...                         catboost_params={'iterations': 50, 'verbose': 0})
    ...     model.fit(X_tgt, y_tgt)
    ...     preds = model.predict(X_tgt)
    ...     assert preds.shape == (80,)
    ... except ImportError:
    ...     pass
    """

    def __init__(
        self,
        source_model: Any,
        mode: TransferMode = "offset",
        loss_function: str = "Poisson",
        catboost_params: Optional[Dict[str, Any]] = None,
        cat_features: Optional[List] = None,
        log_scale_source: bool = True,
    ) -> None:
        self.source_model = source_model
        self.mode = mode
        self.loss_function = loss_function
        self.catboost_params = catboost_params or {}
        self.cat_features = cat_features
        self.log_scale_source = log_scale_source

    def _import_catboost(self):
        try:
            import catboost
            return catboost
        except ImportError as exc:
            raise ImportError(
                "GBMTransfer requires catboost. Install it with: "
                "pip install insurance-transfer[catboost]"
            ) from exc

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        exposure: Optional[NDArray] = None,
        sample_weight: Optional[NDArray] = None,
    ) -> "GBMTransfer":
        """Fit the target GBM model.

        Parameters
        ----------
        X:
            Target feature matrix.
        y:
            Target response vector. For Poisson, this should be raw claim
            counts; exposure is handled separately.
        exposure:
            Exposure vector. For Poisson loss, CatBoost accepts a baseline
            offset. Exposure is combined with source offset as log(exposure).
        sample_weight:
            Optional sample weights.

        Returns
        -------
        self
        """
        cb = self._import_catboost()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[0]

        if exposure is None:
            exposure = np.ones(n)
        exposure = np.asarray(exposure, dtype=np.float64)
        log_exp = np.log(np.maximum(exposure, 1e-10))

        # Compute source predictions
        source_preds = np.asarray(self.source_model.predict(X), dtype=np.float64)
        source_preds = np.maximum(source_preds, 1e-10)

        if self.log_scale_source:
            source_log = np.log(source_preds)
        else:
            source_log = source_preds

        self.source_log_offset_train_ = source_log

        # Combine source log-predictions with exposure offset
        combined_offset = source_log + log_exp

        # Build CatBoost pool and model
        params = {
            "loss_function": self.loss_function,
            **self.catboost_params,
        }

        pool_kwargs: Dict[str, Any] = {"data": X, "label": y}
        if self.cat_features is not None:
            pool_kwargs["cat_features"] = self.cat_features
        if sample_weight is not None:
            pool_kwargs["weight"] = sample_weight

        if self.mode == "offset":
            pool_kwargs["baseline"] = combined_offset
            train_pool = cb.Pool(**pool_kwargs)
            model = cb.CatBoostRegressor(**params)
            model.fit(train_pool)

        elif self.mode == "init_model":
            # Warm-start: continue training from source model
            train_pool = cb.Pool(**pool_kwargs)
            model = cb.CatBoostRegressor(**params)
            model.fit(train_pool, init_model=self.source_model)

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'offset' or 'init_model'.")

        self.target_model_ = model
        return self

    def predict(
        self,
        X: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> NDArray:
        """Predict on new data.

        In offset mode, source predictions are added back. In init_model mode,
        the combined model predicts directly.

        Parameters
        ----------
        X:
            Feature matrix.
        exposure:
            Exposure vector. Defaults to ones.

        Returns
        -------
        Predicted values, shape (n,).
        """
        check_is_fitted(self, ["target_model_"])

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        if exposure is None:
            exposure = np.ones(n)
        exposure = np.asarray(exposure, dtype=np.float64)
        log_exp = np.log(np.maximum(exposure, 1e-10))

        if self.mode == "offset":
            # Source offset
            source_preds = np.asarray(self.source_model.predict(X), dtype=np.float64)
            source_preds = np.maximum(source_preds, 1e-10)
            if self.log_scale_source:
                source_log = np.log(source_preds)
            else:
                source_log = source_preds
            combined_offset = source_log + log_exp

            cb = self._import_catboost()
            pool = cb.Pool(data=X, baseline=combined_offset)
            # CatBoost with Poisson loss and baseline returns log(mu) + baseline
            # The raw prediction is the residual; we need to exponentiate
            raw = np.asarray(self.target_model_.predict(pool), dtype=np.float64)
            # For Poisson with log link: prediction = exp(eta + baseline)
            # CatBoost predict() with baseline gives exp(eta + baseline) in natural space
            return raw

        else:
            # init_model mode: direct prediction
            raw = np.asarray(self.target_model_.predict(X), dtype=np.float64)
            return raw * exposure
