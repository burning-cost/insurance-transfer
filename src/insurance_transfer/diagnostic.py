"""
Negative transfer diagnostics for insurance transfer models.

Negative transfer happens when the source model actively hurts performance
on the target segment. This module detects it by comparing three models:

  1. Transfer model (what you built)
  2. Target-only model (baseline — fit only on target data, no transfer)
  3. Source-only model applied to target (worst case for severe shift)

The Negative Transfer Gap (NTG) measures how much worse the transfer model
is compared to target-only:
  NTG = Deviance(transfer) - Deviance(target_only)

NTG > 0 means the transfer model is worse than just fitting on target data.
NTG < 0 means transfer helped.

Poisson deviance is the primary metric for frequency models:
  D(y, mu) = 2 * sum(y * log(y/mu) - (y - mu))

We also report per-feature deviance-weighted residuals to identify where
the transfer model fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class TransferDiagnosticResult:
    """Diagnostic comparison of transfer vs baseline models.

    Attributes
    ----------
    poisson_deviance_transfer:
        Mean Poisson deviance of the transfer model on test data.
    poisson_deviance_target_only:
        Mean Poisson deviance of a target-only model (no transfer).
    poisson_deviance_source_only:
        Mean Poisson deviance of the source model applied directly to target.
        None if not computed.
    ntg:
        Negative Transfer Gap = deviance_transfer - deviance_target_only.
        Positive means transfer hurt performance.
    ntg_relative:
        NTG as a percentage of target-only deviance.
    transfer_is_beneficial:
        True if NTG < 0 (transfer helped).
    per_feature_analysis:
        Dict mapping feature index to mean squared residual for the transfer
        model. High values indicate features where transfer causes errors.
    n_test:
        Number of test observations used.
    """

    poisson_deviance_transfer: float
    poisson_deviance_target_only: float
    poisson_deviance_source_only: Optional[float]
    ntg: float
    ntg_relative: float
    transfer_is_beneficial: bool
    per_feature_analysis: Dict[int, float] = field(default_factory=dict)
    n_test: int = 0

    def __repr__(self) -> str:
        direction = "beneficial" if self.transfer_is_beneficial else "HARMFUL (negative transfer)"
        return (
            f"TransferDiagnosticResult(\n"
            f"  transfer={self.transfer_is_beneficial} [{direction}]\n"
            f"  deviance_transfer={self.poisson_deviance_transfer:.4f}\n"
            f"  deviance_target_only={self.poisson_deviance_target_only:.4f}\n"
            f"  NTG={self.ntg:+.4f} ({self.ntg_relative:+.1f}%)\n"
            f"  n_test={self.n_test}\n"
            f")"
        )


def poisson_deviance(y: NDArray, mu: NDArray) -> float:
    """Mean Poisson deviance: 2 * mean(y * log(y/mu) - (y - mu)).

    Observations with y=0 contribute 2*mu to the deviance (since 0*log(0) = 0
    by convention). This is the standard actuarial deviance formula.

    Parameters
    ----------
    y:
        Observed claim counts, shape (n,).
    mu:
        Predicted expected counts, shape (n,).

    Returns
    -------
    Scalar mean deviance.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.maximum(mu, 1e-10)

    # y * log(y/mu): where y=0, contribution is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(y > 0, y * np.log(y / mu), 0.0)

    deviance = 2.0 * (log_term - (y - mu))
    return float(np.mean(deviance))


def gamma_deviance(y: NDArray, mu: NDArray) -> float:
    """Mean Gamma deviance: 2 * mean(-log(y/mu) + (y/mu) - 1).

    Parameters
    ----------
    y:
        Observed severity values, shape (n,).
    mu:
        Predicted severity, shape (n,).

    Returns
    -------
    Scalar mean deviance.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.maximum(mu, 1e-10)
    y = np.maximum(y, 1e-10)
    deviance = 2.0 * (-np.log(y / mu) + (y / mu) - 1.0)
    return float(np.mean(deviance))


class NegativeTransferDiagnostic:
    """Compare transfer model vs baselines to detect harmful transfer.

    This class does not fit any models itself. It takes pre-fitted model
    objects with ``predict()`` methods and evaluates them on held-out
    target data.

    Parameters
    ----------
    metric:
        Loss metric to use. ``'poisson_deviance'`` (default) or
        ``'gamma_deviance'``. Custom callables (y, mu) -> float also accepted.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_transfer.diagnostic import NegativeTransferDiagnostic, poisson_deviance
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 4))
    >>> y = rng.poisson(np.exp(0.2 * X[:, 0]))
    >>> exposure = np.ones(100)
    >>>
    >>> # Mock models with predict methods
    >>> class ConstModel:
    ...     def __init__(self, val): self.val = val
    ...     def predict(self, X, exposure=None): return np.full(X.shape[0], self.val)
    >>>
    >>> diag = NegativeTransferDiagnostic()
    >>> result = diag.evaluate(
    ...     X, y, exposure,
    ...     transfer_model=ConstModel(1.0),
    ...     target_only_model=ConstModel(1.1),
    ... )
    >>> isinstance(result.ntg, float)
    True
    """

    def __init__(self, metric: str = "poisson_deviance") -> None:
        self.metric = metric

    def _get_metric_fn(self) -> Callable:
        if self.metric == "poisson_deviance":
            return poisson_deviance
        elif self.metric == "gamma_deviance":
            return gamma_deviance
        elif callable(self.metric):
            return self.metric
        else:
            raise ValueError(f"Unknown metric: {self.metric!r}")

    def _call_predict(self, model: Any, X: NDArray, exposure: NDArray) -> NDArray:
        """Call model.predict with or without exposure argument."""
        try:
            return np.asarray(model.predict(X, exposure), dtype=np.float64)
        except TypeError:
            return np.asarray(model.predict(X), dtype=np.float64)

    def evaluate(
        self,
        X_test: NDArray,
        y_test: NDArray,
        exposure_test: Optional[NDArray],
        transfer_model: Any,
        target_only_model: Any,
        source_only_model: Optional[Any] = None,
        feature_names: Optional[List] = None,
    ) -> TransferDiagnosticResult:
        """Evaluate transfer model vs baselines.

        Parameters
        ----------
        X_test:
            Test feature matrix.
        y_test:
            Test response (observed values).
        exposure_test:
            Test exposure. Defaults to ones.
        transfer_model:
            The transfer learning model (must have ``predict()``).
        target_only_model:
            A model trained only on target data (must have ``predict()``).
        source_only_model:
            The source model applied directly to target (optional).
        feature_names:
            Column names for the per-feature analysis output.

        Returns
        -------
        TransferDiagnosticResult
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)
        n = X_test.shape[0]

        if exposure_test is None:
            exposure_test = np.ones(n)
        exposure_test = np.asarray(exposure_test, dtype=np.float64)

        metric_fn = self._get_metric_fn()

        mu_transfer = self._call_predict(transfer_model, X_test, exposure_test)
        mu_target_only = self._call_predict(target_only_model, X_test, exposure_test)

        dev_transfer = metric_fn(y_test, mu_transfer)
        dev_target_only = metric_fn(y_test, mu_target_only)

        dev_source_only = None
        if source_only_model is not None:
            mu_source_only = self._call_predict(source_only_model, X_test, exposure_test)
            dev_source_only = metric_fn(y_test, mu_source_only)

        ntg = dev_transfer - dev_target_only
        ntg_rel = 100.0 * ntg / max(abs(dev_target_only), 1e-10)

        # Per-feature analysis: residual pattern
        per_feature: Dict[int, float] = {}
        residuals = mu_transfer - y_test
        n_cols = X_test.shape[1]
        for col in range(n_cols):
            col_vals = X_test[:, col]
            # Correlation-weighted residual squared
            per_feature[col] = float(np.mean((residuals * col_vals) ** 2))

        # Use feature names as keys if provided
        if feature_names is not None and len(feature_names) == n_cols:
            per_feature = {
                str(feature_names[i]): v for i, v in per_feature.items()
            }

        return TransferDiagnosticResult(
            poisson_deviance_transfer=dev_transfer,
            poisson_deviance_target_only=dev_target_only,
            poisson_deviance_source_only=dev_source_only,
            ntg=ntg,
            ntg_relative=ntg_rel,
            transfer_is_beneficial=(ntg < 0),
            per_feature_analysis=per_feature,
            n_test=n,
        )

    def summary_table(self, result: TransferDiagnosticResult) -> str:
        """Format a text summary of the diagnostic result.

        Parameters
        ----------
        result:
            Output from ``evaluate()``.

        Returns
        -------
        Formatted string for display.
        """
        lines = [
            "Transfer Learning Diagnostic",
            "=" * 40,
            f"{'Model':<25} {'Deviance':>10}",
            "-" * 40,
            f"{'Transfer model':<25} {result.poisson_deviance_transfer:>10.4f}",
            f"{'Target-only baseline':<25} {result.poisson_deviance_target_only:>10.4f}",
        ]
        if result.poisson_deviance_source_only is not None:
            lines.append(
                f"{'Source-only model':<25} {result.poisson_deviance_source_only:>10.4f}"
            )
        lines += [
            "-" * 40,
            f"{'NTG':<25} {result.ntg:>+10.4f} ({result.ntg_relative:>+.1f}%)",
            f"{'Transfer beneficial?':<25} {'Yes' if result.transfer_is_beneficial else 'No':>10}",
            "=" * 40,
        ]
        return "\n".join(lines)
