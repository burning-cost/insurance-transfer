"""
TransferPipeline: orchestrates the full transfer learning workflow.

Sequence:
  1. CovariateShiftTest — check if source and target distributions differ.
  2. Method selection — choose GLM, GBM, or CANN based on config or shift result.
  3. Fit the selected transfer estimator.
  4. NegativeTransferDiagnostic — evaluate on held-out target data.

The pipeline is opinionated: it runs the shift test and warns loudly if the
shift is severe (p < 0.01). It also runs diagnostics automatically unless
you disable them.

Use the pipeline when you want to run the full workflow without manually
chaining the components. Use the individual components when you want more
control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from insurance_transfer.shift import CovariateShiftTest, ShiftTestResult
from insurance_transfer.glm_transfer import GLMTransfer
from insurance_transfer.diagnostic import NegativeTransferDiagnostic, TransferDiagnosticResult


Method = Literal["glm", "gbm", "cann", "auto"]


@dataclass
class PipelineResult:
    """Full result from a TransferPipeline run.

    Attributes
    ----------
    method_used:
        Which transfer method was selected/fitted.
    shift_result:
        CovariateShiftTest output. None if shift_test=False.
    diagnostic_result:
        NegativeTransferDiagnostic output. None if run_diagnostic=False.
    model:
        Fitted transfer model.
    shift_p_value:
        Convenience access to the shift test p-value.
    transfer_is_beneficial:
        Convenience access to diagnostic result.
    """

    method_used: str
    shift_result: Optional[ShiftTestResult]
    diagnostic_result: Optional[TransferDiagnosticResult]
    model: Any

    @property
    def shift_p_value(self) -> Optional[float]:
        if self.shift_result is not None:
            return self.shift_result.p_value
        return None

    @property
    def transfer_is_beneficial(self) -> Optional[bool]:
        if self.diagnostic_result is not None:
            return self.diagnostic_result.transfer_is_beneficial
        return None

    def __repr__(self) -> str:
        lines = [f"PipelineResult(method={self.method_used!r})"]
        if self.shift_result is not None:
            lines.append(f"  shift_p={self.shift_p_value:.4f}")
        if self.diagnostic_result is not None:
            lines.append(f"  NTG={self.diagnostic_result.ntg:+.4f}")
            beneficial = "Yes" if self.transfer_is_beneficial else "No"
            lines.append(f"  transfer_beneficial={beneficial}")
        return "\n".join(lines)


class TransferPipeline:
    """Full transfer learning pipeline for thin insurance segments.

    Orchestrates: shift test -> method selection -> fit -> diagnostics.

    Parameters
    ----------
    method:
        Transfer method to use. ``'glm'`` uses GLMTransfer (Tian & Feng).
        ``'gbm'`` uses GBMTransfer (CatBoost offset). ``'cann'`` uses
        CANNTransfer. ``'auto'`` selects GLM (the most reliable default).
    glm_params:
        Parameters passed to GLMTransfer if method='glm' or 'auto'.
    gbm_params:
        Parameters passed to GBMTransfer if method='gbm'.
    cann_params:
        Parameters passed to CANNTransfer if method='cann'.
    shift_test:
        Whether to run CovariateShiftTest before fitting.
    shift_n_permutations:
        Permutations for the shift test.
    categorical_cols:
        Categorical column indices for the shift test mixed kernel.
    run_diagnostic:
        Whether to run NegativeTransferDiagnostic after fitting.
    diagnostic_test_size:
        Fraction of target data held out for diagnostics. 0.2 is sensible
        for segments with > 50 observations; use 0.0 to skip the split.
    random_state:
        Seed for train/test split and permutation test.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_transfer import TransferPipeline
    >>> rng = np.random.default_rng(42)
    >>> X_src = rng.standard_normal((1000, 5))
    >>> y_src = rng.poisson(np.exp(0.3 * X_src[:, 0]))
    >>> X_tgt = rng.standard_normal((100, 5)) + 0.2
    >>> y_tgt = rng.poisson(np.exp(0.3 * X_tgt[:, 0]))
    >>> exposure_tgt = np.ones(100)
    >>> exposure_src = np.ones(1000)
    >>> pipeline = TransferPipeline(method='glm', shift_test=True,
    ...                              run_diagnostic=True, random_state=0)
    >>> result = pipeline.run(
    ...     X_tgt, y_tgt, exposure_tgt,
    ...     X_source=X_src, y_source=y_src, exposure_source=exposure_src,
    ... )
    >>> isinstance(result.model, object)
    True
    """

    def __init__(
        self,
        method: Method = "glm",
        glm_params: Optional[Dict] = None,
        gbm_params: Optional[Dict] = None,
        cann_params: Optional[Dict] = None,
        shift_test: bool = True,
        shift_n_permutations: int = 200,
        categorical_cols: Optional[List] = None,
        run_diagnostic: bool = True,
        diagnostic_test_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> None:
        self.method = method
        self.glm_params = glm_params or {}
        self.gbm_params = gbm_params or {}
        self.cann_params = cann_params or {}
        self.shift_test = shift_test
        self.shift_n_permutations = shift_n_permutations
        self.categorical_cols = categorical_cols
        self.run_diagnostic = run_diagnostic
        self.diagnostic_test_size = diagnostic_test_size
        self.random_state = random_state

    def _run_shift_test(
        self, X_source: NDArray, X_target: NDArray
    ) -> ShiftTestResult:
        tester = CovariateShiftTest(
            categorical_cols=self.categorical_cols or [],
            n_permutations=self.shift_n_permutations,
            random_state=self.random_state,
        )
        result = tester.test(X_source, X_target)
        if result.p_value < 0.01:
            import warnings
            warnings.warn(
                f"Severe covariate shift detected (MMD p={result.p_value:.4f}). "
                "Transfer learning may be unreliable. Check per_feature_drift_scores "
                "and consider feature alignment before proceeding.",
                UserWarning,
                stacklevel=3,
            )
        return result

    def _build_model(self, source_model: Any = None) -> Any:
        if self.method in ("glm", "auto"):
            return GLMTransfer(**self.glm_params)
        elif self.method == "gbm":
            if source_model is None:
                raise ValueError(
                    "method='gbm' requires a pre-fitted source CatBoost model. "
                    "Pass source_model= to run()."
                )
            from insurance_transfer.gbm_transfer import GBMTransfer
            return GBMTransfer(source_model=source_model, **self.gbm_params)
        elif self.method == "cann":
            from insurance_transfer.cann_transfer import CANNTransfer
            return CANNTransfer(**self.cann_params)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

    def run(
        self,
        X_target: NDArray,
        y_target: NDArray,
        exposure_target: Optional[NDArray] = None,
        X_source: Optional[NDArray] = None,
        y_source: Optional[NDArray] = None,
        exposure_source: Optional[NDArray] = None,
        source_model: Optional[Any] = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Parameters
        ----------
        X_target:
            Target feature matrix.
        y_target:
            Target response.
        exposure_target:
            Target exposure. Defaults to ones.
        X_source:
            Source feature matrix (for GLM transfer and shift test).
        y_source:
            Source response.
        exposure_source:
            Source exposure.
        source_model:
            Pre-fitted source model (required for GBM transfer, optional
            for CANN, ignored for GLM).

        Returns
        -------
        PipelineResult
        """
        X_target = np.asarray(X_target, dtype=np.float64)
        y_target = np.asarray(y_target, dtype=np.float64)
        n = X_target.shape[0]

        if exposure_target is None:
            exposure_target = np.ones(n)
        exposure_target = np.asarray(exposure_target, dtype=np.float64)

        # Step 1: Shift test
        shift_result = None
        if self.shift_test and X_source is not None:
            shift_result = self._run_shift_test(X_source, X_target)

        # Step 2: Split target for diagnostics
        if self.run_diagnostic and self.diagnostic_test_size > 0 and n > 5:
            idx_train, idx_test = train_test_split(
                np.arange(n),
                test_size=self.diagnostic_test_size,
                random_state=self.random_state,
            )
            X_train, X_test = X_target[idx_train], X_target[idx_test]
            y_train, y_test = y_target[idx_train], y_target[idx_test]
            exp_train = exposure_target[idx_train]
            exp_test = exposure_target[idx_test]
        else:
            X_train, y_train, exp_train = X_target, y_target, exposure_target
            X_test = y_test = exp_test = None

        # Step 3: Fit transfer model
        model = self._build_model(source_model=source_model)

        if self.method in ("glm", "auto"):
            model.fit(
                X_train, y_train, exp_train,
                X_source=X_source, y_source=y_source,
                exposure_source=exposure_source,
            )
        elif self.method == "gbm":
            model.fit(X_train, y_train, exp_train)
        elif self.method == "cann":
            if X_source is not None and y_source is not None:
                model.fit_source(X_source, y_source, exposure_source)
            model.fit(X_train, y_train, exp_train)

        # Step 4: Diagnostics
        diag_result = None
        if self.run_diagnostic and X_test is not None:
            # Fit a target-only baseline for comparison
            target_only = GLMTransfer(
                family=self.glm_params.get("family", "poisson"),
                lambda_pool=self.glm_params.get("lambda_pool", 0.01),
                lambda_debias=0.0,
                scale_features=self.glm_params.get("scale_features", True),
            )
            target_only.fit(X_train, y_train, exp_train)

            diag = NegativeTransferDiagnostic()
            diag_result = diag.evaluate(
                X_test, y_test, exp_test,
                transfer_model=model,
                target_only_model=target_only,
            )

        return PipelineResult(
            method_used=self.method,
            shift_result=shift_result,
            diagnostic_result=diag_result,
            model=model,
        )
