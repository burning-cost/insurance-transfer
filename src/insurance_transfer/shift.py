"""
Covariate shift detection for insurance transfer learning.

Uses Maximum Mean Discrepancy (MMD) with a mixed kernel: RBF for continuous
features, indicator kernel for categorical. Permutation test gives an exact
finite-sample p-value without distributional assumptions.

The mixed-kernel approach matters for insurance data: vehicle age is continuous,
but fuel type and body style are nominal. Treating them uniformly degrades MMD
sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class ShiftTestResult:
    """Result from a covariate shift test.

    Attributes
    ----------
    test_statistic:
        The MMD^2 estimate between source and target distributions.
    p_value:
        Permutation-based p-value. Small values (< 0.05) indicate the
        source and target come from meaningfully different distributions.
    per_feature_drift_scores:
        Per-column marginal MMD^2 values. Useful for identifying which
        features drive the shift. Dict keyed by column index or name.
    n_source:
        Number of source observations used.
    n_target:
        Number of target observations used.
    n_permutations:
        Number of permutations used to compute the p-value.
    """

    test_statistic: float
    p_value: float
    per_feature_drift_scores: dict
    n_source: int
    n_target: int
    n_permutations: int

    def __repr__(self) -> str:
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return (
            f"ShiftTestResult(MMD²={self.test_statistic:.4f}, "
            f"p={self.p_value:.4f} [{sig}], "
            f"n_source={self.n_source}, n_target={self.n_target})"
        )


def _rbf_kernel(X: NDArray, Y: NDArray, bandwidth: float) -> NDArray:
    """Compute RBF (Gaussian) kernel matrix between rows of X and Y."""
    # ||x - y||^2 via expansion
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True)
    cross = X @ Y.T
    sq_dist = X_sq + Y_sq.T - 2 * cross
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-sq_dist / (2.0 * bandwidth**2))


def _indicator_kernel(X: NDArray, Y: NDArray) -> NDArray:
    """Indicator (exact match) kernel for categorical columns.

    Returns a matrix K where K[i,j] = 1 if all categorical features
    in row i of X exactly match row j of Y, else 0. This is equivalent
    to a product of per-feature indicator kernels.
    """
    n, m = X.shape[0], Y.shape[0]
    K = np.ones((n, m), dtype=np.float64)
    for col in range(X.shape[1]):
        K *= (X[:, col : col + 1] == Y[:, col].reshape(1, -1)).astype(np.float64)
    return K


def _mixed_kernel(
    X: NDArray,
    Y: NDArray,
    cat_cols: List[int],
    cont_cols: List[int],
    bandwidth: float,
) -> NDArray:
    """Mixed kernel: product of RBF (continuous) and indicator (categorical).

    If one set is empty the kernel reduces to the other type alone.
    """
    n, m = X.shape[0], Y.shape[0]
    K = np.ones((n, m), dtype=np.float64)
    if cont_cols:
        X_cont = X[:, cont_cols].astype(np.float64)
        Y_cont = Y[:, cont_cols].astype(np.float64)
        K *= _rbf_kernel(X_cont, Y_cont, bandwidth)
    if cat_cols:
        X_cat = X[:, cat_cols]
        Y_cat = Y[:, cat_cols]
        K *= _indicator_kernel(X_cat, Y_cat)
    return K


def _mmd_squared(
    X: NDArray,
    Y: NDArray,
    cat_cols: List[int],
    cont_cols: List[int],
    bandwidth: float,
) -> float:
    """Unbiased estimator of MMD^2 between distributions of X and Y rows."""
    n = X.shape[0]
    m = Y.shape[0]

    Kxx = _mixed_kernel(X, X, cat_cols, cont_cols, bandwidth)
    Kyy = _mixed_kernel(Y, Y, cat_cols, cont_cols, bandwidth)
    Kxy = _mixed_kernel(X, Y, cat_cols, cont_cols, bandwidth)

    # Unbiased: zero diagonal in same-sample terms
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_yy = Kyy.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_xy = Kxy.sum() / (n * m)

    return float(term_xx + term_yy - 2.0 * term_xy)


def _estimate_bandwidth(X: NDArray, Y: NDArray, cont_cols: List[int]) -> float:
    """Median heuristic for RBF bandwidth on continuous columns."""
    if not cont_cols:
        return 1.0
    Z = np.vstack([X[:, cont_cols], Y[:, cont_cols]]).astype(np.float64)
    n = Z.shape[0]
    if n > 500:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 500, replace=False)
        Z = Z[idx]
    Z_sq = np.sum(Z**2, axis=1, keepdims=True)
    sq_dist = Z_sq + Z_sq.T - 2.0 * Z @ Z.T
    sq_dist = np.maximum(sq_dist, 0.0)
    triu = sq_dist[np.triu_indices_from(sq_dist, k=1)]
    median_sq = np.median(triu)
    return float(np.sqrt(median_sq)) if median_sq > 0 else 1.0


class CovariateShiftTest:
    """MMD-based covariate shift test with mixed kernel for insurance data.

    Tests whether source and target feature distributions differ significantly.
    Use this before deciding whether to transfer: if p-value is large (say > 0.2)
    the distributions are similar and transfer is likely to help. If p < 0.01
    the shift is severe and you should check which features drift most.

    Parameters
    ----------
    categorical_cols:
        Column names or integer indices to treat as categorical (indicator
        kernel). Everything else is treated as continuous (RBF kernel).
    n_permutations:
        Number of permutations for the p-value. 500 is fast; 2000 gives
        stable estimates.
    bandwidth:
        RBF bandwidth for continuous features. ``None`` uses the median
        heuristic (recommended).
    random_state:
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_transfer import CovariateShiftTest
    >>> rng = np.random.default_rng(42)
    >>> X_src = rng.standard_normal((200, 5))
    >>> X_tgt = rng.standard_normal((50, 5)) + 0.5
    >>> result = CovariateShiftTest(n_permutations=200).test(X_src, X_tgt)
    >>> result.p_value < 0.05
    True
    """

    def __init__(
        self,
        categorical_cols: Optional[Sequence] = None,
        n_permutations: int = 500,
        bandwidth: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else []
        self.n_permutations = n_permutations
        self.bandwidth = bandwidth
        self.random_state = random_state

    def _resolve_cols(self, X: NDArray) -> tuple[List[int], List[int]]:
        """Return (cat_col_indices, cont_col_indices)."""
        n_cols = X.shape[1]
        if self.categorical_cols and isinstance(self.categorical_cols[0], str):
            raise ValueError(
                "categorical_cols must be integer indices when X is a numpy array. "
                "Convert column names to indices before calling test()."
            )
        cat_cols = [int(c) for c in self.categorical_cols]
        cont_cols = [i for i in range(n_cols) if i not in set(cat_cols)]
        return cat_cols, cont_cols

    def test(
        self,
        X_source: NDArray,
        X_target: NDArray,
    ) -> ShiftTestResult:
        """Run the MMD permutation test.

        Parameters
        ----------
        X_source:
            Feature matrix for source population, shape (n_source, p).
        X_target:
            Feature matrix for target population, shape (n_target, p).

        Returns
        -------
        ShiftTestResult
        """
        X_source = np.asarray(X_source)
        X_target = np.asarray(X_target)

        if X_source.ndim == 1:
            X_source = X_source.reshape(-1, 1)
        if X_target.ndim == 1:
            X_target = X_target.reshape(-1, 1)

        if X_source.shape[1] != X_target.shape[1]:
            raise ValueError(
                f"Source has {X_source.shape[1]} columns but target has {X_target.shape[1]}."
            )

        cat_cols, cont_cols = self._resolve_cols(X_source)
        bandwidth = self.bandwidth if self.bandwidth is not None else _estimate_bandwidth(
            X_source, X_target, cont_cols
        )

        observed = _mmd_squared(X_source, X_target, cat_cols, cont_cols, bandwidth)

        # Permutation test: pool and repeatedly split
        rng = np.random.default_rng(self.random_state)
        n_src = X_source.shape[0]
        n_tgt = X_target.shape[0]
        pooled = np.vstack([X_source, X_target])
        null_stats = np.empty(self.n_permutations)

        for i in range(self.n_permutations):
            perm = rng.permutation(n_src + n_tgt)
            X_a = pooled[perm[:n_src]]
            X_b = pooled[perm[n_src:]]
            null_stats[i] = _mmd_squared(X_a, X_b, cat_cols, cont_cols, bandwidth)

        p_value = float(np.mean(null_stats >= observed))

        # Per-feature marginal drift scores
        per_feature = {}
        for col in range(X_source.shape[1]):
            col_cat = [0] if col in cat_cols else []
            col_cont = [] if col in cat_cols else [0]
            score = _mmd_squared(
                X_source[:, col : col + 1],
                X_target[:, col : col + 1],
                col_cat,
                col_cont,
                bandwidth,
            )
            per_feature[col] = float(score)

        return ShiftTestResult(
            test_statistic=observed,
            p_value=p_value,
            per_feature_drift_scores=per_feature,
            n_source=n_src,
            n_target=n_tgt,
            n_permutations=self.n_permutations,
        )

    def most_drifted_features(
        self, result: ShiftTestResult, top_n: int = 5
    ) -> List[tuple]:
        """Return the top-n features with highest marginal drift scores.

        Parameters
        ----------
        result:
            A ShiftTestResult from ``test()``.
        top_n:
            How many features to return.

        Returns
        -------
        List of (feature_index, drift_score) tuples, sorted descending.
        """
        scores = sorted(result.per_feature_drift_scores.items(), key=lambda x: x[1], reverse=True)
        return scores[:top_n]
