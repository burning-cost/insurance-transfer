"""Tests for covariate shift detection (CovariateShiftTest)."""

import numpy as np
import pytest

from insurance_transfer.shift import (
    CovariateShiftTest,
    ShiftTestResult,
    _mmd_squared,
    _rbf_kernel,
    _indicator_kernel,
    _estimate_bandwidth,
)


class TestRBFKernel:
    def test_shape(self):
        X = np.random.randn(10, 3)
        Y = np.random.randn(7, 3)
        K = _rbf_kernel(X, Y, bandwidth=1.0)
        assert K.shape == (10, 7)

    def test_symmetric_for_same_input(self):
        X = np.random.randn(8, 3)
        K = _rbf_kernel(X, X, bandwidth=1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_diagonal_is_one(self):
        X = np.random.randn(5, 3)
        K = _rbf_kernel(X, X, bandwidth=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_values_in_zero_one(self):
        X = np.random.randn(20, 4)
        Y = np.random.randn(15, 4)
        K = _rbf_kernel(X, Y, bandwidth=2.0)
        assert np.all(K >= 0)
        assert np.all(K <= 1.0 + 1e-10)

    def test_wider_bandwidth_gives_higher_similarity(self):
        X = np.ones((5, 2))
        Y = np.ones((5, 2)) * 2
        K_narrow = _rbf_kernel(X, Y, bandwidth=0.5)
        K_wide = _rbf_kernel(X, Y, bandwidth=5.0)
        assert K_wide.mean() > K_narrow.mean()


class TestIndicatorKernel:
    def test_perfect_match(self):
        X = np.array([[1, 2], [3, 4]])
        K = _indicator_kernel(X, X)
        assert K.shape == (2, 2)
        np.testing.assert_array_equal(np.diag(K), [1.0, 1.0])

    def test_no_match(self):
        X = np.array([[1, 2]])
        Y = np.array([[3, 4]])
        K = _indicator_kernel(X, Y)
        assert K[0, 0] == 0.0

    def test_partial_match(self):
        X = np.array([[1, 2]])
        Y = np.array([[1, 99]])  # First col matches, second doesn't
        K = _indicator_kernel(X, Y)
        assert K[0, 0] == 0.0

    def test_shape(self):
        X = np.random.randint(0, 5, (10, 3))
        Y = np.random.randint(0, 5, (7, 3))
        K = _indicator_kernel(X, Y)
        assert K.shape == (10, 7)


class TestMMDSquared:
    def test_same_distribution_near_zero(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        Y = rng.standard_normal((200, 3))
        mmd = _mmd_squared(X, Y, cat_cols=[], cont_cols=[0, 1, 2], bandwidth=1.0)
        assert abs(mmd) < 0.1  # Should be close to zero

    def test_different_distributions_positive(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        Y = rng.standard_normal((100, 3)) + 3.0  # Large shift
        mmd = _mmd_squared(X, Y, cat_cols=[], cont_cols=[0, 1, 2], bandwidth=1.0)
        assert mmd > 0.1

    def test_identical_data_zero(self):
        X = np.ones((10, 2))
        mmd = _mmd_squared(X, X, cat_cols=[], cont_cols=[0, 1], bandwidth=1.0)
        # With unbiased estimator and identical data, MMD^2 ≈ 0
        assert abs(mmd) < 1e-6


class TestEstimateBandwidth:
    def test_returns_positive(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        Y = rng.standard_normal((50, 3))
        bw = _estimate_bandwidth(X, Y, cont_cols=[0, 1, 2])
        assert bw > 0

    def test_no_cont_cols_returns_one(self):
        X = np.random.randn(10, 3)
        Y = np.random.randn(10, 3)
        bw = _estimate_bandwidth(X, Y, cont_cols=[])
        assert bw == 1.0

    def test_scales_with_data(self):
        rng = np.random.default_rng(1)
        X_small = rng.standard_normal((50, 2))
        X_large = rng.standard_normal((50, 2)) * 10
        bw_small = _estimate_bandwidth(X_small, X_small, cont_cols=[0, 1])
        bw_large = _estimate_bandwidth(X_large, X_large, cont_cols=[0, 1])
        assert bw_large > bw_small


class TestCovariateShiftTest:
    def test_basic_same_distribution(self):
        rng = np.random.default_rng(42)
        X_src = rng.standard_normal((300, 4))
        X_tgt = rng.standard_normal((100, 4))
        result = CovariateShiftTest(n_permutations=100, random_state=42).test(X_src, X_tgt)
        assert isinstance(result, ShiftTestResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_different_distributions_low_pvalue(self):
        rng = np.random.default_rng(42)
        X_src = rng.standard_normal((200, 3))
        X_tgt = rng.standard_normal((50, 3)) + 2.0  # Large shift
        result = CovariateShiftTest(n_permutations=200, random_state=42).test(X_src, X_tgt)
        assert result.p_value < 0.05

    def test_same_data_high_pvalue(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 4))
        # Split the same data — should not reject
        result = CovariateShiftTest(n_permutations=200, random_state=0).test(X[:200], X[200:])
        assert result.p_value > 0.05

    def test_result_attributes(self):
        rng = np.random.default_rng(1)
        X_src = rng.standard_normal((100, 3))
        X_tgt = rng.standard_normal((30, 3))
        result = CovariateShiftTest(n_permutations=50, random_state=1).test(X_src, X_tgt)
        assert result.n_source == 100
        assert result.n_target == 30
        assert result.n_permutations == 50
        assert len(result.per_feature_drift_scores) == 3
        assert isinstance(result.test_statistic, float)

    def test_per_feature_scores_positive(self):
        rng = np.random.default_rng(2)
        X_src = rng.standard_normal((100, 4))
        X_tgt = rng.standard_normal((40, 4)) + 1.0
        result = CovariateShiftTest(n_permutations=50, random_state=2).test(X_src, X_tgt)
        for score in result.per_feature_drift_scores.values():
            assert isinstance(score, float)

    def test_categorical_cols(self):
        rng = np.random.default_rng(3)
        # Mix of continuous and categorical
        X_cont = rng.standard_normal((100, 2))
        X_cat = rng.integers(0, 3, (100, 1))
        X_src = np.column_stack([X_cont, X_cat])
        X_cont2 = rng.standard_normal((40, 2))
        X_cat2 = rng.integers(0, 3, (40, 1))
        X_tgt = np.column_stack([X_cont2, X_cat2])
        result = CovariateShiftTest(
            categorical_cols=[2], n_permutations=50, random_state=3
        ).test(X_src, X_tgt)
        assert isinstance(result, ShiftTestResult)
        assert len(result.per_feature_drift_scores) == 3

    def test_1d_input_handled(self):
        rng = np.random.default_rng(4)
        X_src = rng.standard_normal(100)
        X_tgt = rng.standard_normal(30) + 1.0
        result = CovariateShiftTest(n_permutations=50).test(X_src, X_tgt)
        assert isinstance(result, ShiftTestResult)

    def test_mismatched_columns_raises(self):
        X_src = np.random.randn(50, 3)
        X_tgt = np.random.randn(20, 4)
        with pytest.raises(ValueError, match="columns"):
            CovariateShiftTest(n_permutations=10).test(X_src, X_tgt)

    def test_repr(self):
        rng = np.random.default_rng(5)
        X_src = rng.standard_normal((100, 2))
        X_tgt = rng.standard_normal((30, 2))
        result = CovariateShiftTest(n_permutations=50).test(X_src, X_tgt)
        r = repr(result)
        assert "ShiftTestResult" in r
        assert "MMD" in r

    def test_most_drifted_features(self):
        rng = np.random.default_rng(6)
        # Make col 0 drift a lot, cols 1-3 minimal
        X_src = rng.standard_normal((200, 4))
        X_tgt = rng.standard_normal((60, 4))
        X_tgt[:, 0] += 3.0  # Strong drift in feature 0
        tester = CovariateShiftTest(n_permutations=100, random_state=6)
        result = tester.test(X_src, X_tgt)
        top = tester.most_drifted_features(result, top_n=2)
        assert len(top) == 2
        assert top[0][0] == 0  # Feature 0 should be most drifted

    def test_reproducibility(self):
        rng = np.random.default_rng(7)
        X_src = rng.standard_normal((100, 3))
        X_tgt = rng.standard_normal((30, 3)) + 0.5
        r1 = CovariateShiftTest(n_permutations=100, random_state=99).test(X_src, X_tgt)
        r2 = CovariateShiftTest(n_permutations=100, random_state=99).test(X_src, X_tgt)
        assert r1.test_statistic == r2.test_statistic
        assert r1.p_value == r2.p_value

    def test_fixed_bandwidth(self):
        rng = np.random.default_rng(8)
        X_src = rng.standard_normal((100, 2))
        X_tgt = rng.standard_normal((30, 2))
        result = CovariateShiftTest(bandwidth=2.0, n_permutations=50).test(X_src, X_tgt)
        assert isinstance(result.test_statistic, float)

    def test_small_samples(self):
        rng = np.random.default_rng(9)
        X_src = rng.standard_normal((10, 2))
        X_tgt = rng.standard_normal((5, 2))
        result = CovariateShiftTest(n_permutations=20).test(X_src, X_tgt)
        assert isinstance(result, ShiftTestResult)
