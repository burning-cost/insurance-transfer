# insurance-transfer

Transfer learning for thin-segment insurance pricing.

## The problem

Pricing actuaries routinely face the thin-data problem: you want to price young drivers, a new business class, or a pet breed, but you have fewer than 200 claims in the target segment. A model fitted on that data alone will overfit. Credibility blending helps, but it is a blunt instrument that does not respect covariate structure.

Transfer learning is a better answer. You have a large portfolio — say 50,000 motor policies. Some of that information is relevant to your thin segment. The question is how much to borrow, and how to correct for the fact that young drivers are not just a small random sample of all drivers.

This library implements three transfer methods adapted for insurance pricing, plus diagnostics to detect when the transfer is helping versus hurting.

## What it does

**Covariate shift detection** (`CovariateShiftTest`): Before you transfer anything, test whether the source and target distributions are meaningfully different. Uses Maximum Mean Discrepancy with a mixed kernel — RBF for continuous features (age, vehicle value), indicator for categorical ones (fuel type, body style). Returns a permutation-based p-value and per-feature drift scores so you can see which features are driving the divergence.

**Penalised GLM transfer** (`GLMTransfer`): Implements the two-step algorithm from Tian and Feng (JASA 2023). Step 1 pools target and source data and fits an l1-penalised GLM. Step 2 refines the estimate on target data only, penalising the adjustment to prevent overfitting. Supports Poisson (frequency), Gamma (severity), and Gaussian families. Source auto-detection excludes sources where the transfer direction is harmful.

**GBM transfer** (`GBMTransfer`): CatBoost source-as-offset. Generates log-predictions from a fitted source CatBoost model, uses them as a fixed baseline offset when training a residual GBM on target data. Works in two modes: `offset` (explicit offset, more interpretable) or `init_model` (CatBoost warm-start, fewer parameters to tune). CatBoost only.

**CANN transfer** (`CANNTransfer`, requires PyTorch): Pre-train a Combined Actuarial Neural Network on source data, fine-tune on the target segment. Three fine-tuning strategies: `head_only` (safe default for very thin segments), `all` (full fine-tune), `progressive` (head-only then full). Optional dependency.

**Negative transfer diagnostics** (`NegativeTransferDiagnostic`): Compares the transfer model against a target-only baseline and optionally against the source model applied directly. Reports Poisson deviance, the Negative Transfer Gap (NTG = deviance_transfer - deviance_target_only), and per-feature residual patterns.

**Pipeline** (`TransferPipeline`): Orchestrates the full workflow: shift test, method selection, fit, diagnostics. Use it when you want sensible defaults without chaining components manually.

## Install

```bash
pip install insurance-transfer
```

With CatBoost support:
```bash
pip install insurance-transfer[catboost]
```

With PyTorch (CANN):
```bash
pip install insurance-transfer[torch]
```

## Quick start

```python
import numpy as np
from insurance_transfer import (
    CovariateShiftTest,
    GLMTransfer,
    NegativeTransferDiagnostic,
    TransferPipeline,
)

# Shift test
tester = CovariateShiftTest(categorical_cols=[3, 4], n_permutations=500)
result = tester.test(X_source, X_target)
print(result)
# ShiftTestResult(MMD2=0.0312, p=0.004 [significant], n_source=8000, n_target=150)

# See which features drift most
tester.most_drifted_features(result, top_n=3)

# GLM transfer
model = GLMTransfer(family='poisson', lambda_pool=0.01, lambda_debias=0.05)
model.fit(
    X_target, y_target, exposure_target,
    X_source=X_source, y_source=y_source, exposure_source=exposure_source,
)
predictions = model.predict(X_target, exposure_target)

# Full pipeline
pipeline = TransferPipeline(
    method='glm', shift_test=True, run_diagnostic=True,
    glm_params={'family': 'poisson', 'lambda_pool': 0.01},
)
result = pipeline.run(
    X_target, y_target, exposure_target,
    X_source=X_source, y_source=y_source,
)
print(result)
```

## GBM transfer (CatBoost)

```python
from catboost import CatBoostRegressor
from insurance_transfer import GBMTransfer

source_model = CatBoostRegressor(loss_function='Poisson', iterations=500)
source_model.fit(X_source, y_source)

transfer = GBMTransfer(
    source_model=source_model,
    mode='offset',
    catboost_params={'iterations': 100, 'depth': 4},
)
transfer.fit(X_target, y_target, exposure=exposure_target)
predictions = transfer.predict(X_target, exposure=exposure_target)
```

## CANN transfer (PyTorch)

```python
from insurance_transfer import CANNTransfer

model = CANNTransfer(
    hidden_sizes=[32, 16],
    finetune_strategy='head_only',
    pretrain_epochs=50,
    finetune_epochs=30,
)
model.fit_source(X_source, y_source, exposure_source)
model.fit(X_target, y_target, exposure_target)
predictions = model.predict(X_target, exposure_target)
```

## Design choices

**Poisson deviance as primary metric.** Mean squared error is wrong for count data. We use Poisson deviance throughout, including in the NTG calculation.

**Exposure as first-class parameter.** Every method takes `exposure` as a dedicated argument, not as `sample_weight`. The two are not equivalent: exposure enters the log-offset, sample_weight scales the gradient contribution.

**Mixed kernel for MMD.** Insurance data is always mixed: continuous (driver age, vehicle value) and categorical (fuel type, body style, NCD band). A pure RBF kernel on label-encoded categoricals would be meaningless. The mixed kernel treats each type correctly.

**l1 penalty not l2.** The debiasing step in GLMTransfer uses l1 so that zero-correction is exact. If a feature transfers perfectly, its delta coefficient goes exactly to zero rather than shrinking towards it.

**Auto-detection is greedy, not exhaustive.** Checking all 2^k subsets of sources is infeasible for large source sets. The implementation checks each source individually and keeps those where the delta norm is below threshold.

## References

Tian, Y. and Feng, Y. (2023). Transfer Learning under High-Dimensional Generalized Linear Models. Journal of the American Statistical Association, 118(544), 2684-2697.

Loke, S.-H. and Bauer, D. (2025). Transfer Learning in the Actuarial Domain: Foundations and Applications. North American Actuarial Journal. DOI: 10.1080/10920277.2025.2489637.

Schelldorfer, J. and Wuthrich, M. V. (2019). Nesting Classical Actuarial Models into Neural Networks. SSRN 3325285.
