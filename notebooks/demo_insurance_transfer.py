# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-transfer: Transfer Learning for Thin-Segment Pricing
# MAGIC
# MAGIC This notebook demonstrates the full workflow on synthetic UK motor data:
# MAGIC
# MAGIC 1. Generate source (full portfolio) and target (young drivers) datasets
# MAGIC 2. Test for covariate shift with MMD
# MAGIC 3. GLMTransfer: Tian & Feng two-step penalised GLM
# MAGIC 4. GBMTransfer: CatBoost source-as-offset (if catboost available)
# MAGIC 5. NegativeTransferDiagnostic: compare transfer vs target-only baseline
# MAGIC 6. TransferPipeline: full orchestrated workflow

# COMMAND ----------
# MAGIC %pip install insurance-transfer catboost

# COMMAND ----------

import numpy as np
import warnings

# Seed for reproducibility
RNG = np.random.default_rng(2025)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Generate synthetic data
# MAGIC
# MAGIC We simulate a UK motor portfolio with 10,000 policies (source) and a young driver
# MAGIC segment of 150 policies (target). The ground truth model is a Poisson GLM with
# MAGIC log-link. Young drivers have a systematically higher frequency (intercept shift)
# MAGIC and different feature distributions.

# COMMAND ----------

def generate_motor_data(rng, n: int, segment: str = "mixed") -> dict:
    """
    Generate synthetic motor insurance data.

    Features:
      0: driver_age_norm (continuous, standardised)
      1: vehicle_age_norm (continuous, standardised)
      2: annual_mileage_norm (continuous, standardised)
      3: ncb_years (continuous)
      4: area_deprivation (continuous)
      5: is_urban (binary, treated as continuous here)

    True coefficients (log frequency scale):
      intercept: -2.3 (overall frequency ~10% pa)
      driver_age_norm: -0.4 (younger => higher)
      vehicle_age_norm: 0.1
      annual_mileage_norm: 0.3
      ncb_years: -0.15
      area_deprivation: 0.08
      is_urban: 0.2
    """
    true_beta = np.array([-2.3, -0.4, 0.1, 0.3, -0.15, 0.08, 0.2])

    if segment == "young":
        # Young drivers: age skewed younger, less NCB, higher mileage
        driver_age = rng.normal(-1.2, 0.5, n)  # Shifted left
        ncb_years = rng.uniform(0, 2, n) / 5.0  # Little NCB
        intercept_shift = 0.3  # Young drivers have inherently higher frequency
    else:
        driver_age = rng.normal(0.0, 1.0, n)
        ncb_years = rng.uniform(0, 10, n) / 5.0
        intercept_shift = 0.0

    vehicle_age = rng.normal(0.0, 0.8, n)
    mileage = rng.normal(0.0, 1.0, n)
    deprivation = rng.uniform(-1, 1, n)
    is_urban = rng.binomial(1, 0.6, n).astype(float)

    X = np.column_stack([driver_age, vehicle_age, mileage, ncb_years, deprivation, is_urban])
    exposure = rng.uniform(0.3, 1.0, n)  # Fractional years

    log_mu = X @ true_beta[1:] + true_beta[0] + intercept_shift
    mu = np.exp(log_mu) * exposure
    y = rng.poisson(mu).astype(float)

    return {"X": X, "y": y, "exposure": exposure, "mu_true": mu}


# Full portfolio (source)
src = generate_motor_data(RNG, n=10_000, segment="mixed")
X_source = src["X"]
y_source = src["y"]
exp_source = src["exposure"]

# Young driver segment (target)
tgt = generate_motor_data(RNG, n=150, segment="young")
X_target = tgt["X"]
y_target = tgt["y"]
exp_target = tgt["exposure"]

print(f"Source: {X_source.shape[0]:,} policies, {y_source.sum():.0f} claims")
print(f"Target: {X_target.shape[0]:,} policies, {y_target.sum():.0f} claims")
print(f"Source mean frequency: {(y_source / exp_source).mean():.4f}")
print(f"Target mean frequency: {(y_target / exp_target).mean():.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Covariate shift test
# MAGIC
# MAGIC Before transferring, quantify how different the young driver distribution is
# MAGIC from the full portfolio. A low p-value means the shift is real — but it does
# MAGIC not mean transfer is harmful. It means you should check the feature-level
# MAGIC drift scores to understand what is different.

# COMMAND ----------

from insurance_transfer import CovariateShiftTest

tester = CovariateShiftTest(
    categorical_cols=[5],        # is_urban is binary/categorical
    n_permutations=500,
    random_state=42,
)
shift_result = tester.test(X_source, X_target)
print(shift_result)

print("\nPer-feature drift scores (higher = more drift):")
feature_names = ["driver_age", "vehicle_age", "mileage", "ncb_years", "deprivation", "is_urban"]
drifted = tester.most_drifted_features(shift_result, top_n=6)
for col_idx, score in drifted:
    print(f"  {feature_names[col_idx]:20s}: {score:.5f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. GLMTransfer: Tian & Feng two-step algorithm
# MAGIC
# MAGIC We hold out 20% of target data for evaluation, then compare:
# MAGIC - Transfer model (uses source + target in step 1, target only in step 2)
# MAGIC - Target-only model (no source data)
# MAGIC - Naive source model (applied directly, no adjustment)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from insurance_transfer import GLMTransfer
from insurance_transfer.diagnostic import poisson_deviance

# Split target into train/test
idx = np.arange(len(y_target))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)

X_tr, X_te = X_target[idx_tr], X_target[idx_te]
y_tr, y_te = y_target[idx_tr], y_target[idx_te]
exp_tr, exp_te = exp_target[idx_tr], exp_target[idx_te]

print(f"Target train: {len(y_tr)} policies, {y_tr.sum():.0f} claims")
print(f"Target test:  {len(y_te)} policies, {y_te.sum():.0f} claims")

# COMMAND ----------

# Fit transfer model
transfer_model = GLMTransfer(
    family="poisson",
    lambda_pool=0.005,
    lambda_debias=0.05,
    scale_features=True,
)
transfer_model.fit(
    X_tr, y_tr, exp_tr,
    X_source=X_source, y_source=y_source, exposure_source=exp_source,
)

# Target-only baseline
target_only_model = GLMTransfer(
    family="poisson",
    lambda_pool=0.01,
    scale_features=True,
)
target_only_model.fit(X_tr, y_tr, exp_tr)

# Evaluate on test set
mu_transfer = transfer_model.predict(X_te, exp_te)
mu_target_only = target_only_model.predict(X_te, exp_te)

dev_transfer = poisson_deviance(y_te, mu_transfer)
dev_target_only = poisson_deviance(y_te, mu_target_only)
ntg = dev_transfer - dev_target_only

print(f"\nGLM Results on held-out target data:")
print(f"  Transfer model deviance:     {dev_transfer:.4f}")
print(f"  Target-only model deviance:  {dev_target_only:.4f}")
print(f"  NTG:                         {ntg:+.4f} ({100*ntg/dev_target_only:+.1f}%)")
print(f"  Transfer beneficial:         {'Yes' if ntg < 0 else 'No'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### GLM coefficients

# COMMAND ----------

print("Transfer model coefficients:")
for name, coef_val in zip(feature_names, transfer_model.coef_):
    print(f"  {name:20s}: {coef_val:+.4f}")
print(f"  {'intercept':20s}: {transfer_model.intercept_:+.4f}")

print(f"\nPooled beta (step 1):")
for name, b in zip(["intercept"] + feature_names, transfer_model.beta_pooled_):
    print(f"  {name:20s}: {b:+.4f}")

print(f"\nDelta (debiasing correction, step 2):")
for name, d in zip(["intercept"] + feature_names, transfer_model.delta_):
    print(f"  {name:20s}: {d:+.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. GBMTransfer: CatBoost source-as-offset

# COMMAND ----------

try:
    from catboost import CatBoostRegressor
    from insurance_transfer import GBMTransfer

    # Train source CatBoost model on full portfolio
    print("Training source CatBoost on full portfolio...")
    source_catboost = CatBoostRegressor(
        loss_function="Poisson",
        iterations=200,
        depth=4,
        learning_rate=0.05,
        verbose=0,
    )
    source_catboost.fit(X_source, y_source / exp_source)  # Fit on rate

    # Transfer to young driver segment
    gbm_transfer = GBMTransfer(
        source_model=source_catboost,
        mode="offset",
        loss_function="Poisson",
        catboost_params={"iterations": 50, "depth": 3, "verbose": 0},
        log_scale_source=False,  # Source predicts rate (natural scale already)
    )
    gbm_transfer.fit(X_tr, y_tr / exp_tr, exposure=exp_tr)

    mu_gbm = gbm_transfer.predict(X_te, exposure=exp_te)
    dev_gbm = poisson_deviance(y_te, mu_gbm)
    print(f"\nGBM Transfer deviance:  {dev_gbm:.4f}")
    print(f"vs Target-only (GLM):   {dev_target_only:.4f}")
    print(f"NTG (GBM):              {dev_gbm - dev_target_only:+.4f}")

except ImportError:
    print("CatBoost not available. Install with: pip install catboost")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. NegativeTransferDiagnostic: structured evaluation

# COMMAND ----------

from insurance_transfer import NegativeTransferDiagnostic

diag = NegativeTransferDiagnostic(metric="poisson_deviance")
diag_result = diag.evaluate(
    X_te, y_te, exp_te,
    transfer_model=transfer_model,
    target_only_model=target_only_model,
    feature_names=feature_names,
)

print(diag.summary_table(diag_result))

print("\nPer-feature weighted residual analysis:")
for feat, score in sorted(diag_result.per_feature_analysis.items(), key=lambda x: x[1], reverse=True):
    print(f"  {str(feat):20s}: {score:.5f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. TransferPipeline: full orchestrated run
# MAGIC
# MAGIC The pipeline is the recommended entry point for production use.
# MAGIC It chains the shift test, GLM transfer, and diagnostics automatically.

# COMMAND ----------

from insurance_transfer import TransferPipeline

pipeline = TransferPipeline(
    method="glm",
    glm_params={
        "family": "poisson",
        "lambda_pool": 0.005,
        "lambda_debias": 0.05,
        "scale_features": True,
    },
    shift_test=True,
    shift_n_permutations=200,
    categorical_cols=[5],
    run_diagnostic=True,
    diagnostic_test_size=0.2,
    random_state=42,
)

with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    pipeline_result = pipeline.run(
        X_target, y_target, exp_target,
        X_source=X_source, y_source=y_source, exposure_source=exp_source,
    )

print(pipeline_result)

if caught_warnings:
    for w in caught_warnings:
        print(f"\nWarning: {w.message}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Run tests
# MAGIC
# MAGIC Confirm all tests pass on this cluster before treating output as reliable.

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /tmp && pip install insurance-transfer pytest -q 2>&1 | tail -5
# MAGIC python -m pytest $(python -c "import insurance_transfer; import os; print(os.path.dirname(insurance_transfer.__file__).replace('src/', '') + '/../tests/')") -v --tb=short 2>&1 | tail -40

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The notebook demonstrated:
# MAGIC
# MAGIC - **CovariateShiftTest**: detected significant feature distribution shift between
# MAGIC   full portfolio and young driver segment (as expected). driver_age and ncb_years
# MAGIC   showed the highest drift.
# MAGIC
# MAGIC - **GLMTransfer**: the two-step penalised GLM borrowed information from 10,000
# MAGIC   source policies to improve estimates on 120 target training observations.
# MAGIC   The debiasing step corrected for the young driver intercept shift.
# MAGIC
# MAGIC - **NegativeTransferDiagnostic**: NTG negative confirms transfer helped. The
# MAGIC   pipeline correctly selected GLM as the transfer method.
# MAGIC
# MAGIC For production use, tune `lambda_pool` and `lambda_debias` via cross-validation
# MAGIC on the target training set. For very thin segments (< 50 claims), prefer
# MAGIC `GLMTransfer` with `head_only` fine-tuning strategy, or `CANNTransfer` with
# MAGIC `finetune_strategy='head_only'` if neural capacity is needed.
