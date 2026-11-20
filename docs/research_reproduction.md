# DriftSentinel Research Reproduction Guide

This guide describes the audited research workflows, their inputs, dependencies, outputs, and methodological boundaries.

It is intended to make the existing experiments understandable and reproducible. It is not a recommendation to rerun every script in the repository.

## 1. Environment

The Python project requires:

- Python >= 3.11 and < 3.13.
- Core dependencies declared in `pyproject.toml`.
- The `dev` optional dependencies for tests, Parquet processing, and research utilities.
- The `api` optional dependencies for the FastAPI service.

The project has been developed in a Windows environment using `.venv311`.

From the repository root:

```powershell
.\.venv311\Scripts\Activate.ps1
python --version
python -m pip check
```

Install the project and optional dependencies only when setting up a fresh environment:

```powershell
python -m pip install -e ".[dev,api]"
```

Frontend dependencies are managed separately under `frontend/`.

## 2. Data and artifact layout

The research workflows use the IEEE-CIS Fraud Detection dataset.

Important local paths include:

| Path | Purpose |
|---|---|
| `data/processed/splits/train.parquet` | Chronological training partition |
| `data/processed/splits/validation.parquet` | Chronological validation partition |
| `data/processed/splits/test.parquet` | Later chronological holdout |
| `data/processed/temporal/` | Temporal experiment datasets |
| `artifacts/temporal/` | Temporal experiment model artifacts |
| `artifacts/xgboost/` | Main inference model and schema |
| `experiments/results/` | Research summaries and metadata |
| `experiments/results/temporal_predictions/` | Temporal model prediction datasets |
| `experiments/results/rolling_predictions/` | Rolling-origin prediction datasets |

The required XGBoost inference model, feature schema, calibrators, monitoring reference profile, and four temporal research models are committed to the repository.

The IEEE-CIS datasets and generated intermediate prediction datasets are not committed. Obtain the dataset separately, prepare the chronological partitions, and verify that the required inputs exist before running each experiment.

## 3. Chronological data partitions

The original processed dataset is split chronologically by `TransactionDT`.

The recorded split metadata is:

| Partition | Rows | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| Train | 413,379 | 14,538 | 3.5169% |
| Validation | 88,581 | 3,042 | 3.4341% |
| Test | 88,580 | 3,083 | 3.4805% |

The partitions have strictly ordered temporal boundaries.

The split implementation sorts by `TransactionDT` and moves split boundaries forward when necessary to keep identical timestamps together.

`TransactionDT` is a relative time value in this dataset. It must not be presented as a UTC timestamp.

### Test-partition access disclosure

The final test partition was read by two scripts for timestamp-boundary verification:

- `scripts/inspect_temporal_windows.py`
- `scripts/prepare_temporal_experiment.py`

These checks used timestamp information to verify chronological separation. The temporal preparation script also read transaction identifiers for this boundary check.

The final chronological holdout labels were not used for model training, calibration fitting, threshold selection, or model selection. Predictions were frozen before a separate evaluator loaded the labels for final metrics calculation only.

Therefore, describe the test partition as a later chronological holdout that was accessed for boundary verification. Do not describe it as never accessed or completely untouched.

## 4. Temporal experiment design

The temporal experiment is a separate Logistic Regression research workflow. It should not be confused with the XGBoost production inference pipeline.

### 4.1 Initial training sample

`scripts/prepare_temporal_experiment.py` reads the first 20,000 rows from the training Parquet file using `read_parquet_sample()`.

The sampling utility reads the first Parquet batch; it does not perform random sampling.

Given the chronological ordering of the training partition, this is an earliest-history training sample.

### 4.2 Validation windows

The validation partition is sorted by `TransactionDT` and divided using timestamp quantiles into:

1. Initial calibration window.
2. Adaptation window.
3. Evaluation window.

The adaptation window is divided chronologically into:

- Adaptation training.
- Adaptation calibration.

The split uses timestamp comparisons, so identical timestamps are kept together.

### 4.3 Model populations

| Variant | Training data |
|---|---|
| Frozen baseline | Initial 20,000-row training sample |
| Recent-only | Adaptation training window |
| Retrained | Initial training sample combined with adaptation training |
| Matched-size | 20,000 evenly spaced observations from the combined initial and adaptation training data |

The matched-size sample is sorted chronologically before evenly spaced positions are selected. It is deterministic, not random.

The matched-size comparison helps investigate training-volume effects, but it does not make the temporal composition of the training datasets identical.

## 5. Logistic Regression preprocessing

The temporal Logistic Regression experiments use the shared preprocessing implementation in `src/driftsentinel/features/preprocessing.py`.

### Numerical features

1. Median imputation.
2. Standard scaling.
3. Conversion to `float32`.

### Categorical features

1. Missing values are represented by `__MISSING__`.
2. Values are converted to strings.
3. One-hot encoding is applied.
4. `min_frequency=0.001`.
5. Unknown categories use `infrequent_if_exist`.

The following numeric-coded fields are treated as categorical:

- `card1`
- `card2`
- `card3`
- `card5`
- `addr1`
- `addr2`
- `id_18`
- `id_32`

`TransactionID`, `TransactionDT`, and `isFraud` are excluded from model predictors.

The shared Logistic Regression configuration is:

| Setting | Value |
|---|---|
| Solver | `liblinear` |
| Maximum iterations | 1,000 |
| Class weights | None |
| Random state | 42 |

## 6. Temporal experiment reproduction sequence

The following sequence describes the audited dependencies of the temporal model comparison.

Run commands from the repository root with the project environment activated.

### Step 1 — Prepare temporal datasets

```powershell
python scripts/prepare_temporal_experiment.py
```

Inputs:
- `data/processed/splits/train.parquet`
- `data/processed/splits/validation.parquet`
- `data/processed/splits/test.parquet` for timestamp-boundary verification only

Outputs include the initial training sample, calibration window, adaptation partitions, evaluation window, and temporal experiment metadata.

### Step 2 — Train the frozen baseline

```powershell
python scripts/train_temporal_baseline.py
```

Uses the initial training sample and saves the baseline model and metadata under `artifacts/temporal/` and `experiments/results/`.

### Step 3 — Score the baseline windows

```powershell
python scripts/score_temporal_baseline.py
```

Generates baseline prediction datasets used by downstream comparisons.

### Step 4 — Train the retrained model

```powershell
python scripts/train_retrained_model.py
```

Combines the initial training sample and adaptation training data after checking transaction-ID uniqueness and overlap.

### Step 5 — Train the recent-only model

```powershell
python scripts/train_recent_only_model.py
```

Uses the adaptation training partition.

### Step 6 — Prepare the matched-size dataset

```powershell
python scripts/prepare_matched_size_dataset.py
```

Combines initial and adaptation training data and selects 20,000 evenly spaced chronological positions.

### Step 7 — Train the matched-size model

```powershell
python scripts/train_matched_size_model.py
```

Uses `matched_size_train.parquet`.

### Step 8 — Score the model variants

```powershell
python scripts/score_retraining_models.py
python scripts/score_recent_only_model.py
python scripts/score_matched_size_model.py
```

These scripts generate aligned adaptation-calibration and evaluation prediction datasets for the model variants.

### Step 9 — Run the retraining and calibration comparison

```powershell
python scripts/run_retraining_comparison.py
```

The comparison checks transaction IDs, timestamps, and labels across prediction variants before comparing calibration and evaluation results.

It produces:

- `experiments/results/retraining_calibration_comparison.csv`

The evaluation is based on the temporal experiment's evaluation window. It is not a final test-partition evaluation.

## 7. Rolling-origin calibration evaluation

The rolling-origin experiment is a separate workflow.

```powershell
python scripts/run_rolling_experiment.py
```

It loads the training and validation partitions, constructs five chronological folds, trains a fresh Logistic Regression model for each fold, and compares calibration strategies.

The fold design uses expanding training history and a preceding calibration segment. Fold 1 uses the initial calibration data for both static and adaptive calibration because no later adaptation calibration segment is available at that point.

Outputs:

- `experiments/results/rolling_fold_metrics.csv`
- `experiments/results/rolling_experiment_summary.csv`
- `experiments/results/rolling_predictions/`

The five-fold experiment evaluates calibration behavior. It does not independently validate the separate retraining comparison.

Important methodological details:

- Evaluation segments are approximately equal in row count, not necessarily calendar duration.
- Calibration uses the immediately preceding segment rather than a cumulative calibration history.
- Labels from completed historical validation segments are assumed available before the next fold.
- The workflow does not load `test.parquet`.

## 8. Temporal block bootstrap

The temporal block bootstrap operates on saved evaluation prediction datasets. It does not train models.

```powershell
python scripts/run_temporal_block_bootstrap.py
```

It compares aligned prediction datasets for frozen, recent-only, matched-size, and retrained variants.

The implementation:

- Checks row alignment across model prediction datasets.
- Uses paired samples for model comparisons.
- Samples contiguous row blocks with replacement.
- Evaluates block lengths of 250, 500, and 1,000 rows.
- Runs 1,000 replicates per block length.
- Uses seeds derived as 42 plus the block length.
- Computes percentile confidence intervals and paired differences.

Outputs:

- `experiments/results/temporal_block_bootstrap_intervals.csv`
- `experiments/results/temporal_block_bootstrap_differences.csv`

### Bootstrap limitations

The bootstrap samples contiguous row blocks, not timestamp groups. Identical `TransactionDT` values can therefore fall across block boundaries.

Sampling blocks independently with replacement also breaks dependence between adjacent sampled blocks.

The analysis covers one evaluation window. Its confidence intervals do not establish performance across all future periods, and the results should not be interpreted as a guarantee of stationarity.

## 9. Research result interpretation

The repository contains additional experiments for:

- Logistic Regression training-size benchmarks.
- XGBoost training and calibration.
- Threshold and decision-cost analysis.
- Review-capacity sensitivity.
- Static and adaptive calibration comparisons.
- Feature drift analysis.
- Bootstrap uncertainty estimation.

Their outputs are stored in `experiments/results/`.

Interpret comparisons in the context of their specific experimental protocols. Do not combine results from different datasets, windows, capacity assumptions, or model populations as though they came from one controlled experiment.

### Calibration

Calibration methods do not improve every metric uniformly.

A strategy may improve Brier score or log loss while changing expected calibration error or average precision. State the metric and evaluation window when describing an improvement.

### Drift

PSI is a distribution-shift screening statistic. It does not prove concept drift, prediction degradation, or a need to retrain.

Performance drift requires labels and suitable evaluation. Concept drift requires evidence that the relationship between predictors and the target has changed.

### Cost-sensitive decisions

The decision policy uses explicit relative costs for approving, reviewing, and blocking transactions. These values are illustrative policy units, not monetary amounts.

Review capacity can constrain the available decisions and increase expected cost. A capacity limit is not the same as a target utilization requirement.

## 10. Application and runtime artifacts

The application is separate from the temporal Logistic Regression research workflow.

The inference system uses:

- `artifacts/xgboost/fraud_model.json`
- `artifacts/xgboost/feature_schema.json`
- Calibration artifacts under `artifacts/xgboost/calibration/`
- A reference profile for aggregate monitoring

The inference schema contains 431 model features: 392 numerical and 39 categorical.

The backend persists prediction metadata and aggregate monitoring snapshots in SQLite. Raw transaction feature vectors are not stored in the prediction ledger.

The default monitoring database is:

```text
artifacts/monitoring/monitoring.db
```

Environment variables:

- `DRIFTSENTINEL_DB_PATH`
- `DRIFTSENTINEL_REFERENCE_PROFILE_PATH`

Prediction and monitoring records in the local demo database are runtime artifacts, not research ground truth.

## 11. Repository hygiene and reproducibility

The `.gitignore` excludes local datasets and model artifacts, and excludes selected generated Parquet outputs.

Research summaries, metadata, CSVs, and plots are generally retained in `experiments/results/`.

Before reproducing an experiment:

1. Confirm its required input datasets exist.
2. Confirm the expected model and prediction artifacts exist or can be regenerated.
3. Read the script before executing it.
4. Check whether it overwrites an existing output.
5. Preserve existing research outputs if they are needed for comparison.
6. Record the Python and dependency versions used.

Do not execute every script alphabetically. Some scripts are exploratory, some consume outputs generated by other scripts, and some write files used by the application.

## 12. Current research boundary

The research inventory and temporal methodology have been audited and consolidated.

The final chronological holdout evaluation has been completed on 88,580 transactions across 14 strategies. The prediction stage was label-blind. A separate evaluator subsequently loaded the holdout labels for metrics calculation only; no model retraining or calibrator refitting occurred.

The reported final holdout metrics are point estimates. The bootstrap confidence intervals elsewhere in this report apply to the validation evaluation window, not the final holdout.