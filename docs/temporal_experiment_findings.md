# DriftSentinel — Temporal Experiment Findings

## 1. Experiment Overview

**Project:** DriftSentinel

**Dataset:** IEEE-CIS Fraud Detection

**Research area:** Temporal adaptation in fraud detection and capacity-constrained decision systems

**Current status:** Controlled temporal pilot and initial statistical uncertainty analysis completed

**Evaluation approach:** Chronological, out-of-time evaluation

### Research objective

Investigate how fraud-screening systems should adapt when transaction distributions change over time, with particular emphasis on determining when probability recalibration or decision-policy updates may be sufficient and when predictive model retraining may be necessary.

The research investigates four interconnected aspects:

- Probability calibration under temporal separation.
- Predictive model adaptation using newer labelled transactions.
- Operational decision-making under constrained manual-review capacity.
- Statistical uncertainty and consistency of model comparisons under temporal dependence.

The experiments are organized into two major stages:

1. **Frozen-model adaptation experiments:** Compare static and adaptive probability calibration and decision policies while retaining the same predictive model.
2. **Controlled retraining experiments:** Compare frozen, recent-only, matched-size, and retrained predictive models, together with their calibration strategies.

All experiments use chronological partitions to preserve temporal ordering. The final held-out test partition remains untouched.

---

## 2. Dataset and Temporal Splitting

The IEEE-CIS dataset was integrated using `TransactionID` as the join key.

| Dataset property | Value |
|---|---:|
| Integrated training rows | 590,540 |
| Integrated columns | 434 |
| Fraud transactions | 20,663 |
| Legitimate transactions | 569,877 |
| Fraud prevalence | Approximately 3.50% |

### Chronological partitions

| Partition | Rows | Fraud cases | Fraud prevalence |
|---|---:|---:|---:|
| Training | 413,379 | 14,538 | 3.5169% |
| Validation | 88,581 | 3,042 | 3.4341% |
| Test | 88,580 | 3,083 | 3.4805% |

The partitions were created chronologically using `TransactionDT`.

Transaction IDs do not overlap across partitions. Temporal boundaries were validated.

**Important:** The final test partition has not been used in these experiments.

### Validation temporal windows

The validation partition was divided into three chronological windows.

| Window | Rows | Fraud cases | Fraud prevalence |
|---|---:|---:|---:|
| Initial calibration | 29,527 | 1,025 | 3.4714% |
| Adaptation | 29,527 | 1,037 | 3.5120% |
| Evaluation | 29,527 | 980 | 3.3190% |

The windows preserve chronological ordering.

---

## 3. Predictive Model

### Baseline model

A Logistic Regression pipeline was trained using:

- First 20,000 chronological training transactions.
- 562 fraud transactions in the training sample.
- 431 input features.
- Existing feature preprocessing pipeline.
- `class_weight=None`.
- `solver="liblinear"`.
- `max_iter=1000`.
- `random_state=42`.

The trained pipeline was saved as:

`artifacts/temporal/baseline_model.joblib`

Training time was approximately 11.62 seconds.

The same frozen model generated predictions for the initial calibration, adaptation, and evaluation windows.

### Important limitation

The model was trained on an early chronological prefix rather than the complete training partition.

Consequently, these results are pilot findings. They should not be interpreted as representative performance of a model trained on all 413,379 training transactions.

---

## 4. Frozen Baseline Predictions

| Window | Fraud prevalence | Mean predicted probability |
|---|---:|---:|
| Initial calibration | 3.4714% | 0.040537 |
| Adaptation | 3.5120% | 0.038329 |
| Evaluation | 3.3190% | 0.036204 |

Mean predicted probabilities decreased across the three windows.

This observation alone does not establish concept drift or calibration deterioration. Further statistical analysis is required.

---

## 5. Static Calibration Experiment

### Methodology

Two probability calibrators were fitted using the initial calibration window:

1. Sigmoid calibration.
2. Isotonic regression.

Both were evaluated on the later evaluation window.

The uncalibrated frozen model served as the baseline.

The sigmoid implementation uses a logistic mapping on clipped probability log-odds. It is a probability-level Platt-style transformation, not classic Platt scaling fitted to raw classifier margins.

### Results

| Method | Brier score ↓ | Log loss ↓ | ECE ↓ | Average Precision ↑ |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.034328 | 0.186696 | 0.025965 | 0.226144 |
| Sigmoid | 0.028968 | 0.129348 | 0.003746 | 0.224413 |
| Isotonic | 0.028774 | 0.126191 | 0.003998 | 0.209747 |

### Observations

- Both calibration methods improved Brier score, log loss, and ECE.
- Isotonic achieved the lowest Brier score and log loss.
- Sigmoid achieved the lowest ECE among the calibrated methods.
- The uncalibrated model retained the highest Average Precision.
- Isotonic calibration reduced Average Precision, potentially due in part to tied probability values.

These results suggest improved probability quality without an equivalent improvement in ranking performance.

---

## 6. Static Decision-Policy Experiment

### Methodology

The three strategies were evaluated using the same evaluation window.

A capacity-constrained decision policy was applied with:

- Review capacity: 200 transactions.
- Fixed illustrative cost matrix.
- Approve, review, and block actions.

### Illustrative cost assumptions

| Outcome | Cost |
|---|---:|
| Approve legitimate | 0 |
| Approve fraud | 100 |
| Review legitimate | 2 |
| Review fraud | 5 |
| Block legitimate | 10 |
| Block fraud | 0 |

These values are experimental assumptions, not empirically established banking costs.

### Results

| Strategy | Model-estimated expected cost | Realized cost | Fraud approved | Fraud reviewed | Fraud blocked | Legitimate blocked |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 37,639.36 | 76,305 | 596 | 15 | 369 | 1,626 |
| Static sigmoid | 74,237.14 | 77,577 | 734 | 39 | 207 | 366 |
| Static isotonic | 69,946.56 | 76,636 | 702 | 22 | 256 | 597 |

All strategies used 200 review slots.

### Observations

- Static sigmoid substantially reduced legitimate blocking compared with the uncalibrated strategy.
- This reduction came with more fraudulent transactions being approved.
- Uncalibrated decisions achieved the lowest realized cost in this comparison.
- Model-estimated expected cost did not identify the strategy with the lowest realized cost.

The difference between estimated and realized cost deserves further investigation.

---

## 7. Adaptive Calibration Experiment

### Methodology

The frozen model remained unchanged.

Two additional calibrators were fitted using the newer adaptation window:

1. Adaptive sigmoid.
2. Adaptive isotonic.

The original static calibrators were retained for comparison.

All five strategies were evaluated on the same later evaluation window.

### Probability-quality results

| Strategy | Brier score ↓ | Log loss ↓ | ECE ↓ | Average Precision ↑ |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.034328 | 0.186696 | 0.025965 | 0.226144 |
| Static sigmoid | 0.028968 | 0.129348 | 0.003746 | 0.224413 |
| Static isotonic | 0.028774 | 0.126191 | 0.003998 | 0.209747 |
| Adaptive sigmoid | 0.029008 | 0.129224 | 0.004407 | 0.224413 |
| Adaptive isotonic | 0.028841 | 0.125886 | 0.003278 | 0.205037 |

### Observations

- Adaptive calibration did not consistently outperform static calibration.
- Adaptive sigmoid was slightly worse than static sigmoid on Brier score and ECE.
- Adaptive isotonic achieved the best log loss and ECE.
- Static isotonic retained the best Brier score.
- Adaptive isotonic had lower Average Precision than static isotonic.

Updating calibration with newer labelled data does not guarantee improvements across every probability-quality metric.

---

## 8. Adaptive Decision-Policy Experiment

### Methodology

All five strategies were evaluated using:

- The same frozen predictive model.
- The same evaluation transactions.
- The same illustrative cost matrix.
- Review capacity of 200.

### Results

| Strategy | Estimated cost | Realized cost | Fraud approved | Fraud reviewed | Fraud blocked | Legitimate blocked |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 37,639.36 | 76,305 | 596 | 15 | 369 | 1,626 |
| Static sigmoid | 74,237.14 | 77,577 | 734 | 39 | 207 | 366 |
| Static isotonic | 69,946.56 | 76,636 | 702 | 22 | 256 | 597 |
| Adaptive sigmoid | 86,309.48 | 77,508 | 729 | 36 | 215 | 410 |
| Adaptive isotonic | 79,015.29 | 75,586 | 661 | 32 | 287 | 899 |

All five strategies used the full review capacity of 200.

### Key observations

**Adaptive isotonic achieved the lowest realized cost** in this particular evaluation.

Compared with the uncalibrated strategy, realized cost decreased from 76,305 to 75,586, approximately 0.94%.

Compared with static isotonic, realized cost decreased from 76,636 to 75,586, approximately 1.37%.

However:

- Static sigmoid had the lowest legitimate block rate.
- Uncalibrated had the highest fraud intervention rate.
- Adaptive isotonic provided a different balance between fraud intervention and legitimate blocking.

No strategy dominates every metric.

---

## 9. Preliminary Findings from the Frozen-Model Pilot

The experiments support the following preliminary observations:

1. Probability calibration improved probability-quality metrics for the frozen model.
2. Improved calibration did not guarantee lower realized operational cost.
3. Adaptive calibration did not consistently outperform static calibration.
4. Adaptive isotonic achieved the lowest realized cost in this particular pilot evaluation.
5. Review capacity and the cost assumptions strongly influence operational decisions.
6. Model-estimated expected cost and realized cost can differ substantially.

These are observations from a single experimental setup, not general conclusions about fraud detection systems.

---

## 10. Limitations

The following limitations must be considered when interpreting the experimental findings.

### Data and model limitations

- Only one model architecture (Logistic Regression) has been evaluated.
- The frozen baseline model was trained on the first 20,000 chronological training transactions rather than the complete training partition.
- The initial training sample covers an early chronological prefix and is not representative of the entire training period.
- The retrained, recent-only, and matched-size models use different training-data compositions. Their comparisons therefore do not isolate a single causal factor.
- The matched-size experiment uses deterministic temporal sampling and is not a perfectly controlled comparison of temporal coverage.

### Temporal evaluation limitations

- The controlled retraining experiment uses one adaptation/evaluation sequence.
- The evaluation window contains 29,527 transactions and 980 fraud cases.
- The results do not establish generalization across multiple future periods.
- Moving-block bootstrap preserves local chronological dependence within sampled blocks, but it does not establish independence between all sampled blocks or guarantee that the evaluation sequence is stationary.
- The block-length sensitivity analysis covers 250, 500, and 1,000 observations. Other dependence structures and block lengths may produce different uncertainty estimates.
- The transaction-level bootstrap assumes independent observations and may understate uncertainty when temporal dependence is present.

### Statistical limitations

- Confidence intervals are based on bootstrap resampling of a single evaluation window.
- Multiple model and metric comparisons were performed without a formal multiplicity correction.
- Bootstrap intervals should not be interpreted as proof of generalization to unseen future periods.
- No repeated rolling-origin temporal evaluation has yet been completed.

### Operational limitations

- The decision-cost matrix uses illustrative assumptions rather than empirically established banking costs.
- Review capacity is fixed at 200 transactions in the reported decision comparison.
- Operational conclusions may change under different fraud, review, and legitimate-blocking costs.
- The experiments assume that labels become available before recalibration or retraining.
- No complete production monitoring, delayed-label handling, or automated adaptation lifecycle has yet been evaluated.

The final chronological test partition remains untouched and must be reserved for final evaluation after experimental design and policy choices are frozen.

---

## 11. Next Experiments

The initial frozen-model adaptation experiments, controlled retraining comparison, and first-stage statistical uncertainty analysis have been completed.

The following experiments remain necessary to strengthen the research.

### Experiment A — Cost sensitivity analysis

Evaluate whether model and decision-policy rankings change under different assumptions for:

- Fraud approval costs.
- Legitimate transaction blocking costs.
- Manual review costs.
- Review capacity.

### Experiment B — Repeated temporal evaluation

Use additional chronological training, adaptation, calibration, and evaluation windows to investigate whether the observed findings persist across different periods.

### Experiment C — Drift detection and adaptation triggers

Investigate whether measurable changes in feature distributions, prediction distributions, and calibration quality can help determine when recalibration, policy updates, or model retraining should be considered.

### Experiment D — Adaptation lifecycle comparison

Compare controlled adaptation strategies under a consistent experimental protocol, including:

- No adaptation.
- Probability recalibration.
- Decision-policy updates.
- Model retraining.
- Combined calibration and retraining.

### Experiment E — Final untouched test evaluation

After the experimental design, adaptation strategy, and policy choices are frozen, conduct final evaluation on the held-out chronological test partition.

---

## 12. Current Project Status

| Component | Status |
|---|---|
| Dataset integration | Completed |
| Data validation | Completed |
| Chronological splitting | Completed |
| Feature preprocessing | Completed |
| Logistic Regression pilot | Completed |
| Static calibration | Completed |
| Static decision policies | Completed |
| Capacity-constrained decisions | Completed |
| Adaptive calibration | Completed |
| Adaptive decision comparison | Completed |
| Controlled retraining comparison | Completed |
| Recent-only model comparison | Completed |
| Matched-size model comparison | Completed |
| Prediction alignment validation | Completed |
| Transaction-level paired bootstrap | Completed |
| Temporal block bootstrap | Completed |
| Block-length sensitivity analysis | Completed |
| Cost sensitivity analysis | Pending |
| Repeated temporal evaluation | Pending |
| Final untouched test evaluation | Pending |

**Current conclusion:**

DriftSentinel now has a controlled temporal pilot comparing frozen, recent-only, matched-size, and retrained predictive models, alongside static and adaptive calibration strategies.

The preliminary results suggest that retraining can improve predictive performance over a frozen model in the examined evaluation window. However, the extent to which this improvement generalizes across future periods remains unverified.

---

## 13. Controlled Retraining Versus Recalibration Experiment

### 13.1 Research objective

This experiment investigates whether updating the predictive model using newer labelled transactions provides an improvement over retaining a frozen model and updating only its probability calibration.

It also examines whether training exclusively on recent data or combining historical and recent observations produces better performance.

Four model-training strategies were compared:

1. Frozen model trained on an early chronological sample.
2. Recent-only model trained exclusively on newer observations.
3. Matched-size model trained on a temporally distributed sample.
4. Retrained model trained using both historical and newer observations.

All models were evaluated on identical chronological evaluation transactions.

### 13.2 Controlled temporal protocol

The experiment used the following data partitions:

| Dataset | Rows | Fraud cases | Fraud prevalence |
|---|---:|---:|---:|
| Initial training sample | 20,000 | 562 | 2.8100% |
| Adaptation training | 14,764 | 568 | 3.8472% |
| Adaptation calibration | 14,763 | 469 | 3.1769% |
| Evaluation | 29,527 | 980 | 3.3190% |

The frozen model was trained on the initial training sample.

The full retrained model was trained using the combined initial training and adaptation training samples:

- Total training observations: 34,764.
- Fraud observations: 1,130.
- Fraud prevalence: 3.2505%.

The recent-only model was trained exclusively on the adaptation training sample.

The matched-size model used 20,000 deterministically selected observations from the combined historical and adaptation training data. This provided a comparison with the frozen model at an equal training-set size, although the sampling procedure did not create a perfectly controlled temporal-coverage experiment.

The adaptation calibration partition was kept separate from model fitting. It was used to fit the corresponding calibration strategies.

All four models generated predictions for the same adaptation calibration and evaluation observations. Prediction alignment was validated using transaction IDs, timestamps, and ground-truth labels.

The final evaluation partition remained untouched.

### 13.3 Calibration strategies

The experiment evaluated 14 model-calibration combinations.

**Frozen model — five strategies**

1. Frozen uncalibrated.
2. Frozen static sigmoid.
3. Frozen static isotonic.
4. Frozen adaptive sigmoid.
5. Frozen adaptive isotonic.

Static calibrators were fitted using the initial calibration window. Adaptive calibrators were fitted using the newer adaptation calibration window.

**Retrained model — three strategies**

6. Retrained uncalibrated.
7. Retrained sigmoid.
8. Retrained isotonic.

**Recent-only model — three strategies**

9. Recent-only uncalibrated.
10. Recent-only sigmoid.
11. Recent-only isotonic.

**Matched-size model — three strategies**

12. Matched-size uncalibrated.
13. Matched-size sigmoid.
14. Matched-size isotonic.

For the retrained, recent-only, and matched-size models, their sigmoid and isotonic calibrators were fitted using predictions on the adaptation calibration partition.

All strategies were evaluated on the same final evaluation partition.

The sigmoid implementation uses a logistic transformation of clipped probability log-odds. It is a probability-level Platt-style transformation rather than classic Platt scaling fitted to raw classifier margins.

### 13.4 Predictive performance results

| Model strategy | Brier score ↓ | Log loss ↓ | ECE ↓ | Average Precision ↑ | ROC-AUC ↑ |
|---|---:|---:|---:|---:|---:|
| Frozen uncalibrated | 0.034328 | 0.186696 | 0.025965 | 0.226144 | 0.711774 |
| Frozen static sigmoid | 0.028968 | 0.129348 | 0.003746 | 0.224413 | 0.711776 |
| Frozen static isotonic | 0.028774 | 0.126191 | 0.003998 | 0.209747 | 0.715651 |
| Frozen adaptive sigmoid | 0.029183 | 0.129550 | 0.002844 | 0.224413 | 0.711776 |
| Frozen adaptive isotonic | 0.028981 | 0.128693 | 0.002032 | 0.194901 | 0.716154 |
| Retrained uncalibrated | 0.028155 | 0.131994 | 0.014833 | 0.322355 | 0.800850 |
| Retrained sigmoid | 0.026549 | 0.117430 | 0.002291 | 0.322870 | 0.800846 |
| Retrained isotonic | 0.026801 | 0.122343 | 0.005516 | 0.297930 | 0.799383 |
| Recent-only uncalibrated | 0.030053 | 0.162087 | 0.017288 | 0.271685 | 0.792116 |
| Recent-only sigmoid | 0.028255 | 0.121728 | 0.009579 | 0.272583 | 0.792101 |
| Recent-only isotonic | 0.027718 | 0.134343 | 0.003675 | 0.253512 | 0.793836 |
| Matched-size uncalibrated | 0.029203 | 0.149832 | 0.017877 | 0.300825 | 0.765732 |
| Matched-size sigmoid | 0.027132 | 0.121721 | 0.004223 | 0.301616 | 0.765739 |
| Matched-size isotonic | 0.027029 | 0.135594 | 0.002422 | 0.281115 | 0.762543 |

### 13.5 Key comparisons

**A. Frozen versus full retraining**

The retrained uncalibrated model improved over the frozen uncalibrated model:

| Metric | Frozen | Retrained | Difference |
|---|---:|---:|---:|
| Average Precision | 0.226144 | 0.322355 | +0.096211 |
| ROC-AUC | 0.711774 | 0.800850 | +0.089075 |
| Brier score | 0.034328 | 0.028155 | -0.006173 |
| Log loss | 0.186696 | 0.131994 | -0.054702 |

The retrained model showed improvements in both ranking and probability-quality metrics.

**B. Recent-only versus full retraining**

The full retrained model achieved higher Average Precision, ROC-AUC, and better probability-quality metrics than the recent-only model.

The Average Precision difference was +0.050669 in favour of full retraining. The ROC-AUC difference was +0.008734.

The difference in ROC-AUC requires cautious interpretation because its paired bootstrap confidence interval includes zero.

**C. Matched-size versus frozen model**

The matched-size model achieved an Average Precision of 0.300825 compared with 0.226144 for the frozen model.

The difference was +0.074682.

This provides evidence that the temporally distributed training sample performed better than the early frozen sample in this evaluation window. However, differences in sample composition and fraud prevalence mean that temporal coverage cannot be isolated as the only explanation.

**D. Calibration effects**

Sigmoid calibration improved Brier score and log loss for all four model-training strategies.

Isotonic calibration generally improved calibration-related metrics but sometimes reduced Average Precision.

This reinforces the distinction between probability calibration and ranking performance.

The best calibration method should therefore depend on the intended operational objective rather than a single metric.

### 13.6 Preliminary interpretation

The controlled temporal pilot provides evidence that retraining can improve predictive performance over retaining a frozen model when newer labelled observations become available.

The full retrained model performed best among the four uncalibrated models on the reported Average Precision and ROC-AUC metrics.

The results also indicate that updating calibration alone does not necessarily recover the ranking improvements observed after model retraining.

However, these findings do not establish that retraining is always preferable. The models were trained on different sample sizes and temporal compositions, and the experiment evaluates only one future evaluation window.

Further rolling-origin experiments are required to determine whether the observed improvements persist across different periods.

### 13.7 Reproducibility artifacts

The experiment outputs are stored in the following project locations:

- `artifacts/temporal/baseline_model.joblib`
- `artifacts/temporal/recent_only_model.joblib`
- `artifacts/temporal/matched_size_model.joblib`
- `artifacts/temporal/retrained_model.joblib`
- `experiments/results/retraining_calibration_comparison.csv`
- `experiments/results/temporal_predictions/`
- `data/processed/temporal/matched_size_train.parquet`

The experiment scripts include:

- `scripts/train_temporal_baseline.py`
- `scripts/train_retrained_model.py`
- `scripts/train_recent_only_model.py`
- `scripts/train_matched_size_model.py`
- `scripts/score_retraining_models.py`
- `scripts/run_retraining_comparison.py`

---

## 14. Statistical Uncertainty and Temporal Block Bootstrap

### 14.1 Objective

The model comparison experiment produced performance estimates on a single chronological evaluation window.

To investigate the uncertainty around these estimates, two paired bootstrap procedures were conducted:

1. Transaction-level paired bootstrap.
2. Moving temporal block paired bootstrap.

Both methods compare models on the same resampled observations, preserving paired comparisons between their predictions.

The temporal block bootstrap was introduced to account for local chronological dependence that may not be represented by independent transaction-level resampling.

### 14.2 Evaluation Dataset

| Property | Value |
|---|---:|
| Evaluation transactions | 29,527 |
| Fraud transactions | 980 |
| Fraud prevalence | 3.3190% |
| Models compared | 4 |
| Performance metrics | 4 |
| Temporal block lengths | 250, 500, 1,000 |

The four predictive models were:

- Frozen.
- Recent-only.
- Matched-size.
- Retrained.

The evaluation observations and labels were identical across all four models.

### 14.3 Transaction-Level Paired Bootstrap

The initial bootstrap used 2,000 paired resamples of individual transactions, with random seed 42.

For every resample, the same sampled transaction indices were applied to all four models.

The resulting percentile confidence intervals were:

| Model | Average Precision | ROC-AUC | Brier score | Log loss |
|---|---|---|---|---|
| Frozen | 0.2261 [0.1986, 0.2545] | 0.7118 [0.6913, 0.7317] | 0.03433 [0.03246, 0.03611] | 0.18670 [0.17359, 0.20007] |
| Recent-only | 0.2717 [0.2440, 0.3013] | 0.7921 [0.7753, 0.8085] | 0.03005 [0.02834, 0.03178] | 0.16209 [0.15012, 0.17380] |
| Matched-size | 0.3008 [0.2692, 0.3317] | 0.7657 [0.7473, 0.7825] | 0.02920 [0.02745, 0.03088] | 0.14983 [0.13963, 0.16053] |
| Retrained | 0.3224 [0.2911, 0.3526] | 0.8009 [0.7842, 0.8163] | 0.02816 [0.02647, 0.02973] | 0.13199 [0.12362, 0.14075] |

These intervals describe uncertainty under transaction-level resampling. They do not account for temporal dependence between observations.

### 14.4 Moving Temporal Block Bootstrap

To better preserve local chronological dependence, a moving temporal block bootstrap was implemented.

Instead of sampling individual transactions independently, the procedure:

1. Maintains the original chronological order of evaluation observations.
2. Selects contiguous, non-circular blocks of observations.
3. Samples blocks with replacement.
4. Concatenates sampled blocks until the original evaluation sample size is reached.
5. Truncates the final sample to the original number of observations.
6. Applies identical sampled indices to all four models.

Three block lengths were evaluated:

| Block length | Bootstrap replicates | Valid replicates |
|---|---:|---:|
| 250 observations | 1,000 | 1,000 |
| 500 observations | 1,000 | 1,000 |
| 1,000 observations | 1,000 | 1,000 |

Random seed: 42.

The block-length sensitivity analysis investigates whether conclusions are sensitive to the selected local dependence scale.

### 14.5 Temporal Block Bootstrap Confidence Intervals

The following tables present the 95% percentile confidence intervals.

**Block length: 250 observations**

| Model | Average Precision | ROC-AUC | Brier score | Log loss |
|---|---|---|---|---|
| Frozen | 0.2261 [0.1752, 0.2808] | 0.7118 [0.6850, 0.7399] | 0.03433 [0.03124, 0.03701] | 0.18670 [0.16774, 0.20234] |
| Recent-only | 0.2717 [0.2218, 0.3293] | 0.7921 [0.7696, 0.8171] | 0.03005 [0.02727, 0.03265] | 0.16209 [0.14488, 0.17849] |
| Matched-size | 0.3008 [0.2430, 0.3557] | 0.7657 [0.7380, 0.7900] | 0.02920 [0.02664, 0.03149] | 0.14983 [0.13512, 0.16364] |
| Retrained | 0.3224 [0.2624, 0.3777] | 0.8009 [0.7780, 0.8227] | 0.02816 [0.02582, 0.03041] | 0.13199 [0.11971, 0.14268] |

**Block length: 500 observations**

| Model | Average Precision | ROC-AUC | Brier score | Log loss |
|---|---|---|---|---|
| Frozen | 0.2261 [0.1772, 0.2779] | 0.7118 [0.6861, 0.7412] | 0.03433 [0.03115, 0.03688] | 0.18670 [0.16815, 0.20222] |
| Recent-only | 0.2717 [0.2236, 0.3259] | 0.7921 [0.7678, 0.8184] | 0.03005 [0.02711, 0.03256] | 0.16209 [0.14361, 0.17876] |
| Matched-size | 0.3008 [0.2426, 0.3524] | 0.7657 [0.7382, 0.7898] | 0.02920 [0.02667, 0.03123] | 0.14983 [0.13440, 0.16433] |
| Retrained | 0.3224 [0.2628, 0.3774] | 0.8009 [0.7766, 0.8236] | 0.02816 [0.02557, 0.03033] | 0.13199 [0.11936, 0.14406] |

**Block length: 1,000 observations**

| Model | Average Precision | ROC-AUC | Brier score | Log loss |
|---|---|---|---|---|
| Frozen | 0.2261 [0.1748, 0.2719] | 0.7118 [0.6850, 0.7405] | 0.03433 [0.03119, 0.03657] | 0.18670 [0.16793, 0.19775] |
| Recent-only | 0.2717 [0.2198, 0.3245] | 0.7921 [0.7700, 0.8179] | 0.03005 [0.02690, 0.03258] | 0.16209 [0.14138, 0.17957] |
| Matched-size | 0.3008 [0.2402, 0.3459] | 0.7657 [0.7339, 0.7891] | 0.02920 [0.02680, 0.03123] | 0.14983 [0.13450, 0.16326] |
| Retrained | 0.3224 [0.2588, 0.3709] | 0.8009 [0.7749, 0.8239] | 0.02816 [0.02559, 0.03032] | 0.13199 [0.11927, 0.14113] |

### 14.6 Paired Model Differences

Differences are defined as:

Comparison model metric minus reference model metric.

For Brier score and log loss, negative differences favour the comparison model. For Average Precision and ROC-AUC, positive differences favour the comparison model.

The following table reports the full retrained model compared with the frozen model.

| Block length | Metric | Difference | 95% confidence interval |
|---|---|---:|---|
| 250 | Average Precision | +0.096211 | [0.063438, 0.129886] |
| 250 | ROC-AUC | +0.089075 | [0.067080, 0.109973] |
| 250 | Brier score | -0.006173 | [-0.007843, -0.004557] |
| 250 | Log loss | -0.054702 | [-0.064766, -0.043376] |
| 500 | Average Precision | +0.096211 | [0.059704, 0.125351] |
| 500 | ROC-AUC | +0.089075 | [0.064443, 0.111010] |
| 500 | Brier score | -0.006173 | [-0.008083, -0.004405] |
| 500 | Log loss | -0.054702 | [-0.065737, -0.043697] |
| 1,000 | Average Precision | +0.096211 | [0.057632, 0.127141] |
| 1,000 | ROC-AUC | +0.089075 | [0.064419, 0.107921] |
| 1,000 | Brier score | -0.006173 | [-0.007832, -0.004356] |
| 1,000 | Log loss | -0.054702 | [-0.063643, -0.043096] |

All twelve confidence intervals exclude zero.

The results indicate that the retrained model outperformed the frozen model across all four metrics for this evaluation window, with the direction of the differences remaining consistent across all tested block lengths.

### 14.7 Recent-Only Versus Full Retraining

The paired comparison between the recent-only and retrained models provides additional insight.

| Metric | Difference | 95% CI, block 250 | 95% CI, block 500 | 95% CI, block 1,000 |
|---|---:|---|---|---|
| Average Precision | +0.050669 | [0.017353, 0.078439] | [0.015669, 0.076018] | [0.016021, 0.069550] |
| ROC-AUC | +0.008734 | [-0.010884, 0.024507] | [-0.011649, 0.023593] | [-0.011983, 0.021685] |
| Brier score | -0.001898 | [-0.003037, -0.000708] | [-0.003150, -0.000572] | [-0.003068, -0.000512] |
| Log loss | -0.030093 | [-0.042582, -0.018125] | [-0.042700, -0.016685] | [-0.042810, -0.017159] |

The Average Precision, Brier score, and log loss intervals exclude zero across all three block lengths.

The ROC-AUC interval includes zero across all three block lengths. Therefore, the observed ROC-AUC improvement is not clearly distinguishable from zero under these bootstrap comparisons.

### 14.8 Block-Length Sensitivity Observations

The main findings remained directionally consistent across block lengths of 250, 500, and 1,000 observations.

In particular:

- The retrained model consistently outperformed the frozen model on all four metrics.
- The retrained model consistently outperformed the recent-only model on Average Precision, Brier score, and log loss.
- The ROC-AUC difference between recent-only and retrained models remained uncertain because its interval included zero.
- The retrained model retained the highest point estimate for Average Precision and ROC-AUC among the four uncalibrated models.

The confidence intervals changed with block length, but the principal frozen-versus-retrained comparison remained robust to the three tested block sizes.

### 14.9 Statistical Interpretation and Limitations

The temporal block bootstrap provides a more dependence-aware uncertainty analysis than independent transaction-level resampling.

However, it does not eliminate all statistical limitations.

The procedure preserves local order within sampled blocks, but blocks are sampled independently with replacement. Dependence across block boundaries is not fully preserved.

The analysis also relies on the assumption that the observed chronological sequence provides a useful basis for approximating local temporal behaviour.

The evaluation period remains a single chronological window. Bootstrap intervals quantify uncertainty under the chosen resampling procedure; they do not measure performance variation across entirely different future periods.

No multiplicity correction was applied across the collection of model and metric comparisons.

Accordingly, confidence intervals excluding zero should be interpreted as evidence of a difference under the specified evaluation and resampling design, not as proof of universal superiority.

Repeated rolling-origin temporal evaluation remains necessary.

### 14.10 Reproducibility Artifacts

Transaction-level bootstrap outputs:

- `experiments/results/bootstrap_model_intervals.csv`
- `experiments/results/bootstrap_paired_differences.csv`
- `experiments/results/bootstrap_metadata.json`

Temporal block bootstrap outputs:

- `experiments/results/temporal_block_bootstrap_intervals.csv`
- `experiments/results/temporal_block_bootstrap_differences.csv`

Relevant experiment scripts:

- `scripts/run_paired_bootstrap.py`
- `scripts/run_temporal_block_bootstrap.py`
- `scripts/profile_evaluation_time_blocks.py`
- `scripts/profile_six_hour_blocks.py`