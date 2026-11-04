# IEEE Conference Paper Writing Guide

## ForgeHLS Reliability Study

This guide is the working document for preparing the conference manuscript from the
verified ForgeHLS experiment. Use the real CSV, TXT, and PNG artifacts listed below.
Do not invent, smooth, round, or replace values with assumed results.

The current manuscript draft is:

- `ForgeHLS_reliability_paper.tex`

The final conference template always takes precedence over this guide.

---

## 1. Recommended paper title

Use one of these titles:

**Primary title**

> Reliability-Aware Multi-Objective HLS QoR Prediction Under Kernel-Level Distribution Shift

**Alternative title**

> An Uncertainty-Aware Evaluation of HLS QoR Prediction on Unseen Kernels

Do not use a title claiming state-of-the-art accuracy or universal improvement.

---

## 2. Paper positioning

The paper should be positioned as a:

- reliability and generalization study;
- leakage-controlled HLS QoR prediction framework;
- uncertainty, calibration, and selective-prediction evaluation.

It should **not** be positioned as:

- a new HLS compiler;
- a new hardware architecture;
- a universally more accurate predictor;
- a cyber-security attack detector;
- a security-control system.

The attached symposium call is security-focused. Before submission, manually confirm
with the organizers that trustworthy-AI/HLS methodology is within scope. If not, use
an EDA, FPGA, hardware-design-automation, or trustworthy-ML venue.

---

## 3. Verified data inventory

Use these values in the paper:

| Item | Verified value |
|---|---:|
| Real dataset records | 184,717 |
| Unique sources | 11 |
| Unique algorithm names | 471 |
| Logical source--algorithm groups | 491 |
| Unique design IDs | 1,242 |
| Pareto records | 3,803 |
| Input features | 42 |
| QoR targets | 5 |
| E3 grouped seeds | 42, 43, 44, 45, 46 |
| E3 train groups per seed | 392 |
| E3 test groups per seed | 99 |
| E2/E3 group overlap | 0 |

Targets:

1. `Worst-caseLatency`
2. `LUT`
3. `DSP`
4. `FF`
5. `BRAM_18K`

The dataset audit reported zero missing required fields and the rebuilt feature matrix
reported zero NaNs across all 42 features and 5 targets.

Primary data sources:

- `../hls_ml/results/dataset_audit_summary.txt`
- `../hls_ml/results/splits_summary.txt`
- `../hls_ml/data/dataset_schema.json`
- `../hls_ml/results/baseline_results_e1_vs_e2.csv`

---

## 4. Required paper structure

Use the following order in the official IEEE template.

### Abstract

Include:

1. HLS QoR prediction motivation.
2. Problem with random row splits.
3. Real ForgeHLS dataset and kernel-disjoint protocol.
4. Main verified findings.
5. Reliability contribution.

Do not say:

- “all targets achieve high accuracy”;
- “our method is state-of-the-art”;
- “uncertainty is perfectly calibrated”.

### Index Terms

Use:

> high-level synthesis, quality of result, distribution shift, uncertainty quantification, calibration, selective prediction, trustworthy machine learning

### I. Introduction

Explain:

- HLS design-space exploration is expensive.
- ML surrogates can reduce synthesis cost.
- Multiple configurations of one kernel create dependence.
- Random row splits can leak kernel-family information.
- A practical predictor must expose when it is unreliable.

End the introduction with these contributions:

1. leakage-controlled kernel-disjoint evaluation;
2. multi-objective prediction of five QoR targets;
3. repeated grouped evaluation over five seeds;
4. uncertainty, calibration, and selective prediction analysis;
5. target-dependent tuning and limitation analysis.

### II. Related Work

Discuss:

- ForgeHLS as the source dataset;
- classical HLS QoR prediction;
- tree and neural surrogate models;
- uncertainty-aware DSE;
- calibration and conformal/selective prediction.

Do not claim that Random Forest, Gradient Boosting, or MLP is novel by itself.

### III. Dataset and Feature Construction

Report the verified dataset inventory from Section 3 of this guide.

Define the logical group exactly as:

```text
group = (source_name, algo_name)
```

Explain that source names, algorithm names, design IDs, and target-derived strategy
labels are not model input features.

Describe:

- Set A: basic features;
- Set B: pragma-aware features;
- Set C: source-structure features.

### IV. Evaluation Protocol

Define:

- E1: random row split, 80/20, seed 42;
- E2: group-disjoint split;
- E3: group-disjoint splits with seeds 42--46.

State explicitly:

> E1 is used as an optimism diagnostic. E2 and E3 are the primary generalization
> evaluations.

Metrics:

- primary: LogMAE;
- secondary: MAE, median absolute error, normalized MAE, and R2;
- repeated results: mean plus/minus standard deviation over E3 seeds.

### V. Results

Use the subsections below:

1. Random-row versus kernel-disjoint shift.
2. Repeated grouped evaluation.
3. Uncertainty--error association.
4. Calibration and risk--coverage.
5. Validation-only tuning.

### VI. Discussion

Explain:

- latency is more predictable than resources;
- resources show weak or unstable grouped generalization;
- uncertainty is useful for ranking risk;
- uncertainty ranking does not guarantee calibrated intervals;
- zero-heavy BRAM/DSP and heavy-tailed latency require cautious metrics.

### VII. Threats to Validity

Include:

- one public dataset release;
- source--algorithm grouping is a practical proxy;
- standard ensemble uncertainty is not a formal guarantee;
- zero-heavy targets affect ratios and raw metrics;
- conference scope must be confirmed.

### VIII. Reproducibility

State that the workflow:

- rebuilds features from the original JSON;
- regenerates splits;
- asserts zero group overlap;
- trains baselines before UQ;
- exports tables, figures, and arrays.

### IX. Conclusion

Conclude that random row splits can hide generalization degradation and that
uncertainty-aware selective use is promising but calibration remains target-dependent.

---

## 5. Tables to include

### Table 1: Dataset summary

Create manually from the verified inventory:

- records;
- sources;
- algorithms;
- groups;
- design IDs;
- Pareto records;
- features;
- targets.

### Table 2: E1 versus E2

Use:

- `../hls_ml/results/baseline_results_e1_vs_e2.csv`

Recommended columns:

- Model;
- Target;
- E1_R2;
- E2_R2;
- E1_LogMAE;
- E2_LogMAE.

Use Random Forest and Gradient Boosting as the compact main comparison. Put the full
model table in an appendix if page limits require it.

### Table 3: E3 repeated results

Use:

- `../hls_ml/results/baseline_results_e3_repeated.csv`

Report, for each model and target:

- `R2_mean +/- R2_std`;
- `LogMAE_mean +/- LogMAE_std`.

This table is essential. Do not rely only on one E2 split.

### Table 4: Uncertainty results

Use:

- `../hls_ml/results/uncertainty_results.csv`

Report:

- method;
- target;
- Spearman correlation;
- Q1 MAE;
- Q5 MAE;
- Q5/Q1 ratio.

Explain that ratios for zero-heavy targets can be numerically large and should not be
interpreted as universal accuracy multipliers.

### Table 5: Calibration

Use:

- `../hls_ml/results/calibration_metrics.csv`

Report:

- nominal 80% PICP;
- normalized MPIW;
- Tree ENCE;
- Deep Ensemble ENCE.

Do not hide the high ENCE values for latency or any other target.

### Table 6: Tuning ablation

Use:

- `../hls_ml/results/tuned_results_e3_summary.csv`

Include this table as a secondary or appendix table. State that tuning is target
dependent and does not provide universal improvement.

---

## 6. Figures to include

All four figures already exist and should be inserted from the generated PNG files.
Do not redraw them manually unless the final conference requires a different format.

### Figure 1: Distribution shift

File:

- `../hls_ml/figures/fig1_distribution_shift_e1_vs_e2.png`

Place in the Results section immediately after the E1/E2 table.

Caption:

> Performance degradation from random-row evaluation to kernel-disjoint evaluation.

The annotations must be the E2/E1 NMAE ratios, not raw MAE ratios. The verified
ratios are approximately 3.2x for latency, 2.1x for LUT, 1.9x for DSP, 1.6x for FF,
and 7.2x for BRAM.

### Figure 2: Uncertainty and error

File:

- `../hls_ml/figures/fig2_uncertainty_correlation_quintiles.png`

Place after the uncertainty table.

Caption:

> Uncertainty quintiles expose the association between predictive uncertainty and
> absolute QoR error for the four resource targets shown; latency is reported in the
> uncertainty table.

### Figure 3: Risk--coverage

File:

- `../hls_ml/figures/fig3_risk_coverage_curves.png`

Place in the calibration/selective-prediction subsection.

Caption:

> Risk--coverage curves for uncertainty-based selective prediction.

The BRAM error-reduction-factor curve must not be shown: zero retained MAE creates
unstable and misleading ratios. Deep-ensemble selective prediction must be reported
as a negative result at 80% coverage, where the retained-error ratio is approximately
0.83--0.93 for the reported targets.

### Figure 4: Feature ablation

File:

- `../hls_ml/figures/fig4_feature_ablation.png`

Place in an ablation subsection or appendix. Use:

- `../hls_ml/results/feature_ablation_results.csv`

Caption:

> Effect of basic, pragma-aware, and structural-plus-pragma feature sets on
> uncertainty--error ranking under grouped evaluation.

This figure shows uncertainty--error Spearman correlation, not point-prediction
accuracy. State explicitly that Set C can have the strongest uncertainty ranking while
having poor grouped point $R^2$ for LUT (-11.643), DSP (-15.064), and FF (-2.004).

### Required uncertainty caveat

The tree uncertainty--error correlation is partly scale-confounded. In the saved E2
predictions, tree uncertainty correlates with predicted magnitude at approximately
0.945--0.985. Within predicted-magnitude deciles, the mean Spearman association
drops to approximately 0.29--0.56 depending on target. Describe tree uncertainty as
a useful risk-ranking signal, not as an independent scale-invariant error estimator.

---

## 7. Exact verified result statements

These statements can be used after checking the final CSV files:

1. The verified release contains 184,717 real synthesis-derived records and 491
   source--algorithm groups.
2. Every E2/E3 grouped split has zero train--test group overlap.
3. Gradient Boosting gives the strongest fixed-baseline E2 latency result with
   approximately R2 = 0.782.
4. Under E2, Random Forest resource R2 is approximately -0.043 for LUT, -0.372 for
   DSP, 0.012 for FF, and -0.024 for BRAM.
5. Across E3 seeds, Gradient Boosting latency is approximately
   R2 = 0.346 +/- 0.247 and LogMAE = 1.594 +/- 0.181.
6. Tree uncertainty has Spearman correlation from approximately 0.887 to 0.976 with
   absolute error across the five targets in the verified run.
7. Nominal 80% tree interval coverage is approximately 78.6%--83.0%, but ENCE is
   target-dependent.
8. Validation-only tuning does not yield universal improvement.

Before final submission, compare every number with the final Colab-synchronized CSV.

---

## 8. Manual work required from the author

The following items cannot be safely invented:

- author names;
- affiliations;
- email addresses;
- funding and grant information;
- conflict-of-interest statement;
- corresponding-author designation;
- exact conference name and paper ID;
- final dataset URL/version;
- final Git commit hash;
- final Colab package versions;
- page limit and formatting requirements;
- final decision about whether the security symposium accepts this topic.

The author must also:

1. run the final Colab notebook from a fresh runtime;
2. save the final notebook with outputs;
3. confirm that all final CSV values match the manuscript;
4. obtain permission for any dataset or third-party figure usage;
5. use the official conference IEEE template;
6. run the conference PDF checker before submission.

---

## 9. What must not be added manually

Do not manually add:

- fake accuracy values;
- synthetic records presented as real data;
- unverified confidence intervals;
- security attack results;
- claims of state-of-the-art performance;
- claims of perfect calibration;
- claims that tuning improves every target;
- figures not generated from the verified artifacts.

---

## 10. Final submission checklist

- [ ] Author block completed.
- [ ] Conference scope confirmed.
- [ ] Official IEEE template applied.
- [ ] Page limit checked.
- [ ] Final Colab run completed from a fresh runtime.
- [ ] Dataset and split audit outputs archived.
- [ ] E1/E2 table checked against CSV.
- [ ] E3 mean and standard deviation table checked against CSV.
- [ ] UQ table checked against CSV.
- [ ] Calibration table checked against CSV.
- [ ] Figures inserted with readable labels.
- [ ] All claims are target-dependent and evidence-based.
- [ ] Limitations section retained.
- [ ] Dataset and ForgeHLS citation included.
- [ ] Git commit, package versions, and dataset version recorded.
- [ ] PDF checker passed.
