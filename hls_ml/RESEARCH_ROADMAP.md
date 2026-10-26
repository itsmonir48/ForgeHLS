# Research Roadmap: Uncertainty-Aware Multi-Objective HLS QoR Prediction Under Kernel-Level Distribution Shift

## 1. Executive Summary & Thesis Statement
- **Paper Title / Topic**: *A Systematic Reliability Study and Uncertainty-Aware Evaluation of Multi-Objective HLS QoR Prediction Under Kernel-Level Distribution Shift*
- **One-Line Thesis**:
  Predict multi-objective QoR $\hat{y} = [\widehat{\text{Latency}}, \widehat{\text{LUT}}, \widehat{\text{DSP}}, \widehat{\text{FF}}, \widehat{\text{BRAM}}]$ along with prediction uncertainty $u(x)$ to quantify reliability, evaluated on structurally unseen kernels (Leave-Kernel-Out).
- **Core Positioning**:
  Avoid trivial "ML for QoR" or "first uncertainty" claims. The contribution is a **strict reliability-oriented evaluation framework** combining:
  - Kernel-level distribution shift (OOD generalization)
  - Multi-objective QoR prediction
  - Uncertainty quantification & calibration (ENCE, PICP, MPIW)
  - Risk-coverage selective prediction (abstaining on high-uncertainty candidates to trigger selective verification)

---

## 2. Core Scientific Hypotheses
- **H1 (Random Split is Optimistic)**: Pragma variants of the same base kernel leaking into train and test inflate nominal accuracy metrics.
- **H2 (Kernel-Disjoint Split is Harder)**: Testing on unseen logical kernels introduces structural distribution shift, significantly degrading prediction accuracy.
- **H3 (Uncertainty is Informative)**: Higher prediction uncertainty $u(x)$ strongly correlates with higher actual absolute QoR prediction error $|y - \hat{y}|$.
- **H4 (Selective Prediction Helps)**: Abstaining/rejecting the most uncertain predictions progressively lowers the error of retained predictions along a well-calibrated Risk-Coverage curve.

---

## 3. Dataset Profile & Audit Facts (ForgeHLS Release)
- **Source Dataset**: `data_of_designs_forgehls.json` (184,717 records)
- **Unique Algorithms**: 471
- **Unique Sources / Families**: 11
- **Logical Kernel Groups**: 491 groups defined as $G = (\text{source\_name}, \text{algo\_name})$
- **Unique Design IDs**: 1,242
- **Pareto Designs**: 3,803 (True: 3,803 / False: 180,914)
- **Missing Values in Core Fields**: 0 across all 184,717 rows
- **Synthesized Baseline Invariance**: 483 multi-row groups reduce to a single base program after removing HLS pragmas.

---

## 4. Feature & Target Formulation
### Excluded Inputs (Leakage Prevention)
- `source_name`, `algo_name`, `design_id` (identity leakage)
- `latency-resource-strategy` (target-derived label leakage)
- `is_kernel` (all False in this release)

### Feature Sets
- **Feature Set A (Basic)**:
  `TargetClockPeriod`, `code_length`, `pragma_number`
- **Feature Set B (Pragma-Aware)**:
  Set A + counts and parameters of:
  `pipeline_count`, `pipeline_enabled`, `pipeline_disabled`, `unroll_count`, `max_unroll_factor`, `array_partition_count`, `max_partition_factor`, `array_reshape_count`, `dataflow_count`, `inline_count`, `resource_count`, `bind_op_count`, `bind_storage_count`, etc.
- **Feature Set C (Structural + Pragma)**:
  Set B + deterministic C/C++ structural metrics:
  `function_count`, `loop_count`, `nested_loop_count`, `branch_count`, `array_count`, `operator_count`, `memory_access_count`, `max_nesting_depth`.

### Target Variables ($y$)
- `Worst-caseLatency`
- `LUT`
- `DSP`
- `FF`
- `BRAM_18K`

---

## 5. Experimental Matrix

| Exp ID | Split Protocol | Model Family | Objective |
| :--- | :--- | :--- | :--- |
| **E1** | Random Row Split (80/20) | Ridge, RF, XGBoost, MLP | Baseline optimistic performance |
| **E2** | Kernel-Disjoint (LKO) | Ridge, RF, XGBoost, MLP | Generalization under kernel shift |
| **E3** | Repeated Grouped Split (5 seeds) | Best models | Statistical stability (mean ± std) |
| **E4** | Kernel-Disjoint | Tree-Ensemble Disagreement UQ | Prediction interval & epistemic UQ |
| **E5** | Kernel-Disjoint | Deep Ensemble (MLP $\times$ 5) | Multi-model variance UQ |
| **E6** | All Splits | Tree-Ensemble & Deep Ensemble | Calibration analysis (PICP, ENCE) |
| **E7** | All Splits | Uncertainty Predictors | Selective prediction & Risk-Coverage (AURC) |
| **E8** | Kernel-Disjoint | RF / XGBoost | Feature ablation (Set A vs B vs C) |
| **E9** | Kernel-Disjoint | Model Ablations | Nominal vs Calibrated Uncertainty |
| **E10** | Unseen Test Kernels | Predictive Model | (Optional) Uncertainty-guided Pareto filtering |

---

## 6. Implementation Phases & Script Architecture

```text
hls_ml/
├── data/                       # Cached matrices (X.parquet, y.parquet, groups.parquet)
├── figures/                    # Publication-ready plots (calibration, risk-coverage, shift)
├── results/                    # Metric tables, CSV logs, audit reports
└── scripts/
    ├── dataset_audit.py               # [DONE] Phase 1 & 2
    ├── kernel_group_audit.py          # [DONE] Phase 1 & 2
    ├── source_normalization_audit.py  # [DONE] Phase 1 & 2
    ├── pragma_audit.py                # [PHASE 3 - NEXT] Comprehensive pragma vocabulary audit
    ├── feature_extractor.py           # [PHASE 3] Feature Sets A, B, C extraction
    ├── build_dataset.py               # [PHASE 4] Build tabular arrays and save metadata
    ├── make_splits.py                 # [PHASE 4] Generate random, group-disjoint, family splits
    ├── train_baselines.py             # [PHASE 5 & 6] Train Ridge, RF, XGBoost, MLP (E1, E2, E3)
    ├── train_uncertainty.py           # [PHASE 7] Train Tree UQ, Deep Ensemble (E4, E5)
    ├── evaluate.py                    # [PHASE 8] Evaluate calibration, ENCE, Risk-Coverage
    ├── ablation_study.py              # [PHASE 9] Feature & model ablation
    └── make_figures.py                # [PHASE 11] Generate IEEE-ready publication charts
```
