"""Update an unpacked copy of paper v3 with the results in hls_ml/results -> paper v4.

Every number is read from hls_ml/results; no value is typed by hand.

Usage (from the repository root):
  python -c "import zipfile; zipfile.ZipFile('paper/ForgeHLS_IEEE_Conference_Paper_v3 (2).docx').extractall('v4_unpacked')"
  python <docx-skill>/scripts/merge_runs.py v4_unpacked/      # merge Word's split text runs first
  python paper/build_paper_v4.py v4_unpacked
  (cd v4_unpacked && zip -qXrDy ../paper/ForgeHLS_IEEE_Conference_Paper_v4.docx .)

Paragraph indices refer to v3 after run merging; expect() guards every edit.
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "hls_ml/results"
F = ROOT / "hls_ml/figures"
U = Path(sys.argv[1])  # unpacked docx dir
DOC = U / "word/document.xml"
T = ["Worst-caseLatency", "LUT", "DSP", "FF", "BRAM_18K"]

xml = DOC.read_text(encoding="utf-8")
pieces = re.split(r"(<w:p[ >].*?</w:p>)", xml, flags=re.S)
para_pos = [k for k, s in enumerate(pieces) if re.match(r"<w:p[ >]", s)]


def ptext(i):
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", pieces[para_pos[i]]))


def set_para(i, text, run=-1):
    """Replace the text of one run (default: the last text run) of paragraph i."""
    p = pieces[para_pos[i]]
    ts = list(re.finditer(r"<w:t(?: [^>]*)?>[^<]*</w:t>", p))
    m = ts[run]
    new = f'<w:t xml:space="preserve">{escape(text)}</w:t>'
    pieces[para_pos[i]] = p[:m.start()] + new + p[m.end():]


def sub_para(i, old, new):
    p = pieces[para_pos[i]]
    eo = escape(old)
    assert eo in p, f"para {i}: {old!r} not found in {ptext(i)[:80]!r}"
    pieces[para_pos[i]] = p.replace(eo, escape(new), 1)


def last_run_text(i):
    p = pieces[para_pos[i]]
    return re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p)[-1]


def rep(i, old, new):
    """Replace text inside the LAST text run only (keeps e.g. a bold lead-in run)."""
    from xml.sax.saxutils import unescape
    t = unescape(last_run_text(i))
    assert old in t, f"para {i}: {old!r} not in last run {t[:80]!r}"
    set_para(i, t.replace(old, new, 1))


def expect(i, start):
    assert ptext(i).startswith(start), f"para {i} starts {ptext(i)[:50]!r}, expected {start!r}"


f2 = lambda v: f"{v:.2f}".replace("-", "−")
f3 = lambda v: f"{v:.3f}".replace("-", "−")
f4 = lambda v: f"{v:.4f}".replace("-", "−")
pct = lambda v: f"{100 * v:.1f}%"

# ---------------------------------------------------------------- data
ex = pd.read_csv(R / "extra_e1_e3_summary.csv").set_index("Target").loc[T]
facts = pd.read_csv(R / "extra_dataset_facts.csv").set_index("Target").loc[T]
grp = pd.read_csv(R / "extra_group_definitions.csv").set_index("Target").loc[T]
unc = pd.read_csv(R / "extra_uncertainty_magnitude.csv").set_index("Target").loc[T]
quint = pd.read_csv(R / "extra_uncertainty_quintiles.csv")
aurc = pd.read_csv(R / "extra_risk_coverage_aurc.csv")
aurc_raw = aurc[aurc.Risk == "raw_MAE"].set_index("Target").loc[T]
aurc_log = aurc[aurc.Risk == "log_MAE"].set_index("Target").loc[T]
curves = pd.read_csv(R / "extra_risk_coverage_curves.csv")
cal = pd.read_csv(R / "calibration_metrics.csv").set_index("Target").loc[T]
e3 = pd.read_csv(R / "baseline_results_e3_repeated.csv")
tuned = pd.read_csv(R / "tuned_results_e3_summary.csv")
tuned_e2 = pd.read_csv(R / "tuned_results_e2.csv")
abl = pd.read_csv(R / "feature_ablation_e3_summary.csv")
deep = pd.read_csv(R / "deep_ensemble_stability.csv")
loso = pd.read_csv(R / "extra_loso_by_source.csv")

gb = e3[e3.Model == "GradientBoosting"].set_index("Target").loc[T]
rf = e3[e3.Model == "RandomForest"].set_index("Target").loc[T]
# Table II and Table III must agree: both are GB, same seeds, same code.
assert np.allclose(gb["LogMAE_mean"].values, ex["E3_LogMAE_mean"].values, atol=0.006), \
    (gb["LogMAE_mean"].values, ex["E3_LogMAE_mean"].values)

lat, lut, dsp, ff, bram = T
gain = 1 - ex["E3_LogMAE_mean"] / ex["E3_Const_LogMAE_mean"]
ratio = ex["E3_LogMAE_mean"] / ex["E1_LogMAE_mean"]
qbins = quint.groupby("Target")["Quintile_Bin"].nunique().loc[T]
qratio = {t: (lambda s: s.iloc[-1] / s.iloc[0])(quint[quint.Target == t].sort_values("Quintile_Bin")["Mean_Log_Error"]) for t in T}
bram_models = [rf.loc[bram, "LogMAE_mean"], gb.loc[bram, "LogMAE_mean"]] + list(
    tuned[tuned.Target == bram]["LogMAE_mean"])
rng = lambda s: f"{f2(min(s))}–{f2(max(s))}"
deep_rng = deep.groupby("Target").agg(["min", "max"]).loc[T]
at80 = curves[(curves.Method == "Deep_log_sigma") & np.isclose(curves.Coverage, 0.8)].set_index("Target").loc[T]
wins = unc["Tree_beats_magnitude_frac"]
abl_c = abl[abl.Feature_Set == "Set_C_StructuralPragma"].set_index("Target").loc[T]
abl_b = abl[abl.Feature_Set == "Set_B_PragmaAware"].set_index("Target").loc[T]
abl_a = abl[abl.Feature_Set == "Set_A_Basic"].set_index("Target").loc[T]
tgb = tuned[tuned.Model == "TunedGradientBoosting"].set_index("Target").loc[T]
trf = tuned[tuned.Model == "TunedRandomForest"].set_index("Target").loc[T]
two_stage_f1 = tuned_e2[(tuned_e2.Model == "TwoStageZeroAware") & (tuned_e2.Target == dsp)]["Positive_F1"].iloc[0]
two_stage_log = tuned_e2[(tuned_e2.Model == "TwoStageZeroAware") & (tuned_e2.Target == dsp)]["LogMAE"].iloc[0]
tgb_e2_dsp = tuned_e2[(tuned_e2.Model == "TunedGradientBoosting") & (tuned_e2.Target == dsp)]["LogMAE"].iloc[0]
floor = facts["Collision_LogMAE_Floor"]
between = facts["Between_Kernel_Variance_Share"]
wk = lambda t: (ex.loc[t, "E1_WithinKernel_rho_mean"], ex.loc[t, "E3_WithinKernel_rho_mean"])
heur = ex["E3_Heuristic_rho_mean"]
ls = lambda t, c: loso[(loso.Source == c) & (loso.Target == t)].iloc[0]
COLLIDE = 0.821  # from extra_analysis_summary.txt, checked below
summary_txt = (R / "extra_analysis_summary.txt").read_text()
m = re.search(r"same kernel: ([\d,]+) \(([\d.]+)%\)", summary_txt)
COLLIDE = float(m.group(2)) / 100

# ---------------------------------------------------------------- abstract
expect(3, "Abstract")
set_para(3, (
    "Learned models can speed up high-level synthesis (HLS) design-space exploration by predicting quality of "
    "results (QoR) before a design is synthesized. Random row splits remain a common default, but they put pragma "
    "variants of the same kernel in both the training and the test set, so the score mostly reflects interpolation. "
    "We study QoR prediction for unseen kernels on the ForgeHLS release (184,717 records, 491 source–algorithm groups), "
    "predicting latency, LUT, DSP, FF, and BRAM_18K from 42 regex-based source features. Compared with a random row "
    f"split, kernel-disjoint evaluation raises the five-seed log-scale error of gradient boosting by about "
    f"{ratio[lut]:.0f}× for LUT and FF, {ratio[dsp]:.0f}–{ratio[lat]:.0f}× for DSP and latency, and {ratio[bram]:.0f}× for BRAM_18K. "
    "The models beat a constant predictor for latency, LUT, DSP, and FF but not for BRAM_18K, and a leave-one-source-out "
    "test shows that this gain is fragile. Because the features ignore where a pragma is placed, "
    f"{100 * COLLIDE:.0f}% of the records share their feature vector with another variant of the same kernel, which "
    "bounds the attainable accuracy. Inside a held-out kernel, the ranking of pragma variants stays almost as good as "
    f"under a random split (median Spearman {f2(min(wk(lat)[1], wk(lut)[1], wk(ff)[1]))}–{f2(max(wk(lat)[1], wk(lut)[1], wk(ff)[1]))} "
    f"for latency, LUT, and FF), but a single-feature unroll heuristic already reaches {f2(heur[ff])}–{f2(heur[lut])} for LUT and FF, "
    "so the model adds clearly more only for latency and DSP. Random-forest disagreement correlates strongly with "
    f"absolute error (Spearman {f2(unc['Tree_sigma_rho'].min())}–{f2(unc['Tree_sigma_rho'].max())}), but predicted "
    f"magnitude alone correlates about as well ({f2(unc['Magnitude_rho'].min())}–{f2(unc['Magnitude_rho'].max())}), and "
    "only for latency does disagreement clearly add information beyond it. The abstention quality of a minimal "
    "deep ensemble changes with its random seeds and is not reliable. Interval coverage at the 80% level is within "
    "a few points of nominal, while calibration error stays large. We recommend reporting kernel-disjoint, log-scale "
    "results together with a constant baseline and a magnitude baseline for any uncertainty claim."))

# ---------------------------------------------------------------- introduction
expect(12, "An uncertainty analysis")
expect(13, "An assessment of interval")
set_para(13, "An assessment of interval coverage, calibration error, and selective prediction, including a "
             "seed-stability check that shows why a minimal deep ensemble should not be used for abstention here.")

# ---------------------------------------------------------------- related work
expect(17, "ForgeHLS is a large")
sub_para(17, "Our concern is the evaluation:", "Held-out-kernel evaluation itself is not new; our concern is "
         "how much it changes the picture once constant and magnitude baselines are added, in particular")

# ---------------------------------------------------------------- data
expect(28, "where Latency is the worst-case latency")
set_para(28, ptext(28).rstrip() + (
    " Two further properties limit what any model on these features can do. The target clock period is 10 ns and "
    "the device (xcu280) is the same for every record, so the clock feature is constant. And because the features "
    f"count pragmas without recording where they are placed, {pct(COLLIDE)} of the records share their exact feature "
    "vector with another record of the same kernel. Predicting the median of each such cluster still leaves a "
    f"LogMAE of {f2(floor[lat])} (latency), {f2(floor[lut])} (LUT), {f2(floor[dsp])} (DSP), {f2(floor[ff])} (FF), and "
    f"{f2(floor[bram])} (BRAM_18K), even with the kernel known."))
expect(46, "42")
set_para(46, "42 (41 non-constant)")
expect(56, "Set A")
set_para(56, " target clock period (constant in this release), source-code length, and pragma count.")
expect(57, "Set B")
sub_para(57, "resource, binding, and storage pragma counts", "bind-storage and other pragma counts")
expect(58, "Set C")
sub_para(58, "memory accesses", "array subscripts, asterisk/arrow counts")

# ---------------------------------------------------------------- protocol
expect(64, "We use Random Forest")
sub_para(64, "Gradient Boosting (GB) [18]", "histogram-based gradient boosting (GB, scikit-learn "
         "HistGradientBoostingRegressor) [18]")
sub_para(64, "never uses E3 test results", "is repeated for every split, each time on a grouped validation split "
         "of that split's own training kernels, so no test kernel influences model selection")
expect(68, "Tree disagreement is")
sub_para(68, "five independently initialized MLPs", "five independently initialized MLPs (two hidden layers, "
         "15 epochs, Smooth-L1 loss on standardized log targets, no variance head), a minimal version of [9]")
sub_para(68, "and ENCE.", "and ENCE, which compares per-bin RMSE with the root mean predicted variance [13].")

# ---------------------------------------------------------------- V-A + Table II
expect(71, "Table II compares")
set_para(71, (
    "Table II compares GB under random row splits (E1) and kernel-disjoint splits (E3), both over five seeds. "
    "We lead with LogMAE because raw-scale R² is unstable here. Kernel-disjoint LogMAE is about twice the "
    f"random-split value for LUT ({f2(ex.loc[lut,'E3_LogMAE_mean'])} vs {f2(ex.loc[lut,'E1_LogMAE_mean'])}) and FF "
    f"({f2(ex.loc[ff,'E3_LogMAE_mean'])} vs {f2(ex.loc[ff,'E1_LogMAE_mean'])}), more than three times for DSP "
    f"({f2(ex.loc[dsp,'E3_LogMAE_mean'])} vs {f2(ex.loc[dsp,'E1_LogMAE_mean'])}) and latency "
    f"({f2(ex.loc[lat,'E3_LogMAE_mean'])} vs {f2(ex.loc[lat,'E1_LogMAE_mean'])}), and {ratio[bram]:.0f} times for "
    f"BRAM_18K ({f2(ex.loc[bram,'E3_LogMAE_mean'])} vs {f2(ex.loc[bram,'E1_LogMAE_mean'])}). R² shows the same drop "
    f"for latency ({f2(ex.loc[lat,'E1_R2_mean'])} to {f2(ex.loc[lat,'E3_R2_mean'])}) and BRAM_18K "
    f"({f2(ex.loc[bram,'E1_R2_mean'])} to a strongly negative mean), but it is already close to zero for LUT, DSP, "
    "and FF under random splits, so it is a poor guide for those targets (Fig. 1). Even under random splits the "
    "error cannot fall below the feature-collision floor of Section III-A."))
rows = {80: "E1_LogMAE_mean", 86: "E3_LogMAE_mean", 92: "E3_Const_LogMAE_mean", 98: "E1_R2_mean", 104: "E3_R2_mean"}
for start, col in rows.items():
    for k, t in enumerate(T):
        v = ex.loc[t, col]
        set_para(start + 1 + k, f"{v:.1f}".replace("-", "−") if (col == "E3_R2_mean" and v < -1) else f2(v))
for k, t in enumerate(T[:4]):
    a, b = wk(t)
    set_para(111 + k, f"{f2(a)}/{f2(b)}")
    set_para(117 + k, f2(heur[t]))

# ---------------------------------------------------------------- V-B + Table III
expect(126, "Table III gives")
set_para(126, (
    "Table III gives the mean and standard deviation of R² and LogMAE for RF and GB over the five E3 seeds. GB is "
    f"the strongest on latency (R² {f3(gb.loc[lat,'R2_mean'])} ± {f3(gb.loc[lat,'R2_std'])}, LogMAE "
    f"{f3(gb.loc[lat,'LogMAE_mean'])} ± {f3(gb.loc[lat,'LogMAE_std'])}). R² for LUT, DSP, and FF stays near zero. "
    f"For BRAM_18K the GB R² swings wildly ({f2(gb.loc[bram,'R2_mean'])} ± {f2(gb.loc[bram,'R2_std'])}) because a few "
    "test kernels dominate, which a single split would never reveal, and RF latency R² is similarly unstable "
    f"({f2(rf.loc[lat,'R2_mean'])} ± {f2(rf.loc[lat,'R2_std'])}). The LogMAE standard deviations are much smaller "
    "and give the more stable picture."))
idx = 133
for model in (rf, gb):
    for t in T:
        set_para(idx + 2, f"{f3(model.loc[t,'R2_mean'])} ± {f3(model.loc[t,'R2_std'])}")
        set_para(idx + 3, f"{f3(model.loc[t,'LogMAE_mean'])} ± {f3(model.loc[t,'LogMAE_std'])}")
        idx += 4

# ---------------------------------------------------------------- V-C
expect(175, "The models also have")
set_para(175, (
    f"The models also have to be judged against a floor. Against the constant predictor, GB cuts LogMAE by "
    f"{gain[lat]:.0%} for latency and by {min(gain[lut], gain[dsp], gain[ff]):.0%}–{max(gain[lut], gain[dsp], gain[ff]):.0%} "
    f"for LUT, DSP, and FF. For BRAM_18K, which is zero for {pct(facts.loc[bram,'Zero_Fraction'])} of the records, every "
    f"model we tried is worse than predicting the training median (LogMAE {rng(bram_models)} against "
    f"{f2(ex.loc[bram,'E3_Const_LogMAE_mean'])})."))
expect(176, "Because DSE compares")
add = {t: wk(t)[1] - heur[t] for t in T[:4]}
set_para(176, (
    "Because DSE compares variants of one kernel, we also measured how well models rank variants inside a held-out "
    "kernel (Table II). The median within-kernel Spearman correlation barely changes between splits: "
    f"latency {f2(wk(lat)[0])} vs {f2(wk(lat)[1])}, LUT {f2(wk(lut)[0])} vs {f2(wk(lut)[1])}, DSP {f2(wk(dsp)[0])} vs "
    f"{f2(wk(dsp)[1])}, and FF {f2(wk(ff)[0])} vs {f2(wk(ff)[1])}. BRAM_18K is left out because only a handful of "
    "kernels show any variation. What a random split inflates is therefore the kernel-level scale, not the ordering "
    "of pragma variants. A trivial baseline puts this in perspective: ranking variants by the maximum unroll factor "
    f"alone, with the sign taken from the training kernels, gives {f2(heur[lat])} for latency, {f2(heur[lut])} for LUT, "
    f"{f2(heur[dsp])} for DSP, and {f2(heur[ff])} for FF. The model therefore adds {f2(add[lat])} for latency and "
    f"{f2(add[dsp])} for DSP, but only {f2(add[lut])} for LUT and {f2(add[ff])} for FF; for those two targets much of "
    "the transferable ranking signal is a simple unroll effect. The ranking result also fits the variance "
    f"decomposition: on the log scale, {between[lat]:.0%} of latency variance, {between[bram]:.0%} of BRAM_18K, "
    f"{between[dsp]:.0%} of DSP, {between[ff]:.0%} of FF, and {between[lut]:.0%} of LUT variance lies between kernels, "
    "so the pragma choices explain only the remainder."))
expect(177, "Table IV checks")
ch = ls(lat, "CHStone")
set_para(177, (
    "Table IV checks whether these conclusions depend on how groups are formed. Grouping by algorithm name alone, "
    "which also separates the 20 names that occur in two sources, gives errors close to the source–algorithm "
    "grouping, so cross-source name duplication is not driving the results. Leave-one-source-out is much harder. "
    "Median LogMAE over the 11 held-out sources beats the constant predictor clearly for LUT, DSP, and FF "
    f"({f2(grp.loc[lut,'LOSO_Median'])} vs {f2(grp.loc[lut,'LOSO_Const_Median'])}, {f2(grp.loc[dsp,'LOSO_Median'])} vs "
    f"{f2(grp.loc[dsp,'LOSO_Const_Median'])}, {f2(grp.loc[ff,'LOSO_Median'])} vs {f2(grp.loc[ff,'LOSO_Const_Median'])}) "
    "but only marginally for latency and BRAM_18K. Means are worse than medians because a few sources fail badly: "
    f"for latency, CHStone reaches a LogMAE of {ch.GB_LogMAE:.1f} against {ch.Const_LogMAE:.1f} for the constant "
    f"predictor, while PolyBench and rosetta are above {min(ls(lat,'PolyBench').GB_LogMAE, ls(lat,'rosetta').GB_LogMAE):.0f} "
    "for both."))
cols = ["Constant_E3", "SourceAlgo_E3", "AlgoName_5seed", "LOSO_Median", "LOSO_Const_Median"]
for k, t in enumerate(T):
    for j, c in enumerate(cols):
        set_para(187 + 6 * k + j, f2(grp.loc[t, c]))

# ---------------------------------------------------------------- V-D + Table V
expect(218, "Table V gives")
ur = unc["Tree_sigma_rho"]
set_para(218, (
    f"Table V gives the uncertainty–error association on the E2 unseen-kernel test set. Tree disagreement correlates "
    f"with absolute error at ρ = {f3(ur.min())}–{f3(ur.max())}. The highest-uncertainty quintile has far larger error "
    f"than the lowest (Fig. 2); on the log scale the mean error ratio between the top and bottom bin is "
    f"{qratio[lat]:.0f} for latency, {qratio[lut]:.1f} for LUT, and {qratio[ff]:.1f} for FF, and {qratio[dsp]:.1f} for "
    f"DSP, which has only {qbins[dsp]} non-empty bins because of its zeros; for BRAM_18K ({qbins[bram]} bins) the "
    "lowest bin is almost error-free, so its ratio is not meaningful. Deep-ensemble standard deviation is also "
    f"positively correlated with error ({f2(unc['Deep_sigma_rho'].min())}–{f2(unc['Deep_sigma_rho'].max())}), but this "
    "correlation depends on the ensemble's random seeds (Section V-E)."))
expect(219, "Scale confound")
mags = [t for t in (lut, ff, bram) ]
set_para(219, (
    f" Much of this association is a magnitude effect. On the E2 test set, tree uncertainty correlates strongly "
    f"with predicted magnitude (ρ = {f3(unc['Tree_sigma_vs_magnitude_rho'].min())}–{f3(unc['Tree_sigma_vs_magnitude_rho'].max())}), "
    f"and predicted magnitude alone ranks absolute error about as well (ρ = {f3(unc['Magnitude_rho'].min())}–"
    f"{f3(unc['Magnitude_rho'].max())}, last column of Table V). Within deciles of predicted magnitude, the mean tree "
    f"uncertainty–error correlation falls to {f2(unc['Within_decile_tree_rho_mean'].min())}–"
    f"{f2(unc['Within_decile_tree_rho_mean'].max())} (latency {f2(unc.loc[lat,'Within_decile_tree_rho_mean'])}, LUT "
    f"{f2(unc.loc[lut,'Within_decile_tree_rho_mean'])}, DSP {f2(unc.loc[dsp,'Within_decile_tree_rho_mean'])}, FF "
    f"{f2(unc.loc[ff,'Within_decile_tree_rho_mean'])}, BRAM {f2(unc.loc[bram,'Within_decile_tree_rho_mean'])}). When we "
    f"resample test kernels 200 times, tree disagreement beats the magnitude baseline only for latency (in "
    f"{pct(wins[lat])} of resamples). For LUT, FF, and BRAM_18K the magnitude baseline is the better ranker in "
    f"{pct(1 - max(wins[lut], wins[ff], wins[bram]))}–{pct(1 - min(wins[lut], wins[ff], wins[bram]))} of resamples, and "
    f"for DSP in {pct(1 - wins[dsp])}."), run=-1)
for k, t in enumerate(T):
    set_para(227 + 4 * k, f3(unc.loc[t, "Tree_sigma_rho"]))
    set_para(228 + 4 * k, f3(unc.loc[t, "Deep_sigma_rho"]))
    set_para(229 + 4 * k, f3(unc.loc[t, "Magnitude_rho"]))
expect(249, "Fig. 2")
set_para(249, (" Mean log-scale error across tree-uncertainty quintiles (Q1 = lowest) on the E2 test set. DSP and "
               f"BRAM_18K have only {qbins[dsp]} and {qbins[bram]} non-empty bins because of their zeros."))

# ---------------------------------------------------------------- V-E + Tables VI, VII
expect(252, "At the nominal 80% level")
cov80 = cal["Coverage_80"]
set_para(252, (
    f"At the nominal 80% level the tree intervals cover {pct(cov80.min())}–{pct(cov80.max())} of the test designs "
    f"(Table VI), but the picture is less tidy at other levels. Latency covers {pct(cal.loc[lat,'Coverage_50'])} at "
    f"the 50% level and {pct(cal.loc[lat,'Coverage_90'])} at the 90% level, while FF stays at "
    f"{pct(cal.loc[ff,'Coverage_90'])} at 90%. The intervals are wide (normalized width "
    f"{f2(cal['Tree_Normalized_MPIW'].min())}–{f2(cal['Tree_Normalized_MPIW'].max())}). ENCE is large: "
    f"{cal.loc[lat,'Tree_ENCE']:.1f} for latency, {f2(cal.loc[lut,'Tree_ENCE'])} for LUT, {f2(cal.loc[dsp,'Tree_ENCE'])} "
    f"for DSP, {f2(cal.loc[ff,'Tree_ENCE'])} for FF, and {f2(cal.loc[bram,'Tree_ENCE'])} for BRAM_18K; for latency the "
    "tree spread on the log scale is far smaller than the log error on unseen kernels. Good ranking therefore does "
    "not imply calibrated errors, in line with the findings in [10]. Because the intervals use a constant residual "
    "quantile they do not adapt to the uncertainty estimate; conformalized quantile regression [15] is the natural "
    "next step."))
for k, t in enumerate(T):
    base = 262 + 6 * k
    set_para(base, f3(cal.loc[t, "Coverage_50"]))
    set_para(base + 1, f3(cal.loc[t, "Coverage_80"]))
    set_para(base + 2, f3(cal.loc[t, "Coverage_90"]))
    set_para(base + 3, f2(cal.loc[t, "Tree_Normalized_MPIW"]))
    te_, de_ = cal.loc[t, "Tree_ENCE"], cal.loc[t, "Deep_Ensemble_ENCE"]
    set_para(base + 4, f"{te_:.1f}/{f2(de_)}" if te_ >= 10 else f"{f2(te_)}/{f2(de_)}")
expect(292, "Fig. 3 shows")
mag_beats = [t for t in T[:4] if aurc_raw.loc[t, "Magnitude"] >= aurc_raw.loc[t, "Tree_raw_sigma"]]
dr = deep_rng
set_para(292, (
    "Fig. 3 shows risk–coverage curves that accept the lowest-uncertainty predictions first. Tree disagreement does "
    "lower the retained error, but a baseline that simply accepts the smallest predicted values does as well or "
    f"better for {len(mag_beats)} of the four non-BRAM targets (Table VII). Relative to random rejection, the area under "
    f"the risk–coverage curve (raw MAE) shrinks by {f2(aurc_raw.loc[T[:4],'Tree_log_sigma'].min())}–"
    f"{f2(aurc_raw.loc[T[:4],'Tree_log_sigma'].max())}× when we order by log-scale tree σ, by "
    f"{f2(aurc_raw.loc[T[:4],'Tree_raw_sigma'].min())}–{f2(aurc_raw.loc[T[:4],'Tree_raw_sigma'].max())}× with raw-scale "
    f"tree σ, and by {f2(aurc_raw.loc[T[:4],'Magnitude'].min())}–{f2(aurc_raw.loc[T[:4],'Magnitude'].max())}× when we "
    f"order by predicted magnitude (latency, LUT, DSP, FF), against {f2(aurc_raw.loc[T[:4],'Oracle'].min())}–"
    f"{f2(aurc_raw.loc[T[:4],'Oracle'].max())}× for an oracle that knows the true error. On a log-scale risk the gap "
    f"closes: log-scale σ and magnitude give similar reductions ({f2(aurc_log.loc[lat,'Tree_log_sigma'])} vs "
    f"{f2(aurc_log.loc[lat,'Magnitude'])} for latency). The deep ensemble behaves inconsistently. In the run of "
    f"Table VII its log σ is worse than random rejection for LUT and FF ({f2(aurc_raw.loc[lut,'Deep_log_sigma'])}× and "
    f"{f2(aurc_raw.loc[ff,'Deep_log_sigma'])}×) and better for DSP and BRAM_18K. Retraining the same ensemble with "
    "two other seed sets moves these factors over a wide range (latency "
    f"{f2(dr.loc[lat,('AURC_reduction_log_sigma','min')])}–{f2(dr.loc[lat,('AURC_reduction_log_sigma','max')])}×, LUT "
    f"{f2(dr.loc[lut,('AURC_reduction_log_sigma','min')])}–{f2(dr.loc[lut,('AURC_reduction_log_sigma','max')])}×, FF "
    f"{f2(dr.loc[ff,('AURC_reduction_log_sigma','min')])}–{f2(dr.loc[ff,('AURC_reduction_log_sigma','max')])}×), so the "
    "minimal deep ensemble is not a dependable abstention signal in this setting. The large BRAM_18K factors come "
    "from near-zero retained risk at intermediate coverage and should not be read as typical."))
expect(294, "Reduction in Area")
rep(294, "E2 Test Set)", "E2 Test Set, Coverage 5–100%)")
meths = ["Tree_log_sigma", "Tree_raw_sigma", "Magnitude", "Oracle", "Deep_log_sigma"]
for k, t in enumerate(T):
    for j, mth in enumerate(meths):
        set_para(302 + 6 * k + j, f2(aurc_raw.loc[t, mth]))

# ---------------------------------------------------------------- V-F ablation
expect(337, "The ablation compares")
set_para(337, (
    "The ablation compares the basic (Set A), pragma-aware (Set B), and structural-plus-pragma (Set C) features over "
    "the five E3 seeds, using an RF with 30 trees and the uncertainty–error Spearman ρ (log-scale σ) as the quality "
    "measure (Fig. 4). Set C has the highest ρ and the lowest LogMAE for every target; for example, latency LogMAE "
    f"falls from {f2(abl_a.loc[lat,'LogMAE_mean'])} (Set A) to {f2(abl_b.loc[lat,'LogMAE_mean'])} (Set B) and "
    f"{f2(abl_c.loc[lat,'LogMAE_mean'])} (Set C), and DSP ρ rises from {f2(abl_a.loc[dsp,'Spearman_rho_mean'])} to "
    f"{f2(abl_c.loc[dsp,'Spearman_rho_mean'])}. Set B already improves on Set A for every target, so pragma counts and "
    "structural counts both carry information. Single-split ablations disagree with these averages for some targets, "
    "which is why we report the five-seed means."))
expect(339, "Fig. 4")
set_para(339, " Uncertainty–error Spearman correlation (log-scale σ) for the three feature sets, mean ± std over "
              "the five E3 kernel-disjoint seeds.")

# ---------------------------------------------------------------- V-G tuning + Table VIII
expect(341, "Compact validation-only tuning")
set_para(341, (
    "Compact validation-only tuning gives no uniform gain (Table VIII). Hyperparameters are selected separately for "
    "each seed on its own training kernels. Tuned GB raises latency R² slightly, to "
    f"{f3(tgb.loc[lat,'R2_mean'])} ± {f3(tgb.loc[lat,'R2_std'])}, but its latency LogMAE "
    f"({f3(tgb.loc[lat,'LogMAE_mean'])} ± {f3(tgb.loc[lat,'LogMAE_std'])}) is worse than the fixed baseline "
    f"({f3(gb.loc[lat,'LogMAE_mean'])}). FF and BRAM_18K LogMAE improve a little and the other targets do not improve "
    f"consistently. The two-stage zero-aware DSP model reaches a positive-class F1 of about {two_stage_f1:.3f} on E2 "
    f"without beating the plain regressor on the regression metrics (LogMAE {f3(two_stage_log)} vs "
    f"{f3(tgb_e2_dsp)}). We treat tuning as a side analysis."))
idx = 350
for model in (tgb, trf):
    for t in T:
        set_para(idx + 2, f4(model.loc[t, "R2_mean"]))
        set_para(idx + 3, f4(model.loc[t, "R2_std"]))
        set_para(idx + 4, f4(model.loc[t, "LogMAE_mean"]))
        set_para(idx + 5, f4(model.loc[t, "LogMAE_std"]))
        idx += 6

# ---------------------------------------------------------------- discussion, threats, reproducibility, conclusion
expect(412, "Three things stand out")
rep(412, "random-split LogMAE is 2–9 times lower", f"random-split LogMAE is "
         f"{ratio.min():.0f}–{ratio.max():.0f} times lower")
expect(413, "Second, what transfers")
lo, hi = min(wk(lat)[1], wk(lut)[1], wk(ff)[1]), max(wk(lat)[1], wk(lut)[1], wk(ff)[1])
set_para(413, (
    "Second, what transfers is the ranking of pragma variants within a kernel, not absolute QoR. Median "
    f"within-kernel Spearman correlations of {f2(lo)}–{f2(hi)} for latency, LUT, and FF barely change between random "
    "and kernel-disjoint splits. One caveat matters: ranking variants by the maximum unroll factor alone already gives "
    f"{f2(heur[lut])} for LUT and {f2(heur[ff])} for FF, so for those two targets much of the transferable signal is a "
    f"simple unroll effect, while the model adds more for latency ({f2(wk(lat)[1])} against {f2(heur[lat])}) and DSP "
    f"({f2(wk(dsp)[1])} against {f2(heur[dsp])}). For DSE, which compares variants of one kernel, this ranking is the "
    "more relevant result, although we have not tested it inside an actual search loop. Absolute predictions for a "
    f"new kernel deserve caution: the gain over a constant predictor is about {min(gain[:4]):.0%}–{max(gain[:4]):.0%} in "
    "LogMAE for latency, LUT, DSP, and FF, there is none for BRAM_18K, and under leave-one-source-out the latency "
    "model is worse than the constant for CHStone and the Vitis introductory examples while both are poor on "
    "PolyBench and rosetta."))
expect(414, "Third, uncertainty is useful")
set_para(414, (
    "Third, uncertainty is useful, but less so than the headline correlations suggest. Tree disagreement correlates "
    f"with error at {f2(ur.min())}–{f2(ur.max())}, yet predicted magnitude correlates at "
    f"{f2(unc['Magnitude_rho'].min())}–{f2(unc['Magnitude_rho'].max())}, and ordering by magnitude gives larger "
    "risk reductions than the log-scale tree σ. Latency is the one target where disagreement clearly carries extra "
    "information. The minimal deep ensemble is not a dependable abstention signal: its usefulness changes with the "
    "random seeds of its members. Interval coverage is near nominal at 80% but drifts at other levels, and ENCE "
    "stays high."))
expect(419, "The features are regex-based")
set_para(419, ptext(419).rstrip() + (
    f" They also map {pct(COLLIDE)} of the records onto a feature vector shared with another variant of the same "
    "kernel, and the clock feature is constant in this release."))
expect(420, "Tree and deep-ensemble uncertainty")
set_para(420, ptext(420).rstrip() + (
    " The deep-ensemble results change with the members' random seeds and with the PyTorch version, so we report "
    "them only as a seed-stability check."))
expect(424, "The workflow rebuilds")
set_para(424, (
    "The workflow rebuilds the feature matrix and split files from the original JSON, checks required fields, "
    "asserts zero group overlap for E2 and E3, trains baselines before the uncertainty analysis, and exports CSV, "
    "NPZ, figure, and table artifacts. The package contains a Colab notebook (ForgeHLS_colab.ipynb) and scripts for "
    "feature extraction, dataset building, split generation, baseline and tuned training, uncertainty estimation, "
    "and evaluation. The additional checks (five-seed random-split repeat, constant and magnitude baselines, "
    "algorithm-name grouping, leave-one-source-out, within-kernel ranking, the unroll heuristic, and the "
    "feature-collision floor) are in scripts/extra_analysis.py, the deep-ensemble seed check is in "
    "scripts/deep_ensemble_stability.py, and scripts/make_paper_figures.py regenerates the figures. All tables and "
    "figures come from one run in one software environment, whose exact package versions are listed in "
    "requirements-lock.txt. Code and artifacts: https://github.com/itsmonir48/ForgeHLS."))
expect(426, "We studied QoR prediction")
set_para(426, (
    "We studied QoR prediction for unseen kernels on 184,717 ForgeHLS records. Random row splits understate the "
    f"log-scale error by a factor of {ratio.min():.0f}–{ratio.max():.0f} because they leak kernel-level scale. "
    "Kernel-disjoint models still beat a constant predictor for latency, LUT, DSP, and FF but not for BRAM_18K, and "
    "leave-one-source-out shows how fragile that gain is. The ranking of pragma variants within a kernel transfers "
    "well, although a simple unroll heuristic explains most of it for LUT and FF. Tree disagreement correlates "
    "strongly with error, but much of that is predicted magnitude, with latency the clear exception; a minimal deep "
    "ensemble gives seed-dependent abstention quality, and interval calibration stays target-dependent. We suggest "
    "reporting kernel-disjoint LogMAE next to a constant baseline, and checking uncertainty against a magnitude "
    "baseline."))

# ---------------------------------------------------------------- Pareto DSE proxy (Section V-H)
pz = pd.read_csv(R / "pareto_dse_summary.csv").set_index("Budget")
pz_leak = pd.read_csv(R / "pareto_dse_summary_row_ordered.csv").set_index("Budget")
heur_feats = pd.read_csv(R / "pareto_dse_heuristic_features.csv")
by_kernel = pd.read_csv(R / "pareto_dse_by_kernel.csv")
audit = (R / "row_order_audit.txt").read_text()
pos = float(re.search(r"within their kernel: ([\d.]+)", audit).group(1))
first_p = float(re.search(r"max unroll factor: ([\d.]+)", audit).group(1))
first_all = float(re.search(r"\(all variants: ([\d.]+)\)", audit).group(1))
nk = by_kernel[by_kernel.Budget == 0.05].groupby("Seed").size()
med_front = by_kernel[by_kernel.Budget == 0.05]["N_Pareto"].median()
B = [0.05, 0.10, 0.20]
rec = lambda who, b, d=pz: f"{d.loc[b, f'{who}_Recall_mean']:.2f}"
adm = lambda who, b: f"{pz.loc[b, f'{who}_ADRSMedian_mean']:.2f}"
wins_k = by_kernel.assign(W=by_kernel.Model_Recall > by_kernel.Random_Recall).groupby("Budget")["W"].mean()
hf = sorted(set(heur_feats["Feature"]))

# Final, shorter abstract (IEEE: about 150-250 words).
set_para(3, (
    "Learned QoR predictors can speed up high-level synthesis (HLS) design-space exploration, but random row "
    "splits put pragma variants of the same kernel in both training and test sets. On the ForgeHLS release "
    "(184,717 records, 491 source–algorithm groups) we predict latency, LUT, DSP, FF, and BRAM_18K for unseen "
    "kernels from 42 regex-based source features. Kernel-disjoint evaluation raises the five-seed log-scale error "
    f"of gradient boosting {ratio.min():.0f}–{ratio.max():.0f}× over random splits; the models beat a constant "
    "predictor except for BRAM_18K, and leave-one-source-out shows that this gain is fragile. Because the features "
    f"ignore where a pragma is placed, {100 * COLLIDE:.0f}% of the records share their feature vector with another "
    "variant of the same kernel. The ranking of variants inside a held-out kernel transfers well, but a "
    "single-feature unroll heuristic explains most of it for LUT and FF. In a budgeted exploration, synthesizing "
    f"the 10% of variants with the best predicted Pareto rank recovers {rec('Model', 0.10)} of the true Pareto "
    f"designs, against {rec('Random', 0.10)} for random choice; ForgeHLS lists Pareto variants early within each "
    "kernel, so such comparisons must break ties at random. Random-forest disagreement correlates strongly with "
    "error, but predicted magnitude correlates about as well, and only for latency does disagreement add clear "
    "information. A minimal deep ensemble gives seed-dependent abstention quality, and calibration error stays "
    "large. We recommend kernel-disjoint, log-scale reporting with constant and magnitude baselines."))
expect(11, "A five-seed paired comparison")
set_para(11, ptext(11).rstrip(".") + ", and a budgeted DSE proxy showing that this ranking helps choose which "
         "variants to synthesize.")

heading_xml = pieces[para_pos[340]]
body_xml = pieces[para_pos[341]]


def clone(xml_par, text):
    ts = list(re.finditer(r"<w:t(?: [^>]*)?>[^<]*</w:t>", xml_par))
    m_ = ts[-1]
    out = xml_par[:m_.start()] + f'<w:t xml:space="preserve">{escape(text)}</w:t>' + xml_par[m_.end():]
    for m2 in reversed(ts[:-1]):  # clear any other text runs
        out = out[:m2.start()] + '<w:t xml:space="preserve"></w:t>' + out[m2.end():]
    return out


h_text = "H. Budgeted DSE on Unseen Kernels"
b1 = (
    "Section V-C showed that the ranking of pragma variants transfers to unseen kernels. To check whether this "
    "helps exploration, we simulate a synthesis budget inside every held-out kernel with at least 30 variants "
    f"({nk.min()}–{nk.max()} kernels per E3 seed; median {med_front:.0f} true Pareto designs per kernel). The two "
    "objectives are worst-case latency and the ForgeHLS resource metric, the mean LUT, FF, DSP, and BRAM_18K "
    "utilization of the device; the released Pareto labels use the same resource metric with best-case latency. "
    "GB ranks the variants of a kernel by non-dominated sorting of its predicted latency and predicted resource "
    "metric, and the first k variants are “synthesized”. We compare it with random choice, with an oracle that "
    "sorts on the true values, and with the same sorting applied to the single pragma feature per objective that "
    f"ranks best on the training kernels ({', '.join(f.replace('pragma_', '') for f in hf)}). We report the share "
    "of true Pareto designs among the k chosen (recall) and the average distance from the true front to the best "
    "chosen design (ADRS; median over kernels)."
)
b2 = (
    f"With a budget of 5%, 10%, and 20% of the variants the model recovers {rec('Model', 0.05)}, {rec('Model', 0.10)}, "
    f"and {rec('Model', 0.20)} of the true Pareto designs, against {rec('Random', 0.05)}, {rec('Random', 0.10)}, and "
    f"{rec('Random', 0.20)} for random choice and {rec('Heuristic', 0.05)}, {rec('Heuristic', 0.10)}, and "
    f"{rec('Heuristic', 0.20)} for the pragma heuristic; it beats random choice in "
    f"{wins_k.loc[0.05]:.0%}–{wins_k.loc[0.20]:.0%} of the kernels. Median ADRS at the 10% budget is "
    f"{adm('Model', 0.10)} for the model, {adm('Heuristic', 0.10)} for the heuristic, and {adm('Random', 0.10)} for "
    f"random choice (oracle {adm('Oracle', 0.10)}). The model is therefore useful for exploration on new kernels, "
    "while staying far from the oracle. One detail of the data matters here. Within each kernel, ForgeHLS lists "
    f"the Pareto variants early (mean relative row position {pos:.2f} instead of 0.5), and {first_p:.0%} of the "
    f"Pareto variants are the first row with their unroll factor, against {first_all:.1%} of all variants. Many "
    "variants share a feature vector and therefore a prediction, so breaking ties by row index leaks the label: "
    f"without shuffling, the model's recall at 5% rises to {rec('Model', 0.05, pz_leak)} and the heuristic's to "
    f"{rec('Heuristic', 0.05, pz_leak)}. All selections and risk–coverage curves in this paper break ties at random."
)
new_xml = clone(heading_xml, h_text) + clone(body_xml, b1) + clone(body_xml, b2)

expect(413, "Second, what transfers")
rep(413, "although we have not tested it inside an actual search loop",
    "and the budgeted DSE proxy of Section V-H shows that it helps choose which variants to synthesize, "
    "although we have not run a search with synthesis in the loop")
expect(422, "We evaluate prediction quality")
set_para(422, (
    "We evaluate prediction quality, ranking, and a budgeted selection proxy, not an end-to-end DSE search with "
    "synthesis in the loop. The ForgeHLS row order is not random: Pareto variants appear early within each kernel, "
    "so any procedure that breaks ties by row index is biased; all our selections and risk–coverage curves break "
    "ties at random."))
rep(424, "the deep-ensemble seed check is in scripts/deep_ensemble_stability.py,",
    "the budgeted DSE proxy is in scripts/pareto_dse.py, the row-order audit in scripts/row_order_audit.py, the "
    "deep-ensemble seed check in scripts/deep_ensemble_stability.py,")
rep(426, "although a simple unroll heuristic explains most of it for LUT and FF.",
    "although a simple unroll heuristic explains most of it for LUT and FF, and in a budgeted selection it "
    f"recovers {float(rec('Model', 0.10)) / float(rec('Random', 0.10)):.1f} times as many Pareto designs as random "
    "choice.")

discussion_pos = para_pos[411]
assert ptext(411).startswith("VI. Discussion")
pieces.insert(discussion_pos, new_xml)

DOC.write_text("".join(pieces), encoding="utf-8")

# ---------------------------------------------------------------- figures
media = {"image1.png": "paper_fig1_e1_vs_e3_logmae.png", "image2.png": "paper_fig2_uncertainty_quintiles.png",
         "image3.png": "paper_fig3_risk_coverage.png", "image4.png": "paper_fig4_feature_ablation.png"}
rid = {"image1.png": "rId5", "image2.png": "rId6", "image3.png": "rId7", "image4.png": "rId8"}
xml = DOC.read_text(encoding="utf-8")
for dst, src in media.items():
    data = (F / src).read_bytes()
    (U / "word/media" / dst).write_bytes(data)
    w, h = Image.open(F / src).size
    # resize the drawing that embeds this image to the new aspect ratio, keeping its width
    m = re.search(r'<w:drawing>.*?r:embed="%s".*?</w:drawing>' % rid[dst], xml, re.S)
    block = m.group(0)
    cx = int(re.search(r'<wp:extent cx="(\d+)"', block).group(1))
    cy = int(round(cx * h / w))
    nb = re.sub(r'(<wp:extent cx="\d+" cy=")\d+(")', rf"\g<1>{cy}\g<2>", block)
    nb = re.sub(r'(<a:ext cx="\d+" cy=")\d+(")', rf"\g<1>{cy}\g<2>", nb)
    xml = xml.replace(block, nb)
DOC.write_text(xml, encoding="utf-8")
print("v4 document.xml and figures updated")
