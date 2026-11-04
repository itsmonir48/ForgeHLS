"""
Validation-only tuning and zero-aware resource baselines.

The E2 test groups are never used for model selection. Hyperparameters are
selected on a grouped validation split carved out of the E2 training groups,
then refit on all E2 training groups before evaluation.
"""

from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES


def metrics(y_true, prediction):
    prediction = np.clip(np.asarray(prediction, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "R2": r2_score(y_true, prediction),
        "MAE": mean_absolute_error(y_true, prediction),
        "MedianAE": median_absolute_error(y_true, prediction),
        "LogMAE": mean_absolute_error(
            np.log1p(y_true), np.log1p(prediction)
        ),
    }


def fit_log_model(model, x_train, y_train):
    model.fit(x_train, np.log1p(np.clip(y_train, 0, None)))
    return model


def predict_log_model(model, x_test):
    return np.expm1(model.predict(x_test))


def model_candidates(model_name):
    if model_name == "RandomForest":
        for n_estimators, max_depth, min_samples_leaf in product(
            (50, 100), (16, None), (1,)
        ):
            yield {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "n_jobs": -1,
                "random_state": 42,
            }
    elif model_name == "GradientBoosting":
        for learning_rate, max_leaf_nodes, max_iter in product(
            (0.05, 0.1), (31,), (100, 200)
        ):
            yield {
                "learning_rate": learning_rate,
                "max_leaf_nodes": max_leaf_nodes,
                "max_iter": max_iter,
                "random_state": 42,
            }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def make_model(model_name, params):
    if model_name == "RandomForest":
        return RandomForestRegressor(**params)
    return HistGradientBoostingRegressor(**params)


def select_params(model_name, x_fit, y_fit, x_val, y_val):
    candidates = list(model_candidates(model_name))
    scores = []
    for candidate in candidates:
        model = fit_log_model(make_model(model_name, candidate), x_fit, y_fit)
        score = metrics(y_val, predict_log_model(model, x_val))["LogMAE"]
        scores.append((score, candidate))
    return min(scores, key=lambda item: item[0])


def fit_two_stage(x_train, y_train, x_eval):
    positive = y_train > 0
    classifier = DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=10, random_state=42
    )
    classifier.fit(x_train, positive.astype(np.int8))

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=31, max_iter=200, random_state=42
    )
    if positive.any():
        regressor.fit(x_train[positive], np.log1p(y_train[positive]))
        positive_prediction = np.expm1(regressor.predict(x_eval))
    else:
        positive_prediction = np.zeros(len(x_eval))

    prediction = np.where(
        classifier.predict(x_eval).astype(bool), positive_prediction, 0.0
    )
    return prediction


def main():
    data = np.load(DATA_DIR / "dataset_arrays.npz", allow_pickle=False)
    splits = np.load(DATA_DIR / "splits.npz", allow_pickle=False)
    x = data["X_features"]
    y = data["y_targets"]
    groups = data["group_ids"]

    e2_train = splits["e2_grouped_train_idx"]
    e2_test = splits["e2_grouped_test_idx"]
    validation = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=2026
    )
    fit_local, val_local = next(
        validation.split(e2_train, groups=groups[e2_train])
    )
    fit_idx = e2_train[fit_local]
    val_idx = e2_train[val_local]

    records = []
    selected = []
    selected_params_by_target = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        y_target = y[:, target_index]
        for model_name in ("RandomForest", "GradientBoosting"):
            score, params = select_params(
                model_name,
                x[fit_idx],
                y_target[fit_idx],
                x[val_idx],
                y_target[val_idx],
            )
            final_model = fit_log_model(
                make_model(model_name, params), x[e2_train], y_target[e2_train]
            )
            prediction = predict_log_model(final_model, x[e2_test])
            result = metrics(y_target[e2_test], prediction)
            records.append({
                "Model": f"Tuned{model_name}",
                "Target": target_name,
                "Selection_LogMAE": score,
                **result,
            })
            selected.append({
                "Model": f"Tuned{model_name}",
                "Target": target_name,
                "Params": repr(params),
            })
            selected_params_by_target[(target_name, model_name)] = params

        if target_name in {"DSP", "BRAM_18K"}:
            two_stage = fit_two_stage(
                x[e2_train], y_target[e2_train], x[e2_test]
            )
            result = metrics(y_target[e2_test], two_stage)
            records.append({
                "Model": "TwoStageZeroAware",
                "Target": target_name,
                "Selection_LogMAE": np.nan,
                "Positive_F1": f1_score(
                    y_target[e2_test] > 0, two_stage > 0, zero_division=0
                ),
                **result,
            })

        train_median = float(np.median(y_target[e2_train]))
        median_prediction = np.full(len(e2_test), train_median)
        records.append({
            "Model": "TrainMedian",
            "Target": target_name,
            "Selection_LogMAE": np.nan,
            **metrics(y_target[e2_test], median_prediction),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        RESULTS_DIR / "tuned_results_e2.csv", index=False
    )
    pd.DataFrame(selected).to_csv(
        RESULTS_DIR / "tuned_model_selection.csv", index=False
    )
    repeated_records = []
    for seed in [42, 43, 44, 45, 46]:
        train_idx = splits[f"e3_grouped_seed_{seed}_train_idx"]
        test_idx = splits[f"e3_grouped_seed_{seed}_test_idx"]
        for target_index, target_name in enumerate(TARGET_NAMES):
            y_target = y[:, target_index]
            for model_name in ("RandomForest", "GradientBoosting"):
                params = selected_params_by_target[(target_name, model_name)]
                model = fit_log_model(
                    make_model(model_name, params),
                    x[train_idx],
                    y_target[train_idx],
                )
                prediction = predict_log_model(model, x[test_idx])
                result = metrics(y_target[test_idx], prediction)
                repeated_records.append({
                    "Seed": seed,
                    "Model": f"Tuned{model_name}",
                    "Target": target_name,
                    **result,
                })

    repeated = pd.DataFrame(repeated_records)
    repeated_summary = repeated.groupby(
        ["Model", "Target"], as_index=False
    ).agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        LogMAE_mean=("LogMAE", "mean"),
        LogMAE_std=("LogMAE", "std"),
    )
    repeated.to_csv(RESULTS_DIR / "tuned_results_e3_by_seed.csv", index=False)
    repeated_summary.to_csv(
        RESULTS_DIR / "tuned_results_e3_summary.csv", index=False
    )
    print(pd.DataFrame(records).to_string(index=False))
    print(repeated_summary.to_string(index=False))
    print(f"Saved tuned results to {RESULTS_DIR / 'tuned_results_e2.csv'}")


if __name__ == "__main__":
    main()
