import pandas as pd
import numpy as np
import time
import re
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import plotly.express as px


warnings.filterwarnings("ignore")


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet9_deal5"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:
        print(f"Failed to read the data: {exc}")
        return

    if "Rejection" not in df.columns:
        print("Target column 'Rejection' was not found")
        return

    X = df.drop(columns=["Rejection"])
    y = df["Rejection"]

    X.columns = [
        re.sub(r"[\[\]<>]", "", column)
        for column in X.columns
    ]

    print("Dataset overview")
    print(
        f"Number of samples: {X.shape[0]}, "
        f"number of input features: {X.shape[1]}"
    )

    n_iterations = 20
    rng = np.random.RandomState(42)
    seeds = rng.randint(
        0,
        10000,
        size=n_iterations,
    ).tolist()

    print("Starting Monte Carlo cross-validation")
    print(
        f"Number of validation iterations: "
        f"{n_iterations} using an 80:20 split"
    )
    print(f"Random seeds: {seeds}")

    def get_models(seed):
        return {
            "Random Forest (RF)": RandomForestRegressor(
                random_state=seed
            ),
            "Extra Trees (XT)": ExtraTreesRegressor(
                random_state=seed
            ),
            "XGBoost (XGB)": XGBRegressor(
                random_state=seed,
                objective="reg:squarederror",
            ),
            "LightGBM (LGBM)": LGBMRegressor(
                random_state=seed,
                verbose=-1,
            ),
        }

    print("Stage 1: Baseline model screening")

    metric_keys = [
        "train_R2",
        "train_RMSE",
        "train_MAE",
        "test_R2",
        "test_RMSE",
        "test_MAE",
    ]
    base_records = {
        name: {
            key: []
            for key in metric_keys
        }
        for name in get_models(42).keys()
    }

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
        )
        models = get_models(seed)

        for name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            base_records[name]["train_R2"].append(
                r2_score(y_train, y_train_pred)
            )
            base_records[name]["train_RMSE"].append(
                np.sqrt(
                    mean_squared_error(
                        y_train,
                        y_train_pred,
                    )
                )
            )
            base_records[name]["train_MAE"].append(
                mean_absolute_error(
                    y_train,
                    y_train_pred,
                )
            )

            y_test_pred = model.predict(X_test)
            base_records[name]["test_R2"].append(
                r2_score(y_test, y_test_pred)
            )
            base_records[name]["test_RMSE"].append(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_test_pred,
                    )
                )
            )
            base_records[name]["test_MAE"].append(
                mean_absolute_error(
                    y_test,
                    y_test_pred,
                )
            )

    base_mean_test_r2 = {
        name: np.mean(records["test_R2"])
        for name, records in base_records.items()
    }
    best_model_name = max(
        base_mean_test_r2,
        key=base_mean_test_r2.get,
    )

    for name, records in base_records.items():
        print(
            f"[{name}] Mean performance over "
            f"{n_iterations} iterations"
        )

        tr_r2_m = np.mean(records["train_R2"])
        tr_r2_s = np.std(records["train_R2"])
        tr_rmse_m = np.mean(records["train_RMSE"])
        tr_rmse_s = np.std(records["train_RMSE"])
        tr_mae_m = np.mean(records["train_MAE"])
        tr_mae_s = np.std(records["train_MAE"])

        te_r2_m = np.mean(records["test_R2"])
        te_r2_s = np.std(records["test_R2"])
        te_rmse_m = np.mean(records["test_RMSE"])
        te_rmse_s = np.std(records["test_RMSE"])
        te_mae_m = np.mean(records["test_MAE"])
        te_mae_s = np.std(records["test_MAE"])

        print(
            "Training set: "
            f"R2 = {tr_r2_m:.4f} +/- {tr_r2_s:.4f}, "
            f"RMSE = {tr_rmse_m:.4f} +/- {tr_rmse_s:.4f}, "
            f"MAE = {tr_mae_m:.4f} +/- {tr_mae_s:.4f}"
        )
        print(
            "Test set: "
            f"R2 = {te_r2_m:.4f} +/- {te_r2_s:.4f}, "
            f"RMSE = {te_rmse_m:.4f} +/- {te_rmse_s:.4f}, "
            f"MAE = {te_mae_m:.4f} +/- {te_mae_s:.4f}"
        )

    print(f"Selected baseline model: {best_model_name}")
    print(
        f"Hyperparameter optimization will be applied "
        f"to {best_model_name}"
    )

    print("Stage 2: Grid search and holdout evaluation")

    param_grids = {
        "Random Forest (RF)": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 15, 30],
            "min_samples_split": [2, 5],
        },
        "Extra Trees (XT)": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 15, 25],
            "min_samples_split": [2, 5],
        },
        "XGBoost (XGB)": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
        "LightGBM (LGBM)": {
            "n_estimators": [100, 200],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 50],
        },
    }
    param_grid = param_grids[best_model_name]

    results_base = {
        key: []
        for key in metric_keys
    }
    results_opt = {
        key: []
        for key in metric_keys
    }

    all_best_params = []
    all_search_history = []

    start_time = time.time()

    for iteration, seed in enumerate(seeds, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
        )

        base_model = get_models(seed)[best_model_name]
        base_model.fit(X_train, y_train)

        pred_base_train = base_model.predict(X_train)
        results_base["train_R2"].append(
            r2_score(y_train, pred_base_train)
        )
        results_base["train_RMSE"].append(
            np.sqrt(
                mean_squared_error(
                    y_train,
                    pred_base_train,
                )
            )
        )
        results_base["train_MAE"].append(
            mean_absolute_error(
                y_train,
                pred_base_train,
            )
        )

        pred_base_test = base_model.predict(X_test)
        results_base["test_R2"].append(
            r2_score(y_test, pred_base_test)
        )
        results_base["test_RMSE"].append(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_base_test,
                )
            )
        )
        results_base["test_MAE"].append(
            mean_absolute_error(
                y_test,
                pred_base_test,
            )
        )

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
        grid_search = GridSearchCV(
            estimator=get_models(seed)[best_model_name],
            param_grid=param_grid,
            scoring="r2",
            cv=kf,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_params_for_table = {
            key: "None" if value is None else value
            for key, value in best_params.items()
        }
        all_best_params.append(best_params_for_table)

        for params, score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
        ):
            row = params.copy()
            row["Objective Value (R2)"] = score
            all_search_history.append(row)

        optimized_model = grid_search.best_estimator_

        pred_opt_train = optimized_model.predict(X_train)
        results_opt["train_R2"].append(
            r2_score(y_train, pred_opt_train)
        )
        results_opt["train_RMSE"].append(
            np.sqrt(
                mean_squared_error(
                    y_train,
                    pred_opt_train,
                )
            )
        )
        results_opt["train_MAE"].append(
            mean_absolute_error(
                y_train,
                pred_opt_train,
            )
        )

        pred_opt_test = optimized_model.predict(X_test)
        results_opt["test_R2"].append(
            r2_score(y_test, pred_opt_test)
        )
        results_opt["test_RMSE"].append(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_opt_test,
                )
            )
        )
        results_opt["test_MAE"].append(
            mean_absolute_error(
                y_test,
                pred_opt_test,
            )
        )

        print(
            f"Iteration {iteration}/{n_iterations}, "
            f"seed {seed:4d}, best parameters: {best_params}"
        )

    elapsed_time = time.time() - start_time
    print(
        "Hyperparameter validation completed in "
        f"{elapsed_time:.2f} seconds"
    )

    print("Stage 3: Visualization and SI table generation")

    plot_df = pd.DataFrame(all_search_history)

    if "max_depth" in plot_df.columns:
        plot_df["max_depth"] = plot_df["max_depth"].fillna(0)

    param_cols = [
        column
        for column in plot_df.columns
        if column != "Objective Value (R2)"
    ]

    fig = px.parallel_coordinates(
        plot_df,
        color="Objective Value (R2)",
        dimensions=param_cols,
        color_continuous_scale=px.colors.diverging.Tealrose,
        title=(
            "Hyperparameter Search Space and Objective Value "
            "(Note: max_depth=0 represents 'None')"
        ),
    )

    html_file = "Hyperparameter_Search_Plot.html"
    fig.write_html(html_file)
    print(
        "Hyperparameter parallel-coordinate plot "
        f"saved as {html_file}"
    )

    params_df = pd.DataFrame(all_best_params)

    print(
        f"Hyperparameter distribution over "
        f"{n_iterations} validation iterations"
    )
    print(
        f"{'Hyperparameter':<20} | "
        f"{'Most Frequent (Mode)':<20} | "
        f"{'Distribution (Count)'}"
    )
    print("-" * 80)

    for column in params_df.columns:
        counts = params_df[column].value_counts()
        most_frequent = counts.index[0]
        distribution = ", ".join(
            f"{value} ({count})"
            for value, count in counts.items()
        )
        print(
            f"{column:<20} | "
            f"{str(most_frequent):<20} | "
            f"{distribution}"
        )

    print("-" * 80)

    print("Stage 4: Final performance comparison")
    print(
        f"Selected model: {best_model_name}, "
        f"based on the mean of {n_iterations} "
        "Monte Carlo splits"
    )
    print("-" * 95)
    print(
        "Metric | Baseline Train | Baseline Test | "
        "Optimized Train | Optimized Test"
    )
    print("-" * 95)

    for metric in ["R2", "RMSE", "MAE"]:
        baseline_train_mean = np.mean(
            results_base[f"train_{metric}"]
        )
        baseline_train_std = np.std(
            results_base[f"train_{metric}"]
        )
        baseline_test_mean = np.mean(
            results_base[f"test_{metric}"]
        )
        baseline_test_std = np.std(
            results_base[f"test_{metric}"]
        )
        optimized_train_mean = np.mean(
            results_opt[f"train_{metric}"]
        )
        optimized_train_std = np.std(
            results_opt[f"train_{metric}"]
        )
        optimized_test_mean = np.mean(
            results_opt[f"test_{metric}"]
        )
        optimized_test_std = np.std(
            results_opt[f"test_{metric}"]
        )

        direction = "higher" if metric == "R2" else "lower"

        print(
            f"{metric:4s} ({direction}) | "
            f"{baseline_train_mean:7.4f} "
            f"+/- {baseline_train_std:.4f} | "
            f"{baseline_test_mean:7.4f} "
            f"+/- {baseline_test_std:.4f} | "
            f"{optimized_train_mean:7.4f} "
            f"+/- {optimized_train_std:.4f} | "
            f"{optimized_test_mean:7.4f} "
            f"+/- {optimized_test_std:.4f}"
        )

    print("-" * 95)
    print("Analysis completed")


if __name__ == "__main__":
    main()
