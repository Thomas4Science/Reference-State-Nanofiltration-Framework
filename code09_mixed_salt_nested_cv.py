import pandas as pd
import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings


warnings.filterwarnings("ignore")


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet14_Mix_Resi_ML"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:
        print(f"Failed to read the data: {exc}")
        return

    y = df["Residual_Rejection"]
    comp_features = [
        "Comp_Ratio_Valence",
        "Comp_Ratio_D",
        "Comp_Conc_Fraction",
    ]
    X = df[comp_features]

    print("=" * 54)
    print(
        "Mixed-salt residual modeling: four tree-based "
        "models with nested CV"
    )
    print("=" * 54)
    print(
        "This process may take some time. "
        "Please wait."
    )

    param_grids = {
        "Random Forest": {
            "model": RandomForestRegressor(
                random_state=None,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6],
                "min_samples_leaf": [2, 4],
            },
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(
                random_state=None,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6],
                "min_samples_leaf": [2, 4],
            },
        },
        "XGBoost": {
            "model": XGBRegressor(
                random_state=None,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [100, 150],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.05],
                "reg_lambda": [1.0, 3.0],
            },
        },
        "LightGBM": {
            "model": LGBMRegressor(
                random_state=None,
                n_jobs=-1,
                verbose=-1,
            ),
            "params": {
                "n_estimators": [100, 150],
                "max_depth": [3, 4],
                "num_leaves": [5, 10],
                "learning_rate": [0.03, 0.05],
            },
        },
    }

    n_iterations = 20

    results = {
        name: {
            "Train_R2": [],
            "Test_R2": [],
            "Train_RMSE": [],
            "Test_RMSE": [],
            "Train_MAE": [],
            "Test_MAE": [],
        }
        for name in param_grids.keys()
    }

    for iteration in range(n_iterations):
        print(
            f"Running Monte Carlo iteration "
            f"{iteration + 1}/{n_iterations}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=iteration * 66,
        )

        for name, config in param_grids.items():
            grid = GridSearchCV(
                config["model"],
                config["params"],
                cv=3,
                scoring="r2",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            results[name]["Train_R2"].append(
                max(r2_score(y_train, y_train_pred), -0.5)
            )
            results[name]["Test_R2"].append(
                max(r2_score(y_test, y_test_pred), -0.5)
            )
            results[name]["Train_RMSE"].append(
                np.sqrt(
                    mean_squared_error(
                        y_train,
                        y_train_pred,
                    )
                )
            )
            results[name]["Test_RMSE"].append(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_test_pred,
                    )
                )
            )
            results[name]["Train_MAE"].append(
                mean_absolute_error(
                    y_train,
                    y_train_pred,
                )
            )
            results[name]["Test_MAE"].append(
                mean_absolute_error(
                    y_test,
                    y_test_pred,
                )
            )

    print("Nested CV performance report")
    print("=" * 135)
    print(
        f"{'Model':<15} | {'Train R2':<17} | "
        f"{'Test R2':<17} | {'Train RMSE':<17} | "
        f"{'Test RMSE':<17} | {'Train MAE':<17} | "
        f"{'Test MAE':<17}"
    )
    print("-" * 135)

    for name in param_grids.keys():
        tr_r2_m = np.mean(results[name]["Train_R2"])
        tr_r2_s = np.std(results[name]["Train_R2"])
        te_r2_m = np.mean(results[name]["Test_R2"])
        te_r2_s = np.std(results[name]["Test_R2"])

        tr_rmse_m = np.mean(results[name]["Train_RMSE"])
        tr_rmse_s = np.std(results[name]["Train_RMSE"])
        te_rmse_m = np.mean(results[name]["Test_RMSE"])
        te_rmse_s = np.std(results[name]["Test_RMSE"])

        tr_mae_m = np.mean(results[name]["Train_MAE"])
        tr_mae_s = np.std(results[name]["Train_MAE"])
        te_mae_m = np.mean(results[name]["Test_MAE"])
        te_mae_s = np.std(results[name]["Test_MAE"])

        print(
            f"{name:<15} | "
            f"{tr_r2_m:>7.4f} +/- {tr_r2_s:<6.4f} | "
            f"{te_r2_m:>7.4f} +/- {te_r2_s:<6.4f} | "
            f"{tr_rmse_m:>7.4f} +/- {tr_rmse_s:<6.4f} | "
            f"{te_rmse_m:>7.4f} +/- {te_rmse_s:<6.4f} | "
            f"{tr_mae_m:>7.4f} +/- {tr_mae_s:<6.4f} | "
            f"{te_mae_m:>7.4f} +/- {te_mae_s:<6.4f}"
        )

    print("=" * 135)

    summary_data = []

    for name in param_grids.keys():
        summary_data.append(
            {
                "Model": name,
                "Train_R2_Mean": np.round(
                    np.mean(results[name]["Train_R2"]),
                    4,
                ),
                "Train_R2_Std": np.round(
                    np.std(results[name]["Train_R2"]),
                    4,
                ),
                "Test_R2_Mean": np.round(
                    np.mean(results[name]["Test_R2"]),
                    4,
                ),
                "Test_R2_Std": np.round(
                    np.std(results[name]["Test_R2"]),
                    4,
                ),
                "Train_RMSE_Mean": np.round(
                    np.mean(results[name]["Train_RMSE"]),
                    4,
                ),
                "Train_RMSE_Std": np.round(
                    np.std(results[name]["Train_RMSE"]),
                    4,
                ),
                "Test_RMSE_Mean": np.round(
                    np.mean(results[name]["Test_RMSE"]),
                    4,
                ),
                "Test_RMSE_Std": np.round(
                    np.std(results[name]["Test_RMSE"]),
                    4,
                ),
                "Train_MAE_Mean": np.round(
                    np.mean(results[name]["Train_MAE"]),
                    4,
                ),
                "Train_MAE_Std": np.round(
                    np.std(results[name]["Train_MAE"]),
                    4,
                ),
                "Test_MAE_Mean": np.round(
                    np.mean(results[name]["Test_MAE"]),
                    4,
                ),
                "Test_MAE_Std": np.round(
                    np.std(results[name]["Test_MAE"]),
                    4,
                ),
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(
        "ML_Performance_Summary.csv",
        index=False,
    )

    print(
        "Output file 1 generated: "
        "ML_Performance_Summary.csv"
    )

    detailed_data = {
        "Split_ID": list(range(1, n_iterations + 1))
    }

    for name in param_grids.keys():
        for metric in [
            "Train_R2",
            "Test_R2",
            "Train_RMSE",
            "Test_RMSE",
            "Train_MAE",
            "Test_MAE",
        ]:
            column_name = f"{name}_{metric}"
            detailed_data[column_name] = results[name][metric]

    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(
        "ML_Detailed_20Splits_Data.csv",
        index=False,
    )

    print(
        "Output file 2 generated: "
        "ML_Detailed_20Splits_Data.csv"
    )

    print(
        "Generating the Train vs Test boxplot "
        "with scatter points"
    )

    long_data = []
    for iteration in range(n_iterations):
        for name in param_grids.keys():
            long_data.append(
                {
                    "Model": name,
                    "Phase": "Training Set",
                    "R2 Score": results[name]["Train_R2"][iteration],
                }
            )
            long_data.append(
                {
                    "Model": name,
                    "Phase": "Testing Set",
                    "R2 Score": results[name]["Test_R2"][iteration],
                }
            )

    df_long = pd.DataFrame(long_data)

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        x="Model",
        y="R2 Score",
        hue="Phase",
        data=df_long,
        palette=["#5c9eb2", "#e17b5e"],
        width=0.6,
        showmeans=True,
        meanprops={
            "marker": "*",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "12",
        },
    )

    sns.stripplot(
        x="Model",
        y="R2 Score",
        hue="Phase",
        data=df_long,
        dodge=True,
        marker="o",
        alpha=0.4,
        color="black",
        jitter=0.2,
        legend=False,
    )

    plt.title(
        "Distribution of $R^2$ Scores Across "
        "20 Monte Carlo Splits",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel(
        "$R^2$ Score",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(
        "Machine Learning Models",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylim(
        max(-0.2, df_long["R2 Score"].min() - 0.1),
        1.1,
    )

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.6,
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend(
        loc="lower right",
        fontsize=11,
    )

    plt.tight_layout()

    plot_filename = "ML_TrainTest_BoxPlot.png"
    plt.savefig(
        plot_filename,
        dpi=300,
    )
    plt.close()

    print(
        f"Output file 3 generated: {plot_filename}"
    )
    print("All tasks completed")


if __name__ == "__main__":
    main()
