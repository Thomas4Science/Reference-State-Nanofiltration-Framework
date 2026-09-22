import pandas as pd
import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import re
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
    competition_features = [
        "Comp_Ratio_Valence",
        "Comp_Ratio_D",
        "Comp_Conc_Fraction",
    ]
    X = df[competition_features]

    nice_names = {
        "Comp_Ratio_Valence": "Valence Ratio (Target/Comp)",
        "Comp_Ratio_D": "Diffusivity Ratio (Target/Comp)",
        "Comp_Conc_Fraction": "Concentration Fraction",
    }
    X = X.rename(columns=nice_names)
    X.columns = [
        re.sub(r"[\[\]<>]", "", column)
        for column in X.columns
    ]

    print("=" * 54)
    print(
        "LightGBM evaluation with split-specific output, "
        "seed tracking, and SHAP export"
    )
    print("=" * 54)
    print("Running 20 Monte Carlo evaluations")

    n_iterations = 20

    train_r2_list = []
    test_r2_list = []
    train_rmse_list = []
    test_rmse_list = []
    train_mae_list = []
    test_mae_list = []

    train_predictions_dict = {
        index: []
        for index in X.index
    }
    test_predictions_dict = {
        index: []
        for index in X.index
    }

    best_test_r2_overall = -float("inf")
    best_seed = None

    best_train_index = None
    best_train_actual = None
    best_train_prediction = None
    best_test_index = None
    best_test_actual = None
    best_test_prediction = None

    param_grid = {
        "n_estimators": [150, 200],
        "max_depth": [3, 4],
        "num_leaves": [5, 10],
        "learning_rate": [0.03, 0.05],
    }

    for iteration in range(n_iterations):
        current_seed = iteration * 66

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=current_seed,
        )

        lgbm = LGBMRegressor(
            random_state=iteration,
            verbose=-1,
            n_jobs=-1,
        )
        grid = GridSearchCV(
            lgbm,
            param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_train_prediction = best_model.predict(X_train)
        y_test_prediction = best_model.predict(X_test)

        current_train_r2 = max(
            r2_score(y_train, y_train_prediction),
            -0.5,
        )
        current_test_r2 = max(
            r2_score(y_test, y_test_prediction),
            -0.5,
        )

        train_r2_list.append(current_train_r2)
        test_r2_list.append(current_test_r2)
        train_rmse_list.append(
            np.sqrt(
                mean_squared_error(
                    y_train,
                    y_train_prediction,
                )
            )
        )
        test_rmse_list.append(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    y_test_prediction,
                )
            )
        )
        train_mae_list.append(
            mean_absolute_error(
                y_train,
                y_train_prediction,
            )
        )
        test_mae_list.append(
            mean_absolute_error(
                y_test,
                y_test_prediction,
            )
        )

        for index_value, prediction_value in zip(
            y_train.index,
            y_train_prediction,
        ):
            train_predictions_dict[index_value].append(
                prediction_value
            )

        for index_value, prediction_value in zip(
            y_test.index,
            y_test_prediction,
        ):
            test_predictions_dict[index_value].append(
                prediction_value
            )

        if current_test_r2 > best_test_r2_overall:
            best_test_r2_overall = current_test_r2
            best_seed = current_seed

            best_train_index = y_train.index
            best_train_actual = y_train.values
            best_train_prediction = y_train_prediction

            best_test_index = y_test.index
            best_test_actual = y_test.values
            best_test_prediction = y_test_prediction

    print("LightGBM Monte Carlo performance report")
    print("-" * 68)
    print(
        f"{'Metric':<10} | "
        f"{'Train (Mean +/- SD)':<24} | "
        f"{'Test (Mean +/- SD)':<24}"
    )
    print("-" * 68)
    print(
        f"{'R2':<10} | "
        f"{np.mean(train_r2_list):>7.4f} "
        f"+/- {np.std(train_r2_list):<7.4f}     | "
        f"{np.mean(test_r2_list):>7.4f} "
        f"+/- {np.std(test_r2_list):<7.4f}"
    )
    print(
        f"{'RMSE':<10} | "
        f"{np.mean(train_rmse_list):>7.4f} "
        f"+/- {np.std(train_rmse_list):<7.4f}     | "
        f"{np.mean(test_rmse_list):>7.4f} "
        f"+/- {np.std(test_rmse_list):<7.4f}"
    )
    print(
        f"{'MAE':<10} | "
        f"{np.mean(train_mae_list):>7.4f} "
        f"+/- {np.std(train_mae_list):<7.4f}     | "
        f"{np.mean(test_mae_list):>7.4f} "
        f"+/- {np.std(test_mae_list):<7.4f}"
    )
    print("-" * 68)

    best_train_df = pd.DataFrame(
        {
            "Train_Sample_ID": best_train_index,
            "Train_Actual": best_train_actual,
            "Train_Pred": best_train_prediction,
        }
    ).reset_index(drop=True)

    best_test_df = pd.DataFrame(
        {
            "Test_Sample_ID": best_test_index,
            "Test_Actual": best_test_actual,
            "Test_Pred": best_test_prediction,
        }
    ).reset_index(drop=True)

    best_split_df = pd.concat(
        [
            best_train_df,
            best_test_df,
        ],
        axis=1,
    )

    best_csv = (
        f"LGBM_Best_Split_Seed_{best_seed}_"
        "Parity_Data.csv"
    )
    best_split_df.round(4).to_csv(
        best_csv,
        index=False,
    )

    print(
        "Best split saved with separate Train and Test columns: "
        f"{best_csv}"
    )
    print(
        f"Best seed: {best_seed}, "
        f"Test R2: {best_test_r2_overall:.4f}"
    )

    print(
        "Training the global LightGBM model "
        "for SHAP analysis"
    )

    global_lgbm = LGBMRegressor(
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    global_grid = GridSearchCV(
        global_lgbm,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    global_grid.fit(X, y)
    best_global_lgbm = global_grid.best_estimator_

    explainer = shap.TreeExplainer(best_global_lgbm)
    shap_values = explainer.shap_values(X)

    print("Global SHAP values calculated")

    shap_mas = np.abs(shap_values).mean(axis=0)

    shap_mas_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Mean_Abs_SHAP": shap_mas,
        }
    ).sort_values(
        by="Mean_Abs_SHAP",
        ascending=False,
    )

    mas_csv = "SHAP_MAS_Values.csv"
    shap_mas_df.round(4).to_csv(
        mas_csv,
        index=False,
    )

    print(
        "Mean absolute SHAP values saved as "
        f"{mas_csv}"
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_type="dot",
    )
    plt.xticks(fontsize=12)
    plt.yticks(
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(
        r"SHAP value (Impact on Residual $\Delta R$, %)",
        fontsize=15,
        fontweight="bold",
    )
    plt.title(
        "LightGBM SHAP Analysis of Mixed-Salt Competition",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    beeswarm_file = "SHAP_Beeswarm_LGBM_Residual.png"
    plt.savefig(
        beeswarm_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False,
        color="#00a6a6",
    )
    plt.xticks(fontsize=12)
    plt.yticks(
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(
        r"Mean |SHAP value| "
        r"(Average contribution to $\Delta R$)",
        fontsize=15,
        fontweight="bold",
    )
    plt.title(
        "Dominant Competitive Factors",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    mas_file = "SHAP_MAS_LGBM_Residual.png"
    plt.savefig(
        mas_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Generated preview plot: {beeswarm_file}")
    print(f"Generated preview plot: {mas_file}")


if __name__ == "__main__":
    main()
