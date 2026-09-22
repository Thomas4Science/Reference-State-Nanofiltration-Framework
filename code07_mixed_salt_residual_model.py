import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings


warnings.filterwarnings("ignore")


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_mixed_raw = "Sheet3_Mixed"
    sheet_single = "Sheet9_deal5"
    sheet_mix3 = "Sheet10_Mix3"
    sheet_final = "Sheet14_Mix_Resi_ML"

    print("Stage 1: Mixed-salt data reconstruction")

    try:
        df_mixed = pd.read_excel(
            file_path,
            sheet_name=sheet_mixed_raw,
        )
    except Exception as exc:
        print(
            f"Failed to read mixed-salt data "
            f"from {sheet_mixed_raw}: {exc}"
        )
        return

    augmented_rows = []

    for _, row in df_mixed.iterrows():
        target_name_1 = str(
            row["Targeted ion/cation"]
        ).strip()
        comp_name_1 = str(
            row["Competing ion/cation"]
        ).strip()

        conc_target_1 = row[
            "Single salt concentration (M)"
        ]
        conc_comp_1 = (
            row["Total concentration (M)"]
            - conc_target_1
        )

        rejection_1 = row.get(
            target_name_1,
            np.nan,
        )

        if pd.notna(rejection_1):
            dict_1 = row.to_dict()
            dict_1["Actual_Rejection"] = rejection_1
            dict_1["Target_Ion_Name"] = target_name_1

            dict_1["Target_Mw"] = row["Mw (g/mol)"]
            dict_1["Target_Ads"] = row[
                "Adsorption energy"
            ]
            dict_1["Target_Valence"] = row["Valence"]
            dict_1["Target_Hydration"] = row[
                "Hydration Energy -\u0394G (kJ/mol)"
            ]
            dict_1["Target_D"] = row[
                "Pore_diffusion coefficient D (10-9m2/s)"
            ]
            dict_1["Target_r_s"] = row[
                "Stokes radius (nm)"
            ]
            dict_1["Target_Conc"] = conc_target_1

            dict_1["Comp_Valence"] = row["Valence.1"]
            dict_1["Comp_Hydration"] = row[
                "Hydration Energy -\u0394G (kJ/mol).1"
            ]
            dict_1["Comp_D"] = row[
                "Pore_diffusion coefficient D (10-9m2/s).1"
            ]
            dict_1["Comp_r_s"] = row[
                "Stokes radius (nm).1"
            ]
            dict_1["Comp_Conc"] = conc_comp_1

            augmented_rows.append(dict_1)

        rejection_2 = row.get(
            comp_name_1,
            np.nan,
        )

        if pd.notna(rejection_2):
            dict_2 = row.to_dict()
            dict_2["Actual_Rejection"] = rejection_2
            dict_2["Target_Ion_Name"] = comp_name_1

            dict_2["Target_Mw"] = row["Mw (g/mol).1"]
            dict_2["Target_Ads"] = row[
                "Adsorption energy.1"
            ]
            dict_2["Target_Valence"] = row[
                "Valence.1"
            ]
            dict_2["Target_Hydration"] = row[
                "Hydration Energy -\u0394G (kJ/mol).1"
            ]
            dict_2["Target_D"] = row[
                "Pore_diffusion coefficient D (10-9m2/s).1"
            ]
            dict_2["Target_r_s"] = row[
                "Stokes radius (nm).1"
            ]
            dict_2["Target_Conc"] = conc_comp_1

            dict_2["Comp_Valence"] = row["Valence"]
            dict_2["Comp_Hydration"] = row[
                "Hydration Energy -\u0394G (kJ/mol)"
            ]
            dict_2["Comp_D"] = row[
                "Pore_diffusion coefficient D (10-9m2/s)"
            ]
            dict_2["Comp_r_s"] = row[
                "Stokes radius (nm)"
            ]
            dict_2["Comp_Conc"] = conc_target_1

            augmented_rows.append(dict_2)

    df_aug = pd.DataFrame(augmented_rows)

    print(
        "Ion-identity exchange completed. "
        f"Mixed-salt rows extracted: {len(df_aug)}"
    )

    F = 96485.332
    R = 8.314

    temperature = df_aug["Absolute temperature (K)"]
    zeta_v = df_aug["zeta potential (mV)"] * 1e-3
    pore_radius = df_aug["Pore radius (nm)"]

    delta_pressure_raw = (
        df_aug["Hydraulic pressure (bar)"]
        - df_aug["Overall osmotic pressure (atm)"]
    )
    delta_pressure = np.clip(
        delta_pressure_raw,
        1e-4,
        None,
    )

    water_flux_raw = (
        df_aug["Water permeability (L/m2*h*bar)"]
        * delta_pressure
    )
    water_flux_factor = 1e-3 / 3600

    df_aug["Steric Number (Anion)"] = (
        df_aug["Stokes radius (nm).2"]
        / pore_radius
    )

    anion_diffusion_safe = np.clip(
        df_aug[
            "Pore_diffusion coefficient D (10-9m2/s).2"
        ],
        1e-9,
        None,
    )

    df_aug["Pe Number (Anion)"] = (
        water_flux_raw
        * water_flux_factor
        * pore_radius
    ) / anion_diffusion_safe

    df_aug["Donnan Number (Anion)"] = (
        df_aug["Valence.2"]
        * F
        * zeta_v
    ) / (R * temperature)

    df_aug["Hydration Number (Anion)"] = (
        df_aug[
            "Hydration Energy -\u0394G (kJ/mol).2"
        ]
        * 1000
    ) / (R * temperature)

    df_aug["Steric Number (Cation)"] = (
        df_aug["Target_r_s"]
        / pore_radius
    )

    target_diffusion_safe = np.clip(
        df_aug["Target_D"],
        1e-9,
        None,
    )

    df_aug["Pe Number (Cation)"] = (
        water_flux_raw
        * water_flux_factor
        * pore_radius
    ) / target_diffusion_safe

    df_aug["Donnan Number (Cation)"] = (
        df_aug["Target_Valence"]
        * F
        * zeta_v
    ) / (R * temperature)

    df_aug["Hydration Number (Cation)"] = (
        df_aug["Target_Hydration"]
        * 1000
    ) / (R * temperature)

    df_aug["Water contact angle (\u00b0)"] = (
        df_aug["Water contact angle (\u00b0)"]
    )
    df_aug["Charge density (C/m2)"] = (
        df_aug["Charge density (C/m2)"]
    )
    df_aug["Mw (g/mol)"] = df_aug["Target_Mw"]
    df_aug["Adsorption energy"] = (
        df_aug["Target_Ads"]
    )

    df_aug["Comp_Ratio_Steric"] = (
        df_aug["Target_r_s"]
        / df_aug["Comp_r_s"]
    )
    df_aug["Comp_Ratio_Valence"] = (
        df_aug["Target_Valence"]
        / df_aug["Comp_Valence"]
    )
    df_aug["Comp_Diff_Hydration"] = (
        df_aug["Target_Hydration"]
        - df_aug["Comp_Hydration"]
    )

    competing_diffusion_safe = np.clip(
        df_aug["Comp_D"],
        1e-9,
        None,
    )

    df_aug["Comp_Ratio_D"] = np.clip(
        df_aug["Target_D"]
        / competing_diffusion_safe,
        0.0,
        999.0,
    )

    df_aug["Comp_Conc_Fraction"] = (
        df_aug["Target_Conc"]
        / (
            df_aug["Target_Conc"]
            + df_aug["Comp_Conc"]
        )
    )

    features_12 = [
        "Water contact angle (\u00b0)",
        "Charge density (C/m2)",
        "Mw (g/mol)",
        "Adsorption energy",
        "Steric Number (Anion)",
        "Donnan Number (Anion)",
        "Hydration Number (Anion)",
        "Pe Number (Anion)",
        "Steric Number (Cation)",
        "Donnan Number (Cation)",
        "Hydration Number (Cation)",
        "Pe Number (Cation)",
    ]

    competition_features = [
        "Comp_Ratio_Steric",
        "Comp_Ratio_Valence",
        "Comp_Diff_Hydration",
        "Comp_Ratio_D",
        "Comp_Conc_Fraction",
    ]

    final_cols_mix3 = [
        "Target_Ion_Name",
        "Actual_Rejection",
    ] + features_12 + competition_features

    df_mix3 = df_aug[final_cols_mix3].copy()

    try:
        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            df_mix3.to_excel(
                writer,
                sheet_name=sheet_mix3,
                index=False,
            )
    except Exception as exc:
        print(
            f"Failed to write {sheet_mix3}: {exc}"
        )
        return

    print(
        "Aligned mixed-salt data saved to "
        f"{sheet_mix3}"
    )

    print("Stage 2: Single-salt baseline model evaluation")

    try:
        df_single = pd.read_excel(
            file_path,
            sheet_name=sheet_single,
        )
    except Exception as exc:
        print(
            f"Failed to read single-salt data "
            f"from {sheet_single}: {exc}"
        )
        return

    df_single = (
        df_single.dropna(
            subset=features_12 + ["Rejection"]
        )
        .reset_index(drop=True)
    )

    X_single = df_single[features_12]
    y_single = df_single["Rejection"]

    print(
        "Running 20 Monte Carlo evaluations "
        "using an 80:20 split"
    )

    rf_eval = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    r2_list = []
    rmse_list = []
    mae_list = []

    for index in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X_single,
            y_single,
            test_size=0.2,
            random_state=index * 42,
        )

        rf_eval.fit(
            X_train,
            y_train,
        )

        predictions = rf_eval.predict(X_test)

        r2_list.append(
            r2_score(
                y_test,
                predictions,
            )
        )
        rmse_list.append(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )
        mae_list.append(
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

    print("Single-salt baseline model performance")
    print("-" * 65)
    print(f"Mean Test R2   : {np.mean(r2_list):.4f}")
    print(f"Mean Test RMSE : {np.mean(rmse_list):.4f}")
    print(f"Mean Test MAE  : {np.mean(mae_list):.4f}")
    print("-" * 65)

    if np.mean(r2_list) > 0.75:
        print(
            "The mean Monte Carlo test R2 is above 0.75"
        )
    else:
        print(
            "The model may require additional data "
            "to improve stability"
        )

    print(
        "Running GridSearchCV on the complete "
        "single-salt dataset"
    )

    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 3],
    }

    rf_grid = GridSearchCV(
        RandomForestRegressor(
            random_state=42
        ),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )

    rf_grid.fit(
        X_single,
        y_single,
    )

    final_best_rf = rf_grid.best_estimator_

    print(
        "Selected single-salt model parameters: "
        f"{rf_grid.best_params_}"
    )

    print(
        "Stage 3: Mixed-salt prediction "
        "and residual calculation"
    )

    X_mixed = df_mix3[features_12]

    print(
        f"Predicting baseline rejection for "
        f"{len(X_mixed)} mixed-salt rows"
    )

    df_mix3["Predicted_Base_Rejection"] = (
        final_best_rf.predict(X_mixed)
    )

    df_mix3["Residual_Rejection"] = (
        df_mix3["Actual_Rejection"]
        - df_mix3["Predicted_Base_Rejection"]
    )

    mean_residual = df_mix3[
        "Residual_Rejection"
    ].mean()

    print(
        "Mean mixed-salt residual: "
        f"{mean_residual:.2f}%"
    )

    final_cols_residual = (
        competition_features
        + ["Residual_Rejection"]
    )

    df_final = df_mix3[
        final_cols_residual
    ].copy()

    try:
        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            df_final.to_excel(
                writer,
                sheet_name=sheet_final,
                index=False,
            )
    except Exception as exc:
        print(
            f"Failed to write {sheet_final}: {exc}"
        )
        return

    print(
        "Residual machine-learning dataset "
        f"saved to {sheet_final}"
    )

    print("First five rows of the residual dataset")
    print("-" * 80)
    print(df_final.head(5))
    print("-" * 80)
    print("Processing completed")


if __name__ == "__main__":
    main()
