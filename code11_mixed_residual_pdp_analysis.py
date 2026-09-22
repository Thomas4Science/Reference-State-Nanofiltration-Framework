import pandas as pd
import numpy as np

def process_mix1(file_path, sheet_name):
    print("\n" + "="*60)
    print("Generating Sheet10_Mix1: Data augmentation and descriptors")
    print("="*60)
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    augmented_rows = []
    
    # Data augmentation via targeted/competing ion identity swapping
    for idx, row in df.iterrows():
        target_name_1 = str(row['Targeted ion/cation']).strip()
        comp_name_1 = str(row['Competing ion/cation']).strip()

        conc_target_1 = row['Single salt concentration (M)']
        conc_comp_1 = row['Total concentration (M)'] - conc_target_1

        # Perspective A: Treat the first cation as the target
        rej_1 = row.get(target_name_1, np.nan)
        if pd.notna(rej_1):
            dict_1 = row.to_dict()
            dict_1['Actual_Rejection'] = rej_1
            dict_1['Target_Ion_Name'] = target_name_1

            dict_1['Target_Valence'] = row['Valence']
            dict_1['Target_Hydration'] = row['Hydration Energy -ΔG (kJ/mol)']
            dict_1['Target_D'] = row['Pore_diffusion coefficient D (10-9m2/s)']
            dict_1['Target_r_s'] = row['Stokes radius (nm)']
            dict_1['Target_Conc'] = conc_target_1

            dict_1['Comp_Valence'] = row['Valence.1']
            dict_1['Comp_Hydration'] = row['Hydration Energy -ΔG (kJ/mol).1']
            dict_1['Comp_D'] = row['Pore_diffusion coefficient D (10-9m2/s).1']
            dict_1['Comp_r_s'] = row['Stokes radius (nm).1']
            dict_1['Comp_Conc'] = conc_comp_1
            augmented_rows.append(dict_1)

        # Perspective B: Treat the second cation as the target
        rej_2 = row.get(comp_name_1, np.nan)
        if pd.notna(rej_2):
            dict_2 = row.to_dict()
            dict_2['Actual_Rejection'] = rej_2
            dict_2['Target_Ion_Name'] = comp_name_1

            dict_2['Target_Valence'] = row['Valence.1']
            dict_2['Target_Hydration'] = row['Hydration Energy -ΔG (kJ/mol).1']
            dict_2['Target_D'] = row['Pore_diffusion coefficient D (10-9m2/s).1']
            dict_2['Target_r_s'] = row['Stokes radius (nm).1']
            dict_2['Target_Conc'] = conc_comp_1

            dict_2['Comp_Valence'] = row['Valence']
            dict_2['Comp_Hydration'] = row['Hydration Energy -ΔG (kJ/mol)']
            dict_2['Comp_D'] = row['Pore_diffusion coefficient D (10-9m2/s)']
            dict_2['Comp_r_s'] = row['Stokes radius (nm)']
            dict_2['Comp_Conc'] = conc_target_1
            augmented_rows.append(dict_2)

    df_aug = pd.DataFrame(augmented_rows)
    print(f"Data augmentation completed. Total rows: {len(df_aug)}")

    # Constants for physical descriptors
    F, R = 96485.332, 8.314
    T = df_aug['Absolute temperature (K)']
    zeta_V = df_aug['zeta potential (mV)'] * 1e-3
    r_p = df_aug['Pore radius (nm)']

    # Calculate transmembrane parameters
    delta_P = df_aug['Hydraulic pressure (bar)'] - df_aug['Overall osmotic pressure (atm)']
    J_v_raw = df_aug['Water permeability (L/m2*h*bar)'] * delta_P
    J_v_factor = 1e-3 / 3600

    # Dimensionless features (Anion)
    df_aug['St_A (Anion)'] = df_aug['Stokes radius (nm).2'] / r_p
    df_aug['Pe_A (Anion)'] = (J_v_raw * J_v_factor * r_p) / df_aug['Pore_diffusion coefficient D (10-9m2/s).2']
    df_aug['Do_A (Anion)'] = (df_aug['Valence.2'] * F * zeta_V) / (R * T)
    df_aug['CA (Contact Angle)'] = df_aug['Water contact angle (°)']

    # Dimensionless features (Target Cation)
    df_aug['St_T (Target)'] = df_aug['Target_r_s'] / r_p
    df_aug['Pe_T (Target)'] = (J_v_raw * J_v_factor * r_p) / df_aug['Target_D']
    df_aug['Do_T (Target)'] = (df_aug['Target_Valence'] * F * zeta_V) / (R * T)
    df_aug['Hy_T (Target)'] = (df_aug['Target_Hydration'] * 1000) / (R * T)

    # Competitive transport features
    df_aug['Comp_Ratio_Steric'] = df_aug['Target_r_s'] / df_aug['Comp_r_s']
    df_aug['Comp_Ratio_Valence'] = df_aug['Target_Valence'] / df_aug['Comp_Valence']
    df_aug['Comp_Diff_Hydration'] = df_aug['Target_Hydration'] - df_aug['Comp_Hydration']
    df_aug['Comp_Ratio_D'] = df_aug['Target_D'] / df_aug['Comp_D']
    df_aug['Comp_Conc_Fraction'] = df_aug['Target_Conc'] / (df_aug['Target_Conc'] + df_aug['Comp_Conc'])

    final_cols = [
        'Target_Ion_Name', 'Actual_Rejection',
        'St_A (Anion)', 'Pe_A (Anion)', 'Do_A (Anion)', 'CA (Contact Angle)',
        'St_T (Target)', 'Pe_T (Target)', 'Do_T (Target)', 'Hy_T (Target)',
        'Comp_Ratio_Steric', 'Comp_Ratio_Valence', 'Comp_Diff_Hydration',
        'Comp_Ratio_D', 'Comp_Conc_Fraction'
    ]

    df_final = df_aug[final_cols].copy()
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_final.to_excel(writer, sheet_name="Sheet10_Mix1", index=False)
    print("Sheet10_Mix1 successfully generated and saved.")


def process_mix2(file_path, sheet_name):
    print("\n" + "="*60)
    print("Generating Sheet10_Mix2: Physics-constrained clipping")
    print("="*60)
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    augmented_rows = []

    # Data augmentation block
    for idx, row in df.iterrows():
        target_name_1 = str(row['Targeted ion/cation']).strip()
        comp_name_1 = str(row['Competing ion/cation']).strip()
        conc_target_1 = row['Single salt concentration (M)']
        conc_comp_1 = row['Total concentration (M)'] - conc_target_1

        rej_1 = row.get(target_name_1, np.nan)
        if pd.notna(rej_1):
            dict_1 = row.to_dict()
            dict_1['Actual_Rejection'] = rej_1
            dict_1['Target_Ion_Name'] = target_name_1
            dict_1['Target_Valence'] = row['Valence']
            dict_1['Target_Hydration'] = row['Hydration Energy -ΔG (kJ/mol)']
            dict_1['Target_D'] = row['Pore_diffusion coefficient D (10-9m2/s)']
            dict_1['Target_r_s'] = row['Stokes radius (nm)']
            dict_1['Target_Conc'] = conc_target_1
            dict_1['Comp_Valence'] = row['Valence.1']
            dict_1['Comp_Hydration'] = row['Hydration Energy -ΔG (kJ/mol).1']
            dict_1['Comp_D'] = row['Pore_diffusion coefficient D (10-9m2/s).1']
            dict_1['Comp_r_s'] = row['Stokes radius (nm).1']
            dict_1['Comp_Conc'] = conc_comp_1
            augmented_rows.append(dict_1)

        rej_2 = row.get(comp_name_1, np.nan)
        if pd.notna(rej_2):
            dict_2 = row.to_dict()
            dict_2['Actual_Rejection'] = rej_2
            dict_2['Target_Ion_Name'] = comp_name_1
            dict_2['Target_Valence'] = row['Valence.1']
            dict_2['Target_Hydration'] = row['Hydration Energy -ΔG (kJ/mol).1']
            dict_2['Target_D'] = row['Pore_diffusion coefficient D (10-9m2/s).1']
            dict_2['Target_r_s'] = row['Stokes radius (nm).1']
            dict_2['Target_Conc'] = conc_comp_1
            dict_2['Comp_Valence'] = row['Valence']
            dict_2['Comp_Hydration'] = row['Hydration Energy -ΔG (kJ/mol)']
            dict_2['Comp_D'] = row['Pore_diffusion coefficient D (10-9m2/s)']
            dict_2['Comp_r_s'] = row['Stokes radius (nm)']
            dict_2['Comp_Conc'] = conc_target_1
            augmented_rows.append(dict_2)

    df_aug = pd.DataFrame(augmented_rows)
    print(f"Data augmentation completed. Total rows: {len(df_aug)}")

    F, R = 96485.332, 8.314
    T = df_aug['Absolute temperature (K)']
    zeta_V = df_aug['zeta potential (mV)'] * 1e-3
    r_p = df_aug['Pore radius (nm)']

    # Handle infinity values by applying clipping constraints
    delta_P_raw = df_aug['Hydraulic pressure (bar)'] - df_aug['Overall osmotic pressure (atm)']
    delta_P = np.clip(delta_P_raw, 1e-4, None)
    J_v_raw = df_aug['Water permeability (L/m2*h*bar)'] * delta_P
    J_v_factor = 1e-3 / 3600

    df_aug['St_A (Anion)'] = df_aug['Stokes radius (nm).2'] / r_p
    D_anion_safe = np.clip(df_aug['Pore_diffusion coefficient D (10-9m2/s).2'], 1e-9, None)
    df_aug['Pe_A (Anion)'] = (J_v_raw * J_v_factor * r_p) / D_anion_safe
    df_aug['Do_A (Anion)'] = (df_aug['Valence.2'] * F * zeta_V) / (R * T)
    df_aug['Hy_A (Anion)'] = (df_aug['Hydration Energy -ΔG (kJ/mol).2'] * 1000) / (R * T)
    df_aug['CA (Contact Angle)'] = df_aug['Water contact angle (°)']

    df_aug['St_T (Target)'] = df_aug['Target_r_s'] / r_p
    D_target_safe = np.clip(df_aug['Target_D'], 1e-9, None)
    df_aug['Pe_T (Target)'] = (J_v_raw * J_v_factor * r_p) / D_target_safe
    df_aug['Do_T (Target)'] = (df_aug['Target_Valence'] * F * zeta_V) / (R * T)
    df_aug['Hy_T (Target)'] = (df_aug['Target_Hydration'] * 1000) / (R * T)

    df_aug['Comp_Ratio_Steric'] = df_aug['Target_r_s'] / df_aug['Comp_r_s']
    df_aug['Comp_Ratio_Valence'] = df_aug['Target_Valence'] / df_aug['Comp_Valence']
    df_aug['Comp_Diff_Hydration'] = df_aug['Target_Hydration'] - df_aug['Comp_Hydration']
    D_comp_safe = np.clip(df_aug['Comp_D'], 1e-9, None)
    df_aug['Comp_Ratio_D'] = np.clip(df_aug['Target_D'] / D_comp_safe, 0.0, 999.0)
    df_aug['Comp_Conc_Fraction'] = df_aug['Target_Conc'] / (df_aug['Target_Conc'] + df_aug['Comp_Conc'])

    final_cols = [
        'Target_Ion_Name', 'Actual_Rejection',
        'St_A (Anion)', 'Pe_A (Anion)', 'Do_A (Anion)', 'Hy_A (Anion)', 'CA (Contact Angle)',
        'St_T (Target)', 'Pe_T (Target)', 'Do_T (Target)', 'Hy_T (Target)',
        'Comp_Ratio_Steric', 'Comp_Ratio_Valence', 'Comp_Diff_Hydration',
        'Comp_Ratio_D', 'Comp_Conc_Fraction'
    ]

    df_final = df_aug[final_cols].copy()
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_final.to_excel(writer, sheet_name="Sheet10_Mix2", index=False)
    print("Sheet10_Mix2 successfully generated and saved.")


def process_mix3(file_path, sheet_name):
    print("\n" + "="*60)
    print("Generating Sheet10_Mix3: Feature alignment for ML pipeline")
    print("="*60)
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    augmented_rows = []

    for idx, row in df.iterrows():
        target_name_1 = str(row['Targeted ion/cation']).strip()
        comp_name_1 = str(row['Competing ion/cation']).strip()
        conc_target_1 = row['Single salt concentration (M)']
        conc_comp_1 = row['Total concentration (M)'] - conc_target_1

        rej_1 = row.get(target_name_1, np.nan)
        if pd.notna(rej_1):
            dict_1 = row.to_dict()
            dict_1['Actual_Rejection'] = rej_1
            dict_1['Target_Ion_Name'] = target_name_1
            dict_1['Target_Mw'] = row['Mw (g/mol)']
            dict_1['Target_Ads'] = row['Adsorption energy']
            dict_1['Target_Valence'] = row['Valence']
            dict_1['Target_Hydration'] = row['Hydration Energy -ΔG (kJ/mol)']
            dict_1['Target_D'] = row['Pore_diffusion coefficient D (10-9m2/s)']
            dict_1['Target_r_s'] = row['Stokes radius (nm)']
            dict_1['Target_Conc'] = conc_target_1

            dict_1['Comp_Valence'] = row['Valence.1']
            dict_1['Comp_Hydration'] = row['Hydration Energy -ΔG (kJ/mol).1']
            dict_1['Comp_D'] = row['Pore_diffusion coefficient D (10-9m2/s).1']
            dict_1['Comp_r_s'] = row['Stokes radius (nm).1']
            dict_1['Comp_Conc'] = conc_comp_1
            augmented_rows.append(dict_1)

        rej_2 = row.get(comp_name_1, np.nan)
        if pd.notna(rej_2):
            dict_2 = row.to_dict()
            dict_2['Actual_Rejection'] = rej_2
            dict_2['Target_Ion_Name'] = comp_name_1
            dict_2['Target_Mw'] = row['Mw (g/mol).1']
            dict_2['Target_Ads'] = row['Adsorption energy.1']
            dict_2['Target_Valence'] = row['Valence.1']
            dict_2['Target_Hydration'] = row['Hydration Energy -ΔG (kJ/mol).1']
            dict_2['Target_D'] = row['Pore_diffusion coefficient D (10-9m2/s).1']
            dict_2['Target_r_s'] = row['Stokes radius (nm).1']
            dict_2['Target_Conc'] = conc_comp_1

            dict_2['Comp_Valence'] = row['Valence']
            dict_2['Comp_Hydration'] = row['Hydration Energy -ΔG (kJ/mol)']
            dict_2['Comp_D'] = row['Pore_diffusion coefficient D (10-9m2/s)']
            dict_2['Comp_r_s'] = row['Stokes radius (nm)']
            dict_2['Comp_Conc'] = conc_target_1
            augmented_rows.append(dict_2)

    df_aug = pd.DataFrame(augmented_rows)
    print(f"Data augmentation completed. Total rows: {len(df_aug)}")

    F, R = 96485.332, 8.314
    T = df_aug['Absolute temperature (K)']
    zeta_V = df_aug['zeta potential (mV)'] * 1e-3
    r_p = df_aug['Pore radius (nm)']

    delta_P_raw = df_aug['Hydraulic pressure (bar)'] - df_aug['Overall osmotic pressure (atm)']
    delta_P = np.clip(delta_P_raw, 1e-4, None)
    J_v_raw = df_aug['Water permeability (L/m2*h*bar)'] * delta_P
    J_v_factor = 1e-3 / 3600

    df_aug['Steric Number (Anion)'] = df_aug['Stokes radius (nm).2'] / r_p
    D_anion_safe = np.clip(df_aug['Pore_diffusion coefficient D (10-9m2/s).2'], 1e-9, None)
    df_aug['Pe Number (Anion)'] = (J_v_raw * J_v_factor * r_p) / D_anion_safe
    df_aug['Donnan Number (Anion)'] = (df_aug['Valence.2'] * F * zeta_V) / (R * T)
    df_aug['Hydration Number (Anion)'] = (df_aug['Hydration Energy -ΔG (kJ/mol).2'] * 1000) / (R * T)

    df_aug['Steric Number (Cation)'] = df_aug['Target_r_s'] / r_p
    D_target_safe = np.clip(df_aug['Target_D'], 1e-9, None)
    df_aug['Pe Number (Cation)'] = (J_v_raw * J_v_factor * r_p) / D_target_safe
    df_aug['Donnan Number (Cation)'] = (df_aug['Target_Valence'] * F * zeta_V) / (R * T)
    df_aug['Hydration Number (Cation)'] = (df_aug['Target_Hydration'] * 1000) / (R * T)

    df_aug['Water contact angle (°)'] = df_aug['Water contact angle (°)']
    df_aug['Charge density (C/m2)'] = df_aug['Charge density (C/m2)']
    df_aug['Mw (g/mol)'] = df_aug['Target_Mw']
    df_aug['Adsorption energy'] = df_aug['Target_Ads']

    df_aug['Comp_Ratio_Steric'] = df_aug['Target_r_s'] / df_aug['Comp_r_s']
    df_aug['Comp_Ratio_Valence'] = df_aug['Target_Valence'] / df_aug['Comp_Valence']
    df_aug['Comp_Diff_Hydration'] = df_aug['Target_Hydration'] - df_aug['Comp_Hydration']
    D_comp_safe = np.clip(df_aug['Comp_D'], 1e-9, None)
    df_aug['Comp_Ratio_D'] = np.clip(df_aug['Target_D'] / D_comp_safe, 0.0, 999.0)
    df_aug['Comp_Conc_Fraction'] = df_aug['Target_Conc'] / (df_aug['Target_Conc'] + df_aug['Comp_Conc'])

    final_cols = [
        'Target_Ion_Name', 'Actual_Rejection',
        'Water contact angle (°)', 'Charge density (C/m2)', 'Mw (g/mol)', 'Adsorption energy',
        'Steric Number (Anion)', 'Donnan Number (Anion)', 'Hydration Number (Anion)', 'Pe Number (Anion)',
        'Steric Number (Cation)', 'Donnan Number (Cation)', 'Hydration Number (Cation)', 'Pe Number (Cation)',
        'Comp_Ratio_Steric', 'Comp_Ratio_Valence', 'Comp_Diff_Hydration',
        'Comp_Ratio_D', 'Comp_Conc_Fraction'
    ]

    df_final = df_aug[final_cols].copy()
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_final.to_excel(writer, sheet_name="Sheet10_Mix3", index=False)
    print("Sheet10_Mix3 successfully generated and saved.")


if __name__ == "__main__":
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet3_Mixed"
    
    # Execute sequential data processing steps
    process_mix1(file_path, sheet_name)
    process_mix2(file_path, sheet_name)
    process_mix3(file_path, sheet_name)
    
    print("\nAll processing steps completed successfully. Results appended to the target Excel file.")

import re
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import (
    partial_dependence,
    PartialDependenceDisplay,
)
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")


def clean_filename(name):
    return re.sub(r"[^A-Za-z0-9]", "_", name)


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

    features_list = list(X.columns)

    print("=" * 54)
    print(
        "Training the global LightGBM model "
        "for PDP and ICE analysis"
    )

    param_grid = {
        "n_estimators": [150, 200],
        "max_depth": [3, 4],
        "num_leaves": [5, 10],
        "learning_rate": [0.03, 0.05],
    }

    lgbm = LGBMRegressor(
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    grid = GridSearchCV(
        lgbm,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_

    print("Model training completed")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
        }
    )

    print(
        f"Generating {len(features_list)} "
        "one-dimensional ICE plots"
    )

    for feature in features_list:
        fig, ax = plt.subplots(figsize=(8, 6))

        try:
            pd_results = partial_dependence(
                best_model,
                X,
                [feature],
                kind="both",
                grid_resolution=100,
                percentiles=(0.02, 0.98),
            )
            x_grid = pd_results.get(
                "grid_values",
                pd_results.get("values"),
            )[0]
            pdp_mean = pd_results["average"][0]
            ice_values = pd_results["individual"][0]

            for index in range(ice_values.shape[0]):
                ax.plot(
                    x_grid,
                    ice_values[index, :],
                    color="lightblue",
                    alpha=0.15,
                    linewidth=1,
                )

            ax.plot(
                x_grid,
                pdp_mean,
                color="#105b9e",
                linewidth=3,
                label="Mean Impact",
            )

        except Exception:
            PartialDependenceDisplay.from_estimator(
                best_model,
                X,
                [feature],
                ax=ax,
                line_kw={
                    "color": "#105b9e",
                    "linewidth": 3,
                },
            )

        ax.set_title(
            f"1D PDP and ICE: {feature}",
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel(
            r"Impact on Residual $\Delta R$ (%)",
            fontweight="bold",
        )
        ax.set_xlabel(
            feature,
            fontweight="bold",
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        filename = (
            f"Mixed_PDP_1D_{clean_filename(feature)}.png"
        )
        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Saved one-dimensional plot: {filename}")

    print("Generating three two-dimensional PDP plots")

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
        }
    )

    features_2d_pairs = [
        (features_list[0], features_list[2]),
        (features_list[0], features_list[1]),
        (features_list[1], features_list[2]),
    ]

    for feature_1, feature_2 in features_2d_pairs:
        fig, ax = plt.subplots(figsize=(9, 7))

        pd_results = partial_dependence(
            best_model,
            X,
            [feature_1, feature_2],
            grid_resolution=80,
            percentiles=(0.02, 0.98),
        )
        x_grid = pd_results.get(
            "grid_values",
            pd_results.get("values"),
        )[0]
        y_grid = pd_results.get(
            "grid_values",
            pd_results.get("values"),
        )[1]
        z_values = pd_results["average"][0]

        zz = z_values.T
        xx, yy = np.meshgrid(
            x_grid,
            y_grid,
        )

        filled_contours = ax.contourf(
            xx,
            yy,
            zz,
            levels=8,
            cmap="viridis",
            alpha=0.9,
        )
        ax.contour(
            xx,
            yy,
            zz,
            levels=8,
            colors="black",
            linewidths=0.8,
            alpha=0.8,
        )

        colorbar = fig.colorbar(
            filled_contours,
            ax=ax,
            pad=0.04,
            fraction=0.046,
        )
        colorbar.set_label(
            r"Impact on Residual $\Delta R$ (%)",
            fontweight="bold",
            rotation=270,
            labelpad=25,
        )
        colorbar.ax.tick_params(labelsize=12)

        ax.plot(
            X[feature_1],
            np.full_like(
                X[feature_1],
                y_grid.min(),
            ),
            "|",
            color="k",
            alpha=0.6,
            markersize=12,
            markeredgewidth=1.5,
        )
        ax.plot(
            np.full_like(
                X[feature_2],
                x_grid.min(),
            ),
            X[feature_2],
            "_",
            color="k",
            alpha=0.6,
            markersize=12,
            markeredgewidth=1.5,
        )

        ax.set_xlabel(
            feature_1,
            fontweight="bold",
            labelpad=10,
        )
        ax.set_ylabel(
            feature_2,
            fontweight="bold",
            labelpad=10,
        )
        ax.set_title(
            f"2D PDP: {feature_1}\nvs {feature_2}",
            fontweight="bold",
            pad=15,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        plt.tight_layout()

        safe_feature_1 = clean_filename(feature_1)
        safe_feature_2 = clean_filename(feature_2)
        filename = (
            f"Mixed_PDP_2D_{safe_feature_1}_vs_"
            f"{safe_feature_2}.png"
        )

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Saved two-dimensional plot: {filename}")

    print(
        "Generated three one-dimensional plots "
        "and three two-dimensional plots"
    )


if __name__ == "__main__":
    main()
