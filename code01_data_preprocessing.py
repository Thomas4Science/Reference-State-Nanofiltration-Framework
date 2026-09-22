import pandas as pd
import numpy as np


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
sheet_name = "Sheet2_Single"
df = pd.read_excel(file_path, sheet_name=sheet_name)

print("Data information")
df.info()


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"

    df = pd.read_excel(file_path, sheet_name="Sheet2_Single")

    ion_group_1 = [
        "Targeted ion/cation",
        "Mw (g/mol)",
        "Valence",
        "Ion charge density (C/nm2)",
        "Hydration enthalpy -\u0394H (kJ/mol)",
        "Hydration entropy -\u0394S (J/(K*mol))",
        "Hydration Energy -\u0394G (kJ/mol)",
        "Diffusion coefficient D (10-9m2/s)",
        "Pore_diffusion coefficient D (10-9m2/s)",
        "Ionic radius (nm)",
        "Stokes radius (nm)",
        "Hydrated radius (nm)",
        "Adsorption energy",
    ]

    ion_group_2 = [
        "Competing ion/cation",
        "Mw (g/mol).1",
        "Valence.1",
        "Ion charge density (C/nm).1",
        "Hydration enthalpy -\u0394H (kJ/mol).1",
        "Hydration entropy -\u0394S (J/(K*mol)).1",
        "Hydration Energy -\u0394G (kJ/mol).1",
        "Diffusion coefficient D (10-9m2/s).1",
        "Pore_diffusion coefficient D (10-9m2/s).1",
        "Ionic radius (nm).1",
        "Stokes radius (nm).1",
        "Hydrated radius (nm).1",
        "Adsorption energy.1",
    ]

    standard_ion_cols = [
        "Cation Type",
        "Mw (g/mol)",
        "Valence",
        "Ion charge density (C/nm2)",
        "Hydration enthalpy -\u0394H (kJ/mol)",
        "Hydration entropy -\u0394S (J/(K*mol))",
        "Hydration Energy -\u0394G (kJ/mol)",
        "Diffusion coefficient D (10-9m2/s)",
        "Pore_diffusion coefficient D (10-9m2/s)",
        "Ionic radius (nm)",
        "Stokes radius (nm)",
        "Hydrated radius (nm)",
        "Adsorption energy",
    ]

    common_cols = [
        col
        for col in df.columns
        if col not in ion_group_1 and col not in ion_group_2
    ]

    df_part1 = df[common_cols + ion_group_1].copy()
    df_part2 = df[common_cols + ion_group_2].copy()

    df_part1.rename(
        columns=dict(zip(ion_group_1, standard_ion_cols)),
        inplace=True,
    )
    df_part2.rename(
        columns=dict(zip(ion_group_2, standard_ion_cols)),
        inplace=True,
    )

    df_part1["Ion Source"] = "Targeted"
    df_part2["Ion Source"] = "Competing"

    df_part1.index = df.index * 2
    df_part2.index = df.index * 2 + 1

    df_long = (
        pd.concat([df_part1, df_part2])
        .sort_index()
        .reset_index(drop=True)
    )

    rejection_cols = ["Na+", "Li+", "Ca2+", "Mg2+"]

    def extract_rejection(row):
        cation = str(row["Cation Type"]).strip()
        if cation in rejection_cols:
            return row[cation]
        return np.nan

    df_long["Rejection"] = df_long.apply(extract_rejection, axis=1)
    df_long.drop(
        columns=rejection_cols,
        inplace=True,
        errors="ignore",
    )

    print("Long-format conversion completed")

    with pd.ExcelWriter(
        file_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df_long.to_excel(
            writer,
            sheet_name="Sheet4_deal1",
            index=False,
        )

    print("Processed data saved to Sheet4_deal1")

    df = pd.read_excel(file_path, sheet_name="Sheet4_deal1")

    columns_to_drop = [
        "Number",
        "Membrane",
        "Total concentration (M)",
        "Ionic strength (I, M)",
        "Targeted ion/cation activity coefficient  (\u03b3\u00b1)",
        "Targeted ion/cation activity difference  (a)",
        "Overall ion/cation activity difference  (a)",
        "Specific ion osmotic pressure (atm)",
        "Active area (cm2)",
        "Single salt (1)/Mixed salts (2)",
        "Co-anion/anion",
        "Cation Type",
        "Ion Source",
    ]

    df_cleaned = df.drop(columns=columns_to_drop, errors="ignore")
    missing_rates = df_cleaned.isnull().mean()
    high_missing_cols = missing_rates[missing_rates > 0.4].index.tolist()
    df_cleaned = df_cleaned.drop(columns=high_missing_cols)

    print("Missing-value cleaning report")
    if high_missing_cols:
        print(
            f"Removed {len(high_missing_cols)} features "
            "with more than 40% missing values:"
        )
        for col in high_missing_cols:
            print(
                f"- {col} "
                f"(missing rate: {missing_rates[col] * 100:.2f}%)"
            )
    else:
        print("No features exceeded the 40% missing-value threshold")

    final_missing_rates = df_cleaned.isnull().mean() * 100
    final_missing_counts = df_cleaned.isnull().sum()

    missing_info = (
        pd.DataFrame(
            {
                "Feature": final_missing_rates.index,
                "Missing Count": final_missing_counts.values,
                "Missing Rate (%)": final_missing_rates.values.round(2),
            }
        )
        .sort_values(
            by="Missing Rate (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("Missing values in retained features")
    print(missing_info)

    print("Final data dimensions")
    print(f"Original number of columns: {df.shape[1]}")
    print(f"Retained number of columns: {df_cleaned.shape[1]}")
    print(f"Number of rows: {df_cleaned.shape[0]}")

    with pd.ExcelWriter(
        file_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df_cleaned.to_excel(
            writer,
            sheet_name="Sheet5_deal2",
            index=False,
        )

    print("Cleaned data saved to Sheet5_deal2")

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet5_deal2")
    except Exception as exc:
        print(
            "Failed to read Sheet5_deal2. "
            f"Confirm that the file exists and is closed: {exc}"
        )
        return

    F = 96485.332
    R = 8.314

    print("Calculating dimensionless numbers")

    T = df["Absolute temperature (K)"]
    zeta_V = df["zeta potential (mV)"] * 1e-3
    r_p = df["Pore radius (nm)"]

    delta_P = (
        df["Hydraulic pressure (bar)"]
        - df["Overall osmotic pressure (atm)"]
    )
    J_v_raw = (
        df["Water permeability (L/m2*h*bar)"]
        * delta_P
    )
    J_v_factor = 1e-3 / 3600

    df["Steric Number (Anion)"] = (
        df["Stokes radius (nm).2"] / r_p
    )
    df["Donnan Number (Anion)"] = (
        df["Valence.2"] * F * zeta_V
    ) / (R * T)
    df["Hydration Number (Anion)"] = (
        df["Hydration Energy -\u0394G (kJ/mol).2"] * 1000
    ) / (R * T)
    df["Pe Number (Anion)"] = (
        J_v_raw * J_v_factor * r_p
    ) / df["Pore_diffusion coefficient D (10-9m2/s).2"]

    df["Steric Number (Cation)"] = (
        df["Stokes radius (nm)"] / r_p
    )
    df["Donnan Number (Cation)"] = (
        df["Valence"] * F * zeta_V
    ) / (R * T)
    df["Hydration Number (Cation)"] = (
        df["Hydration Energy -\u0394G (kJ/mol)"] * 1000
    ) / (R * T)
    df["Pe Number (Cation)"] = (
        J_v_raw * J_v_factor * r_p
    ) / df["Pore_diffusion coefficient D (10-9m2/s)"]

    new_cols = [
        "Steric Number (Anion)",
        "Donnan Number (Anion)",
        "Hydration Number (Anion)",
        "Pe Number (Anion)",
        "Steric Number (Cation)",
        "Donnan Number (Cation)",
        "Hydration Number (Cation)",
        "Pe Number (Cation)",
    ]

    manual_drops = [
        "Ionic radius (nm).2",
        "Hydrated radius (nm).2",
        "Ionic radius (nm)",
        "Hydrated radius (nm)",
        "Ion charge density (C/nm).2",
        "Ion charge density (C/nm2)",
        "Hydration enthalpy -\u0394H (kJ/mol).2",
        "Hydration entropy -\u0394S (J/(K*mol)).2",
        "Hydration enthalpy -\u0394H (kJ/mol)",
        "Hydration entropy -\u0394S (J/(K*mol))",
        "Diffusion coefficient D (10-9m2/s).2",
        "Diffusion coefficient D (10-9m2/s)",
        "Single salt concentration (M)",
    ]

    calc_vars_drops = [
        "Pore radius (nm)",
        "zeta potential (mV)",
        "Absolute temperature (K)",
        "Water permeability (L/m2*h*bar)",
        "Hydraulic pressure (bar)",
        "Overall osmotic pressure (atm)",
        "Valence.2",
        "Hydration Energy -\u0394G (kJ/mol).2",
        "Pore_diffusion coefficient D (10-9m2/s).2",
        "Stokes radius (nm).2",
        "Valence",
        "Hydration Energy -\u0394G (kJ/mol)",
        "Pore_diffusion coefficient D (10-9m2/s)",
        "Stokes radius (nm)",
    ]

    cols_to_drop = list(set(manual_drops + calc_vars_drops))
    cols_to_drop_actual = [
        col for col in cols_to_drop if col in df.columns
    ]

    df_calculated = df.drop(columns=cols_to_drop_actual)

    print("Removed variables")
    for col in cols_to_drop_actual:
        if col in manual_drops:
            print(f"- Manually removed: {col}")
        else:
            print(f"- Removed after calculation: {col}")

    if "Rejection" in df_calculated.columns:
        all_cols = list(df_calculated.columns)
        for col in new_cols:
            all_cols.remove(col)

        rejection_index = all_cols.index("Rejection")
        final_col_order = (
            all_cols[:rejection_index]
            + new_cols
            + all_cols[rejection_index:]
        )
        df_calculated = df_calculated[final_col_order]
    else:
        print(
            "Rejection column not found; "
            "new features remain at the end"
        )

    print("Dimensionless-number calculation completed")
    if "Rejection" in df_calculated.columns:
        print(df_calculated[new_cols + ["Rejection"]].head())
    else:
        print(df_calculated[new_cols].head())

    with pd.ExcelWriter(
        file_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df_calculated.to_excel(
            writer,
            sheet_name="Sheet6_deal3",
            index=False,
        )

    print("Final data saved to Sheet6_deal3")


if __name__ == "__main__":
    main()
