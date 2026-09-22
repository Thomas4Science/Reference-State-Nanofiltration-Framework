import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


FILE_PATH = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
SOURCE_SHEET = "Sheet6_deal3"


def analyze_pearson_correlations():
    try:
        df = pd.read_excel(FILE_PATH, sheet_name=SOURCE_SHEET)
    except Exception as exc:
        print(f"Failed to read the Excel file: {exc}")
        return

    rename_dict = {
        "Mw (g/mol).2": "Anion_Mw (g/mol)",
        "Adsorption energy.2": "Anion_Adsorption_Energy",
        "Mw (g/mol)": "Cation_Mw (g/mol)",
        "Adsorption energy": "Cation_Adsorption_Energy",
        "Volumetric charge density (mol/m3)": (
            "Volumetric charge density"
        ),
    }
    df = df.rename(columns=rename_dict)

    corr_matrix = df.corr(method="pearson")

    print("Highly correlated feature pairs (|r| > 0.8)")
    high_corr_pairs = []
    columns = corr_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr_matrix.iloc[i, j]
            if abs(value) > 0.8:
                high_corr_pairs.append(
                    (columns[i], columns[j], value)
                )

    if not high_corr_pairs:
        print(
            "No feature pairs with an absolute "
            "correlation above 0.8 were found"
        )
    else:
        high_corr_pairs.sort(
            key=lambda item: abs(item[2]),
            reverse=True,
        )
        for feature_1, feature_2, value in high_corr_pairs:
            print(
                f"- [{feature_1}] <-> [{feature_2}] "
                f"(r = {value:.4f})"
            )

    correlation_sheet = "Sheet7_CorrMatrix"
    with pd.ExcelWriter(
        FILE_PATH,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        corr_matrix.to_excel(
            writer,
            sheet_name=correlation_sheet,
            index=True,
        )

    print(
        f"Correlation matrix saved to "
        f"{FILE_PATH}, sheet {correlation_sheet}"
    )

    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool)
    )

    plt.figure(figsize=(14, 12))
    sns.set_theme(style="white")

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=1.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(
        "Pearson Correlation Matrix",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(rotation=90, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    output_image = "Pearson_Matrix.png"
    plt.savefig(
        output_image,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Pearson correlation heatmap saved as {output_image}")

    plt.show()


def remove_collinear_features():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    try:
        df = pd.read_excel(FILE_PATH, sheet_name=SOURCE_SHEET)
    except Exception as exc:
        print(f"Failed to read the Excel file: {exc}")
        return

    columns_to_drop = [
        "Mw (g/mol).2",
        "Adsorption energy.2",
        "Volumetric charge density (mol/m3)",
    ]

    existing_columns_to_drop = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    df_reduced = df.drop(columns=existing_columns_to_drop)

    print("Collinearity removal report")
    print(
        "The following original features were removed "
        "while retaining the calculated dimensionless numbers:"
    )
    for column in existing_columns_to_drop:
        print(f"- Removed: {column}")

    print("Data dimensions")
    print(
        "Number of columns before removal, "
        f"including Rejection: {df.shape[1]}"
    )
    print(
        "Number of columns after removal, "
        f"including Rejection: {df_reduced.shape[1]}"
    )

    print("Retained input features")
    remaining_features = list(df_reduced.columns)
    remaining_features.remove("Rejection")

    for index, feature in enumerate(
        remaining_features,
        start=1,
    ):
        print(f"{index}. {feature}")

    output_sheet = "Sheet8_deal4"
    with pd.ExcelWriter(
        FILE_PATH,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df_reduced.to_excel(
            writer,
            sheet_name=output_sheet,
            index=False,
        )

    print(
        f"Reduced data saved to "
        f"{FILE_PATH}, sheet {output_sheet}"
    )


if __name__ == "__main__":
    analyze_pearson_correlations()
    remove_collinear_features()
