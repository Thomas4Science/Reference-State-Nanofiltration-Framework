import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
from sklearn.ensemble import RandomForestRegressor
import shap
import warnings


warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Arial Unicode MS",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


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

    rename_mapping = {
        "Mw (g/mol)": "Cation_Mw (g/mol)",
        "Adsorption energy": "Cation_Adsorption_Energy",
    }
    df = df.rename(columns=rename_mapping)

    X = df.drop(columns=["Rejection"])
    y = df["Rejection"]

    original_feature_names = X.columns.tolist()

    X.columns = [
        re.sub(r"[\[\]<>]", "", column)
        for column in X.columns
    ]

    print("Training the Random Forest explanation model")

    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X, y)

    print(
        "Model training completed. "
        f"Full-data R2 score: {final_model.score(X, y):.4f}"
    )

    print("Calculating SHAP values using TreeExplainer")

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    print("Calculating and exporting mean absolute SHAP values")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    mas_df = pd.DataFrame(
        {
            "Feature": original_feature_names,
            "Mean_Absolute_SHAP": mean_abs_shap,
        }
    )

    mas_df = mas_df.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False,
    )

    mas_csv_file = "SHAP_MAS_Values_For_Origin.csv"
    mas_df.to_csv(
        mas_csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Mean absolute SHAP data saved as "
        f"{mas_csv_file}"
    )

    colors = [
        "#99CCFF",
        "#FFFFFF",
        "#FF9999",
    ]
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_blue_red",
        colors,
    )

    print("Generating the high-resolution SHAP beeswarm plot")

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=original_feature_names,
        show=False,
        max_display=12,
        cmap=custom_cmap,
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(
        "SHAP value (Impact on model output Rejection)",
        fontsize=14,
        fontweight="bold",
    )
    plt.title(
        "SHAP Beeswarm Plot",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    beeswarm_file = "SHAP_Beeswarm_RF.tif"

    plt.savefig(
        beeswarm_file,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.savefig(
        "SHAP_Beeswarm_RF.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()

    print("Generating the high-resolution SHAP bar plot")

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=original_feature_names,
        plot_type="bar",
        show=False,
        max_display=12,
        color="#FF9999",
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(
        "Mean |SHAP value| "
        "(Average impact on model output magnitude)",
        fontsize=14,
        fontweight="bold",
    )
    plt.title(
        "SHAP Mean Absolute (MAS) Plot",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    mas_file = "SHAP_MAS_Bar_RF.tif"

    plt.savefig(
        mas_file,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.savefig(
        "SHAP_MAS_Bar_RF.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()

    print("SHAP analysis completed")
    print(f"Beeswarm TIFF: {beeswarm_file} at 600 DPI")
    print(f"Bar plot TIFF: {mas_file} at 600 DPI")
    print("SVG files and the CSV file were also generated")


if __name__ == "__main__":
    main()
