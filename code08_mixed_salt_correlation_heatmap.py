import pandas as pd
import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet14_Mix_Resi_ML"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:
        print(f"Failed to read the data: {exc}")
        return

    nice_names = {
        "Comp_Ratio_Steric": "Steric Ratio (Target/Comp)",
        "Comp_Ratio_Valence": "Valence Ratio (Target/Comp)",
        "Comp_Diff_Hydration": (
            "Hydration Diff (Target-Comp)"
        ),
        "Comp_Ratio_D": "Diffusivity Ratio (Target/Comp)",
        "Comp_Conc_Fraction": "Concentration Fraction",
        "Residual_Rejection": (
            "Rejection Residual (\u0394R)"
        ),
    }
    df = df.rename(columns=nice_names)

    corr = df.corr(method="pearson")

    mask_full = np.triu(
        np.ones_like(corr, dtype=bool),
        k=0,
    )

    corr_sliced = corr.iloc[1:, :-1]
    mask_sliced = mask_full[1:, :-1]

    print("Generating the lower-triangle Pearson heatmap")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_sliced,
        mask=mask_sliced,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        vmin=-1.0,
        vmax=1.0,
        center=0,
        square=True,
        linewidths=2,
        linecolor="white",
        cbar_kws={
            "shrink": 0.8,
            "ticks": [
                -1.0,
                -0.75,
                -0.5,
                -0.25,
                0,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
        },
        annot_kws={
            "size": 13,
            "weight": "bold",
        },
        ax=ax,
    )

    ax.set_title(
        "Pearson Correlation Matrix",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=90, fontsize=13)
    plt.yticks(rotation=0, fontsize=13)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "Pearson_Correlation_Triangle.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close()


    print(
        "Processing completed. "
        "Check Pearson_Correlation_Triangle.png"
    )


if __name__ == "__main__":
    main()
