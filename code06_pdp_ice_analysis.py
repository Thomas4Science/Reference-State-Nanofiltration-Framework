import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clean_filename(name):
    return re.sub(r"[^A-Za-z0-9]", "_", name)


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet9_deal5"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:
        print(f"Failed to read the data: {exc}")
        return

    features_12 = [
        "Steric Number (Anion)",
        "Pe Number (Anion)",
        "Hydration Number (Cation)",
        "Steric Number (Cation)",
        "Water contact angle (\u00b0)",
        "Donnan Number (Cation)",
        "Charge density (C/m2)",
        "Pe Number (Cation)",
        "Hydration Number (Anion)",
        "Mw (g/mol)",
        "Donnan Number (Anion)",
        "Adsorption energy",
    ]

    df_clean = (
        df.dropna(subset=features_12 + ["Rejection"])
        .reset_index(drop=True)
    )
    X = df_clean[features_12]
    y = df_clean["Rejection"]

    X.columns = [
        re.sub(r"[\[\]<>]", "", column)
        for column in X.columns
    ]
    features_1d = list(X.columns)

    print("Training the Random Forest model for PDP and ICE analysis")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    print("Model training completed")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
        }
    )

    print(
        f"Generating {len(features_1d)} "
        "one-dimensional ICE plots"
    )

    for feature in features_1d:
        fig, ax = plt.subplots(figsize=(8, 6))

        try:
            pd_results = partial_dependence(
                model,
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
                    alpha=0.08,
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
                model,
                X,
                [feature],
                ax=ax,
                line_kw={
                    "color": "#105b9e",
                    "linewidth": 3,
                },
            )

        ax.set_title(
            f"ICE curve: {feature}",
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel(
            "Relative impact on Rejection",
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
            f"ICE_1D_{clean_filename(feature)}.png"
        )
        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Saved one-dimensional plot: {filename}")

    print("Generating two-dimensional PDP plots")

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
        }
    )

    feature_names = X.columns.tolist()

    steric_anion = [
        feature
        for feature in feature_names
        if "Steric" in feature and "Anion" in feature
    ][0]
    pe_anion = [
        feature
        for feature in feature_names
        if "Pe Number" in feature and "Anion" in feature
    ][0]
    steric_cation = [
        feature
        for feature in feature_names
        if "Steric" in feature and "Cation" in feature
    ][0]
    hydration_cation = [
        feature
        for feature in feature_names
        if "Hydration" in feature and "Cation" in feature
    ][0]
    contact_angle = [
        feature
        for feature in feature_names
        if "contact angle" in feature
    ][0]

    features_2d = [
        (steric_anion, pe_anion),
        (steric_anion, steric_cation),
        (steric_cation, hydration_cation),
        (contact_angle, pe_anion),
    ]

    for feature_1, feature_2 in features_2d:
        fig, ax = plt.subplots(figsize=(9, 7))

        pd_results = partial_dependence(
            model,
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
            "Relative impact on Rejection",
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
            f"2D PDP: {feature_1} vs {feature_2}",
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
            f"PDP_2D_{safe_feature_1}_vs_"
            f"{safe_feature_2}_Premium.png"
        )

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Saved two-dimensional plot: {filename}")

    print(
        "Generated 12 one-dimensional ICE plots "
        "and 4 two-dimensional PDP plots"
    )


if __name__ == "__main__":
    main()
