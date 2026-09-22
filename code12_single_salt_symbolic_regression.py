import pandas as pd
import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings


warnings.filterwarnings("ignore")


def select_balanced_equation(
    model,
    X,
    y_true,
    gain_threshold=0.015,
    csv_path="SingleSalt_Pareto_Complexity_R2.csv",
):
    eqs = (
        model.equations_
        .sort_values(by="complexity")
        .reset_index()
    )

    complexities = []
    r2_values = []
    equation_indices = []
    equation_strings = []

    for _, row in eqs.iterrows():
        equation_index = row["index"]

        try:
            z_prediction = model.predict(
                X,
                index=equation_index,
            )
            y_prediction = 100.0 / (
                1.0 + np.exp(-z_prediction)
            )

            complexities.append(row["complexity"])
            r2_values.append(
                r2_score(
                    y_true,
                    y_prediction,
                )
            )
            equation_indices.append(equation_index)
            equation_strings.append(row["equation"])

        except Exception:
            pass

    complexities = np.array(
        complexities,
        dtype=float,
    )
    r2_values = np.array(
        r2_values,
        dtype=float,
    )

    marginal_gain = np.zeros_like(complexities)
    marginal_gain[1:] = (
        r2_values[1:] - r2_values[:-1]
    ) / (
        complexities[1:] - complexities[:-1]
    )

    acceptable_positions = np.where(
        marginal_gain[1:] > gain_threshold
    )[0]

    if len(acceptable_positions) > 0:
        best_position = int(
            acceptable_positions[-1] + 1
        )
    else:
        best_position = int(
            np.argmax(r2_values)
        )

    best_index = equation_indices[best_position]

    output_df = pd.DataFrame(
        {
            "complexity": complexities.astype(int),
            "R2": r2_values,
            "marginal_gain": marginal_gain,
            "is_best_balance": [
                complexity == complexities[best_position]
                for complexity in complexities
            ],
            "equation": equation_strings,
        }
    )
    output_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Complexity and R2 data exported to "
        f"{csv_path}"
    )

    print("R2 and complexity balance analysis")
    print(
        f"{'Complexity':<12}"
        f"{'True R2':<12}"
        f"{'Gain per Unit':<16}"
    )
    print("-" * 50)

    for complexity, r2_value, gain in zip(
        complexities,
        r2_values,
        marginal_gain,
    ):
        marker = ""
        if complexity == complexities[best_position]:
            marker = "  <-- Best balance"

        print(
            f"{complexity:<12.0f}"
            f"{r2_value:<12.4f}"
            f"{gain:<16.4f}"
            f"{marker}"
        )

    print("-" * 50)
    print(
        "Threshold: gain per unit of complexity > "
        f"{gain_threshold}"
    )

    return (
        best_index,
        complexities,
        r2_values,
        best_position,
    )


def plot_pareto_front(
    complexities,
    r2_values,
    best_position,
    save_path="SingleSalt_PySR_Pareto_Front.png",
):
    plt.figure(figsize=(8, 6))

    plt.plot(
        complexities,
        r2_values,
        "-o",
        color="#1E88E5",
        markersize=8,
        linewidth=2,
        label="Pareto Front",
    )

    plt.scatter(
        complexities[best_position],
        r2_values[best_position],
        s=320,
        marker="*",
        color="#ff7f00",
        edgecolor="k",
        zorder=5,
        label=(
            "Best Balance "
            f"(C={complexities[best_position]:.0f}, "
            f"$R^2$={r2_values[best_position]:.3f})"
        ),
    )

    for complexity, r2_value in zip(
        complexities,
        r2_values,
    ):
        plt.annotate(
            f"{r2_value:.3f}",
            (complexity, r2_value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
        )

    plt.title(
        "Pareto Front: Complexity vs $R^2$",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel(
        "Equation Complexity",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel(
        "Rejection $R^2$",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )
    plt.legend(
        loc="lower right",
        fontsize=11,
        framealpha=0.9,
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
    )
    plt.close()

    print(
        f"Pareto-front plot saved as {save_path}"
    )


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet9_deal5"

    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
        )
    except Exception as exc:
        print(f"Failed to read the data: {exc}")
        return

    features = [
        "Steric Number (Anion)",
        "Pe Number (Anion)",
        "Hydration Number (Cation)",
        "Steric Number (Cation)",
        "Water contact angle (\u00b0)",
        "Donnan Number (Anion)",
        "Donnan Number (Cation)",
        "Hydration Number (Anion)",
    ]

    df = (
        df.dropna(
            subset=features + ["Rejection"]
        )
        .reset_index(drop=True)
    )

    feature_mapping = {
        "Steric Number (Anion)": "St_A",
        "Pe Number (Anion)": "Pe_A",
        "Hydration Number (Cation)": "Hy_C",
        "Steric Number (Cation)": "St_C",
        "Water contact angle (\u00b0)": "CA",
        "Donnan Number (Anion)": "Do_A",
        "Donnan Number (Cation)": "Do_C",
        "Hydration Number (Anion)": "Hy_A",
    }

    X = df[features].rename(
        columns=feature_mapping
    )
    y = df["Rejection"]

    print(
        f"Data loaded successfully. "
        f"Number of samples: {len(X)}"
    )

    y_scaled = np.clip(
        y / 100.0,
        0.001,
        0.999,
    )
    z_target = np.log(
        y_scaled / (1 - y_scaled)
    )

    print(
        "Starting PySR with nested constraints"
    )

    model = PySRRegressor(
        niterations=300,
        populations=40,
        binary_operators=[
            "+",
            "-",
            "*",
            "/",
        ],
        unary_operators=[
            "exp",
            "square",
            "abs",
        ],
        nested_constraints={
            "exp": {
                "exp": 0,
                "/": 0,
            },
            "square": {
                "square": 0,
                "exp": 0,
            },
        },
        model_selection="best",
        elementwise_loss=(
            "loss(prediction, target) = "
            "(prediction - target)^2"
        ),
        maxsize=30,
        parsimony=0.001,
        random_state=42,
        deterministic=False,
    )

    model.fit(
        X,
        z_target,
    )

    print("Pareto equation candidates")

    equations_df = model.equations_
    sorted_equations = equations_df.sort_values(
        by="complexity"
    )

    print(
        f"{'Complexity':<12} | "
        f"{'True Rejection R2':<18} | "
        f"{'Z-potential Equation'}"
    )
    print("-" * 110)

    for equation_index, row in sorted_equations.iterrows():
        complexity = row["complexity"]
        equation_string = row["equation"]

        try:
            z_prediction = model.predict(
                X,
                index=equation_index,
            )
            y_prediction = 100.0 / (
                1.0 + np.exp(-z_prediction)
            )
            true_r2 = r2_score(
                y,
                y_prediction,
            )

            if true_r2 >= 0.6:
                print(
                    f"[{complexity:2d}] | "
                    f"R2: {true_r2:.4f} | "
                    f"Z = {equation_string}"
                )
            else:
                print(
                    f"[{complexity:2d}] | "
                    f"R2: {true_r2:.4f} | "
                    f"Z = {equation_string}"
                )

        except Exception:
            pass

    print("-" * 110)

    (
        best_index,
        complexity_array,
        r2_array,
        best_position,
    ) = select_balanced_equation(
        model,
        X,
        y,
        gain_threshold=0.015,
    )

    plot_pareto_front(
        complexity_array,
        r2_array,
        best_position,
    )

    best_equation_sympy = model.sympy(
        index=best_index
    )
    best_complexity = equations_df.loc[
        best_index,
        "complexity",
    ]
    best_true_r2 = r2_array[best_position]

    print(
        "Recommended equation after balancing "
        "R2 and complexity"
    )
    print(
        f"Complexity: {best_complexity}, "
        f"True R2: {best_true_r2:.4f}"
    )
    print(f"Z = {best_equation_sympy}")

    z_prediction_best = model.predict(
        X,
        index=best_index,
    )
    y_prediction_best = 100.0 / (
        1.0 + np.exp(-z_prediction_best)
    )

    parity_df = pd.DataFrame(
        {
            "Measured_Rejection": np.asarray(y),
            "Predicted_Rejection": np.asarray(
                y_prediction_best
            ),
        }
    )
    parity_df.to_csv(
        "SingleSalt_Parity_Plot_Data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Parity-plot data exported to "
        "SingleSalt_Parity_Plot_Data.csv"
    )

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y,
        y_prediction_best,
        alpha=0.7,
        color="#1E88E5",
        edgecolor="k",
        s=60,
        label=(
            "Balanced Logit-PySR\n"
            f"($R^2$={best_true_r2:.4f})"
        ),
    )

    minimum_value = min(
        y.min(),
        y_prediction_best.min(),
    )
    maximum_value = max(
        y.max(),
        y_prediction_best.max(),
    )

    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        "r--",
        linewidth=2,
        label="1:1 Line",
    )

    plt.title(
        "Prediction with Balanced PySR Model",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel(
        "Measured Rejection (%)",
        fontsize=14,
    )
    plt.ylabel(
        "Predicted Rejection (%)",
        fontsize=14,
    )
    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )
    plt.legend(
        loc="upper left",
        fontsize=12,
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "Logit_PySR_Parity_Plot_Balanced.png",
        dpi=300,
    )
    plt.close()

    print(
        "Generated the Pareto-front plot, parity plot, "
        "and two CSV data files"
    )


if __name__ == "__main__":
    main()
