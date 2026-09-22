# Mixed salt residual symbolic regression (Full power + Division safety lock version)
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


def select_balanced_equation(model, X, y, gain_threshold=0.015):
    """
    Select the best balance point between R2 and complexity.
    Strategy: Find the last equation where 'gain per unit complexity > gain_threshold',
    which is the inflection point before adding more complexity yields little R2 gain.
    """
    eqs = model.equations_.sort_values(by='complexity').reset_index()

    comp, r2s, idxs = [], [], []
    for _, row in eqs.iterrows():
        idx = row['index']
        try:
            y_pred = model.predict(X, index=idx)
            comp.append(row['complexity'])
            r2s.append(r2_score(y, y_pred))
            idxs.append(idx)
        except Exception:
            pass

    comp = np.array(comp, dtype=float)
    r2s = np.array(r2s, dtype=float)

    # R2 gain per unit complexity
    marginal = np.zeros_like(comp)
    marginal[1:] = (r2s[1:] - r2s[:-1]) / (comp[1:] - comp[:-1])

    # Find the last position where the gain still exceeds the threshold (inflection point)
    good = np.where(marginal[1:] > gain_threshold)[0]
    best_pos = int(good[-1] + 1) if len(good) > 0 else int(np.argmax(r2s))
    best_idx = idxs[best_pos]

    print("\n================== R2 and Complexity Balance Analysis ==================")
    print(f"{'Comp':<8}{'R2':<12}{'Marginal Gain':<14}")
    print("-" * 50)
    for c, r, m in zip(comp, r2s, marginal):
        mark = "  <-- Best Balance Point" if c == comp[best_pos] else ""
        print(f"{c:<8.0f}{r:<12.4f}{m:<14.4f}{mark}")
    print("-" * 50)
    print(f"(Threshold: marginal gain > {gain_threshold}, below this is considered not worth adding complexity)")

    return best_idx, comp, r2s, best_pos


def plot_pareto_front(comp, r2s, best_pos, save_path="PySR_Pareto_Front.png"):
    """Plot the Pareto front of complexity vs R2, and mark the best balance point."""
    plt.figure(figsize=(8, 6))
    plt.plot(comp, r2s, '-o', color='#1f77b4', markersize=8, linewidth=2,
             label='Pareto Front')

    # Highlight the best balance point
    plt.scatter(comp[best_pos], r2s[best_pos], s=320, marker='*',
                color='#ff7f00', edgecolor='k', zorder=5,
                label=f'Best Balance (C={comp[best_pos]:.0f}, '
                      f'$R^2$={r2s[best_pos]:.3f})')

    for c, r in zip(comp, r2s):
        plt.annotate(f'{r:.3f}', (c, r), textcoords="offset points",
                     xytext=(0, 9), ha='center', fontsize=9)

    plt.title('Pareto Front: Complexity vs $R^2$', fontsize=15,
              fontweight='bold', pad=15)
    plt.xlabel('Equation Complexity', fontsize=14, fontweight='bold')
    plt.ylabel('Residual $R^2$', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Pareto front plot saved: {save_path}")


def main():
    # ================= 1. Load Data =================
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet14_Mix_Resi_ML"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    y_residual = df['Residual_Rejection']

    # Contains all 3 golden competitive features
    comp_features = [
        'Comp_Ratio_Valence',
        'Comp_Ratio_D',
        'Comp_Conc_Fraction'
    ]
    X = df[comp_features]

    feature_mapping = {
        'Comp_Ratio_Valence': 'Z_ratio',
        'Comp_Ratio_D': 'D_ratio',
        'Comp_Conc_Fraction': 'C_frac'
    }
    X = X.rename(columns=feature_mapping)

    print("======================================================")
    print("Explore the limits: Launch full power PySR (including division operator)")
    print("======================================================")

    # ================= 2. Launch Full Power PySR =================
    model = PySRRegressor(
        niterations=300,  # Increase iterations to 300 for thorough search
        populations=40,
        binary_operators=["+", "-", "*", "/"],  # Reintroduce division
        unary_operators=["abs", "square", "exp"],
        # Strict engineering safety locks to prevent denominator explosion and exponential overflow
        nested_constraints={
            "exp": {"exp": 0},
            "/": {"/": 0}
        },
        model_selection="best",
        maxsize=25,  # Allow longer equations to accommodate fractions
        parsimony=0.005,  # Slightly lower penalty to encourage physical meaning
        random_state=42,
        deterministic=False,  # Enable random perturbation
        parallelism='serial',
        verbosity=0
    )

    print("Searching for minimalist algebraic formulas, estimated time 2-5 minutes...")
    model.fit(X, y_residual)

    # ================= 3. Output Pareto Menu =================
    print("\n================== Delta R Analytical Equation Menu (with Division) ==================")
    equations_df = model.equations_
    sorted_eqs = equations_df.sort_values(by='complexity')

    print(f"{'Comp':<5} | {'Residual R2':<12} | {'Delta R Equation'}")
    print("-" * 120)

    for idx, row in sorted_eqs.iterrows():
        comp = row['complexity']
        eq_str = row['equation']
        try:
            y_pred_resi = model.predict(X, index=idx)
            r2 = r2_score(y_residual, y_pred_resi)
            if r2 >= 0.30:
                print(f"[{comp:2d}]   | * {r2:.4f}   | Delta R = {eq_str}")
            else:
                print(f"[{comp:2d}]   |   {r2:.4f}   | Delta R = {eq_str}")
        except Exception:
            pass
    print("-" * 120)

    # ================= 3.5 Balance R2 and Complexity =================
    # Use 0.015 as marginal gain threshold to find the inflection point
    best_idx, comp_arr, r2_arr, best_pos = select_balanced_equation(
        model, X, y_residual, gain_threshold=0.015)
    plot_pareto_front(comp_arr, r2_arr, best_pos)

    # ================= 4. Output and Plot (with the balanced formula) =================
    best_eq_sympy = model.sympy(index=best_idx)
    best_complexity = equations_df.loc[best_idx, 'complexity']
    best_r2 = r2_arr[best_pos]

    print(f"\nRecommended residual formula after balancing (Comp: {best_complexity}, R2: {best_r2:.4f}):")
    print(f"Delta R = {best_eq_sympy}")

    # Predict and plot parity plot
    y_pred_best = model.predict(X, index=best_idx)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_residual, y_pred_best, alpha=0.8, color='#ff7f00', edgecolor='k', s=70,
                label=f'Optimal Formula\n($R^2$={best_r2:.4f})')

    min_val = min(y_residual.min(), y_pred_best.min()) - 5
    max_val = max(y_residual.max(), y_pred_best.max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Parity')

    plt.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)

    plt.title(r'Symbolic Prediction of Residual ($\Delta R$)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(r'Actual ML Residual ($\Delta R$) (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Formula Predicted Residual (%)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("PySR_Balanced_Residual_Parity_Plot.png", dpi=300)
    plt.close()

    print("\nDone! Pareto front plot and balanced formula fitting scatter plot have been output.")


if __name__ == "__main__":
    main()


# ==============================================================================
# Script 2: Directly lock and verify the optimal formula
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main_verify():
    # ================= 1. Load Data =================
    file_path = "0607-Paper Data.xlsx"
    sheet_name = "Sheet14_Mix_Resi_ML"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    y_actual = df['Residual_Rejection']

    # Extract three main features
    Z_ratio = df['Comp_Ratio_Valence']
    D_ratio = df['Comp_Ratio_D']
    C_frac = df['Comp_Conc_Fraction']

    print("======================================================")
    print("Directly lock and verify the optimal formula (Complexity 18, R2 0.7407)")
    print("======================================================")

    # ================= 2. Deploy the optimal formula for the paper =================
    # Delta R = -0.01025*DR - 0.7042/|CF + VR - 0.6892| - 0.2501/(DR - 0.0933)
    y_pred = (D_ratio * -0.0102546625) - \
             (0.7042217 / np.abs(C_frac + Z_ratio - 0.6892325)) - \
             (0.25014773 / (D_ratio - 0.093290076))

    # ================= 3. Verify Score =================
    final_r2 = r2_score(y_actual, y_pred)
    print(f"Verification successful! True R2 on full data = {final_r2:.4f}")

    # ================= 4. Export CSV for Origin Plotting =================
    parity_df = pd.DataFrame({
        'Actual_Residual_R': np.asarray(y_actual),
        'Predicted_Residual_R': np.asarray(y_pred)
    })
    csv_filename = "MixedSalt_Parity_Plot_Data_Final.csv"
    parity_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"[Important Output] 1:1 scatter plot data exported to: {csv_filename}")
    print("Please drag this file directly into Origin to plot the inset of Figure 4b!")

    # ================= 5. Generate Python Preview Plot =================
    plt.figure(figsize=(7, 7))
    plt.scatter(y_actual, y_pred, alpha=0.8, color='#ff7f00', edgecolor='k', s=70,
                label=f'Equation 12\n($R^2$={final_r2:.4f})')

    min_val = min(y_actual.min(), y_pred.min()) - 5
    max_val = max(y_actual.max(), y_pred.max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Parity')

    plt.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)

    plt.title(r'Symbolic Prediction of Residual ($\Delta R$)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(r'Actual ML Residual ($\Delta R$) (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Formula Predicted Residual (%)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("PySR_Final_Residual_Parity_Plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main_verify()
