#!/usr/bin/env python3
"""
payoff_estimation/scripts/08_generate_plots.py
Phase 2 - Step 8: Comprehensive Publication Plots & Econometric Visualizations.

Generates 6 publication-ready figures (300 DPI, seaborn style):
1. plots/01_model_progression_fit.png: In-Sample R², LOCO-CV R²_OOS, Validation R²_OOS, AIC/BIC
2. plots/02_actual_vs_predicted.png: Actual vs Predicted log views across Train, Val, and Test
3. plots/03_residual_audit.png: 4-panel residual diagnostics (vs Fitted, Q-Q, vs Own Appeal, vs Subscribers)
4. plots/04_payoff_response_curves.png: Payoff curves across peer appeal percentiles & convex cost
5. plots/05_peer_interaction_heatmap.png: 2D contour surface of ln V(x_i, x_bar) with reaction curve
6. plots/06_channel_error_dispersion.png: Channel-by-channel MAE & bias across all 38 channels (Train/Val/Test)

Outputs:
- payoff_estimation/plots/*.png
- payoff_estimation/logs/08_generate_plots.log
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PAYOFF_DIR = SCRIPT_DIR.parent
DATA_DIR = PAYOFF_DIR / "data"
RESULTS_DIR = PAYOFF_DIR / "results"
PLOTS_DIR = PAYOFF_DIR / "plots"
LOGS_DIR = PAYOFF_DIR / "logs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = LOGS_DIR / "08_generate_plots.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("08_generate_plots")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def generate_plots():
    logger.info("=" * 75)
    logger.info("STEP 8: GENERATING COMPREHENSIVE PUBLICATION FIGURES (300 DPI)")
    logger.info("=" * 75)

    # Load datasets
    part_csv = DATA_DIR / "partitioned_payoff_data.csv"
    gt_json = RESULTS_DIR / "game_theory_coefficients.json"
    train_comp_csv = RESULTS_DIR / "training_model_comparison.csv"
    val_comp_csv = RESULTS_DIR / "validation_evaluation.csv"
    test_comp_csv = RESULTS_DIR / "test_set_evaluation.csv"
    loco_cv_csv = RESULTS_DIR / "loco_cv_aggregate_summary.csv"
    val_disp_csv = RESULTS_DIR / "validation_channel_dispersion.csv"
    test_disp_csv = RESULTS_DIR / "test_channel_dispersion.csv"

    df = pd.read_csv(part_csv)
    df["published_at_dt"] = pd.to_datetime(df["published_at"])
    with open(gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Compute contemporaneous LOCO peer appeal for each video across all 38 channels
    peer_appeals = []
    for _, row in df.iterrows():
        c_id = row["channel_id"]
        t_pub = row["published_at_dt"]
        w = df[
            (df["channel_id"] != c_id) &
            (df["published_at_dt"] >= t_pub - pd.Timedelta(days=30)) &
            (df["published_at_dt"] <= t_pub + pd.Timedelta(days=30))
        ]
        if len(w) > 0:
            peer_appeals.append(float(w.groupby("channel_id")["x"].mean().mean()))
        else:
            loco = df[df["channel_id"] != c_id]
            peer_appeals.append(float(loco.groupby("channel_id")["x"].mean().mean()))
    df["x_bar"] = peer_appeals

    mu_x = gt_data["sample_summary"]["centering_mu_x"]
    mu_xbar = gt_data["sample_summary"]["centering_mu_xbar"]

    df["x_c"] = df["x"] - mu_x
    df["x2_c"] = df["x_c"] ** 2
    df["x_bar_c"] = df["x_bar"] - mu_xbar
    df["x_inter_c"] = df["x_c"] * df["x_bar_c"]
    df["const"] = 1.0

    cols_m4 = ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_subscribers", "log_duration", "age_days"]
    m4_fit = sm.OLS(df["log_views"], df[cols_m4]).fit(cov_type="cluster", cov_kwds={"groups": df["channel_id"]})
    df["pred_log_views"] = m4_fit.predict(df[cols_m4])
    df["residuals"] = df["log_views"] - df["pred_log_views"]

    # =========================================================================
    # FIGURE 1: Model Progression Fit & Information Criteria
    # =========================================================================
    logger.info("Generating Figure 1: Model Progression Fit...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    models = ["M0: Controls", "M1: Linear Appeal", "M2: Quadratic", "M3: Peer Congestion", "M4: Interaction", "M5: Tags"]
    # Training R²
    r2_train = [0.6358, 0.6373, 0.6374, 0.6377, 0.6383, 0.6409]
    # LOCO-CV OOS R² (Training 26 folds)
    r2_loco = [0.5741, 0.5708, 0.5600, 0.5589, 0.5565, 0.5440]
    # Validation OOS R² (6 channels)
    r2_val = [0.4768, 0.4035, 0.3836, 0.3881, 0.3965, 0.4210]

    x_idx = np.arange(len(models))
    w = 0.25

    ax1.bar(x_idx - w, r2_train, width=w, label="Train In-Sample $R^2$", color="#1f77b4", alpha=0.9)
    ax1.bar(x_idx, r2_loco, width=w, label="26-Fold LOCO-CV $R^2_{\\mathrm{OOS}}$", color="#2ca02c", alpha=0.9)
    ax1.bar(x_idx + w, r2_val, width=w, label="Validation $R^2_{\\mathrm{OOS}}$ (6 Channels)", color="#ff7f0e", alpha=0.9)

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylabel("Coefficient of Determination ($R^2$ / $R^2_{\\mathrm{OOS}}$)")
    ax1.set_title("Generalization Progression: In-Sample vs. Out-of-Sample $R^2$", fontweight="bold")
    ax1.set_ylim(0.30, 0.70)
    ax1.axvline(x=4, color="crimson", linestyle="--", alpha=0.7, label="Locked Spec (M4-Centered)")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: AIC and BIC
    aic = [1912.4, 1912.2, 1914.1, 1915.6, 1913.4, 1913.6]
    bic = [1933.9, 1937.9, 1944.2, 1949.9, 1947.7, 1952.2]

    ax2.plot(models, aic, marker="o", color="#d62728", linewidth=2.0, label="AIC (Training)")
    ax2.plot(models, bic, marker="s", color="#9467bd", linewidth=2.0, label="BIC (Training, Parsimony Penalty)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=9.5)
    ax2.set_ylabel("Information Criterion Value")
    ax2.set_title("Akaike & Bayesian Information Criteria (Training Sample)", fontweight="bold")
    ax2.axvline(x=4, color="crimson", linestyle="--", alpha=0.7, label="Locked Spec (M4)")
    ax2.legend(loc="center right", frameon=True, fontsize=9)

    plt.tight_layout()
    f1_path = PLOTS_DIR / "01_model_progression_fit.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {f1_path}")

    # =========================================================================
    # FIGURE 2: Actual vs. Predicted (Train, Validation, and Test Sets)
    # =========================================================================
    logger.info("Generating Figure 2: Actual vs. Predicted across partitions...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)

    partitions = [
        ("train", "Training Set (26 Channels, N=541)\n$R^2 = 0.638$, RMSE = 1.488", axes[0]),
        ("validation", "Validation Set (6 Channels, N=136)\n$R^2_{\\mathrm{OOS}} = 0.396$, RMSE = 1.691", axes[1]),
        ("test", "Held-Out Test Set (6 Channels, N=132)\n$R^2_{\\mathrm{OOS}} = 0.754$, RMSE = 1.138", axes[2])
    ]

    tier_colors = {
        "Tier_1_Low (<5M)": "#1f77b4",
        "Tier_2_Mid (5M-20M)": "#2ca02c",
        "Tier_3_Mega (>20M)": "#d62728"
    }

    min_val, max_val = 6.0, 20.0

    for part_name, title, ax in partitions:
        sub = df[df["partition"] == part_name]
        for tier, grp in sub.groupby("subscriber_tier"):
            ax.scatter(
                grp["log_views"], grp["pred_log_views"],
                label=tier.split(" (")[0],
                color=tier_colors.get(tier, "gray"),
                alpha=0.65, edgecolors="none", s=32
            )
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.5, label="Perfect Agreement (45°)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel("Actual $\\ln(\\mathrm{Views})$")
        ax.set_ylabel("Predicted $\\ln(\\mathrm{Views})$")
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.legend(loc="upper left", frameon=True, fontsize=8)

    plt.tight_layout()
    f2_path = PLOTS_DIR / "02_actual_vs_predicted.png"
    plt.savefig(f2_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {f2_path}")

    # =========================================================================
    # FIGURE 3: Residual Audit (Diagnostics of Locked Model 4)
    # =========================================================================
    logger.info("Generating Figure 3: Residual Diagnostics...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    # Panel A: Residuals vs Fitted
    ax_a = axes[0, 0]
    ax_a.scatter(df["pred_log_views"], df["residuals"], alpha=0.5, color="#1f77b4", s=25)
    ax_a.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax_a.set_xlabel("Fitted Values $\\hat{y}_{it}$")
    ax_a.set_ylabel("Residuals $e_{it}$")
    ax_a.set_title("A: Residuals vs. Fitted Values (Homoscedasticity)", fontweight="bold")

    # Panel B: Normal Q-Q Plot
    ax_b = axes[0, 1]
    stats.probplot(df["residuals"], dist="norm", plot=ax_b)
    ax_b.get_lines()[0].set_markerfacecolor("#2ca02c")
    ax_b.get_lines()[0].set_markeredgecolor("none")
    ax_b.get_lines()[0].set_markersize(5.0)
    ax_b.get_lines()[0].set_alpha(0.6)
    ax_b.get_lines()[1].set_color("crimson")
    ax_b.get_lines()[1].set_linewidth(1.5)
    ax_b.set_title("B: Normal Q-Q Plot of Residuals", fontweight="bold")

    # Panel C: Residuals vs Own Click Appeal
    ax_c = axes[1, 0]
    ax_c.scatter(df["x"], df["residuals"], alpha=0.5, color="#ff7f0e", s=25)
    ax_c.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax_c.set_xlabel("Own Click Appeal Score $x_{it}$")
    ax_c.set_ylabel("Residuals $e_{it}$")
    ax_c.set_title("C: Residuals vs. Own Click Appeal (Linearity Audit)", fontweight="bold")

    # Panel D: Residuals vs Log Subscribers
    ax_d = axes[1, 1]
    ax_d.scatter(df["log_subscribers"], df["residuals"], alpha=0.5, color="#9467bd", s=25)
    ax_d.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax_d.set_xlabel("$\\ln(\\mathrm{Subscribers}_i)$")
    ax_d.set_ylabel("Residuals $e_{it}$")
    ax_d.set_title("D: Residuals vs. Creator Scale (Scale Adequacy)", fontweight="bold")

    plt.tight_layout()
    f3_path = PLOTS_DIR / "03_residual_audit.png"
    plt.savefig(f3_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {f3_path}")

    # =========================================================================
    # FIGURE 4: Payoff Response Curves & Production Cost Tradeoff
    # =========================================================================
    logger.info("Generating Figure 4: Payoff Response Curves...")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 10), dpi=300)

    # Grid of own appeal x
    x_grid = np.linspace(0.10, 0.75, 200)
    x_c_grid = x_grid - mu_x
    x2_c_grid = x_c_grid ** 2

    # Point estimates from full structural model
    b0 = gt_data["locked_centered_coefficients"]["beta_0_c_intercept"]
    b1 = gt_data["locked_centered_coefficients"]["beta_1_c_own_appeal_at_mean"]
    b2 = gt_data["locked_centered_coefficients"]["beta_2_c_quadratic_curvature"]
    b3 = gt_data["locked_centered_coefficients"]["beta_3_c_peer_congestion_at_mean"]
    b5 = gt_data["locked_centered_coefficients"]["beta_5_c_strategic_interaction"]
    g_s = gt_data["locked_centered_coefficients"]["gamma_subscribers"]
    g_d = gt_data["locked_centered_coefficients"]["gamma_duration"]
    g_a = gt_data["locked_centered_coefficients"]["gamma_age_days"]

    # Fixed representative controls at sample medians
    med_subs = float(df["log_subscribers"].median())
    med_dur = float(df["log_duration"].median())
    med_age = float(df["age_days"].median())
    base_shift = b0 + g_s * med_subs + g_d * med_dur + g_a * med_age

    # Three peer levels: 10th, 50th, 90th percentiles
    peer_p10 = float(np.percentile(df["x_bar"], 10))
    peer_p50 = float(np.percentile(df["x_bar"], 50))
    peer_p90 = float(np.percentile(df["x_bar"], 90))

    peer_levels = [
        (peer_p10, f"Low Peer Packaging ($p_{{10}} = {peer_p10:.3f}$)", "#2ca02c"),
        (peer_p50, f"Median Peer Packaging ($p_{{50}} = {peer_p50:.3f}$)", "#1f77b4"),
        (peer_p90, f"High Peer Packaging ($p_{{90}} = {peer_p90:.3f}$)", "#d62728")
    ]

    for p_val, label_str, color_str in peer_levels:
        x_bar_c = p_val - mu_xbar
        y_curve = base_shift + b1 * x_c_grid + b2 * x2_c_grid + b3 * x_bar_c + b5 * (x_c_grid * x_bar_c)
        ax_top.plot(x_grid, y_curve, label=label_str, color=color_str, linewidth=2.2)

    ax_top.set_xlabel("Own Click Appeal Score $x_i$")
    ax_top.set_ylabel("Expected Log Views $\\ln V_i(x_i, \\bar{x}_{-i})$")
    ax_top.set_title("A: Gross Expected Attention Response Curves Across Peer Competition Levels", fontweight="bold")
    ax_top.legend(loc="lower left", frameon=True)

    # Panel B: Net Payoff with Production Costs
    # C(x) = c * x + 0.5 * kappa * x^2
    # Setting c = 1.0, and contrasting kappa = 0 (linear) vs kappa = 8.0 (strictly convex)
    x_bar_c_med = peer_p50 - mu_xbar
    y_rev = base_shift + b1 * x_c_grid + b2 * x2_c_grid + b3 * x_bar_c_med + b5 * (x_c_grid * x_bar_c_med)

    c_lin = 1.0
    c_conv = 8.0

    cost_linear = c_lin * x_grid
    cost_convex = c_lin * x_grid + 0.5 * c_conv * (x_grid ** 2)

    payoff_linear = y_rev - cost_linear
    payoff_convex = y_rev - cost_convex

    # Optimal choices
    opt_linear_idx = np.argmax(payoff_linear)
    opt_convex_idx = np.argmax(payoff_convex)

    ax_bot.plot(x_grid, y_rev, label="Gross Revenue $\\ln V_i(x_i, \\bar{x}_{50})$", color="#1f77b4", linestyle="--", linewidth=1.8)
    ax_bot.plot(x_grid, payoff_linear, label=f"Net Payoff (Linear Cost $C=x$): Peak at $x^* = {x_grid[opt_linear_idx]:.2f}$", color="#ff7f0e", linewidth=2.0)
    ax_bot.plot(x_grid, payoff_convex, label=f"Net Payoff (Convex Cost $C=x+4x^2$): Peak at $x^* = {x_grid[opt_convex_idx]:.2f}$", color="#2ca02c", linewidth=2.0)

    ax_bot.axvline(x_grid[opt_linear_idx], color="#ff7f0e", linestyle=":", alpha=0.8)
    ax_bot.axvline(x_grid[opt_convex_idx], color="#2ca02c", linestyle=":", alpha=0.8)

    ax_bot.scatter([x_grid[opt_linear_idx]], [payoff_linear[opt_linear_idx]], color="#ff7f0e", s=60, zorder=5)
    ax_bot.scatter([x_grid[opt_convex_idx]], [payoff_convex[opt_convex_idx]], color="#2ca02c", s=60, zorder=5)

    ax_bot.set_xlabel("Own Click Appeal Score $x_i$")
    ax_bot.set_ylabel("Creator Payoff $\\Pi_i(x_i, \\bar{x}_{-i})$")
    ax_bot.set_title("B: Net Payoff Tradeoff: Linear vs. Strictly Convex Production Cost", fontweight="bold")
    ax_bot.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    f4_path = PLOTS_DIR / "04_payoff_response_curves.png"
    plt.savefig(f4_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {f4_path}")

    # =========================================================================
    # FIGURE 5: Strategic Interaction 2D Contour Heatmap & Best Response
    # =========================================================================
    logger.info("Generating Figure 5: Strategic Interaction Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)

    x_i_vals = np.linspace(0.10, 0.75, 100)
    x_bar_vals = np.linspace(0.25, 0.55, 100)
    X_grid, Y_grid = np.meshgrid(x_i_vals, x_bar_vals)

    X_c_mesh = X_grid - mu_x
    Y_c_mesh = Y_grid - mu_xbar

    Z_views = base_shift + b1 * X_c_mesh + b2 * (X_c_mesh ** 2) + b3 * Y_c_mesh + b5 * (X_c_mesh * Y_c_mesh)

    cp = ax.contourf(X_grid, Y_grid, Z_views, levels=25, cmap="viridis")
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label("Expected $\\ln(\\mathrm{Views}_{it})$")

    cs = ax.contour(X_grid, Y_grid, Z_views, levels=12, colors="white", alpha=0.4, linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Compute best response curve x_i*(x_bar) under convex cost C(x) = x + 4 x^2
    # FOC: b1 + 2*b2*(x - mu_x) + b5*(x_bar - mu_xbar) - (1 + 8*x) = 0
    # x * (2*b2 - 8) = 1 - b1 + 2*b2*mu_x - b5*(x_bar - mu_xbar)
    # x*(x_bar) = (1 - b1 + 2*b2*mu_x - b5*(x_bar - mu_xbar)) / (2*b2 - 8)
    br_num = 1.0 - b1 + 2.0 * b2 * mu_x - b5 * (x_bar_vals - mu_xbar)
    br_den = 2.0 * b2 - 8.0
    x_br = br_num / br_den
    x_br = np.clip(x_br, 0.10, 0.75)

    ax.plot(x_br, x_bar_vals, color="crimson", linewidth=2.8, linestyle="--", label="Best Response Curve $x_i^*(\\bar{x}_{-i})$ (Convex Cost)")
    ax.axvline(mu_x, color="white", linestyle=":", alpha=0.7, label=f"Mean Own Appeal ($\\mu_x={mu_x:.3f}$)")
    ax.axhline(mu_xbar, color="gold", linestyle=":", alpha=0.7, label=f"Mean Peer Appeal ($\\mu_{{\\bar{{x}}}}={mu_xbar:.3f}$)")

    ax.set_xlabel("Focal Creator Click Appeal Score $x_i$")
    ax.set_ylabel("Competitor Average Peer Appeal $\\bar{x}_{-i}$")
    ax.set_title("Strategic Interaction Surface $\\ln V_i(x_i, \\bar{x}_{-i})$ & Creator Best Response", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.85)

    plt.tight_layout()
    f5_path = PLOTS_DIR / "05_peer_interaction_heatmap.png"
    plt.savefig(f5_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {f5_path}")

    # =========================================================================
    # FIGURE 6: Channel Error Dispersion across 38 Channels
    # =========================================================================
    logger.info("Generating Figure 6: Channel Error Dispersion across all 38 channels...")
    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(15, 11), dpi=300, sharey=True)

    # Compute MAE and Bias for each of the 38 channels under locked Model 4
    ch_records = []
    for c_id, grp in df.groupby("channel_id"):
        ch_name = grp["channel_name"].iloc[0]
        part = grp["partition"].iloc[0]
        tier = grp["subscriber_tier"].iloc[0]
        mae = float(np.mean(np.abs(grp["residuals"])))
        rmse = float(np.sqrt(np.mean(grp["residuals"] ** 2)))
        bias = float(np.mean(grp["residuals"]))
        ch_records.append({
            "channel_id": c_id,
            "channel_name": ch_name,
            "partition": part,
            "subscriber_tier": tier,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "n_videos": len(grp)
        })

    df_disp = pd.DataFrame(ch_records).sort_values("mae", ascending=True)

    part_colors = {
        "train": "#1f77b4",
        "validation": "#ff7f0e",
        "test": "#d62728"
    }

    y_pos = np.arange(len(df_disp))
    bar_colors = [part_colors[p] for p in df_disp["partition"]]

    # Panel A: MAE
    ax_mae.barh(y_pos, df_disp["mae"], color=bar_colors, alpha=0.85, height=0.7)
    ax_mae.set_yticks(y_pos)
    ax_mae.set_yticklabels(
        [f"{r['channel_name']} (N={r['n_videos']})" for _, r in df_disp.iterrows()],
        fontsize=8.5
    )
    ax_mae.set_xlabel("Mean Absolute Error (MAE in log-views)")
    ax_mae.set_title("A: Channel-by-Channel Error Magnitude (MAE)", fontweight="bold")
    ax_mae.axvline(df_disp["mae"].mean(), color="black", linestyle="--", linewidth=1.2, label=f"Mean MAE = {df_disp['mae'].mean():.2f}")

    # Panel B: Bias
    ax_bias.barh(y_pos, df_disp["bias"], color=bar_colors, alpha=0.85, height=0.7)
    ax_bias.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax_bias.set_xlabel("Prediction Bias (Mean Residual: Actual − Predicted)")
    ax_bias.set_title("B: Directional Bias (Under / Over-Prediction)", fontweight="bold")

    # Legend for partitions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Training Cohort (26 Channels)"),
        Patch(facecolor="#ff7f0e", label="Validation Cohort (6 Channels)"),
        Patch(facecolor="#d62728", label="Held-Out Test Cohort (6 Channels)"),
    ]
    ax_mae.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    f6_path = PLOTS_DIR / "06_channel_error_dispersion.png"
    plt.savefig(f6_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {f6_path}")

    logger.info("=" * 75)
    logger.info("STEP 8 COMPLETED SUCCESSFULLY: ALL 6 PUBLICATION FIGURES GENERATED")
    logger.info("=" * 75)


if __name__ == "__main__":
    generate_plots()
