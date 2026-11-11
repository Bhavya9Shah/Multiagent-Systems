#!/usr/bin/env python3
"""
payoff_estimation/scripts/generate_plots.py
Stage 3 of Payoff Estimation: Publication-Quality Econometric & Diagnostic Visualizations.

Generates:
1. plots/01_actual_vs_predicted.png
2. plots/02_residual_diagnostics.png
3. plots/03_payoff_response_curves.png
4. plots/04_peer_interaction_heatmap.png
5. plots/05_variable_importance_vif.png
6. plots/06_subscribers_dominance.png
"""

import sys
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
PAYOFF_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PAYOFF_DIR / "data"
RESULTS_DIR = PAYOFF_DIR / "results"
PLOTS_DIR = PAYOFF_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PAYOFF_DIR / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "generate_plots.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("generate_plots")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.titlesize"] = 14

PREPARED_CSV = DATA_DIR / "prepared_payoff_data.csv"


def generate_all_plots():
    logger.info("Loading prepared data...")
    if not PREPARED_CSV.exists():
        logger.error(f"Prepared data not found at {PREPARED_CSV}")
        sys.exit(1)

    df = pd.read_csv(PREPARED_CSV)

    # Re-estimate Full Model (M4) for residuals and predictions
    cols = ["x", "x2", "x_bar_roll90", "x_inter_roll90", "log_subscribers", "log_duration", "age_days"]
    X = sm.add_constant(df[cols])
    y = df["log_views"]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["channel_id"]})

    df["predicted_log_views"] = model.predict(X)
    df["residuals"] = model.resid
    std_resid = (df["residuals"] - df["residuals"].mean()) / df["residuals"].std()
    df["std_residuals"] = std_resid

    # Extract coefficients
    b0 = model.params["const"]
    b1 = model.params["x"]
    b2 = model.params["x2"]
    b3 = model.params["x_bar_roll90"]
    b5 = model.params["x_inter_roll90"]
    b4 = model.params["log_subscribers"]
    g1 = model.params["log_duration"]
    g2 = model.params["age_days"]

    # =========================================================================
    # PLOT 1: Actual vs. Predicted Log(Views)
    # =========================================================================
    logger.info("Generating Plot 1: Actual vs. Predicted...")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    scatter = ax.scatter(
        df["predicted_log_views"], df["log_views"],
        c=df["x"], cmap="viridis", alpha=0.75, edgecolors="none", s=55
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Click Appeal Score ($x_{it}$)", fontsize=10)

    # 45-degree reference line
    min_val = min(df["predicted_log_views"].min(), df["log_views"].min()) - 0.5
    max_val = max(df["predicted_log_views"].max(), df["log_views"].max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit ($y = \\hat{y}$)")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Fitted Payoff $\\widehat{\\log(V_{it})}$", fontsize=11)
    ax.set_ylabel("Observed Payoff $\\log(V_{it})$", fontsize=11)
    ax.set_title("Empirical Model Fit: Observed vs. Predicted Log(Views)\nFull Proposed Specification (Eq. 1)", fontsize=12, pad=12)

    # Annotation box with stats
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    rmse = np.sqrt(np.mean(df["residuals"]**2))
    stat_text = (
        f"$R^2 = {r2:.4f}$\n"
        f"Adj. $R^2 = {adj_r2:.4f}$\n"
        f"RMSE $= {rmse:.3f}$\n"
        f"$N = {len(df)}$ videos\n"
        f"Clusters $= {df['channel_id'].nunique()}$ channels"
    )
    ax.text(
        0.05, 0.95, stat_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plot1_path = PLOTS_DIR / "01_actual_vs_predicted.png"
    plt.savefig(plot1_path)
    plt.close()

    # =========================================================================
    # PLOT 2: Residual Diagnostics (4 Panels)
    # =========================================================================
    logger.info("Generating Plot 2: Residual Diagnostics...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    # 2A: Residuals vs Fitted
    ax_rf = axes[0, 0]
    ax_rf.scatter(df["predicted_log_views"], df["residuals"], alpha=0.6, color="#1f77b4", edgecolors="none")
    ax_rf.axhline(0, color="red", linestyle="--", linewidth=1.2)
    # Lowess smooth
    lowess = sm.nonparametric.lowess(df["residuals"], df["predicted_log_views"], frac=0.6)
    ax_rf.plot(lowess[:, 0], lowess[:, 1], color="darkorange", linewidth=2, label="LOWESS trend")
    ax_rf.set_xlabel("Fitted Values $\\widehat{\\log(V_{it})}$")
    ax_rf.set_ylabel("Residuals $\\hat{\\varepsilon}_{it}$")
    ax_rf.set_title("(A) Residuals vs. Fitted Values")
    ax_rf.legend(loc="upper right")

    # 2B: Normal Q-Q Plot
    ax_qq = axes[0, 1]
    stats.probplot(df["residuals"], dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_markerfacecolor("#1f77b4")
    ax_qq.get_lines()[0].set_markeredgecolor("none")
    ax_qq.get_lines()[0].set_alpha(0.6)
    ax_qq.get_lines()[1].set_color("red")
    ax_qq.get_lines()[1].set_linewidth(1.5)
    ax_qq.set_title("(B) Normal Q-Q Plot")

    # 2C: Histogram & Density
    ax_hist = axes[1, 0]
    count, bins, _ = ax_hist.hist(df["residuals"], bins=20, density=True, alpha=0.5, color="#2ca02c", edgecolor="black")
    # Kernel density
    kde_x = np.linspace(df["residuals"].min() - 1, df["residuals"].max() + 1, 200)
    kde = stats.gaussian_kde(df["residuals"])
    ax_hist.plot(kde_x, kde(kde_x), color="darkgreen", linewidth=2, label="Residual KDE")
    # Normal overlay
    norm_pdf = stats.norm.pdf(kde_x, loc=df["residuals"].mean(), scale=df["residuals"].std())
    ax_hist.plot(kde_x, norm_pdf, "r--", linewidth=1.5, label="Theoretical Normal")
    ax_hist.set_xlabel("Residual Value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("(C) Residual Distribution")
    ax_hist.legend(loc="upper right")

    # 2D: Residuals by Channel Cluster (Top 15 channels by video count / views)
    ax_clust = axes[1, 1]
    top_channels = df["channel_name"].value_counts().head(12).index
    df_top_clust = df[df["channel_name"].isin(top_channels)]
    sns_order = df_top_clust.groupby("channel_name")["residuals"].median().sort_values().index
    channel_resids = [df[df["channel_name"] == c]["residuals"].values for c in sns_order]
    
    ax_clust.boxplot(channel_resids, labels=[c[:14] for c in sns_order], vert=False, patch_artist=True,
                     boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                     medianprops=dict(color="red", linewidth=1.5))
    ax_clust.axvline(0, color="black", linestyle=":", alpha=0.7)
    ax_clust.set_xlabel("Residuals $\\hat{\\varepsilon}_{it}$")
    ax_clust.set_title("(D) Within-Channel Residual Dispersion (Sample Creators)")
    ax_clust.tick_params(axis="y", labelsize=8)

    plt.suptitle("Econometric Residual Diagnostics: Validating OLS Assumptions", y=1.005)
    plt.tight_layout()
    plot2_path = PLOTS_DIR / "02_residual_diagnostics.png"
    plt.savefig(plot2_path, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT 3: Empirical Payoff Response Curves
    # =========================================================================
    logger.info("Generating Plot 3: Payoff Response Curves...")
    fig, (ax_curve, ax_dist) = plt.subplots(
        2, 1, figsize=(9, 8), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08}
    )

    x_grid = np.linspace(0.0, 1.0, 300)
    # Competitor benchmark levels: 10th percentile, Median, 90th percentile
    p10_peer = float(df["x_bar_roll90"].quantile(0.10))
    p50_peer = float(df["x_bar_roll90"].median())
    p90_peer = float(df["x_bar_roll90"].quantile(0.90))

    # Mean controls
    mean_controls = (
        b0 + b4 * df["log_subscribers"].mean() +
        g1 * df["log_duration"].mean() +
        g2 * df["age_days"].mean()
    )

    def calc_payoff(x_val, peer_val):
        return (
            b1 * x_val + b2 * (x_val ** 2) +
            b3 * peer_val + b5 * (x_val * peer_val) +
            mean_controls
        )

    # Plot response curves
    ax_curve.plot(x_grid, calc_payoff(x_grid, p10_peer), label=f"Low Peer Intensity (10th %ile: $\\bar{{x}}_{{-i}}={p10_peer:.3f}$)", color="#2ca02c", linewidth=2.2)
    ax_curve.plot(x_grid, calc_payoff(x_grid, p50_peer), label=f"Median Peer Intensity (50th %ile: $\\bar{{x}}_{{-i}}={p50_peer:.3f}$)", color="#1f77b4", linewidth=2.5)
    ax_curve.plot(x_grid, calc_payoff(x_grid, p90_peer), label=f"High Peer Intensity (90th %ile: $\\bar{{x}}_{{-i}}={p90_peer:.3f}$)", color="#d62728", linewidth=2.2, linestyle="--")

    # Mark Peak (Best Response) for median peer
    denom = 2 * abs(b2)
    br_p50 = (b1 + b5 * p50_peer) / denom
    payoff_peak = calc_payoff(br_p50, p50_peer)
    ax_curve.plot(br_p50, payoff_peak, "ko", markersize=8, label=f"Empirical Peak at $\\bar{{x}}_{{50}}$: $x^*={br_p50:.3f}$")
    ax_curve.axvline(br_p50, color="gray", linestyle=":", alpha=0.8)

    # Mark Symmetric Nash Equilibrium
    denom_nash = 2 * abs(b2) - b5
    x_nash_val = b1 / denom_nash
    ax_curve.axvline(x_nash_val, color="purple", linestyle="-.", linewidth=1.8, label=f"Symmetric Nash Equilibrium $x_{{Nash}} = {x_nash_val:.3f}$")

    ax_curve.set_ylabel("Expected Video Payoff $\\pi_i(x_i, \\bar{x}_{-i}) = \\mathbb{E}[\\log(V_i)]$", fontsize=11)
    ax_curve.set_title("Empirical Payoff Function $\\pi_i(x_i, \\bar{x}_{-i})$ Across Click Appeal Strategies\nDemonstrating Quadratic Concavity $(\\beta_2 = -9.168 < 0)$ and Interior Best Response", fontsize=12, pad=10)
    ax_curve.legend(loc="lower center", fontsize=9.5, frameon=True)
    ax_curve.set_ylim(bottom=calc_payoff(0.0, p50_peer) - 0.5)

    # Bottom Panel: Empirical Distribution of Click Appeal
    ax_dist.hist(df["x"], bins=25, color="#4a7bb0", edgecolor="black", alpha=0.7, density=True)
    ax_dist.axvline(df["x"].mean(), color="black", linestyle="-", linewidth=1.5, label=f"Empirical Mean: {df['x'].mean():.3f}")
    ax_dist.set_xlabel("Own Click Appeal Score ($x_{it} \\in [0, 1]$)", fontsize=11)
    ax_dist.set_ylabel("Empirical\nDensity", fontsize=9)
    ax_dist.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plot3_path = PLOTS_DIR / "03_payoff_response_curves.png"
    plt.savefig(plot3_path)
    plt.close()

    # =========================================================================
    # PLOT 4: 2D Strategic Heatmap & Best Response Function
    # =========================================================================
    logger.info("Generating Plot 4: Strategic Interaction Heatmap...")
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

    x_vals = np.linspace(0.0, 1.0, 150)
    peer_vals = np.linspace(0.2, 0.6, 150)
    X_grid, Y_grid = np.meshgrid(x_vals, peer_vals)

    Z_payoff = (
        b1 * X_grid + b2 * (X_grid ** 2) +
        b3 * Y_grid + b5 * (X_grid * Y_grid) +
        mean_controls
    )

    contour = ax.contourf(X_grid, Y_grid, Z_payoff, levels=25, cmap="Spectral_r", alpha=0.85)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Conditional Expected Payoff $\\mathbb{E}[\\log(V)]$", fontsize=11)

    # Contour lines
    cs = ax.contour(X_grid, Y_grid, Z_payoff, levels=12, colors="black", linewidths=0.5, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Best Response Line: BR(peer) = (b1 + b5 * peer) / (2 * |b2|)
    br_line = (b1 + b5 * peer_vals) / (2 * abs(b2))
    ax.plot(br_line, peer_vals, "b-", linewidth=2.8, label="Best-Response Function $BR_i(\\bar{x}_{-i})$")

    # Symmetric profile line (x_i = peer)
    ax.plot(peer_vals, peer_vals, "k--", linewidth=1.5, label="Symmetric Profile ($x_i = \\bar{x}_{-i}$)")

    # Nash Equilibrium Intersection Point
    ax.plot([x_nash_val], [x_nash_val], "r*", markersize=14, label=f"Symmetric Nash Equilibrium $(x^* = {x_nash_val:.3f})$")

    # Scatter of actual creator observations (own score vs peer average)
    ax.scatter(df["x"], df["x_bar_roll90"], color="black", alpha=0.35, s=20, label="Observed YouTube Videos ($N=186$)")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.2, 0.6)
    ax.set_xlabel("Own Click Appeal Strategy ($x_i$)", fontsize=11)
    ax.set_ylabel("Competitor Average Click Appeal ($\\bar{x}_{-i}$)", fontsize=11)
    ax.set_title("Strategic Payoff Landscape and Nash Equilibrium in Action Space\nContour Map of $\\pi_i(x_i, \\bar{x}_{-i})$ with Contraction Best-Response Dynamics", fontsize=12, pad=12)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    plot4_path = PLOTS_DIR / "04_peer_interaction_heatmap.png"
    plt.savefig(plot4_path)
    plt.close()

    # =========================================================================
    # PLOT 5: Multicollinearity Diagnostics & VIF Comparison
    # =========================================================================
    logger.info("Generating Plot 5: VIF & Collinearity Diagnostics...")
    vif_csv = RESULTS_DIR / "multicollinearity_diagnostics.csv"
    df_vif = pd.read_csv(vif_csv)

    m4_vifs = df_vif[df_vif["model_name"] == "M4_Full_Proposed"]
    m4c_vifs = df_vif[df_vif["model_name"] == "M4_Full_Centered"]

    fig, (ax_vif1, ax_vif2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 5A: Raw Uncentered VIFs
    bars1 = ax_vif1.barh(m4_vifs["variable"], m4_vifs["vif"], color="#d62728", alpha=0.85)
    ax_vif1.axvline(10, color="black", linestyle="--", linewidth=1.2, label="Rule of Thumb (VIF = 10)")
    ax_vif1.set_xlabel("Variance Inflation Factor (Log Scale)", fontsize=10)
    ax_vif1.set_xscale("log")
    ax_vif1.set_title("(A) Raw Uncentered Specification\nHigh Structural Collinearity between $x$, $x^2$, and $x \\cdot \\bar{x}$", fontsize=11)
    ax_vif1.legend(loc="lower right")
    for bar in bars1:
        w = bar.get_width()
        ax_vif1.text(w * 1.1, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va="center", fontsize=8.5)

    # 5B: Mean-Centered VIFs
    bars2 = ax_vif2.barh(m4c_vifs["variable"], m4c_vifs["vif"], color="#2ca02c", alpha=0.85)
    ax_vif2.axvline(10, color="black", linestyle="--", linewidth=1.2, label="Rule of Thumb (VIF = 10)")
    ax_vif2.set_xlabel("Variance Inflation Factor (Linear Scale)", fontsize=10)
    ax_vif2.set_xlim(0, 5)
    ax_vif2.set_title("(B) Mean-Centered Specification\nStructural Collinearity Eliminated (All VIFs < 1.30)", fontsize=11)
    ax_vif2.legend(loc="upper right")
    for bar in bars2:
        w = bar.get_width()
        ax_vif2.text(w + 0.08, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=9)

    plt.suptitle("Multicollinearity Diagnostics: Structural Polynomial Collinearity vs. Centering Resolution", y=1.02)
    plt.tight_layout()
    plot5_path = PLOTS_DIR / "05_variable_importance_vif.png"
    plt.savefig(plot5_path, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT 6: Subscriber Base Dominance vs. Marginal Click Appeal
    # =========================================================================
    logger.info("Generating Plot 6: Subscriber Dominance & Partial Regression...")
    fig, (ax_sub, ax_appeal) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=300)

    # 6A: Log Views vs Log Subscribers (Primary Driver)
    sns_fit1 = np.polyfit(df["log_subscribers"], df["log_views"], 1)
    fit_line1 = np.poly1d(sns_fit1)
    ax_sub.scatter(df["log_subscribers"], df["log_views"], color="#1f77b4", alpha=0.7, edgecolors="none", s=50)
    x_sub_grid = np.linspace(df["log_subscribers"].min(), df["log_subscribers"].max(), 100)
    ax_sub.plot(x_sub_grid, fit_line1(x_sub_grid), "r-", linewidth=2.2,
                label=f"Linear Fit: Slope = {sns_fit1[0]:.2f}\n$r = {df['log_subscribers'].corr(df['log_views']):.3f}$ ($R^2 \\approx 63.2\\%$)")
    ax_sub.set_xlabel("Log(Subscribers) $\\log(s_i)$", fontsize=11)
    ax_sub.set_ylabel("Log(Views) $\\log(V_{it})$", fontsize=11)
    ax_sub.set_title("(A) Dominant Platform Driver: Channel Subscriber Base\nStrong Linear Elasticity $(\\beta_4 \\approx 1.07, p < 10^{-20})$", fontsize=11)
    ax_sub.legend(loc="upper left", fontsize=10)

    # 6B: Partial Regression: Residual Views (controlling for scale & controls) vs Click Appeal
    # Partial out log_subscribers, log_duration, age_days from log_views and x
    exog_controls = sm.add_constant(df[["log_subscribers", "log_duration", "age_days"]])
    res_y_given_z = sm.OLS(df["log_views"], exog_controls).fit().resid
    res_x_given_z = sm.OLS(df["x"], exog_controls).fit().resid

    fit_part = np.polyfit(res_x_given_z, res_y_given_z, 1)
    fit_line_part = np.poly1d(fit_part)
    x_part_grid = np.linspace(res_x_given_z.min(), res_x_given_z.max(), 100)

    ax_appeal.scatter(res_x_given_z, res_y_given_z, color="#ff7f0e", alpha=0.7, edgecolors="none", s=50)
    ax_appeal.plot(x_part_grid, fit_line_part(x_part_grid), "b-", linewidth=2.2,
                   label=f"Partial Slope = {fit_part[0]:+.2f} (marginal packaging effect)")
    ax_appeal.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax_appeal.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax_appeal.set_xlabel("Residual Click Appeal $x_{it} \\mid Z$", fontsize=11)
    ax_appeal.set_ylabel("Residual Log(Views) $\\log(V_{it}) \\mid Z$", fontsize=11)
    ax_appeal.set_title("(B) Marginal Incentive Effect: Residual Click Appeal\nOperates at the Margin after Accounting for Creator Fixed Scale", fontsize=11)
    ax_appeal.legend(loc="upper left", fontsize=10)

    plt.suptitle("Empirical Decomposition: Platform Scale Effect vs. Marginal Packaging Incentive", y=1.02)
    plt.tight_layout()
    plot6_path = PLOTS_DIR / "06_subscribers_dominance.png"
    plt.savefig(plot6_path, bbox_inches="tight")
    plt.close()

    logger.info("All 6 diagnostic and publication plots generated successfully.")
    print("\n================ DIAGNOSTIC PLOTS GENERATED ================")
    print(f"  1. {plot1_path.name}  (Actual vs Predicted Log Views)")
    print(f"  2. {plot2_path.name} (4-Panel Residual Diagnostics & Cluster Dispersion)")
    print(f"  3. {plot3_path.name} (Payoff Curves, Diminishing Returns, Interior Peak)")
    print(f"  4. {plot4_path.name} (2D Strategic Landscape, Best Response & Nash Eq)")
    print(f"  5. {plot5_path.name} (VIF Multicollinearity Diagnostics)")
    print(f"  6. {plot6_path.name} (Scale Dominance & Partial Regression Decomposition)")
    print("============================================================\n")


if __name__ == "__main__":
    generate_all_plots()
