#!/usr/bin/env python3
"""
payoff_estimation/scripts/generate_plots.py
Stage 3 of Payoff Estimation: Diagnostic Visualizations, Model Progression,
and Grouped Cross-Validation Evaluation with Explicit Uncertainty Representation.

Generates:
1. plots/01_model_progression_comparison.png
2. plots/02_actual_vs_predicted.png
3. plots/03_residual_diagnostics.png
4. plots/04_payoff_response_curves.png
5. plots/05_peer_interaction_heatmap.png
6. plots/06_variable_importance_vif.png
7. plots/07_grouped_cv_channel_errors.png
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

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

PREPARED_CSV = DATA_DIR / "prepared_payoff_data.csv"
MODEL_COMP_CSV = RESULTS_DIR / "model_comparison.csv"
CV_RESULTS_CSV = RESULTS_DIR / "cross_validation_results.csv"
VIF_CSV = RESULTS_DIR / "multicollinearity_diagnostics.csv"


def generate_all_plots():
    logger.info("Loading prepared data and results...")
    if not PREPARED_CSV.exists() or not MODEL_COMP_CSV.exists():
        logger.error("Required data/results missing. Run run_regressions.py first.")
        sys.exit(1)

    df = pd.read_csv(PREPARED_CSV)
    df_comp = pd.read_csv(MODEL_COMP_CSV)
    df_cv = pd.read_csv(CV_RESULTS_CSV)

    # Re-estimate Model 5 for residuals and parameter covariance
    cols_m5 = ["x", "x2", "x_bar_roll90", "x_inter_roll90", "log_subscribers", "log_duration", "age_days"]
    X5 = sm.add_constant(df[cols_m5])
    y = df["log_views"]
    m5 = sm.OLS(y, X5).fit(cov_type="cluster", cov_kwds={"groups": df["channel_id"]})

    df["fitted_m5"] = m5.predict(X5)
    df["resid_m5"] = m5.resid
    b0 = m5.params["const"]
    b1 = m5.params["x"]
    b2 = m5.params["x2"]
    b3 = m5.params["x_bar_roll90"]
    b5 = m5.params["x_inter_roll90"]
    b4 = m5.params["log_subscribers"]
    g1 = m5.params["log_duration"]
    g2 = m5.params["age_days"]

    # Parameter covariance for delta method standard errors
    cov_m5 = m5.cov_params()

    # =========================================================================
    # PLOT 1: Progressive Model Comparison (In-Sample R2 vs Grouped-CV OOS R2)
    # =========================================================================
    logger.info("Generating Plot 1: Model Progression Comparison...")
    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    core_models = ["Model_1_OwnScore", "Model_2_Quadratic", "Model_3_CompetitorAvg", 
                   "Model_4_StrategicInteraction", "Model_5_FullProposed"]
    labels = ["M1: Own Score (x)", "M2: Quadratic (x+x²)", "M3: Peer Avg (+x̄)", 
              "M4: Interaction (+x·x̄)", "M5: Full (+Controls)"]
    
    sub_comp = df_comp[df_comp["model_name"].isin(core_models)].set_index("model_name").loc[core_models]
    
    x_pos = np.arange(len(core_models))
    width = 0.35

    # Panel A: R-squared Comparison
    bars_in = ax_r2.bar(x_pos - width/2, sub_comp["r_squared"], width, label="In-Sample $R^2$ (Full Dataset)", color="#1f77b4", alpha=0.85)
    bars_oos = ax_r2.bar(x_pos + width/2, sub_comp["oos_r_squared_loco_cv"].fillna(0), width, label="Out-of-Sample $R^2$ (Strict LOCO-CV)", color="#2ca02c", alpha=0.85)
    ax_r2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax_r2.set_ylabel("$R^2$ Value", fontsize=11)
    ax_r2.set_title("(A) Explanatory Power Across Progressive Models\nIn-Sample vs. Leave-One-Channel-Out Cross-Validation", fontsize=11)
    ax_r2.legend(loc="upper left", fontsize=9.5)
    ax_r2.set_ylim(-0.16, 0.75)

    for bar in bars_in:
        h = bar.get_height()
        ax_r2.text(bar.get_x() + bar.get_width()/2, max(0.01, h + 0.02), f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars_oos:
        h = bar.get_height()
        y_pos = h + 0.02 if h >= 0 else h - 0.04
        ax_r2.text(bar.get_x() + bar.get_width()/2, y_pos, f"{h:.3f}", ha="center", va="bottom" if h>=0 else "top", fontsize=8.5)

    # Panel B: RMSE Comparison
    bars_rmse_in = ax_rmse.bar(x_pos - width/2, sub_comp["rmse_in_sample"], width, label="In-Sample RMSE", color="#4a7bb0", alpha=0.85)
    bars_rmse_oos = ax_rmse.bar(x_pos + width/2, sub_comp["oos_rmse_loco_cv"], width, label="OOS RMSE (Strict LOCO-CV)", color="#ff7f0e", alpha=0.85)
    ax_rmse.set_xticks(x_pos)
    ax_rmse.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax_rmse.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
    ax_rmse.set_title("(B) Prediction Error Progression\nModels 1–4 Suffer Negative OOS $R^2$; Scale Stabilizes Fit", fontsize=11)
    ax_rmse.legend(loc="upper right", fontsize=9.5)
    ax_rmse.set_ylim(0, 3.0)

    for bar in bars_rmse_in:
        h = bar.get_height()
        ax_rmse.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars_rmse_oos:
        h = bar.get_height()
        ax_rmse.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    plt.suptitle("Econometric Progression: From Univariate Appeal to Full Strategic Payoff Model", y=1.02, fontsize=13)
    plt.tight_layout()
    plot1_path = PLOTS_DIR / "01_model_progression_comparison.png"
    plt.savefig(plot1_path, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT 2: Actual vs. Fitted Log(Views) for Model 5
    # =========================================================================
    logger.info("Generating Plot 2: Actual vs. Fitted...")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    scatter = ax.scatter(
        df["fitted_m5"], df["log_views"],
        c=df["x"], cmap="viridis", alpha=0.75, s=60, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Click Appeal Score ($x_{it}$)", fontsize=10)

    min_v = min(df["fitted_m5"].min(), df["log_views"].min()) - 0.5
    max_v = max(df["fitted_m5"].max(), df["log_views"].max()) + 0.5
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect Fit ($y = \\hat{y}$)")

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Fitted Values $\\widehat{\\log(V_{it})}$ (Model 5)", fontsize=11)
    ax.set_ylabel("Observed Values $\\log(V_{it})$", fontsize=11)
    ax.set_title("Full Payoff Model Fit: Observed vs. Predicted Log(Views)\nModel 5 (Full Proposed Specification, $N=186$)", fontsize=12, pad=12)

    stat_box = (
        f"$R^2 = {m5.rsquared:.4f}$\n"
        f"Adj. $R^2 = {m5.rsquared_adj:.4f}$\n"
        f"In-Sample RMSE $= {np.sqrt(np.mean(df['resid_m5']**2)):.3f}$\n"
        f"OOS LOCO-CV $R^2 = {sub_comp.loc['Model_5_FullProposed', 'oos_r_squared_loco_cv']:.4f}$\n"
        f"$N = 186$ videos across 38 channels"
    )
    ax.text(
        0.05, 0.95, stat_box, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plot2_path = PLOTS_DIR / "02_actual_vs_predicted.png"
    plt.savefig(plot2_path)
    plt.close()

    # =========================================================================
    # PLOT 3: 4-Panel Econometric Residual Diagnostics
    # =========================================================================
    logger.info("Generating Plot 3: Residual Diagnostics...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    # 3A: Residuals vs Fitted
    ax_rf = axes[0, 0]
    ax_rf.scatter(df["fitted_m5"], df["resid_m5"], alpha=0.6, color="#1f77b4", edgecolors="none")
    ax_rf.axhline(0, color="red", linestyle="--", linewidth=1.2)
    lowess = sm.nonparametric.lowess(df["resid_m5"], df["fitted_m5"], frac=0.6)
    ax_rf.plot(lowess[:, 0], lowess[:, 1], color="darkorange", linewidth=2, label="LOWESS trend")
    ax_rf.set_xlabel("Fitted Values $\\widehat{\\log(V_{it})}$")
    ax_rf.set_ylabel("Residuals $\\hat{\\varepsilon}_{it}$")
    ax_rf.set_title("(A) Residuals vs. Fitted Values")
    ax_rf.legend(loc="upper right")

    # 3B: Normal Q-Q Plot
    ax_qq = axes[0, 1]
    stats.probplot(df["resid_m5"], dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_markerfacecolor("#1f77b4")
    ax_qq.get_lines()[0].set_markeredgecolor("none")
    ax_qq.get_lines()[0].set_alpha(0.6)
    ax_qq.get_lines()[1].set_color("red")
    ax_qq.get_lines()[1].set_linewidth(1.5)
    ax_qq.set_title("(B) Normal Q-Q Plot")

    # 3C: Histogram & Density
    ax_hist = axes[1, 0]
    ax_hist.hist(df["resid_m5"], bins=20, density=True, alpha=0.5, color="#2ca02c", edgecolor="black")
    kde_x = np.linspace(df["resid_m5"].min() - 1, df["resid_m5"].max() + 1, 200)
    kde = stats.gaussian_kde(df["resid_m5"])
    ax_hist.plot(kde_x, kde(kde_x), color="darkgreen", linewidth=2, label="Residual KDE")
    norm_pdf = stats.norm.pdf(kde_x, loc=df["resid_m5"].mean(), scale=df["resid_m5"].std())
    ax_hist.plot(kde_x, norm_pdf, "r--", linewidth=1.5, label="Theoretical Normal")
    ax_hist.set_xlabel("Residual Value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("(C) Residual Distribution")
    ax_hist.legend(loc="upper right")

    # 3D: Channel Cluster Residual Dispersion
    ax_clust = axes[1, 1]
    top_channels = df["channel_name"].value_counts().head(12).index
    df_top_clust = df[df["channel_name"].isin(top_channels)]
    sns_order = df_top_clust.groupby("channel_name")["resid_m5"].median().sort_values().index
    channel_resids = [df[df["channel_name"] == c]["resid_m5"].values for c in sns_order]
    
    ax_clust.boxplot(channel_resids, tick_labels=[c[:14] for c in sns_order], vert=False, patch_artist=True,
                     boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                     medianprops=dict(color="red", linewidth=1.5))
    ax_clust.axvline(0, color="black", linestyle=":", alpha=0.7)
    ax_clust.set_xlabel("Residuals $\\hat{\\varepsilon}_{it}$")
    ax_clust.set_title("(D) Within-Channel Residual Dispersion (Sample Creators)")
    ax_clust.tick_params(axis="y", labelsize=8)

    plt.suptitle("Econometric Residual Diagnostics: Validating OLS Assumptions for Model 5", y=1.005)
    plt.tight_layout()
    plot3_path = PLOTS_DIR / "03_residual_diagnostics.png"
    plt.savefig(plot3_path, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT 4: Empirical Payoff Response Curves with Confidence Bands
    # =========================================================================
    logger.info("Generating Plot 4: Payoff Response Curves with Confidence Bands...")
    fig, (ax_curve, ax_dist) = plt.subplots(
        2, 1, figsize=(9, 8.5), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08}
    )

    x_grid = np.linspace(0.0, 1.0, 300)
    p10_peer = float(df["x_bar_roll90"].quantile(0.10))
    p50_peer = float(df["x_bar_roll90"].median())
    p90_peer = float(df["x_bar_roll90"].quantile(0.90))

    mean_sub = float(df["log_subscribers"].mean())
    mean_dur = float(df["log_duration"].mean())
    mean_age = float(df["age_days"].mean())

    def calc_payoff_and_se(x_arr, peer_val):
        y_hat = []
        se_arr = []
        for x_val in x_arr:
            # Linear combination vector: const, x, x2, x_bar, x_inter, log_sub, log_dur, age
            w = np.array([1.0, x_val, x_val**2, peer_val, x_val * peer_val, mean_sub, mean_dur, mean_age])
            pred = np.dot(w, m5.params.values)
            var_pred = np.dot(w, np.dot(cov_m5.values, w))
            se_pred = np.sqrt(max(0, var_pred))
            y_hat.append(pred)
            se_arr.append(se_pred)
        return np.array(y_hat), np.array(se_arr)

    y_p50, se_p50 = calc_payoff_and_se(x_grid, p50_peer)
    y_p10, _ = calc_payoff_and_se(x_grid, p10_peer)
    y_p90, _ = calc_payoff_and_se(x_grid, p90_peer)

    # Plot response curves
    ax_curve.plot(x_grid, y_p10, label=f"Low Peer Intensity (10th %ile: $\\bar{{x}}_{{-i}}={p10_peer:.3f}$)", color="#2ca02c", linewidth=2.0)
    ax_curve.plot(x_grid, y_p50, label=f"Median Peer Intensity (50th %ile: $\\bar{{x}}_{{-i}}={p50_peer:.3f}$)", color="#1f77b4", linewidth=2.5)
    ax_curve.plot(x_grid, y_p90, label=f"High Peer Intensity (90th %ile: $\\bar{{x}}_{{-i}}={p90_peer:.3f}$)", color="#d62728", linewidth=2.0, linestyle="--")

    # Shaded 95% Confidence Band around Median Peer curve
    ax_curve.fill_between(x_grid, y_p50 - 1.96 * se_p50, y_p50 + 1.96 * se_p50, color="#1f77b4", alpha=0.15, label=r"95% Confidence Interval ($\pm 1.96$ Cluster SE)")

    # Mark Peak at median peer
    denom = 2 * abs(b2)
    br_p50 = (b1 + b5 * p50_peer) / denom
    payoff_peak, _ = calc_payoff_and_se(np.array([br_p50]), p50_peer)
    ax_curve.plot(br_p50, payoff_peak[0], "ko", markersize=8, label=f"Point-Estimate Peak at $\\bar{{x}}_{{50}}$: $x^*={br_p50:.3f}$")
    ax_curve.axvline(br_p50, color="gray", linestyle=":", alpha=0.8)

    ax_curve.set_ylabel("Expected Video Payoff $\\pi_i(x_i, \\bar{x}_{-i}) = \\mathbb{E}[\\log(V_i)]$", fontsize=11)
    ax_curve.set_title("Empirical Payoff Function $\\pi_i(x_i, \\bar{x}_{-i})$ with 95% Confidence Band\nPoint Estimates Show Concavity $(\\beta_2 = -9.168)$, but Confidence Band Reflects Statistical Uncertainty", fontsize=11, pad=10)
    ax_curve.legend(loc="lower center", fontsize=9.0, frameon=True)
    ax_curve.set_ylim(bottom=y_p50.min() - 0.8, top=y_p50.max() + 1.5)

    # Text box flagging statistical uncertainty
    uncert_box = (
        "STATISTICAL AUDIT NOTE:\n"
        "• Point estimates: beta_1 = +8.30, beta_2 = -9.17 (concave).\n"
        "• 95% CI for beta_2 = [-25.71, +7.37] (crosses zero, p = 0.28).\n"
        "• Concavity is mathematically satisfied on point estimates,\n"
        "  but NOT statistically established at alpha = 0.05 in Model 5."
    )
    ax_curve.text(
        0.03, 0.96, uncert_box, transform=ax_curve.transAxes,
        fontsize=8.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6", edgecolor="#d4b106", alpha=0.95)
    )

    # Bottom Panel: Empirical Distribution of Click Appeal
    ax_dist.hist(df["x"], bins=25, color="#4a7bb0", edgecolor="black", alpha=0.7, density=True)
    ax_dist.axvline(df["x"].mean(), color="black", linestyle="-", linewidth=1.5, label=f"Empirical Mean: {df['x'].mean():.3f}")
    ax_dist.set_xlabel("Own Click Appeal Score ($x_{it} \\in [0, 1]$)", fontsize=11)
    ax_dist.set_ylabel("Empirical\nDensity", fontsize=9)
    ax_dist.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plot4_path = PLOTS_DIR / "04_payoff_response_curves.png"
    plt.savefig(plot4_path)
    plt.close()

    # =========================================================================
    # PLOT 5: 2D Strategic Payoff Surface / Contour Map
    # =========================================================================
    logger.info("Generating Plot 5: Strategic Interaction Heatmap...")
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

    x_vals = np.linspace(0.0, 1.0, 150)
    peer_vals = np.linspace(0.2, 0.6, 150)
    X_grid, Y_grid = np.meshgrid(x_vals, peer_vals)

    mean_controls = (
        b0 + b4 * mean_sub + g1 * mean_dur + g2 * mean_age
    )
    Z_payoff = (
        b1 * X_grid + b2 * (X_grid ** 2) +
        b3 * Y_grid + b5 * (X_grid * Y_grid) +
        mean_controls
    )

    contour = ax.contourf(X_grid, Y_grid, Z_payoff, levels=25, cmap="Spectral_r", alpha=0.85)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Conditional Expected Payoff $\\mathbb{E}[\\log(V)]$", fontsize=11)

    cs = ax.contour(X_grid, Y_grid, Z_payoff, levels=12, colors="black", linewidths=0.5, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Marginal return ridge: partial d_pi / d_xi = b1 + 2*b2*x + b5*peer = 0
    br_curve = (b1 + b5 * peer_vals) / (2 * abs(b2))
    ax.plot(br_curve, peer_vals, "b-", linewidth=2.8, label="Marginal Return Ridge: $\\partial\\pi_i/\\partial x_i = 0$ (Point Est)")
    ax.plot(peer_vals, peer_vals, "k--", linewidth=1.5, label="Symmetric Profile Line ($x_i = \\bar{x}_{-i}$)")

    # Scatter of observed points
    ax.scatter(df["x"], df["x_bar_roll90"], color="black", alpha=0.35, s=20, label="Observed YouTube Videos ($N=186$)")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.2, 0.6)
    ax.set_xlabel("Own Click Appeal Strategy ($x_i$)", fontsize=11)
    ax.set_ylabel("Competitor Average Click Appeal ($\\bar{x}_{-i}$)", fontsize=11)
    ax.set_title("Empirical Strategic Payoff Landscape in Action Space $[0, 1] \\times [0.2, 0.6]$\nContour Map of $\\pi_i(x_i, \\bar{x}_{-i})$ Derived from Model 5 Point Estimates", fontsize=11, pad=12)
    ax.legend(loc="upper left", fontsize=9.0, framealpha=0.9)

    # Uncertainty note
    ax.text(
        0.03, 0.05,
        "NOTE: Surface derived from point estimates (beta_5 = -0.78, p = 0.99).\nStrategic interaction is statistically indistinguishable from zero.",
        transform=ax.transAxes, fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85)
    )

    plt.tight_layout()
    plot5_path = PLOTS_DIR / "05_peer_interaction_heatmap.png"
    plt.savefig(plot5_path)
    plt.close()

    # =========================================================================
    # PLOT 6: Multicollinearity Diagnostics (Uncentered vs Centered)
    # =========================================================================
    logger.info("Generating Plot 6: VIF & Collinearity Diagnostics...")
    df_vif = pd.read_csv(VIF_CSV)
    m5_vifs = df_vif[df_vif["model_name"] == "Model_5_FullProposed"]
    m5c_vifs = df_vif[df_vif["model_name"] == "Model_5_Centered"]

    fig, (ax_vif1, ax_vif2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    bars1 = ax_vif1.barh(m5_vifs["variable"], m5_vifs["vif"], color="#d62728", alpha=0.85)
    ax_vif1.axvline(10, color="black", linestyle="--", linewidth=1.2, label="VIF = 10 Threshold")
    ax_vif1.set_xlabel("Variance Inflation Factor (Log Scale)", fontsize=10)
    ax_vif1.set_xscale("log")
    ax_vif1.set_title("(A) Raw Uncentered Specification\nHigh Structural Collinearity between $x$, $x^2$, and $x \\cdot \\bar{x}$", fontsize=11)
    ax_vif1.legend(loc="lower right")
    for bar in bars1:
        w = bar.get_width()
        ax_vif1.text(w * 1.1, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va="center", fontsize=8.5)

    bars2 = ax_vif2.barh(m5c_vifs["variable"], m5c_vifs["vif"], color="#2ca02c", alpha=0.85)
    ax_vif2.axvline(10, color="black", linestyle="--", linewidth=1.2, label="VIF = 10 Threshold")
    ax_vif2.set_xlabel("Variance Inflation Factor (Linear Scale)", fontsize=10)
    ax_vif2.set_xlim(0, 5)
    ax_vif2.set_title("(B) Mean-Centered Specification\nStructural Collinearity Eliminated (All VIFs < 1.30)", fontsize=11)
    ax_vif2.legend(loc="upper right")
    for bar in bars2:
        w = bar.get_width()
        ax_vif2.text(w + 0.08, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=9)

    plt.suptitle("Multicollinearity Diagnostics: Structural Polynomial Collinearity vs. Centering Resolution", y=1.02)
    plt.tight_layout()
    plot6_path = PLOTS_DIR / "06_variable_importance_vif.png"
    plt.savefig(plot6_path, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT 7: Grouped Cross-Validation Channel Error Distribution
    # =========================================================================
    logger.info("Generating Plot 7: Grouped CV Channel Errors...")
    fig, ax_cv = plt.subplots(figsize=(10, 6), dpi=300)

    channels = df["channel_id"].unique()
    channel_errors_m5 = []
    channel_errors_m2 = []
    channel_names = []

    for c in channels:
        sub = df[df["channel_id"] == c]
        cname = sub["channel_name"].iloc[0]
        rmse5 = np.sqrt(np.mean((sub["log_views"] - sub["oos_pred_Model_5_FullProposed"]) ** 2))
        rmse2 = np.sqrt(np.mean((sub["log_views"] - sub["oos_pred_Model_2_Quadratic"]) ** 2))
        channel_errors_m5.append(rmse5)
        channel_errors_m2.append(rmse2)
        channel_names.append(cname)

    df_err = pd.DataFrame({
        "channel": channel_names,
        "rmse_m5": channel_errors_m5,
        "rmse_m2": channel_errors_m2
    }).sort_values("rmse_m5")

    idx_pos = np.arange(len(df_err))
    ax_cv.bar(idx_pos - 0.2, df_err["rmse_m2"], 0.4, label="Model 2 (Quadratic Appeal Only)", color="#d62728", alpha=0.7)
    ax_cv.bar(idx_pos + 0.2, df_err["rmse_m5"], 0.4, label="Model 5 (Full Proposed Model with Audience Scale)", color="#1f77b4", alpha=0.85)

    ax_cv.set_xticks(idx_pos[::2])
    ax_cv.set_xticklabels([c[:12] for c in df_err["channel"].iloc[::2]], rotation=45, ha="right", fontsize=8.5)
    ax_cv.set_ylabel("Out-of-Sample RMSE per Channel", fontsize=11)
    ax_cv.set_title("Leave-One-Channel-Out Cross-Validation: Generalization Error across 38 Channels\nDemonstrating How Audience Scale Rescues Out-of-Sample Prediction", fontsize=12, pad=12)
    ax_cv.axhline(np.mean(df_err["rmse_m5"]), color="blue", linestyle="--", linewidth=1.2, label=f"Mean M5 OOS RMSE: {np.mean(df_err['rmse_m5']):.2f}")
    ax_cv.axhline(np.mean(df_err["rmse_m2"]), color="red", linestyle=":", linewidth=1.2, label=f"Mean M2 OOS RMSE: {np.mean(df_err['rmse_m2']):.2f}")
    ax_cv.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    plot7_path = PLOTS_DIR / "07_grouped_cv_channel_errors.png"
    plt.savefig(plot7_path, bbox_inches="tight")
    plt.close()

    logger.info("All 7 diagnostic and publication plots generated successfully.")
    print("\n================ DIAGNOSTIC PLOTS GENERATED ================")
    print(f"  1. {plot1_path.name}  (Progression Comparison: In-Sample vs OOS LOCO-CV)")
    print(f"  2. {plot2_path.name}  (Observed vs Fitted Log Views for Model 5)")
    print(f"  3. {plot3_path.name}  (4-Panel Residual Diagnostics for Model 5)")
    print(f"  4. {plot4_path.name}  (Payoff Curves with 95% Confidence Band & Uncertainty)")
    print(f"  5. {plot5_path.name}  (2D Strategic Interaction Landscape with Caveats)")
    print(f"  6. {plot6_path.name}  (VIF Multicollinearity Diagnostics: Raw vs Centered)")
    print(f"  7. {plot7_path.name}  (Grouped-CV Out-of-Sample Error Distribution)")
    print("============================================================\n")


if __name__ == "__main__":
    generate_all_plots()
