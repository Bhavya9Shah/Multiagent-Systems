"""
Plotting Script: Publication-Quality Theoretical Game Theory Figures
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Component: Game-Theoretic Formulation & Price of Anarchy Visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from game_theory_solver import PotentialGame

# Set publication style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "figure.dpi": 300,
})


def plot_01_best_response_and_nash(empirical_game: PotentialGame, congestion_game: PotentialGame, save_path: str):
    """Plot 1: Best-Response functions and Nash equilibrium fixed points."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x_grid = np.linspace(0, 1, 300)

    # Panel A: Regime A (Empirical)
    ax = axes[0]
    br_empirical = empirical_game.best_response(x_grid)
    ax.plot(x_grid, x_grid, "k--", alpha=0.6, label="45° Line (Symmetric Profile $x_i = \\bar{x}_{-i}$)")
    ax.plot(x_grid, br_empirical, color="#1f77b4", label=f"Best Response $\\text{{BR}}_i(\\bar{{x}}_{{-i}})$ (slope $s = {empirical_game.best_response_slope:.3f}$)")
    ax.scatter([empirical_game.x_nash], [empirical_game.x_nash], color="#d62728", s=90, zorder=5,
               label=f"Nash Equilibrium $x^* = {empirical_game.x_nash:.4f}$")
    ax.axvline(empirical_game.x_nash, color="#d62728", linestyle=":", alpha=0.6)
    ax.axvline(empirical_game.x_so, color="#2ca02c", linestyle=":", alpha=0.6)
    ax.scatter([empirical_game.x_so], [empirical_game.x_so], color="#2ca02c", s=70, marker="s", zorder=5,
               label=f"Social Optimum $x^{{\\text{{SO}}}} = {empirical_game.x_so:.4f}$")
    ax.set_title("Panel A: Regime A (Empirical Point Estimates)\n$\\beta_1=8.302, \\beta_2=-9.168, \\beta_5=-0.782$ (Substitutes)")
    ax.set_xlabel("Competitor Leave-One-Out Average $\\bar{x}_{-i}$")
    ax.set_ylabel("Focal Creator Strategy $x_i$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.95)

    # Panel B: Regime B (Congestion)
    ax = axes[1]
    br_congestion = congestion_game.best_response(x_grid)
    ax.plot(x_grid, x_grid, "k--", alpha=0.6, label="45° Line ($x_i = \\bar{x}_{-i}$)")
    ax.plot(x_grid, br_congestion, color="#ff7f0e", label=f"Best Response $\\text{{BR}}_i(\\bar{{x}}_{{-i}})$ (slope $s = {congestion_game.best_response_slope:.3f}$)")
    ax.scatter([congestion_game.x_nash], [congestion_game.x_nash], color="#d62728", s=90, zorder=5,
               label=f"Nash Equilibrium $x^* = {congestion_game.x_nash:.4f}$")
    ax.scatter([congestion_game.x_so], [congestion_game.x_so], color="#2ca02c", s=70, marker="s", zorder=5,
               label=f"Social Optimum $x^{{\\text{{SO}}}} = {congestion_game.x_so:.4f}$")
    ax.axvline(congestion_game.x_nash, color="#d62728", linestyle=":", alpha=0.6)
    ax.axvline(congestion_game.x_so, color="#2ca02c", linestyle=":", alpha=0.6)
    # Highlight overshooting
    ax.annotate("Attention Congestion Overshoot\n($x^* > x^{\\text{SO}}$)",
                xy=(congestion_game.x_so, 0.4), xytext=(0.05, 0.65),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2cc", edgecolor="#d6b656"))
    ax.set_title("Panel B: Regime B (Theoretical Attention Congestion)\n$\\beta_1=8.302, \\beta_2=-9.168, \\beta_5=+0.782$ (Complements)")
    ax.set_xlabel("Competitor Leave-One-Out Average $\\bar{x}_{-i}$")
    ax.set_ylabel("Focal Creator Strategy $x_i$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 1: {save_path}")


def plot_02_welfare_nash_vs_social_optimum(empirical_game: PotentialGame, congestion_game: PotentialGame, save_path: str):
    """Plot 2: Symmetric Welfare curves, Nash vs Social Optimum, and Price of Anarchy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x_grid = np.linspace(0, 1, 300)

    # Panel A: Regime A
    ax = axes[0]
    w_emp = empirical_game.social_welfare_symmetric(x_grid)
    w_nash = empirical_game.social_welfare_symmetric(empirical_game.x_nash)
    w_so = empirical_game.social_welfare_symmetric(empirical_game.x_so)
    ax.plot(x_grid, w_emp, color="#1f77b4", label="Symmetric Welfare $\\pi(x, x)$")
    ax.scatter([empirical_game.x_nash], [w_nash], color="#d62728", s=90, zorder=5,
               label=f"Nash Outcome ($x^*={empirical_game.x_nash:.3f}, \\pi={w_nash:.2f}$)")
    ax.scatter([empirical_game.x_so], [w_so], color="#2ca02c", s=90, marker="s", zorder=5,
               label=f"Social Optimum ($x^{{\\text{{SO}}}}={empirical_game.x_so:.3f}, \\pi={w_so:.2f}$)")
    ax.axvline(empirical_game.x_nash, color="#d62728", linestyle=":", alpha=0.7)
    ax.axvline(empirical_game.x_so, color="#2ca02c", linestyle=":", alpha=0.7)
    gap_a = w_so - w_nash
    ax.fill_between(x_grid, w_nash, w_emp, where=(x_grid >= empirical_game.x_nash) & (x_grid <= empirical_game.x_so),
                    color="#2ca02c", alpha=0.15, label=f"Welfare Gap = {gap_a:.3f}\n$\\text{{PoA}} = {empirical_game.price_of_anarchy:.3f}$")
    ax.set_title("Panel A: Regime A (Empirical Estimates)\nUnder-Investment relative to Optimum ($E > 0$)")
    ax.set_xlabel("Common Click Appeal Score $x$")
    ax.set_ylabel("Expected Log-Views $\\pi(x, x)$")
    ax.legend(loc="lower center", framealpha=0.95)

    # Panel B: Regime B
    ax = axes[1]
    w_cong = congestion_game.social_welfare_symmetric(x_grid)
    w_nash_b = congestion_game.social_welfare_symmetric(congestion_game.x_nash)
    w_so_b = congestion_game.social_welfare_symmetric(congestion_game.x_so)
    ax.plot(x_grid, w_cong, color="#ff7f0e", label="Symmetric Welfare $\\pi(x, x)$")
    ax.scatter([congestion_game.x_nash], [w_nash_b], color="#d62728", s=90, zorder=5,
               label=f"Nash Outcome ($x^*={congestion_game.x_nash:.3f}, \\pi={w_nash_b:.2f}$)")
    ax.scatter([congestion_game.x_so], [w_so_b], color="#2ca02c", s=90, marker="s", zorder=5,
               label=f"Social Optimum ($x^{{\\text{{SO}}}}={congestion_game.x_so:.3f}, \\pi={w_so_b:.2f}$)")
    ax.axvline(congestion_game.x_nash, color="#d62728", linestyle=":", alpha=0.7)
    ax.axvline(congestion_game.x_so, color="#2ca02c", linestyle=":", alpha=0.7)
    gap_b = w_so_b - w_nash_b
    ax.fill_between(x_grid, w_nash_b, w_cong, where=(x_grid >= congestion_game.x_so) & (x_grid <= congestion_game.x_nash),
                    color="#d62728", alpha=0.15, label=f"Congestion Gap = {gap_b:.3f}\n$\\text{{PoA}} = {congestion_game.price_of_anarchy:.3f}$")
    ax.set_title("Panel B: Regime B (Attention Congestion $H_1: E < 0$)\nClickbait Arms Race Overshoots Optimum ($x^* > x^{\\text{SO}}$)")
    ax.set_xlabel("Common Click Appeal Score $x$")
    ax.set_ylabel("Expected Log-Views $\\pi(x, x)$")
    ax.legend(loc="lower center", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 2: {save_path}")


def plot_03_potential_landscape(empirical_game: PotentialGame, save_path: str):
    """Plot 3: 2D Contour map of Exact Potential Function Phi(x1, x2)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 200)
    X1, X2 = np.meshgrid(grid, grid)
    
    # Evaluate potential on 2-agent slice
    b1 = empirical_game.beta_1
    b2 = empirical_game.beta_2
    b5 = empirical_game.beta_5
    
    # Phi(x_1, x_2) = (b1*x1 + b2*x1^2) + (b1*x2 + b2*x2^2) + b5*x1*x2
    Phi = (b1 * X1 + b2 * (X1 ** 2)) + (b1 * X2 + b2 * (X2 ** 2)) + b5 * X1 * X2
    
    cp = ax.contourf(X1, X2, Phi, levels=25, cmap="viridis")
    fig.colorbar(cp, ax=ax, label="Potential Function $\\Phi(x_1, x_2)$")
    
    # Nash equilibrium point
    x_n = empirical_game.x_nash
    ax.scatter([x_n], [x_n], color="#d62728", s=120, edgecolors="white", linewidths=1.5, zorder=6,
               label=f"Unique Maximizer: Nash $(x^*, x^*) = ({x_n:.3f}, {x_n:.3f})$")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.7, label="Symmetric Diagonal ($x_1 = x_2$)")
    
    ax.set_title(r"Exact Potential Game Landscape: Concavity of $\Phi(x)$" + "\n" + r"Strictly Negative Hessian Eigenvalues ($\lambda_1 < 0, \lambda_2 < 0$)")
    ax.set_xlabel("Creator 1 Click Appeal $x_1$")
    ax.set_ylabel("Creator 2 Click Appeal $x_2$")
    ax.legend(loc="upper left", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 3: {save_path}")


def plot_04_poa_sensitivity_heatmap(save_path: str):
    """Plot 4: 2D Heatmaps of Price of Anarchy across (beta_2, beta_5) for beta_3 > 0 and beta_3 < 0."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    b2_vals = np.linspace(-20, -3, 100)
    b5_vals = np.linspace(-5, 5, 100)
    B2, B5 = np.meshgrid(b2_vals, b5_vals)
    b1 = 8.3020

    panels = [
        {"ax": axes[0], "b3": 3.8021, "title": r"Panel A: Empirical Peer Spillover ($\beta_3 = +3.80$)" + "\n" + r"Empirical Point ($\text{PoA} = 1.3515$)", "show_marker": True},
        {"ax": axes[1], "b3": -3.8021, "title": r"Panel B: Hypothetical Congestion ($\beta_3 = -3.80$)" + "\n" + r"Attention Overshoot Inefficiency", "show_marker": False},
    ]

    for p in panels:
        ax = p["ax"]
        b3 = p["b3"]
        PoA = np.zeros_like(B2)
        for i in range(len(b5_vals)):
            for j in range(len(b2_vals)):
                b2 = B2[i, j]
                b5 = B5[i, j]
                abs_b2 = abs(b2)
                B = abs_b2 - b5
                # Stability & concavity
                if B > 0 and abs(b5) < (2.0 * abs_b2):
                    denom_n = 2.0 * abs_b2 - b5
                    x_n = np.clip(b1 / denom_n, 0.0, 1.0)
                    denom_s = 2.0 * B
                    x_s = np.clip((b1 + b3) / denom_s, 0.0, 1.0)
                    w_s = (b1 + b3) * x_s + (b2 + b5) * (x_s ** 2)
                    w_n = (b1 + b3) * x_n + (b2 + b5) * (x_n ** 2)
                    PoA[i, j] = np.exp(w_s - w_n)
                else:
                    PoA[i, j] = np.nan

        im = ax.imshow(PoA, extent=[b2_vals[0], b2_vals[-1], b5_vals[0], b5_vals[-1]],
                       origin="lower", aspect="auto", cmap="magma", vmin=1.0, vmax=3.0)
        fig.colorbar(im, ax=ax, label=r"Price of Anarchy ($\text{PoA} = V(x^{\text{SO}}) / V(x^*)$)")

        if p["show_marker"]:
            ax.scatter([-9.1677], [-0.7822], color="#00ffcc", s=130, marker="o", edgecolors="black", linewidths=1.5, zorder=6,
                       label=r"Empirical Point ($\beta_2 = -9.17, \beta_5 = -0.78$)")
            ax.legend(loc="upper left", framealpha=0.9)

        ax.set_title(p["title"])
        ax.set_xlabel(r"Diminishing Returns Parameter $\beta_2$ (Curvature)")
        ax.set_ylabel(r"Strategic Interaction $\beta_5$ (Complements / Substitutes)")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 4: {save_path}")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    json_path = os.path.abspath(os.path.join(base_dir, "..", "payoff_estimation", "results", "game_theory_coefficients.json"))
    
    empirical_game = PotentialGame.from_json(json_path, n_creators=38)
    congestion_game = PotentialGame(
        beta_1=8.301959,
        beta_2=-9.167702,
        beta_3=-3.802120,
        beta_5=0.782244,
        n_creators=38,
        name="Congestion Regime",
    )
    
    plot_01_best_response_and_nash(empirical_game, congestion_game, os.path.join(plots_dir, "01_best_response_and_nash.png"))
    plot_02_welfare_nash_vs_social_optimum(empirical_game, congestion_game, os.path.join(plots_dir, "02_welfare_nash_vs_social_optimum.png"))
    plot_03_potential_landscape(empirical_game, os.path.join(plots_dir, "03_potential_function_landscape.png"))
    plot_04_poa_sensitivity_heatmap(os.path.join(plots_dir, "04_poa_sensitivity_heatmap.png"))
    
    print("\nAll Part 3 theoretical figures generated successfully!")


if __name__ == "__main__":
    main()

