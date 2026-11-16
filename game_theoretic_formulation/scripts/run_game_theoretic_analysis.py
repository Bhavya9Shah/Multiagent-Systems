"""
Execution Script: Game-Theoretic Formulation & Price of Anarchy Analysis
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar

Performs:
1. Ingestion of empirical coefficients from payoff_estimation/results/game_theory_coefficients.json
2. Exact analytical derivation of Nash Equilibrium, Social Optimum, and Price of Anarchy
3. Parametric bootstrap (10,000 draws) using cluster-robust covariance matrix to construct 95% CIs
4. Systematic sensitivity grid analysis across curvature (beta_2), peer interaction (beta_5), and congestion (beta_3)
5. Export of machine-readable payloads for downstream simulation modules
"""

import os
import json
import numpy as np
import pandas as pd
from game_theory_solver import PotentialGame


def load_empirical_inputs(json_path: str):
    """Load empirical coefficients and cluster-robust covariance matrix."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if "point_estimates_uncentered_model_5" not in data:
        raise KeyError("Missing required key 'point_estimates_uncentered_model_5' in JSON")
    uncentered = data["point_estimates_uncentered_model_5"]
    
    if "parameter_covariance_matrix_cluster_robust" not in data:
        raise KeyError("Missing 'parameter_covariance_matrix_cluster_robust' in JSON")
    cov_dict = data["parameter_covariance_matrix_cluster_robust"]
    
    # Extract parameter vectors for bootstrap: [beta_1, beta_2, beta_3, beta_5]
    # Corresponding keys in covariance matrix: 'x', 'x2', 'x_bar_roll90', 'x_inter_roll90'
    key_mapping = {
        "beta_1": "x",
        "beta_2": "x2",
        "beta_3": "x_bar_roll90",
        "beta_5": "x_inter_roll90",
    }
    
    for ki, row_k in key_mapping.items():
        if row_k not in uncentered and f"{ki}_own_click_appeal" not in uncentered:
            pass  # uncentered verified below
    
    mean_vec = np.array([
        uncentered["beta_1_own_click_appeal"],
        uncentered["beta_2_quadratic_own_appeal"],
        uncentered["beta_3_competitor_click_appeal"],
        uncentered["beta_5_strategic_interaction"],
    ], dtype=float)
    
    cov_sub = np.zeros((4, 4))
    param_keys = ["beta_1", "beta_2", "beta_3", "beta_5"]
    for i, ki in enumerate(param_keys):
        row_key = key_mapping[ki]
        if row_key not in cov_dict:
            raise KeyError(f"Missing row key '{row_key}' in covariance dictionary.")
        for j, kj in enumerate(param_keys):
            col_key = key_mapping[kj]
            if col_key not in cov_dict[row_key]:
                raise KeyError(f"Missing col key '{col_key}' in covariance dictionary for row '{row_key}'.")
            cov_sub[i, j] = cov_dict[row_key][col_key]
            
    # Symmetrize
    cov_sub = 0.5 * (cov_sub + cov_sub.T)
    min_eig = float(np.min(np.real(np.linalg.eigvals(cov_sub))))
    eigenvalue_shift_triggered = False
    shift_amount = 0.0
    if min_eig < 1e-8:
        shift_amount = float(abs(min_eig) + 1e-6)
        cov_sub += shift_amount * np.eye(4)
        eigenvalue_shift_triggered = True
        
    diag_info = {
        "covariance_parametrization": "Uncentered Model 5 (param_cov from m5, matching uncentered SEs)",
        "min_eigenvalue_before_shift": min_eig,
        "eigenvalue_shift_triggered": eigenvalue_shift_triggered,
        "shift_amount": shift_amount,
    }
        
    return uncentered, cov_sub, data, diag_info


def run_parametric_bootstrap(mean_vec: np.ndarray, cov_mat: np.ndarray, n_draws: int = 10000, seed: int = 42):
    """
    Parametric bootstrap for Nash, Social Optimum, and PoA over the cluster-robust parameter distribution.
    Draws beta ~ N(beta_hat, Sigma_hat) and accounts for finite-sample sampling variation.
    Evaluates:
      - Over all draws: P(b2 < 0), P(B > 0), P(|b5| < 2|b2|), P(well-posed), P(E < 0)
      - Conditional on well-posedness: percentiles for x_N, x_SO, E, Delta, and PoA (by definition, no cap)
      - Fraction of well-posed draws with clipped equilibria
    """
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean_vec, cov_mat, size=n_draws)
    
    b1_all = draws[:, 0]
    b2_all = draws[:, 1]
    b3_all = draws[:, 2]
    b5_all = draws[:, 3]
    
    # Probabilities over ALL 10,000 draws
    cond_b2_neg = b2_all < 0.0
    cond_B_pos = (np.abs(b2_all) - b5_all) > 0.0
    cond_contraction = np.abs(b5_all) < (2.0 * np.abs(b2_all))
    cond_well_posed = cond_b2_neg & cond_B_pos & cond_contraction
    
    p_b2_neg = float(np.mean(cond_b2_neg))
    p_B_pos = float(np.mean(cond_B_pos))
    p_contraction_all = float(np.mean(cond_contraction))
    p_well_posed = float(np.mean(cond_well_posed))
    
    # Compute Nash and SO for well-posed draws
    valid_indices = np.where(cond_well_posed)[0]
    n_valid = len(valid_indices)
    
    records_valid = []
    e_all = []
    
    # Track E across all draws where Nash denominator != 0
    denom_nash_all = 2.0 * np.abs(b2_all) - b5_all
    safe_nash_all = np.where(denom_nash_all != 0.0, b1_all / np.where(denom_nash_all != 0.0, denom_nash_all, 1.0), 0.5)
    clipped_nash_all = np.clip(safe_nash_all, 0.0, 1.0)
    E_all_draws = b3_all + b5_all * clipped_nash_all
    p_E_neg_all = float(np.mean(E_all_draws < 0.0))
    
    n_clipped_nash = 0
    n_clipped_so = 0
    n_clipped_any = 0
    
    for idx in valid_indices:
        b1 = b1_all[idx]
        b2 = b2_all[idx]
        b3 = b3_all[idx]
        b5 = b5_all[idx]
        B = np.abs(b2) - b5
        
        denom_n = 2.0 * np.abs(b2) - b5
        x_n_raw = b1 / denom_n
        x_n = float(np.clip(x_n_raw, 0.0, 1.0))
        
        denom_s = 2.0 * B
        x_s_raw = (b1 + b3) / denom_s
        x_s = float(np.clip(x_s_raw, 0.0, 1.0))
        
        is_n_clipped = bool(x_n == 0.0 or x_n == 1.0)
        is_s_clipped = bool(x_s == 0.0 or x_s == 1.0)
        if is_n_clipped:
            n_clipped_nash += 1
        if is_s_clipped:
            n_clipped_so += 1
        if is_n_clipped or is_s_clipped:
            n_clipped_any += 1
            
        E = b3 + b5 * x_n
        delta = x_s - x_n
        
        # PoA by definition with clipped values: exp(pi(x_s, x_s) - pi(x_n, x_n))
        # pi(x, x) = (b1 + b3)*x + (b2 + b5)*x^2
        w_s = (b1 + b3) * x_s + (b2 + b5) * (x_s ** 2)
        w_n = (b1 + b3) * x_n + (b2 + b5) * (x_n ** 2)
        poa = float(np.exp(w_s - w_n))
        
        records_valid.append({
            "beta_1": b1,
            "beta_2": b2,
            "beta_3": b3,
            "beta_5": b5,
            "x_nash_clipped": x_n,
            "x_so_clipped": x_s,
            "marginal_E": E,
            "delta": delta,
            "poa": poa,
            "is_n_clipped": is_n_clipped,
            "is_s_clipped": is_s_clipped,
        })
        
    df_valid = pd.DataFrame(records_valid)
    p_E_neg_valid = float(np.mean(df_valid["marginal_E"] < 0.0)) if n_valid > 0 else 0.0
    
    # Calculate bootstrap percentiles conditional on well-posedness
    percentiles = [2.5, 25, 50, 75, 97.5]
    summary_boot = {
        "diagnostics_over_all_draws": {
            "n_draws_total": n_draws,
            "P_beta_2_negative": p_b2_neg,
            "P_welfare_concave_B_pos": p_B_pos,
            "P_contraction_all": p_contraction_all,
            "P_well_posed_all_conditions": p_well_posed,
            "valid_draws_count": n_valid,
            "P_marginal_E_negative_all_draws": p_E_neg_all,
            "P_marginal_E_negative_conditional_well_posed": p_E_neg_valid,
            "fraction_valid_with_clipped_x_nash": (n_clipped_nash / n_valid) if n_valid > 0 else 0.0,
            "fraction_valid_with_clipped_x_so": (n_clipped_so / n_valid) if n_valid > 0 else 0.0,
            "fraction_valid_with_any_clipped": (n_clipped_any / n_valid) if n_valid > 0 else 0.0,
        },
        "conditional_on_well_posedness": {},
    }
    
    for col in ["x_nash_clipped", "x_so_clipped", "marginal_E", "delta", "poa"]:
        vals = np.percentile(df_valid[col], percentiles)
        summary_boot["conditional_on_well_posedness"][col] = {
            "p025": float(vals[0]),
            "p25": float(vals[1]),
            "median": float(vals[2]),
            "p75": float(vals[3]),
            "p975": float(vals[4]),
            "mean": float(np.mean(df_valid[col])),
            "std": float(np.std(df_valid[col])),
        }
        
    return df_valid, summary_boot


def build_sensitivity_analysis_grid(b1_base: float = 8.301959):
    """
    Construct sensitivity grid spanning:
      beta_2 in [-20.0, -15.0, -9.1677, -5.0, -3.0]
      beta_5 in [-5.0, -0.7822, 0.0, 3.0, 5.0]
      beta_3 in [-5.0, -3.8021, 0.0, 3.8021, 5.0]
    """
    b2_list = [-20.0, -15.0, -9.1677, -5.0, -3.0]
    b5_list = [-5.0, -0.7822, 0.0, 3.0, 5.0]
    b3_list = [-5.0, -3.8021, 0.0, 3.8021, 5.0]
    
    rows = []
    for b2 in b2_list:
        for b5 in b5_list:
            for b3 in b3_list:
                game = PotentialGame(
                    beta_1=b1_base,
                    beta_2=b2,
                    beta_3=b3,
                    beta_5=b5,
                    n_creators=38,
                )
                wp = game.check_well_posedness()
                is_valid = wp["all_conditions_satisfied"]
                
                if is_valid:
                    x_n = game.x_nash
                    x_s = game.x_so
                    E = game.marginal_external_congestion
                    delta = game.nash_so_gap
                    poa = game.price_of_anarchy
                    lam_max_01 = game.mediator_max_lambda(0.01)
                    r_rec_01 = game.welfare_recovery(lam_max_01)
                else:
                    x_n = np.nan
                    x_s = np.nan
                    E = np.nan
                    delta = np.nan
                    poa = np.nan
                    lam_max_01 = np.nan
                    r_rec_01 = np.nan
                    
                rows.append({
                    "beta_1": b1_base,
                    "beta_2": b2,
                    "beta_3": b3,
                    "beta_5": b5,
                    "well_posed": is_valid,
                    "concavity_satisfied": wp["condition_1_individual_concavity"]["satisfied"],
                    "contraction_satisfied": wp["condition_2_contraction_unique_nash"]["satisfied"],
                    "welfare_concavity_satisfied": wp["condition_3_social_welfare_concavity"]["satisfied"],
                    "contraction_rate": game.contraction_rate,
                    "x_nash_clipped": x_n,
                    "x_so_clipped": x_s,
                    "marginal_E": E,
                    "delta_nash_so": delta,
                    "price_of_anarchy": poa,
                    "overshooting_congestion": (E < 0) if is_valid else False,
                    "mediator_lambda_max_eps01": lam_max_01,
                    "welfare_recovery_eps01": r_rec_01,
                })
                
    return pd.DataFrame(rows)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Path to upstream payoff estimation handover
    json_path = os.path.abspath(
        os.path.join(base_dir, "..", "payoff_estimation", "results", "game_theory_coefficients.json")
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Upstream file not found: {json_path}")
        
    print(f"Loading empirical parameters from: {json_path}")
    uncentered, cov_sub, raw_json, diag_info = load_empirical_inputs(json_path)
    print(f"Covariance Check: {diag_info['covariance_parametrization']}")
    print(f"Min Eigenvalue: {diag_info['min_eigenvalue_before_shift']:.6f} | Shift triggered: {diag_info['eigenvalue_shift_triggered']} (shift: {diag_info['shift_amount']:.6e})")
    
    # -------------------------------------------------------------------------
    # 1. Primary Empirical Model (Regime A: Calibrated Empirical Point Estimates)
    # -------------------------------------------------------------------------
    empirical_game = PotentialGame.from_json(json_path, n_creators=38)
    empirical_summary = empirical_game.to_dict()
    
    print("\n" + "=" * 70)
    print("REGIME A: EMPIRICAL POINT ESTIMATES (MODEL 5)")
    print("=" * 70)
    print(f"Parameters: beta_1 = {empirical_game.beta_1:.4f}, beta_2 = {empirical_game.beta_2:.4f}, "
          f"beta_3 = {empirical_game.beta_3:.4f}, beta_5 = {empirical_game.beta_5:.4f}")
    print(f"Well-Posed Potential Game: {empirical_summary['well_posedness']['all_conditions_satisfied']}")
    print(f"Contraction Rate:          {empirical_game.contraction_rate:.4f} (< 1.0 -> Unique Stable Nash)")
    print(f"Symmetric Nash (x*):       {empirical_game.x_nash:.4f} (unclipped: {empirical_game.x_nash_unclipped:.4f})")
    print(f"Social Optimum (x^SO):     {empirical_game.x_so:.4f} (unclipped: {empirical_game.x_so_unclipped:.4f})")
    print(f"Marginal Externality (E):  {empirical_game.marginal_external_congestion:.4f}")
    print(f"Nash-SO Gap (Delta):       {empirical_game.nash_so_gap:.4f}")
    print(f"Price of Anarchy (PoA):    {empirical_game.price_of_anarchy:.4f} (closed form: {empirical_game.price_of_anarchy_closed_form:.4f})")
    print(f"Equilibria Interior:       {empirical_game.equilibria_interior}")
    print(f"Potential Hessian Eigs:    lambda_1={empirical_summary['well_posedness']['potential_game_hessian_eigenvalues']['lambda_all_ones']:.4f}, "
          f"lambda_2={empirical_summary['well_posedness']['potential_game_hessian_eigenvalues']['lambda_orthogonal']:.4f}")

    # -------------------------------------------------------------------------
    # 2. Theoretical Attention-Congestion Models (Regimes B and B' - Hypothetical)
    # -------------------------------------------------------------------------
    congestion_game = PotentialGame(
        beta_1=8.301959,
        beta_2=-9.167702,
        beta_3=-3.802120,   # Negative peer externality (attention congestion)
        beta_5=0.782244,    # Strategic complements
        n_creators=38,
        name="Regime B (Hypothetical): Attention Congestion (Strategic Complements)",
    )
    congestion_summary = congestion_game.to_dict()

    congestion_game_b_prime = PotentialGame(
        beta_1=8.301959,
        beta_2=-9.167702,
        beta_3=-3.802120,   # Negative peer externality (attention congestion)
        beta_5=-0.782244,   # Keep empirical strategic substitutes
        n_creators=38,
        name="Regime B' (Hypothetical): Attention Congestion (Empirical Interaction)",
    )
    congestion_b_prime_summary = congestion_game_b_prime.to_dict()
    
    print("\n" + "=" * 70)
    print("REGIME B (HYPOTHETICAL): ATTENTION CONGESTION + STRATEGIC COMPLEMENTS")
    print("=" * 70)
    print(f"Parameters: beta_1 = {congestion_game.beta_1:.4f}, beta_2 = {congestion_game.beta_2:.4f}, "
          f"beta_3 = {congestion_game.beta_3:.4f}, beta_5 = {congestion_game.beta_5:.4f}")
    print(f"Symmetric Nash (x*):       {congestion_game.x_nash:.4f}")
    print(f"Social Optimum (x^SO):     {congestion_game.x_so:.4f}  <-- Overshoots optimum!")
    print(f"Marginal Externality (E):  {congestion_game.marginal_external_congestion:.4f}")
    print(f"Nash-SO Gap (Delta):       {congestion_game.nash_so_gap:.4f}")
    print(f"Price of Anarchy (PoA):    {congestion_game.price_of_anarchy:.4f}")

    print("\n" + "=" * 70)
    print("REGIME B' (HYPOTHETICAL): ATTENTION CONGESTION + EMPIRICAL INTERACTION")
    print("=" * 70)
    print(f"Parameters: beta_1 = {congestion_game_b_prime.beta_1:.4f}, beta_2 = {congestion_game_b_prime.beta_2:.4f}, "
          f"beta_3 = {congestion_game_b_prime.beta_3:.4f}, beta_5 = {congestion_game_b_prime.beta_5:.4f}")
    print(f"Symmetric Nash (x*):       {congestion_game_b_prime.x_nash:.4f}")
    print(f"Social Optimum (x^SO):     {congestion_game_b_prime.x_so:.4f}  <-- Overshoots optimum!")
    print(f"Marginal Externality (E):  {congestion_game_b_prime.marginal_external_congestion:.4f}")
    print(f"Nash-SO Gap (Delta):       {congestion_game_b_prime.nash_so_gap:.4f}")
    print(f"Price of Anarchy (PoA):    {congestion_game_b_prime.price_of_anarchy:.4f}")

    # -------------------------------------------------------------------------
    # 3. Parametric Bootstrap for 95% Confidence Intervals
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PARAMETRIC BOOTSTRAP (10,000 DRAWS OVER CLUSTER-ROBUST COVARIANCE)")
    print("=" * 70)
    mean_vec = np.array([
        uncentered["beta_1_own_click_appeal"],
        uncentered["beta_2_quadratic_own_appeal"],
        uncentered["beta_3_competitor_click_appeal"],
        uncentered["beta_5_strategic_interaction"],
    ])
    df_boot_valid, boot_summary = run_parametric_bootstrap(mean_vec, cov_sub, n_draws=10000, seed=42)
    
    diag = boot_summary["diagnostics_over_all_draws"]
    perc = boot_summary["conditional_on_well_posedness"]
    print(f"Over ALL 10,000 draws:")
    print(f"  P(beta_2 < 0):                {diag['P_beta_2_negative'] * 100:.2f}%")
    print(f"  P(B > 0):                     {diag['P_welfare_concave_B_pos'] * 100:.2f}%")
    print(f"  P(|beta_5| < 2|beta_2|):      {diag['P_contraction_all'] * 100:.2f}%")
    print(f"  P(Well-Posed Game):           {diag['P_well_posed_all_conditions'] * 100:.2f}% ({diag['valid_draws_count']} valid draws)")
    print(f"  P(E < 0) [All draws]:         {diag['P_marginal_E_negative_all_draws'] * 100:.2f}%")
    print(f"  P(E < 0) [Well-posed draws]:  {diag['P_marginal_E_negative_conditional_well_posed'] * 100:.2f}%")
    print(f"  Fraction valid with clipped x_Nash: {diag['fraction_valid_with_clipped_x_nash'] * 100:.2f}%")
    print(f"  Fraction valid with clipped x_SO:   {diag['fraction_valid_with_clipped_x_so'] * 100:.2f}%")
    print(f"  Fraction valid with any clipped:    {diag['fraction_valid_with_any_clipped'] * 100:.2f}%")

    print(f"\nPercentile Intervals (Conditional on Well-Posedness):")
    print(f"  x^Nash 95% CI: [{perc['x_nash_clipped']['p025']:.4f}, {perc['x_nash_clipped']['p975']:.4f}] (median: {perc['x_nash_clipped']['median']:.4f})")
    print(f"  x^SO   95% CI: [{perc['x_so_clipped']['p025']:.4f}, {perc['x_so_clipped']['p975']:.4f}] (median: {perc['x_so_clipped']['median']:.4f})")
    print(f"  Delta  95% CI: [{perc['delta']['p025']:.4f}, {perc['delta']['p975']:.4f}] (median: {perc['delta']['median']:.4f})")
    print(f"  PoA    95% CI: [{perc['poa']['p025']:.4f}, {perc['poa']['p975']:.4f}] (median: {perc['poa']['median']:.4f})")

    # -------------------------------------------------------------------------
    # 4. Sensitivity Analysis Grid
    # -------------------------------------------------------------------------
    print("\nGenerating systematic sensitivity grid across (beta_2, beta_5, beta_3)...")
    df_grid = build_sensitivity_analysis_grid(b1_base=uncentered["beta_1_own_click_appeal"])
    grid_csv_path = os.path.join(results_dir, "sensitivity_analysis_grid.csv")
    df_grid.to_csv(grid_csv_path, index=False)
    print(f"Saved sensitivity grid ({len(df_grid)} combinations) to: {grid_csv_path}")

    # -------------------------------------------------------------------------
    # 5. Export Handover Payloads for Part 4 (Simulations)
    # -------------------------------------------------------------------------
    equilibrium_summary_csv = pd.DataFrame([
        {
            "regime": "Regime A: Empirical Point Estimate",
            "is_hypothetical": False,
            "beta_1": empirical_game.beta_1,
            "beta_2": empirical_game.beta_2,
            "beta_3": empirical_game.beta_3,
            "beta_5": empirical_game.beta_5,
            "x_nash": empirical_game.x_nash,
            "x_so": empirical_game.x_so,
            "marginal_E": empirical_game.marginal_external_congestion,
            "delta_gap": empirical_game.nash_so_gap,
            "price_of_anarchy": empirical_game.price_of_anarchy,
            "price_of_anarchy_closed_form": empirical_game.price_of_anarchy_closed_form,
            "contraction_rate": empirical_game.contraction_rate,
            "overshooting": empirical_game.marginal_external_congestion < 0,
            "equilibria_interior": empirical_game.equilibria_interior,
        },
        {
            "regime": "Regime B (Hypothetical): Attention Congestion (Strategic Complements)",
            "is_hypothetical": True,
            "beta_1": congestion_game.beta_1,
            "beta_2": congestion_game.beta_2,
            "beta_3": congestion_game.beta_3,
            "beta_5": congestion_game.beta_5,
            "x_nash": congestion_game.x_nash,
            "x_so": congestion_game.x_so,
            "marginal_E": congestion_game.marginal_external_congestion,
            "delta_gap": congestion_game.nash_so_gap,
            "price_of_anarchy": congestion_game.price_of_anarchy,
            "price_of_anarchy_closed_form": congestion_game.price_of_anarchy_closed_form,
            "contraction_rate": congestion_game.contraction_rate,
            "overshooting": congestion_game.marginal_external_congestion < 0,
            "equilibria_interior": congestion_game.equilibria_interior,
        },
        {
            "regime": "Regime B' (Hypothetical): Attention Congestion (Empirical Interaction)",
            "is_hypothetical": True,
            "beta_1": congestion_game_b_prime.beta_1,
            "beta_2": congestion_game_b_prime.beta_2,
            "beta_3": congestion_game_b_prime.beta_3,
            "beta_5": congestion_game_b_prime.beta_5,
            "x_nash": congestion_game_b_prime.x_nash,
            "x_so": congestion_game_b_prime.x_so,
            "marginal_E": congestion_game_b_prime.marginal_external_congestion,
            "delta_gap": congestion_game_b_prime.nash_so_gap,
            "price_of_anarchy": congestion_game_b_prime.price_of_anarchy,
            "price_of_anarchy_closed_form": congestion_game_b_prime.price_of_anarchy_closed_form,
            "contraction_rate": congestion_game_b_prime.contraction_rate,
            "overshooting": congestion_game_b_prime.marginal_external_congestion < 0,
            "equilibria_interior": congestion_game_b_prime.equilibria_interior,
        },
    ])
    summary_csv_path = os.path.join(results_dir, "equilibrium_summary.csv")
    equilibrium_summary_csv.to_csv(summary_csv_path, index=False)
    print(f"Saved equilibrium summary table to: {summary_csv_path}")

    # Bootstrap summary export
    boot_csv_path = os.path.join(results_dir, "bootstrap_poa_distribution.csv")
    pd.DataFrame(boot_summary["conditional_on_well_posedness"]).to_csv(boot_csv_path)
    print(f"Saved bootstrap distribution percentiles to: {boot_csv_path}")

    # Comprehensive JSON Handover for Part 4
    handover_payload = {
        "metadata": {
            "component": "Game-Theoretic Formulation & Potential Game Analysis",
            "date": "November 2026",
            "course": "CS 329: Foundations of AI: Multi-Agent Systems",
        },
        "covariance_diagnostics": diag_info,
        "regime_A_empirical": empirical_summary,
        "regime_B_hypothetical_congestion_complements": congestion_summary,
        "regime_B_prime_hypothetical_congestion_empirical_interaction": congestion_b_prime_summary,
        "bootstrap_uncertainty_intervals": boot_summary,
        "mediator_formulas_guide": {
            "recommendation": "x_lambda = (1 - lambda) * x_nash + lambda * x_so",
            "deviation_gain": "G(lambda) = pi(BR(x_lambda), x_lambda) - pi(x_lambda, x_lambda)",
            "deviation_gain_closed_form": "G_cf(lambda) = |beta_2| * kappa^2 * lambda^2 * Delta^2",
            "lambda_max_eps": "lambda_max = min(1, sqrt(eps / |beta_2|) / (kappa * |Delta|))",
            "welfare_recovery_R": "R(lambda) = 2*lambda - lambda^2",
            "view_recovery_R_V": "R_V(lambda) = (PoA^(2*lambda - lambda^2) - 1) / (PoA - 1)",
        },
    }
    json_export_path = os.path.join(results_dir, "equilibrium_results.json")
    with open(json_export_path, "w", encoding="utf-8") as f:
        json.dump(handover_payload, f, indent=2)
    print(f"Saved comprehensive JSON handover to: {json_export_path}")
    print("\nPart 3 game-theoretic analysis executed successfully!")


if __name__ == "__main__":
    main()

