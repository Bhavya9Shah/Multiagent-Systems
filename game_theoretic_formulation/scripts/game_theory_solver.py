"""
Game-Theoretic Formulation & Potential Game Solver
CS 329: Foundations of AI: Multi-Agent Systems (2026)
Indian Institute of Technology Gandhinagar


Component: Game-Theoretic Formulation, Nash Equilibrium, Social Optimum, and Price of Anarchy


Mathematical References:
- Monderer & Shapley (1996): Potential Games
- Roughgarden (2005): Selfish Routing and the Price of Anarchy
- Project Proposal Equations (2) to (15)
"""

import json
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np


class PotentialGame:
    """
    Symmetric continuous exact potential game representing competition among
    YouTube content creators under attention-congestion externalities.
    
    Each creator i chooses a click appeal score x_i in [0, 1].
    The payoff function is:
        pi_i(x_i, x_bar_-i) = beta_1 * x_i + beta_2 * x_i^2 + beta_3 * x_bar_-i + beta_5 * x_i * x_bar_-i + c_i
    where x_bar_-i = 1/(n-1) * sum_{j != i} x_j.
    """

    def __init__(
        self,
        beta_1: float,
        beta_2: float,
        beta_3: float,
        beta_5: float,
        beta_0: float = 0.0,
        beta_4: float = 0.0,
        n_creators: int = 38,
        name: str = "Empirical YouTube Game",
        cov_matrix: Optional[np.ndarray] = None,
        parameter_names: Optional[list] = None,
    ):
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.beta_3 = float(beta_3)
        self.beta_5 = float(beta_5)
        self.beta_0 = float(beta_0)
        self.beta_4 = float(beta_4)
        self.n_creators = int(n_creators)
        self.name = name
        self.cov_matrix = cov_matrix
        self.parameter_names = parameter_names

    @classmethod
    def from_json(cls, json_path: str, n_creators: int = 38) -> "PotentialGame":
        """Load empirical parameters from estimated payoff JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "point_estimates_uncentered_model_5" not in data:
            raise KeyError("Missing required section 'point_estimates_uncentered_model_5' in JSON")
        uncentered = data["point_estimates_uncentered_model_5"]
        
        required_keys = [
            "beta_1_own_click_appeal",
            "beta_2_quadratic_own_appeal",
            "beta_3_competitor_click_appeal",
            "beta_5_strategic_interaction",
            "beta_0_intercept",
            "beta_4_log_subscribers",
        ]
        for k in required_keys:
            if k not in uncentered:
                raise KeyError(f"Missing required parameter '{k}' in JSON payload")
                
        beta_1 = uncentered["beta_1_own_click_appeal"]
        beta_2 = uncentered["beta_2_quadratic_own_appeal"]
        beta_3 = uncentered["beta_3_competitor_click_appeal"]
        beta_5 = uncentered["beta_5_strategic_interaction"]
        beta_0 = uncentered["beta_0_intercept"]
        beta_4 = uncentered["beta_4_log_subscribers"]
        
        # Load covariance matrix if available
        cov_dict = data.get("parameter_covariance_matrix_cluster_robust")
        cov_matrix = None
        param_names = None
        if cov_dict is not None:
            param_names = list(cov_dict.keys())
            p = len(param_names)
            cov_matrix = np.zeros((p, p))
            for i, p_row in enumerate(param_names):
                for j, p_col in enumerate(param_names):
                    if p_col not in cov_dict[p_row]:
                        raise KeyError(f"Missing covariance entry for ({p_row}, {p_col}) in JSON")
                    cov_matrix[i, j] = cov_dict[p_row][p_col]
                    
        return cls(
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3,
            beta_5=beta_5,
            beta_0=beta_0,
            beta_4=beta_4,
            n_creators=n_creators,
            name="Empirical YouTube Potential Game (Model 5)",
            cov_matrix=cov_matrix,
            parameter_names=param_names,
        )

    # -------------------------------------------------------------------------
    # Mathematical Conditions & Game-Theoretic Well-Posedness (Proposal Eq. 3)
    # -------------------------------------------------------------------------
    def check_well_posedness(self) -> Dict[str, Any]:
        """
        Verify the well-posedness conditions from Proposal Eq. (3):
          1. Strict individual concavity: beta_2 < 0
          2. Contraction mapping / Unique stable Nash: |beta_5| < 2|beta_2|
          3. Social welfare strict concavity: -(n-1)|beta_2| < beta_5 < |beta_2|
          4. Combined Eq. (3): beta_2 < 0 and -min(2, n-1)|beta_2| < beta_5 < |beta_2|
        """
        abs_beta_2 = abs(self.beta_2)
        n = self.n_creators
        lower_bound_eq3 = -min(2, n - 1) * abs_beta_2
        upper_bound_eq3 = abs_beta_2

        concavity_ok = self.beta_2 < 0
        contraction_ok = abs(self.beta_5) < (2.0 * abs_beta_2)
        welfare_lower = -(n - 1) * abs_beta_2 if n > 1 else -abs_beta_2
        welfare_concavity_ok = (welfare_lower < self.beta_5 < abs_beta_2) and ((self.beta_2 + self.beta_5) < 0)
        combined_eq3_ok = concavity_ok and (lower_bound_eq3 < self.beta_5 < upper_bound_eq3)

        # Exact potential game eigenvalues of Hessian of Phi(x)
        lam_1 = 2 * self.beta_2 + self.beta_5
        lam_2 = 2 * self.beta_2 - (self.beta_5 / (n - 1)) if n > 1 else lam_1
        potential_strictly_concave = (lam_1 < 0) and (lam_2 < 0)

        return {
            "condition_1_individual_concavity": {
                "condition": "beta_2 < 0",
                "value": self.beta_2,
                "satisfied": bool(concavity_ok),
            },
            "condition_2_contraction_unique_nash": {
                "condition": f"|beta_5| < 2|beta_2| ({abs(self.beta_5):.4f} < {2.0 * abs_beta_2:.4f})",
                "slope": self.best_response_slope,
                "contraction_rate": self.contraction_rate,
                "satisfied": bool(contraction_ok),
            },
            "condition_3_social_welfare_concavity": {
                "condition": f"-(n-1)|beta_2| < beta_5 < |beta_2| ({welfare_lower:.4f} < {self.beta_5:.4f} < {abs_beta_2:.4f})",
                "value": self.beta_2 + self.beta_5,
                "satisfied": bool(welfare_concavity_ok),
            },
            "condition_combined_eq3": {
                "condition": f"beta_2 < 0 and -min(2, n-1)|beta_2| < beta_5 < |beta_2| ({lower_bound_eq3:.4f} < {self.beta_5:.4f} < {upper_bound_eq3:.4f})",
                "satisfied": bool(combined_eq3_ok),
            },
            "potential_game_hessian_eigenvalues": {
                "lambda_all_ones": lam_1,
                "lambda_orthogonal": lam_2,
                "strictly_concave_potential": bool(potential_strictly_concave),
            },
            "all_conditions_satisfied": bool(
                concavity_ok and contraction_ok and welfare_concavity_ok and potential_strictly_concave and combined_eq3_ok
            ),
        }

    # -------------------------------------------------------------------------
    # Core Payoff, Welfare, Potential, and Best-Response Maps
    # -------------------------------------------------------------------------
    def payoff(
        self,
        x_i: Union[float, np.ndarray],
        x_bar_minus_i: Union[float, np.ndarray],
        c_i: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """
        Individual creator payoff (Eq. 2):
          pi_i(x_i, x_bar_-i) = beta_1 * x_i + beta_2 * x_i^2 + beta_3 * x_bar_-i + beta_5 * x_i * x_bar_-i + c_i
        """
        return (
            self.beta_1 * x_i
            + self.beta_2 * (x_i ** 2)
            + self.beta_3 * x_bar_minus_i
            + self.beta_5 * x_i * x_bar_minus_i
            + c_i
        )

    def best_response(
        self,
        x_bar_minus_i: Union[float, np.ndarray],
        clip: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Best response map (Eq. 4):
          BR_i(x_bar_-i) = clip((beta_1 + beta_5 * x_bar_-i) / (2 * |beta_2|), 0, 1)
        """
        abs_beta_2 = abs(self.beta_2)
        denom = 2.0 * abs_beta_2
        if denom == 0.0:
            raise ZeroDivisionError("beta_2 is zero; best response undefined.")
        unclipped = (self.beta_1 + self.beta_5 * x_bar_minus_i) / denom
        if clip:
            return np.clip(unclipped, 0.0, 1.0)
        return unclipped

    @property
    def best_response_slope(self) -> float:
        """Slope of best response in x_bar_-i: s = beta_5 / (2 * |beta_2|)."""
        return self.beta_5 / (2.0 * abs(self.beta_2))

    @property
    def contraction_rate(self) -> float:
        """Contraction rate |s| = |beta_5| / (2 * |beta_2|)."""
        return abs(self.best_response_slope)

    def potential(self, x_profile: np.ndarray) -> float:
        """
        Exact potential function Phi(x) (Section 3.2):
          Phi(x) = sum_{i=1}^n (beta_1 * x_i + beta_2 * x_i^2) + (beta_5 / (n-1)) * sum_{i < j} x_i * x_j
        Satisfies: dPhi / dx_i = dpi_i / dx_i.
        """
        x = np.asarray(x_profile, dtype=float)
        n = len(x)
        term1 = np.sum(self.beta_1 * x + self.beta_2 * (x ** 2))
        if n > 1 and self.beta_5 != 0.0:
            # sum_{i < j} x_i * x_j = 0.5 * ((sum x_i)^2 - sum x_i^2)
            sum_x = np.sum(x)
            sum_x_sq = np.sum(x ** 2)
            cross_terms = 0.5 * (sum_x ** 2 - sum_x_sq)
            term2 = (self.beta_5 / (n - 1)) * cross_terms
        else:
            term2 = 0.0
        return float(term1 + term2)

    def social_welfare_symmetric(self, x: Union[float, np.ndarray], c: float = 0.0) -> Union[float, np.ndarray]:
        """
        Symmetric social welfare per creator (Section 3.3):
          pi(x, x) = (beta_1 + beta_3) * x + (beta_2 + beta_5) * x^2 + c
        Total welfare for n creators is n * pi(x, x).
        """
        return (self.beta_1 + self.beta_3) * x + (self.beta_2 + self.beta_5) * (x ** 2) + c

    # -------------------------------------------------------------------------
    # Closed-Form Game-Theoretic Equilibria (Proposal Sec. 3.2–3.4)
    # -------------------------------------------------------------------------
    @property
    def x_nash_unclipped(self) -> float:
        """Unclipped symmetric Nash equilibrium (Eq. 5 numerator / denominator)."""
        denom = 2.0 * abs(self.beta_2) - self.beta_5
        if denom == 0.0:
            raise ZeroDivisionError("2|beta_2| - beta_5 == 0; Nash equilibrium undefined.")
        return self.beta_1 / denom

    @property
    def x_nash(self) -> float:
        """Symmetric Nash Equilibrium score clipped to [0, 1] (Eq. 5)."""
        return float(np.clip(self.x_nash_unclipped, 0.0, 1.0))

    @property
    def x_so_unclipped(self) -> float:
        """Unclipped Social Optimum score (Eq. 6)."""
        denom = 2.0 * (abs(self.beta_2) - self.beta_5)
        if denom == 0.0:
            raise ZeroDivisionError("2(|beta_2| - beta_5) == 0; Social optimum undefined.")
        return (self.beta_1 + self.beta_3) / denom

    @property
    def x_so(self) -> float:
        """Social Optimum score clipped to [0, 1] (Eq. 6)."""
        return float(np.clip(self.x_so_unclipped, 0.0, 1.0))

    @property
    def equilibria_interior(self) -> bool:
        """True if both symmetric Nash and Social Optimum are strictly interior to [0, 1]."""
        return bool(0.0 < self.x_nash_unclipped < 1.0 and 0.0 < self.x_so_unclipped < 1.0)

    @property
    def marginal_external_congestion(self) -> float:
        """
        Marginal external effect at Nash profile (Eq. 7):
          E = beta_3 + beta_5 * x^Nash
        If E < 0, decentralized competition overshoots social optimum (attention congestion).
        """
        return self.beta_3 + self.beta_5 * self.x_nash

    @property
    def nash_so_gap(self) -> float:
        """Exact gap between clipped Social Optimum and clipped Nash: Delta = x^SO - x^Nash."""
        return float(self.x_so - self.x_nash)

    @property
    def nash_so_gap_closed_form(self) -> float:
        """Closed-form unclipped gap Delta = E / (2*(|beta_2| - beta_5)) (Eq. 8), exact when interior."""
        denom = 2.0 * (abs(self.beta_2) - self.beta_5)
        return float(self.marginal_external_congestion / denom)

    @property
    def price_of_anarchy(self) -> float:
        """
        Price of Anarchy in view welfare computed from definition with clipped equilibria (Eq. 9):
          PoA = V(x^SO) / V(x^Nash) = exp(pi(x^SO, x^SO) - pi(x^Nash, x^Nash))
        Equals price_of_anarchy_closed_form when equilibria are interior.
        """
        w_so = self.social_welfare_symmetric(self.x_so)
        w_nash = self.social_welfare_symmetric(self.x_nash)
        return float(np.exp(w_so - w_nash))

    @property
    def price_of_anarchy_closed_form(self) -> float:
        """
        Closed-form Price of Anarchy (Eq. 10):
          PoA = exp(E^2 / (4 * (|beta_2| - beta_5)))
        Exact when both x^Nash and x^SO are interior.
        """
        B = abs(self.beta_2) - self.beta_5
        if B <= 0:
            raise ValueError(f"Welfare not strictly concave (B = {B} <= 0); PoA undefined.")
        E = self.marginal_external_congestion
        exponent = (E ** 2) / (4.0 * B)
        return float(np.exp(exponent))

    # -------------------------------------------------------------------------
    # Correlated-Equilibrium Platform Mediator (Proposal Sec. 3.5 & 3.6)
    # -------------------------------------------------------------------------
    def mediator_recommendation(self, lam: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolated platform recommendation (Eq. 11):
          x_lambda = (1 - lambda) * x^Nash + lambda * x^SO
        """
        return (1.0 - lam) * self.x_nash + lam * self.x_so

    def mediator_deviation_gain(self, lam: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Deviation gain from ignoring platform recommendation by definition (Eq. 12):
          G(lambda) = pi(BR(x_lambda), x_lambda) - pi(x_lambda, x_lambda)
        Matches closed form |beta_2| * kappa^2 * lambda^2 * Delta^2 when interior.
        """
        x_rec = self.mediator_recommendation(lam)
        br_rec = self.best_response(x_rec, clip=True)
        return self.payoff(br_rec, x_rec) - self.payoff(x_rec, x_rec)

    def mediator_deviation_gain_closed_form(self, lam: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Closed-form quadratic deviation gain |beta_2| * kappa^2 * lambda^2 * Delta^2 (Eq. 12)."""
        s = self.best_response_slope
        kappa = 1.0 - s
        delta = self.nash_so_gap
        abs_beta_2 = abs(self.beta_2)
        return abs_beta_2 * (kappa ** 2) * (lam ** 2) * (delta ** 2)

    def mediator_max_lambda(self, epsilon: float) -> float:
        """
        Largest admissible mediator strength under epsilon-incentive compatibility (Eq. 13):
          lambda_max(eps) = min(1, sqrt(eps / |beta_2|) / (kappa * |Delta|))
        """
        if epsilon <= 0:
            return 0.0
        s = self.best_response_slope
        kappa = 1.0 - s
        delta = abs(self.nash_so_gap)
        abs_beta_2 = abs(self.beta_2)
        if kappa <= 0 or delta == 0 or abs_beta_2 == 0:
            return 1.0
        val = np.sqrt(epsilon / abs_beta_2) / (kappa * delta)
        return float(np.clip(val, 0.0, 1.0))

    def welfare_recovery(self, lam: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Fraction of Nash-Optimum welfare gap recovered by recommendation x_lambda (Eq. 15):
          R(lambda) = 2 * lambda - lambda^2
        Concave: lambda = 0.5 recovers 75% of welfare gap.
        """
        return 2.0 * lam - (lam ** 2)

    def view_welfare_recovery(self, lam: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        View recovery in percentage of view gap (Section 3.6):
          R_V(lambda) = (PoA^(2 * lambda - lambda^2) - 1) / (PoA - 1)
        """
        poa = self.price_of_anarchy
        if abs(poa - 1.0) < 1e-9:
            return 2.0 * lam - (lam ** 2)
        r_lam = self.welfare_recovery(lam)
        return (poa ** r_lam - 1.0) / (poa - 1.0)

    # -------------------------------------------------------------------------
    # Comprehensive Summary Payload
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Generate structured game-theoretic summary for handover."""
        wp = self.check_well_posedness()
        return {
            "game_name": self.name,
            "parameters": {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "beta_3": self.beta_3,
                "beta_5": self.beta_5,
                "beta_0": self.beta_0,
                "beta_4": self.beta_4,
                "n_creators": self.n_creators,
            },
            "well_posedness": wp,
            "equilibria": {
                "x_nash_unclipped": self.x_nash_unclipped,
                "x_nash_clipped": self.x_nash,
                "x_so_unclipped": self.x_so_unclipped,
                "x_so_clipped": self.x_so,
                "marginal_external_congestion_E": self.marginal_external_congestion,
                "nash_so_gap_Delta": self.nash_so_gap,
                "nash_so_gap_closed_form": self.nash_so_gap_closed_form,
                "price_of_anarchy_PoA": self.price_of_anarchy,
                "price_of_anarchy_closed_form": self.price_of_anarchy_closed_form,
                "equilibria_interior": self.equilibria_interior,
                "overshooting_congestion_detected": bool(self.marginal_external_congestion < 0),
            },
            "properties": {
                "best_response_slope_s": self.best_response_slope,
                "contraction_rate": self.contraction_rate,
                "strategic_type": (
                    "complements" if self.beta_5 > 0 else "substitutes" if self.beta_5 < 0 else "independent"
                ),
            },
            "mediator_benchmark_epsilons": {
                f"eps_{eps}": {
                    "lambda_max": self.mediator_max_lambda(eps),
                    "welfare_recovery_R": self.welfare_recovery(self.mediator_max_lambda(eps)),
                    "view_recovery_R_V": self.view_welfare_recovery(self.mediator_max_lambda(eps)),
                }
                for eps in [0.001, 0.005, 0.01, 0.05, 0.10]
            },
        }
