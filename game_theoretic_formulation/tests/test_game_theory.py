"""
Unit Tests for Game-Theoretic Formulation & Potential Game Solver
CS 329: Foundations of AI: Multi-Agent Systems
"""

import sys
import os
import unittest
import numpy as np

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from game_theory_solver import PotentialGame


class TestPotentialGame(unittest.TestCase):

    def setUp(self):
        # Empirical point estimates from estimated Model 5
        self.game = PotentialGame(
            beta_1=8.301959,
            beta_2=-9.167702,
            beta_3=3.802120,
            beta_5=-0.782244,
            beta_0=-7.983732,
            beta_4=1.066772,
            n_creators=38,
        )

    def test_well_posedness(self):
        """Verify strict concavity, contraction mapping, and potential stability."""
        wp = self.game.check_well_posedness()
        self.assertTrue(wp["condition_1_individual_concavity"]["satisfied"])
        self.assertTrue(wp["condition_2_contraction_unique_nash"]["satisfied"])
        self.assertTrue(wp["condition_3_social_welfare_concavity"]["satisfied"])
        self.assertTrue(wp["potential_game_hessian_eigenvalues"]["strictly_concave_potential"])
        self.assertTrue(wp["all_conditions_satisfied"])
        self.assertLess(self.game.contraction_rate, 1.0)

    def test_nash_is_fixed_point_of_best_response(self):
        """Nash equilibrium must be the fixed point: BR(x_nash) == x_nash."""
        x_nash = self.game.x_nash
        br_nash = self.game.best_response(x_nash)
        self.assertAlmostEqual(x_nash, br_nash, places=6)
        self.assertAlmostEqual(x_nash, 0.434256, places=4)

    def test_social_optimum_maximizes_symmetric_welfare(self):
        """First derivative of symmetric welfare at x_so must be zero."""
        x_so = self.game.x_so
        # W'(x) = (beta_1 + beta_3) + 2*(beta_2 + beta_5)*x
        w_prime = (self.game.beta_1 + self.game.beta_3) + 2 * (self.game.beta_2 + self.game.beta_5) * x_so
        self.assertAlmostEqual(w_prime, 0.0, places=6)
        self.assertAlmostEqual(x_so, 0.608248, places=4)

    def test_exact_potential_gradient(self):
        """Exact potential game property: dPhi/dx_i == dpi_i/dx_i."""
        n = 5
        game_n = PotentialGame(
            beta_1=8.301959,
            beta_2=-9.167702,
            beta_3=3.802120,
            beta_5=-0.782244,
            n_creators=n,
        )
        x_profile = np.array([0.2, 0.35, 0.5, 0.4, 0.6])
        h = 1e-6

        for i in range(n):
            # Numerical derivative of Phi with respect to x_i
            x_plus = x_profile.copy()
            x_minus = x_profile.copy()
            x_plus[i] += h
            x_minus[i] -= h
            d_phi_num = (game_n.potential(x_plus) - game_n.potential(x_minus)) / (2 * h)

            # Analytical partial derivative of pi_i: d_pi_i / dx_i = beta_1 + 2*beta_2*x_i + beta_5*x_bar_-i
            x_others = np.delete(x_profile, i)
            x_bar_minus_i = np.mean(x_others)
            d_pi_ana = game_n.beta_1 + 2 * game_n.beta_2 * x_profile[i] + game_n.beta_5 * x_bar_minus_i

            self.assertAlmostEqual(d_phi_num, d_pi_ana, places=4)

    def test_price_of_anarchy_identity(self):
        """PoA formula exp(E^2 / (4B)) must match exp(pi(x_so, x_so) - pi(x_nash, x_nash))."""
        poa_formula = self.game.price_of_anarchy
        w_so = self.game.social_welfare_symmetric(self.game.x_so)
        w_nash = self.game.social_welfare_symmetric(self.game.x_nash)
        poa_definition = np.exp(w_so - w_nash)
        self.assertAlmostEqual(poa_formula, poa_definition, places=5)
        self.assertGreaterEqual(poa_formula, 1.0)

    def test_mediator_properties(self):
        """Verify mediator interpolation, deviation gain, and welfare recovery."""
        # At lambda = 0: recovers Nash
        self.assertAlmostEqual(self.game.mediator_recommendation(0.0), self.game.x_nash, places=6)
        self.assertAlmostEqual(self.game.mediator_deviation_gain(0.0), 0.0, places=6)
        self.assertAlmostEqual(self.game.welfare_recovery(0.0), 0.0, places=6)

        # At lambda = 1: recovers Social Optimum
        self.assertAlmostEqual(self.game.mediator_recommendation(1.0), self.game.x_so, places=6)
        self.assertAlmostEqual(self.game.welfare_recovery(1.0), 1.0, places=6)

        # At lambda = 0.5: R(0.5) = 2(0.5) - 0.25 = 0.75 (75% recovery)
        self.assertAlmostEqual(self.game.welfare_recovery(0.5), 0.75, places=6)

        # Deviation gain test at lambda_max(eps): G(lambda_max) <= eps
        eps = 0.05
        lam_max = self.game.mediator_max_lambda(eps)
        gain_at_max = self.game.mediator_deviation_gain(lam_max)
        self.assertLessEqual(gain_at_max, eps + 1e-7)

    def test_congestion_regime(self):
        """Verify behavior in attention-congestion regime (H1: E < 0, beta_3 < 0)."""
        congestion_game = PotentialGame(
            beta_1=8.301959,
            beta_2=-9.167702,
            beta_3=-3.802120,  # Negative peer externality (congestion)
            beta_5=0.782244,   # Strategic complements
            n_creators=38,
        )
        self.assertLess(congestion_game.marginal_external_congestion, 0.0)
        # Decentralized competition overshoots social optimum: x_so < x_nash
        self.assertLess(congestion_game.x_so, congestion_game.x_nash)
        self.assertGreater(congestion_game.price_of_anarchy, 1.0)

    def test_clipped_nash_case(self):
        """When unclipped Nash exceeds 1, clipped x_nash must equal 1.0."""
        game_clipped = PotentialGame(beta_1=50.0, beta_2=-9.167702, beta_3=0.0, beta_5=0.0)
        self.assertGreater(game_clipped.x_nash_unclipped, 1.0)
        self.assertEqual(game_clipped.x_nash, 1.0)
        self.assertFalse(game_clipped.equilibria_interior)

    def test_clipped_so_case(self):
        """When unclipped Social Optimum exceeds 1 or is negative, x_so must clip to boundary."""
        game_high = PotentialGame(beta_1=50.0, beta_2=-5.0, beta_3=20.0, beta_5=0.0)
        self.assertGreater(game_high.x_so_unclipped, 1.0)
        self.assertEqual(game_high.x_so, 1.0)

        game_low = PotentialGame(beta_1=2.0, beta_2=-5.0, beta_3=-10.0, beta_5=0.0)
        self.assertLess(game_low.x_so_unclipped, 0.0)
        self.assertEqual(game_low.x_so, 0.0)

    def test_poa_definition_vs_closed_form(self):
        """PoA by definition equals closed form when interior, and differs when clipped."""
        # Interior: empirical game
        self.assertTrue(self.game.equilibria_interior)
        self.assertAlmostEqual(self.game.price_of_anarchy, self.game.price_of_anarchy_closed_form, places=6)

        # Clipped: both clip to 1.0, so realized welfare difference is 0 -> PoA = 1.0
        game_both_clipped = PotentialGame(beta_1=50.0, beta_2=-5.0, beta_3=10.0, beta_5=0.0)
        self.assertEqual(game_both_clipped.x_nash, 1.0)
        self.assertEqual(game_both_clipped.x_so, 1.0)
        self.assertAlmostEqual(game_both_clipped.price_of_anarchy, 1.0, places=6)
        # Closed form would assume unclipped vertices and yield > 1.0
        self.assertGreater(game_both_clipped.price_of_anarchy_closed_form, 1.0)

    def test_nash_so_gap_uses_clipped_values(self):
        """nash_so_gap must equal x_so - x_nash (clipped)."""
        self.assertAlmostEqual(self.game.nash_so_gap, self.game.x_so - self.game.x_nash, places=6)
        game_clipped = PotentialGame(beta_1=50.0, beta_2=-5.0, beta_3=10.0, beta_5=0.0)
        self.assertEqual(game_clipped.nash_so_gap, 0.0)
        self.assertGreater(game_clipped.nash_so_gap_closed_form, 0.0)

    def test_from_json_raises_on_missing_key(self):
        """from_json must raise KeyError if required parameters are missing."""
        import tempfile
        import json
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            json.dump({"point_estimates_uncentered_model_5": {"beta_1_own_click_appeal": 1.0}}, tmp)
            tmp_path = tmp.name
        try:
            with self.assertRaises(KeyError):
                PotentialGame.from_json(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_split_well_posedness_conditions(self):
        """check_well_posedness must independently check individual concavity, contraction, welfare concavity, and Eq. 3."""
        wp = self.game.check_well_posedness()
        self.assertIn("condition_1_individual_concavity", wp)
        self.assertIn("condition_2_contraction_unique_nash", wp)
        self.assertIn("condition_3_social_welfare_concavity", wp)
        self.assertIn("condition_combined_eq3", wp)
        self.assertTrue(wp["condition_combined_eq3"]["satisfied"])

    def test_mediator_with_clipped_equilibria(self):
        """Mediator recommendation and recovery formula must remain valid under clipped equilibria."""
        game_clipped = PotentialGame(beta_1=50.0, beta_2=-5.0, beta_3=-20.0, beta_5=0.0)
        self.assertEqual(game_clipped.x_nash, 1.0)
        self.assertEqual(game_clipped.x_so, 1.0)
        # With x_nash == x_so == 1, recommendation is 1 for all lambda
        self.assertEqual(game_clipped.mediator_recommendation(0.5), 1.0)
        self.assertEqual(game_clipped.welfare_recovery(0.5), 0.75)


if __name__ == "__main__":
    unittest.main()

