# Game-Theoretic Formulation & Potential Game Analysis

## 1. Executive Summary & Component Role

This component establishes the rigorous game-theoretic framework for modeling decentralized creator competition on YouTube as an **exact continuous potential game** with attention-congestion externalities. Rather than assuming ad-hoc payoff shapes or hypothetical payoff matrices, this module ingests the empirically estimated econometric parameters $(\beta_1, \beta_2, \beta_3, \beta_5)$ from the upstream payoff estimation results (`payoff_estimation/results/game_theory_coefficients.json`), proves mathematical well-posedness, derives the closed-form symmetric Nash equilibrium ($x^{\text{Nash}}$) and Social Optimum ($x^{\text{SO}}$), quantifies efficiency loss via the Price of Anarchy ($\text{PoA}$), and exposes a clean, machine-readable API and handover for downstream multi-agent simulations and platform mediator interventions.

---

## 2. Mathematical Formulation

### 2.1 Agents, Action Spaces, and Reduced Payoff Function
The agents are $n = 38$ content creators competing in a single niche. Each creator $i \in \{1, \dots, n\}$ independently chooses a continuous click appeal score:
$$x_i \in [0, 1]$$
Let $\boldsymbol{x} = (x_1, \dots, x_n)$ denote the joint strategy profile, and let $\bar{x}_{-i} = \frac{1}{n-1}\sum_{j \neq i} x_j$ denote the leave-one-out niche competitor average faced by creator $i$.

The reduced-form payoff function (Proposal Eq. 2) is:
$$\pi_i(x_i, \bar{x}_{-i}) = \beta_1 x_i + \beta_2 x_i^2 + \beta_3 \bar{x}_{-i} + \beta_5 x_i \bar{x}_{-i} + c_i$$
where $c_i = \beta_0 + \beta_4 \log(s_i) + \boldsymbol{\gamma}^\top \boldsymbol{Z}_i$ captures exogenous channel characteristics (subscriber scale, duration, video age).

The partial derivatives governing strategic behavior are:
$$\frac{\partial \pi_i}{\partial x_i} = \beta_1 + 2\beta_2 x_i + \beta_5 \bar{x}_{-i}, \quad \frac{\partial^2 \pi_i}{\partial x_i \partial \bar{x}_{-i}} = \beta_5$$

### 2.2 Mathematical Well-Posedness Conditions (Proposal Eq. 3)
For the game to be mathematically well-behaved, three conditions must hold:
1. **Strict Individual Concavity ($\beta_2 < 0$):** Guarantees a unique interior best-response for every competitor profile.
2. **Contraction Mapping / Unique & Stable Nash Equilibrium ($|\beta_5| < 2|\beta_2|$):** Ensures the best-response mapping is a contraction with Lipschitz constant $k = \frac{|\beta_5|}{2|\beta_2|} < 1$, preventing cyclic divergence or multiple equilibria.
3. **Strict Social Welfare Concavity ($\beta_2 + \beta_5 < 0$):** Ensures total social welfare $W(\boldsymbol{x})$ is strictly concave, yielding a unique social optimum.

### 2.3 Exact Potential Game Structure (Monderer & Shapley, 1996)
The game is an **exact potential game** admitting the global potential function $\Phi: [0, 1]^n \to \mathbb{R}$:
$$\Phi(\boldsymbol{x}) = \sum_{i=1}^n \left(\beta_1 x_i + \beta_2 x_i^2\right) + \frac{\beta_5}{n-1} \sum_{i < j} x_i x_j$$
which satisfies the exact potential property:
$$\frac{\partial \Phi(\boldsymbol{x})}{\partial x_i} = \beta_1 + 2\beta_2 x_i + \frac{\beta_5}{n-1} \sum_{j \neq i} x_j = \beta_1 + 2\beta_2 x_i + \beta_5 \bar{x}_{-i} = \frac{\partial \pi_i(x_i, \bar{x}_{-i})}{\partial x_i} \quad \forall i$$
The Hessian matrix $\nabla^2 \Phi(\boldsymbol{x})$ has two distinct eigenvalues:
$$\lambda_1 = 2\beta_2 + \beta_5 \quad (\text{along the all-ones direction } \boldsymbol{1}), \quad \lambda_2 = 2\beta_2 - \frac{\beta_5}{n-1} \quad (\text{along the orthogonal subspace } \boldsymbol{1}^\perp)$$
Under the contraction condition $|\beta_5| < 2|\beta_2|$, both $\lambda_1 < 0$ and $\lambda_2 < 0$. Therefore, $\Phi$ is strictly concave on $[0, 1]^n$, its unique global maximizer coincides with the unique Nash equilibrium, and decentralized gradient dynamics converge to $x^*$.

---

## 3. Closed-Form Equilibrium Solutions

### 3.1 Best Response & Symmetric Nash Equilibrium (Proposal Eq. 4 & 5)
Setting $\frac{\partial \pi_i}{\partial x_i} = 0$:
$$\text{BR}_i(\bar{x}_{-i}) = \text{clip}\left(\frac{\beta_1 + \beta_5 \bar{x}_{-i}}{2|\beta_2|}, 0, 1\right)$$
In any symmetric profile $x_1 = \dots = x_n = x$, we have $\bar{x}_{-i} = x$. The unique symmetric Nash equilibrium is the fixed point $\text{BR}(x) = x$:
$$x^{\text{Nash}} = \text{clip}\left(\frac{\beta_1}{2|\beta_2| - \beta_5}, 0, 1\right)$$

### 3.2 Social Optimum (Proposal Eq. 6)
A social planner maximizes total expected log-view welfare $W(\boldsymbol{x}) = \sum_{i=1}^n \pi_i(x_i, \bar{x}_{-i})$. At a common intensity $x_1 = \dots = x_n = x$:
$$\pi(x, x) = (\beta_1 + \beta_3) x + (\beta_2 + \beta_5) x^2 + c$$
Setting the derivative to zero yields:
$$x^{\text{SO}} = \text{clip}\left(\frac{\beta_1 + \beta_3}{2(|\beta_2| - \beta_5)}, 0, 1\right)$$

### 3.3 Marginal External Effect & The Nash-Optimum Gap (Proposal Eq. 7 & 8)
At the Nash profile, the external effect imposed by one creator on peers is:
$$E = \beta_3 + \beta_5 x^{\text{Nash}}$$
Each creator ignores $E$ when choosing $x_i$. The exact equilibrium gap is:
$$\Delta = x^{\text{SO}} - x^{\text{Nash}} = \frac{E}{2(|\beta_2| - \beta_5)}$$
- **Attention Congestion ($H_1: E < 0$):** Decentralized competition overshoots the social optimum ($x^{\text{Nash}} > x^{\text{SO}}$) because negative externalities push creators into an aggressive clickbait arms race.
- **Positive Network Spillover ($E > 0$):** Creators under-invest relative to the social optimum ($x^{\text{Nash}} < x^{\text{SO}}$).

### 3.4 Price of Anarchy (Proposal Eq. 9 & 10)
Measuring social welfare as total predicted views $V(x) = \exp(\pi(x, x))$:
$$\text{PoA} = \frac{V(x^{\text{SO}})}{V(x^{\text{Nash}})} = \exp\left[\pi(x^{\text{SO}}, x^{\text{SO}}) - \pi(x^{\text{Nash}}, x^{\text{Nash}})\right] = \exp\left(\frac{E^2}{4(|\beta_2| - \beta_5)}\right)$$
Because $E^2 \ge 0$ and $|\beta_2| - \beta_5 > 0$, $\text{PoA} \ge 1.0$ structurally, with $\text{PoA} = 1.0$ if and only if $E = 0$.

---

## 4. Empirical Evaluation: Calibrated Strategic Game

To address the empirical econometric findings from the full $N=809$ sample under Option B ($\kappa=8.0$), we evaluate the calibrated empirical game:

### Summary of Strategic Model

| Metric | Calibrated Baseline ($N=809$, Assumed $\kappa=8$) |
| :--- | :---: |
| **Status** | **Calibrated Empirical Baseline (Option B)** |
| **Parameters** | $\beta_1=-2.0190, \beta_2=-12.1974, \beta_3=-5.5424, \beta_5=20.7942$ |
| **Strategic Nature** | Strategic Complements ($\beta_5 > 0$) |
| **Contraction Rate ($k$)** | $0.8524 < 1.0$ (Contraction mapping holds) |
| **Hessian Eigenvalues** | $\lambda_1 = -3.6005, \ \lambda_2 = -24.9567$ |
| **Potential Concavity** | **Strictly Concave** ($\lambda_1, \lambda_2 < 0$) |
| **Welfare Concavity** | **Violated** ($\beta_2 + \beta_5 = +8.5969 > 0$, Convex!) |
| **Well-Posedness** | `all_conditions_satisfied: false` (Convex welfare) |
| **Symmetric Nash ($x^*$)** | **$0.0000$** (Clipped from $-0.5608$) |
| **Social Optimum ($x^{\text{SO}}$)** | **$1.0000$** (Boundary maximizer) |
| **Marginal Externality ($E$)** | **$-5.5424 < 0$** (Negative attention congestion) |
| **Nash-SO Gap ($\Delta$)** | **$+1.0000$** (Severe under-investment) |
| **Price of Anarchy ($\text{PoA}$)** | **$2.8166$** (Definition-based, $181.7\%$ view gap) |

### Key Theoretical Insights & Methodological Caveats:
1. **Regime A is NOT Theoretically Well-Posed in the Classical Concave Sense:**
   While the best-response mapping is contractive ($k = 0.8524 < 1.0$) ensuring a unique symmetric Nash equilibrium at the lower boundary ($x_{\text{Nash}} = 0.0$), **welfare concavity fails** because $\beta_2^{\text{eff}} + \beta_5 = -12.1974 + 20.7942 = +8.5969 > 0$ ($B = |\beta_2^{\text{eff}}| - \beta_5 = -8.5969 < 0$). In `results/equilibrium_results.json`, `all_conditions_satisfied: false`. Total welfare is strictly convex in symmetric appeal, so the social optimum is located at the upper boundary ($x_{\text{SO}} = 1.0$).
2. **Severe Under-Investment Rather than Overshooting:**
   Because $x_{\text{Nash}} = 0.0$ and $x_{\text{SO}} = 1.0$, decentralized competition yields minimal click appeal while social welfare coordinates at maximum appeal ($\Delta = +1.0$).
3. **Convex Welfare Dip Under Weak Interventions:**
   Because welfare is convex in symmetric appeal and $E < 0$, weak mediator recommendations ($\lambda \le 0.07$) produce a temporary welfare dip before recovering toward the social optimum as $\lambda \to 1$.

---

## 5. Parametric Bootstrap Uncertainty Analysis

Because parameter standard errors carry estimation noise once channel scale is controlled, we performed a **10,000-draw parametric bootstrap** drawing parameter vectors directly from $\boldsymbol{\beta} \sim \mathcal{N}(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{\Sigma}}_{\beta})$ using the uncentered cluster-robust covariance matrix of Model 5 ($N=809$, Option B):

```text
Parametric Bootstrap Diagnostics (10,000 draws ~ N(beta_hat, Sigma_hat)):
  P(beta_2 < 0) [Individual Concavity]:         92.96%
  P(B > 0) [Welfare Concavity]:                 44.25%
  P(|beta_5| < 2|beta_2|) [Contraction]:        27.61%
  P(Well-Posed Potential Game - All 3 Conds):   17.96% (1,796 valid draws)
  P(Contraction-Valid, beta_2 < 0 & k < 1):     27.32% (2,732 valid draws)
  P(E < 0) [Congestion Overshoot - All Draws]:  22.95%
  P(E < 0) [Conditional on Well-Posedness]:     32.35%
  Fraction Valid with Clipped x_Nash:           1.95%
  Fraction Valid with Clipped x_SO:             17.98%
  Fraction Valid with Any Clipped Equilibrium:  18.37%

Percentile Intervals (Conditional on Well-Posedness, 1,796 draws):
  x^Nash:  [0.0256,  0.3653]  (Median: 0.2685, Mean: 0.2513)
  x^SO:    [0.0000,  1.0000]  (Median: 0.3715, Mean: 0.3759)
  Delta:   [-0.2647, 0.8358]  (Median: 0.0845, Mean: 0.1247)
  PoA:     [1.0006, 121.430]  (Median: 1.4665, Mean: 129.33)

Percentile Intervals (Contraction-Valid, 2,732 draws - Contains Point Estimate):
  x^Nash:  [0.0000,  0.3569]  (Median: 0.2124, Mean: 0.1788)
  x^SO:    [0.0000,  1.0000]  (Median: 0.4559, Mean: 0.5385)
  Delta:   [-0.2467, 1.0000]  (Median: 0.1802, Mean: 0.3597)
  PoA:     [0.0292, 17952.8]  (Median: 1.6502)
```

### Statistical Interpretation & Dual Conditioning:
- **Conditioning on Well-Posedness ($N=1,796$, 17.96%):** Requires $B > 0$ (concave welfare). This isolates the lower tail of $\beta_5$ and **excludes the empirical point estimate itself** ($B = -8.597 < 0$).
- **Contraction-Valid Set ($N=2,732$, 27.32%):** Requires only $\beta_2 < 0$ and $|\beta_5| < 2|\beta_2|$. **The empirical point estimate belongs to this set.** Equilibria are clipped to $[0, 1]$ and PoA is evaluated definitionally.
- **Inconclusive Direction of Inefficiency:** In both subsets, the confidence interval for $\Delta$ spans both negative (overshoot) and positive (under-investment) values, including zero. Only $22.95\%$ of all draws exhibit negative marginal externality ($E < 0$).

---

## 6. Limitations & Methodological Clarifications

1. **Assumed Regularization ($\kappa=8.0$):** In the unaugmented $N=809$ estimation, individual concavity is not statistically established ($\beta_{2,u} = -8.1974$, $p=0.26$). The quadratic cost parameter $\kappa=8.0$ is an assumed regularizer ($\beta_{2,\text{eff}} = \beta_{2,u} - \kappa/2 = -12.1974$) rather than an empirically estimated cost.
2. **Identification of Peer Packaging:** Within a single niche at any time $t$, competitor leave-one-out averages $\bar{x}_{-i,t}$ vary across creators only through the focal creator's own packaging $x_{it}$ ($\partial \bar{x}_{-i,t} / \partial x_{it} = -1/(n-1)$). Cross-sectional peer variation is therefore $O(1/n)$, meaning $\beta_3$ and $\beta_5$ are identified primarily through aggregate time variation across the 24-month window.
3. **Mediator Interpretation:** Because potential games with unique contraction equilibria have no correlated equilibria outside the Nash equilibrium, the platform mediator is formally an **$\varepsilon$-Nash incentive-compatibility device** ($\varepsilon$-IC) evaluating unilateral deviation gains, rather than an expanded Aumann correlated equilibrium.

---

## 7. Generated Publication Figures

All plots are rendered at 300 DPI and saved in `plots/`:

1. `01_best_response_and_nash.png`:
   - Dual-panel plot contrasting best-response curves $\text{BR}_i(\bar{x}_{-i})$ against the 45-degree line and cobweb contraction iteration for the Calibrated Empirical Baseline ($k = 0.8524$).
2. `02_welfare_nash_vs_social_optimum.png`:
   - Symmetric log-view welfare $\pi(x, x)$ comparing Nash outcomes against Social Optima, shading the Price of Anarchy efficiency loss ($\text{PoA} = 2.817$), along with the mediator recovery curve $R(\lambda)$.
3. `03_potential_function_landscape.png`:
   - 2D contour surface of the exact potential function $\Phi(x_1, x_2)$, proving strict global concavity with its peak at the unique symmetric Nash equilibrium.
4. `04_poa_sensitivity_heatmap.png`:
   - 2D sensitivity heatmaps exploring Price of Anarchy variations across curvature $\beta_2^{\text{eff}}$ and strategic interaction $\beta_5$.

---

## 8. Machine-Readable Handover for Simulations (Part 4)

The outputs are exported and immediately importable by Part 4:
- Python Class: `game_theoretic_formulation.scripts.game_theory_solver.PotentialGame`
- Data Artifacts:
  - `results/equilibrium_results.json`: Full nested dictionary with parameters, equilibria, eigenvalues, covariance diagnostics, and $\varepsilon$-IC mediator limits.
  - `results/equilibrium_summary.csv`: Tabular summary of Calibrated Baseline Regime A.
  - `results/sensitivity_analysis_grid.csv`: 125-point parameter sweep.
  - `results/bootstrap_poa_distribution.csv`: Percentiles of bootstrap distributions.

### Ready-to-Use Code Snippet for Simulations:
```python
import sys
sys.path.append("../game_theoretic_formulation/scripts")
from game_theory_solver import PotentialGame

# Load empirical game (Option B, kappa=8.0)
game = PotentialGame.from_json("../payoff_calculation/results/game_theory_coefficients.json")

# Ground truth benchmarks for Part 4
x_nash = game.x_nash        # 0.0000 (clipped)
x_so = game.x_so            # 1.0000 (boundary maximizer)
poa = game.price_of_anarchy # 2.8166

# Mediator formulas
lam_max = game.mediator_max_lambda(epsilon=0.01) # 0.0051 (admissible lambda at 1% deviation budget)
welfare_rec = game.welfare_recovery(lam_max)     # Welfare recovery R(lambda)
```

---

## 9. Verification & Test Suite

All mathematical identities are tested with `unittest`:
```bash
python game_theoretic_formulation/tests/test_game_theory.py
```
```text
Ran 14 tests in 0.029s: OK
- test_well_posedness: Concavity, contraction rate < 1, potential eigenvalues
- test_nash_is_fixed_point_of_best_response: BR(x*) == x*
- test_social_optimum_maximizes_symmetric_welfare: W'(x^SO) == 0
- test_exact_potential_gradient: dPhi/dx_i == dpi_i/dx_i (finite-difference verified)
- test_price_of_anarchy_identity: exp(E^2 / 4B) == exp(pi(SO) - pi(Nash))
- test_mediator_properties: R(0)=0, R(1)=1, R(0.5)=0.75, G(lam_max) <= eps
- test_congestion_regime: x_so < x_nash verified when E < 0
- test_clipped_nash_case: unclipped Nash > 1 correctly clips to 1.0
- test_clipped_so_case: unclipped SO boundary clipping to 0.0 and 1.0
- test_poa_definition_vs_closed_form: PoA by definition matches closed form when interior, diverges when clipped
- test_nash_so_gap_uses_clipped_values: Delta uses clipped values; closed form preserved
- test_from_json_raises_on_missing_key: Missing keys raise KeyError without fallback
- test_split_well_posedness_conditions: Independent checks for concavity, contraction, welfare concavity, and Eq. 3
- test_mediator_with_clipped_equilibria: Recommendation and welfare recovery remain valid under boundary clipping
```

