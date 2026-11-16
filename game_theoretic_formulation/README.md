# Game-Theoretic Formulation & Potential Game Analysis

**CS 329: Foundations of AI: Multi-Agent Systems (2026)**  
*Indian Institute of Technology Gandhinagar*  
**Project:** Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube Recommendation  
**Component:** *Game-Theoretic Formulation, Nash Equilibrium, Social Optimum & Price of Anarchy*  

---

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

## 4. Empirical Evaluation & Dual-Regime Analysis

To address the empirical econometric findings, we evaluate three strategic regimes:

### Summary of Strategic Regimes

| Metric | Regime A: Empirical Point Estimates (Model 5) | Regime B (Hypothetical): Attention Congestion (Complements) | Regime B' (Hypothetical): Attention Congestion (Empirical Interaction) |
| :--- | :---: | :---: | :---: |
| **Status** | **Calibrated Empirical Baseline** | **Hypothetical Benchmark** | **Hypothetical Benchmark** |
| **Parameters** | $\beta_1=8.3020, \beta_2=-9.1677, \beta_3=3.8021, \beta_5=-0.7822$ | $\beta_1=8.3020, \beta_2=-9.1677, \beta_3=-3.8021, \beta_5=+0.7822$ | $\beta_1=8.3020, \beta_2=-9.1677, \beta_3=-3.8021, \beta_5=-0.7822$ |
| **Strategic Nature** | Strategic Substitutes ($\beta_5 < 0$) | Strategic Complements ($\beta_5 > 0$) | Strategic Substitutes ($\beta_5 < 0$) |
| **Contraction Rate ($k$)** | $0.0427 < 1.0$ (Near dominant-strategy) | $0.0427 < 1.0$ (Strong contraction) | $0.0427 < 1.0$ (Near dominant-strategy) |
| **Hessian Eigenvalues** | $\lambda_1 = -19.1176, \ \lambda_2 = -18.3143$ | $\lambda_1 = -17.5532, \ \lambda_2 = -18.3565$ | $\lambda_1 = -19.1176, \ \lambda_2 = -18.3143$ |
| **Potential Concavity** | **Strictly Concave** ($\lambda_1, \lambda_2 < 0$) | **Strictly Concave** ($\lambda_1, \lambda_2 < 0$) | **Strictly Concave** ($\lambda_1, \lambda_2 < 0$) |
| **Symmetric Nash ($x^*$)** | **$0.4343$** (Interior) | **$0.4730$** (Interior) | **$0.4343$** (Interior) |
| **Social Optimum ($x^{\text{SO}}$)** | **$0.6082$** (Interior) | **$0.2683$** (Interior) | **$0.2261$** (Interior) |
| **Marginal Externality ($E$)** | **$+3.4624 > 0$** (Positive peer spillover) | **$-3.4321 < 0$** (Attention congestion) | **$-4.1418 < 0$** (Attention congestion) |
| **Nash-SO Gap ($\Delta$)** | **$+0.1740$** (Under-investment) | **$-0.2046$** (Clickbait Overshoot!) | **$-0.2081$** (Clickbait Overshoot!) |
| **Price of Anarchy ($\text{PoA}$)** | **$1.3515$** (35.1% view gap) | **$1.4208$** (42.1% view gap) | **$1.5388$** (53.9% view gap) |

### Key Theoretical Insights:
1. **$H_1$ Congestion Hypothesis Not Supported at Point Estimates:** At the calibrated point estimates (Regime A), $\beta_3 = +3.8021 > 0$ and $E = +3.4624 > 0$. Creators generate positive mutual view spillovers. Consequently, decentralized competition results in **under-investment** relative to the social optimum ($x^* = 0.434 < x^{\text{SO}} = 0.608$), meaning a platform planner would nudge creators toward *higher* click appeal.
2. **Near Dominant-Strategy Game:** The contraction rate $k = \frac{|\beta_5|}{2|\beta_2|} \approx 0.0427 \ll 1$ indicates that creators' best responses are nearly independent of peer actions (slope $s = -0.043$). Nearly all of the external gap $E = \beta_3 + \beta_5 x^*$ is driven by the level shift $\beta_3$, which affects group welfare but scarcely alters individual incentives.
3. **Hypothetical Attention Congestion Regimes (B and B'):** Under negative peer externalities ($\beta_3 = -3.8021$), decentralized competition cleanly overshoots the social optimum ($x^* > x^{\text{SO}}$), creating a classic clickbait arms race where Price of Anarchy rises to $1.4208$ (complements) and $1.5388$ (substitutes).

---

## 5. Parametric Bootstrap Uncertainty Analysis

Because parameter standard errors carry estimation noise once channel scale is controlled, we performed a **10,000-draw parametric bootstrap** drawing parameter vectors directly from $\boldsymbol{\beta} \sim \mathcal{N}(\hat{\boldsymbol{\beta}}, \hat{\boldsymbol{\Sigma}}_{\beta})$ using the uncentered cluster-robust covariance matrix:

```text
Parametric Bootstrap Diagnostics (10,000 draws ~ N(beta_hat, Sigma_hat)):
  P(beta_2 < 0) [Individual Concavity]:         86.57%
  P(B > 0) [Welfare Concavity]:                 59.73%
  P(|beta_5| < 2|beta_2|) [Contraction]:        33.76%
  P(Well-Posed Potential Game - All 3 Conds):   25.15% (2,515 valid draws)
  P(E < 0) [Congestion Overshoot - All Draws]:  27.97%
  P(E < 0) [Conditional on Well-Posedness]:     22.47%
  Fraction Valid with Clipped x_Nash:           0.52%
  Fraction Valid with Clipped x_SO:             13.36%
  Fraction Valid with Any Clipped Equilibrium:  13.36%

Percentile Intervals (Conditional on Well-Posedness, 2,515 draws):
  x^Nash:  [0.2928,  0.6532]  (Median: 0.3923)
  x^SO:    [0.0507,  1.0000]  (Median: 0.4816)
  Delta:   [-0.3048, 0.6350]  (Median: 0.0835)
  PoA:     [1.0003, 103.140]  (Median: 1.3225)
```

### Statistical Interpretation & Cautionary Notes:
- **PoA Lower Bound Interpretation:** Structurally, $\text{PoA} \ge 1.0$ by construction. An empirical bootstrap interval whose lower bound touches $1.0003$ indicates that efficiency loss is statistically indistinguishable from zero at conventional significance levels once estimation variance is recognized.
- **Inconclusive Direction of Inefficiency:** The 95% confidence interval for $\Delta = x^{\text{SO}} - x^*$ spans from $-0.3048$ (congestion overshoot) to $+0.6350$ (under-investment), and includes zero. Only $22.47\%$ of well-posed draws exhibit overshooting ($E < 0$), showing that empirical YouTube data cannot statistically confirm whether creator competition over- or under-invests in packaging.
- **Parametric Sampling Nature:** This bootstrap reflects parameter uncertainty from the estimated linear model, not non-parametric channel-level resampling.

---

## 6. Limitations & Methodological Clarifications

1. **Identification of Peer Packaging:** Within a single niche at any time $t$, competitor leave-one-out averages $\bar{x}_{-i,t}$ vary across creators only through the focal creator's own packaging $x_{it}$ ($\partial \bar{x}_{-i,t} / \partial x_{it} = -1/(n-1)$). Cross-sectional peer variation is therefore $O(1/n)$, meaning $\beta_3$ and $\beta_5$ are identified primarily through aggregate time variation across the 24-month window, which may partially absorb unobserved platform algorithm adjustments or temporal trends.
2. **Mediator Interpretation:** Because potential games with unique contraction equilibria have no correlated equilibria outside the Nash equilibrium, the platform mediator is formally an **$\varepsilon$-Nash incentive-compatibility device** ($\varepsilon$-IC) evaluating unilateral deviation gains, rather than an expanded Aumann correlated equilibrium.

---

## 7. Generated Publication Figures

All plots are rendered at 300 DPI and saved in `plots/`:

1. `01_best_response_and_nash.png`:
   - Dual-panel plot contrasting best-response curves $\text{BR}_i(\bar{x}_{-i})$ against the 45-degree line for Regime A and Regime B, illustrating contraction stability ($s = -0.043$ vs $s = +0.043$) and highlighting attention congestion overshooting ($x^* > x^{\text{SO}}$).
2. `02_welfare_nash_vs_social_optimum.png`:
   - Symmetric log-view welfare $\pi(x, x)$ comparing Nash outcomes against Social Optima, shading the Price of Anarchy efficiency loss.
3. `03_potential_function_landscape.png`:
   - 2D contour surface of the exact potential function $\Phi(x_1, x_2)$, proving strict global concavity with its peak at the unique symmetric Nash equilibrium $(0.434, 0.434)$.
4. `04_poa_sensitivity_heatmap.png`:
   - Two-panel 2D sensitivity heatmaps comparing empirical peer spillover ($\beta_3 = +3.80$) with the marked empirical point ($\text{PoA} = 1.3515$) against hypothetical attention congestion ($\beta_3 = -3.80$) using exact clipped Price of Anarchy across curvature $\beta_2 \in [-20, -3]$ and interaction $\beta_5 \in [-5, +5]$.

---

## 8. Machine-Readable Handover for Simulations (Part 4)

The outputs are exported and immediately importable by Part 4:
- Python Class: `game_theoretic_formulation.scripts.game_theory_solver.PotentialGame`
- Data Artifacts:
  - `results/equilibrium_results.json`: Full nested dictionary with parameters, equilibria, eigenvalues, covariance diagnostics, and $\varepsilon$-IC mediator limits.
  - `results/equilibrium_summary.csv`: Tabular comparison of Regime A, Regime B, and Regime B'.
  - `results/sensitivity_analysis_grid.csv`: 125-point parameter sweep.
  - `results/bootstrap_poa_distribution.csv`: Percentiles of bootstrap distributions.

### Ready-to-Use Code Snippet for Simulations:
```python
import sys
sys.path.append("../game_theoretic_formulation/scripts")
from game_theory_solver import PotentialGame

# Load empirical game
game = PotentialGame.from_json("../payoff_estimation/results/game_theory_coefficients.json")

# Ground truth benchmarks for Part 4
x_nash = game.x_nash        # 0.4343
x_so = game.x_so            # 0.6082
poa = game.price_of_anarchy # 1.3515

# Mediator formulas
lam_max = game.mediator_max_lambda(epsilon=0.01) # Max admissible lambda at 1% deviation budget
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

