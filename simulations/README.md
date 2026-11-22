# Simulation Experiments: Multi-Agent Best-Response Dynamics, Learning & Mediator Analysis

**CS 329: Foundations of AI: Multi-Agent Systems (2026)**  
*Indian Institute of Technology Gandhinagar*  
**Component:** *Simulation Experiments -- Best-Response Dynamics, Payoff-Based Learning (RQ2), Welfare Verification, and Mediator Intervention*

---

## 1. Overview & Role

This component implements **empirical multi-agent simulations** using the game-theoretic framework from Part 3. All parameters flow from `payoff_calculation/results/game_theory_coefficients.json` ($N=809$, Option B cost-augmented $\kappa=8.0$) through the `PotentialGame` API and `make_regime` factory -- **no new data inputs required.**

It addresses two core research questions and empirical uncertainty:

- **RQ1 (Mediator):** At what recommendation strength $\lambda$ does the platform mediator remain $\epsilon$-incentive-compatible, and what log-view and view welfare recovery does it achieve?
- **RQ2 (Learning):** Does decentralised payoff-based trial-and-error learning converge to the Nash equilibrium without knowledge of the payoff function's parameters?
- **Parametric Uncertainty (Bootstrap):** Evaluates finite-sample sampling uncertainty across 10,000 draws from the cluster-robust parameter covariance matrix, reporting both all-draw diagnostics and dual conditional subsets (well-posed $N=1,796$ vs. contraction-valid $N=2,732$).

Under the calibrated $N=809$ parameters with Option B cost-augmentation ($\kappa=8.0$), the unique Nash equilibrium is at $x_{\mathrm{Nash}} = 0.0000$ and the Social Optimum is at $x_{\mathrm{SO}} = 1.0000$ ($\mathrm{PoA} = 2.8161$). Decentralized competition yields zero click appeal while the social optimum coordinates at maximum click appeal.

---

## 2. Folder Structure

```
simulations/
├── README.md
├── requirements.txt
├── scripts/
│   ├── sim_dynamics.py          # Exp 1 & 2: BR dynamics + make_regime factory
│   ├── sim_mediator.py          # Exp 3: mediator lambda-sweep (+ numeric columns)
│   ├── sim_regime_compare.py    # Master runner (--with-learning, --with-bootstrap, --with-sweeps)
│   ├── sim_learning.py          # RQ2: payoff-based learning + sweep helper
│   ├── sim_sweeps.py            # Task G: multi-parameter sweeps and plots 09-11
│   ├── sim_numeric.py           # Independent numeric verification helpers
│   └── sim_bootstrap.py         # Parametric bootstrap uncertainty analysis
├── results/
│   ├── dynamics_regime_A_empirical.csv
│   ├── mediator_sweep.csv
│   ├── regime_comparison_summary.csv
│   ├── learning_A_empirical.csv                       [--with-learning, sigma=0.05]
│   ├── learning_A_empirical_sigma_1.14.csv            [--with-learning, sigma=1.14]
│   ├── learning_trajectories_*.csv                    [--with-learning]
│   ├── bootstrap_draws.csv                            [--with-bootstrap]
│   ├── bootstrap_mediator_summary.csv                 [--with-bootstrap]
│   └── sweeps/                                        [via sim_sweeps.py]
│       ├── learning_sigma.csv / _summary.csv
│       ├── learning_eta.csv / _summary.csv
│       ├── learning_beta_5.csv / _summary.csv
│       ├── learning_n_creators.csv / _summary.csv
│       └── learning_kappa.csv / _summary.csv
├── plots/
│   ├── 01_br_convergence.png
│   ├── 02_welfare_trajectory.png
│   ├── 03_mediator_sweep.png
│   ├── 04_regime_comparison.png
│   ├── 05_learning_convergence.png                    [--with-learning]
│   ├── 06_learning_welfare.png                        [--with-learning]
│   ├── 07_learning_dispersion.png                     [--with-learning]
│   ├── 08_bootstrap_mediator.png                      [--with-bootstrap]
│   ├── 09_noise_sweep.png                             [via sim_sweeps.py]
│   ├── 10_step_size_sweep.png                         [via sim_sweeps.py]
│   └── 11_kappa_sensitivity.png                      [via sim_sweeps.py]
└── tests/
    └── test_simulations.py                            # 41 tests (all passing)
```

---

## 3. Mathematical Background

### 3.1 Synchronous Best-Response Dynamics

Agents iterate:

$$
x_i^{t+1} = \text{BR}_i\!\left(\bar{x}_{-i}^t\right) = \text{clip}\!\left(\frac{\beta_1 + \beta_5 \bar{x}_{-i}^t}{2|\beta_2^{\mathrm{eff}}|}, 0, 1\right)
$$

By the **contraction mapping theorem**, with $k = \frac{|\beta_5|}{2|\beta_2^{\mathrm{eff}}|} = 0.8524 < 1.0$ under Regime A Option B, dynamics contract to $x_{\mathrm{Nash}} = 0.0$ from any starting point. Independent numeric verification (`sim_numeric.py`) confirms closed-form equilibria via bracketed bisection on $g(x) = \text{BR}(x) - x$.

Theoretical error decay: $`\|x_t - x_{\mathrm{Nash}}\|_\infty \le k^t \cdot \|x_0 - x_{\mathrm{Nash}}\|_\infty`$.

### 3.2 Platform Mediator (RQ1)

Recommendation: $x_\lambda = (1-\lambda) x_{\mathrm{Nash}} + \lambda x_{\mathrm{SO}}$.

Deviation gain ($\epsilon$-IC check): $G(\lambda) = \pi(\text{BR}(x_\lambda), x_\lambda) - \pi(x_\lambda, x_\lambda)$.

Log-view welfare recovery ($R$ answers RQ1 for log-view welfare):

$$
R(\lambda) = \frac{W(x_\lambda) - W(x_{\mathrm{Nash}})}{W(x_{\mathrm{SO}}) - W(x_{\mathrm{Nash}})}
$$

*(Note: When welfare is strictly concave and equilibria are interior, this reduces to $2\lambda - \lambda^2$. When welfare is convex, as in Regime A, $R(\lambda) \ne 2\lambda - \lambda^2$ and is evaluated definitionally.)*

View welfare recovery ($R_V$ answers RQ1 for actual views, exp-scale):

$$
R_V(\lambda) = \frac{\exp(W(x_\lambda)) - \exp(W(x_{\mathrm{Nash}}))}{\exp(W(x_{\mathrm{SO}})) - \exp(W(x_{\mathrm{Nash}}))}
$$

### 3.3 Payoff-Based Trial-and-Error Learning (RQ2, Paper Eq. 16)

Creators observe **only** their own noisy payoff $y_i^t = \pi_i(x_i^t, \bar{x}_{-i}^t) + \epsilon$, with $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and $c_i = 0$. They never observe $\beta$ coefficients or best response functions; the learner receives a pure payoff callable.

- **Initialisation ($t=0$):** $x_i^0 \sim U[\text{range}]$, $d_i^0 \in \{-1, +1\}$ uniformly. Config supports `"uniform"`, `"low"` ($U[0, 0.3]$), `"high"` ($U[0.7, 1.0]$); pipeline calls use `init="both"` in `run_learning_regime` which runs both and reports each separately (not aggregated).
- **Boundary condition ($y^0$):** Evaluated at the round-0 profile so round $t=1$ has a valid comparison baseline.
- **Update rule ($t=1, 2, \dots, T$):**
  - Step size: $\eta_t = \frac{\eta}{\sqrt{t}}$
  - Direction: $d_i^t = d_i^{t-1} \text{ if } y_i^t \ge y_i^{t-1}, \text{ else } -d_i^{t-1}$
  - Action update: $x_i^{t+1} = \text{clip}\left(x_i^t + \eta_t d_i^t, 0, 1\right)$

- **Convergence Metrics:**
  - $`D_T = \max_i |x_i^T - x_{\mathrm{Nash}}|`$ (*primary* sup-norm dispersion metric at horizon $T$)
  - $`\text{frac\_creators\_within\_delta\_T} = \frac{1}{n} \sum_i \mathbf{1}\{|x_i^T - x_{\mathrm{Nash}}| < \delta\}`$ (*primary* creator share in $\delta$-ball)
  - $`\text{std\_across\_creators\_T} = \text{std}_i(x_i^T)`$
  - $`\Delta_T = |\bar{x}_T - x_{\mathrm{Nash}}|`$ (secondary population mean gap)
  - $`T_\delta = \min\{t : |\bar{x}_{t'} - x_{\mathrm{Nash}}| < \delta \text{ for ALL } t' \ge t \text{ up to } T\}`$; `NaN` if not within $\delta$ at $T$.
  - $`t_{\mathrm{first\_hit}} = \text{first round } t \text{ with } |\bar{x}_t - x_{\mathrm{Nash}}| < \delta`$ (appended column)
  - $`S_T = \text{Var}(\bar{x}_t \text{ over last } k \text{ rounds})`$

---

## 4. Experiments & Results

### Experiment 1 & 2 -- Best-Response Dynamics

**Setup:** 100 random starting profiles, $n = 38$ agents, synchronous BR, max 500 iterations.

| Metric | Calibrated Baseline ($N=809, \kappa=8$) |
|--------|-----------------------------------------|
| Converged (of 100) | 100 |
| Mean iterations | 36.2 |
| Mean error to Nash | < 1e-6 |
| Simulated PoA | 2.8166 |
| Theoretical PoA | 2.8166 (verified) |
| Numeric PoA (independent check) | 2.8166 (verified) |

**Key finding:** Under the calibrated empirical parameters ($N=809$ Option B, $\kappa=8.0$), contraction holds at $k = 0.8524 < 1.0$, converging to the unique stable Nash equilibrium $x_{\mathrm{Nash}} = 0.0000$ with social optimum $x_{\mathrm{SO}} = 1.0000$, yielding a Price of Anarchy $\mathrm{PoA} = 2.8166$.

### Experiment 3 -- Mediator Sweep (Regime A)

| $\epsilon$ | $\lambda_{\max}$ | $R(\lambda_{\max})$ log-view | $R_V(\lambda_{\max})$ view | Interpretation |
|---|-------|-------------------|-----------------|----------------|
| 0.001 | 0.0005 | -0.37% | -0.21% | Near-ineffective; negative recovery |
| 0.01 | 0.0051 | -3.68% | -2.06% | Near-ineffective; negative recovery |
| 0.05 | 0.0281 | -19.89% | -11.66% | Negative welfare dip |
| 0.10 | 0.0710 | -47.65% | -29.28% | Substantial initial welfare loss |

> [!NOTE]
> In Regime A, moving along the recommendation path $x_\lambda = (1-\lambda)x_{\text{Nash}} + \lambda x_{\text{SO}}$ from $x_{\text{Nash}}=0$ toward $x_{\text{SO}}=1$ encounters negative initial marginal externality ($E = \beta_3 = -5.5424 < 0$). Because welfare is convex ($\beta_2^{\mathrm{eff}} + \beta_5 = +8.597 > 0$), welfare initially decreases before rising to exceed $W(0)$ at $\lambda \approx 0.145$. Thus, small mediator interventions $(\lambda \le 0.07)$ achieve **negative** welfare recovery.

### RQ2 -- Payoff-Based Learning & Dispersion

**Summary of Learning Performance at Horizon $T=500$ (low and high starts reported separately):**

| Regime | Init | Payoff Noise $\sigma$ | Mean $D_T$ | Mean $\Delta_T$ | Frac Conv ($T_\delta$) | Share Within $\delta$ | Mean $T_\delta$ |
|--------|------|-----------------------|------------|------------------|------------------------|-----------------------|-----------------|
| **A** | low  | 0.05 | 0.0518 | 0.0148 | 1.00 | 97.9% | 5.1 |
| **A** | high | 0.05 | 0.3437 | 0.2656 | 0.00 | 0.0% | NaN |
| **A** | low  | 1.14 | 0.4533 | 0.1614 | 0.00 | 17.8% | NaN |
| **A** | high | 1.14 | 0.9617 | 0.7255 | 0.00 | 0.0% | NaN |
| **B** | low  | 0.05 | 0.0955 | 0.0046 | 1.00 | 75.1% | 18.8 |
| **B** | high | 0.05 | 0.1061 | 0.0062 | 1.00 | 75.9% | 8.4 |
| **B** | low  | 1.14 | 0.4438 | 0.1983 | 0.00 | 10.5% | NaN |
| **B** | high | 1.14 | 0.4955 | 0.2340 | 0.00 | 6.8% | NaN |
| **B'** | low  | 0.05 | 0.0941 | 0.0058 | 1.00 | 76.7% | 16.5 |
| **B'** | high | 0.05 | 0.0986 | 0.0051 | 1.00 | 75.1% | 9.4 |

> [!NOTE]
> **Regime A asymmetry:** With low starts ($U[0, 0.3]$, near $x_{\mathrm{Nash}}=0$), creators stay near Nash and $D_T \approx 0.05$. With high starts ($U[0.7, 1.0]$), the slow contraction ($k = 0.852$) and boundary dynamics prevent convergence within $T=500$ rounds ($D_T \approx 0.34$). This asymmetry is specific to Regime A's boundary equilibrium. Regimes B and B' (interior equilibrium, $k \approx 0.043$) converge from both directions with similar performance.

```python
LEARNING_CONFIG = {
    "eta": 0.1,
    "sigma": 0.05,
    "T_horizon": 500,
    "n_trials": 20,
    "delta": 0.05,
    "k_window": 50,
    "seed_base": 42,
    "init": "uniform",
}
```

---

## 5. Bootstrap Uncertainty Distribution

Parametric bootstrap draws ($N=10,000$) were generated from $\mathcal{N}(\hat{\beta}, \hat{\Sigma})$ using the cluster-robust parameter covariance matrix of Model 5 (uncentered).

### Full Population Diagnostics (Across ALL 10,000 Draws)
- $P(\beta_2 < 0) = 92.96\%$
- $P(B > 0) = 44.25\%$
- $P(\text{contraction}, |\beta_5| < 2|\beta_2|) = 27.61\%$
- $P(\text{well-posed}, \beta_2 < 0 \land B > 0 \land |\beta_5| < 2|\beta_2|) = 17.96\%$ (1,796 draws)
- $P(E < 0) = 22.95\%$
- $P(x_{\mathrm{SO}} \text{ clipped to } \{0, 1\}) = 59.45\%$
- $P(x_{\mathrm{Nash}} \text{ clipped to } \{0, 1\}) = 18.20\%$

### Dual Conditioning Subsets:
1. **Conditional on Well-Posed ($N=1,796$, 17.96%):** Enforces concavity of welfare ($B > 0$). This subset isolates the lower tail of $\beta_5$ and **excludes the empirical point estimate** (which has $B = -8.597 < 0$).
2. **Contraction-Valid ($N=2,732$, 27.32%):** Requires only $\beta_2 < 0$ and $|\beta_5| < 2|\beta_2|$ (no $B > 0$ requirement). **The empirical point estimate belongs to this set.**

| Metric | Point Estimate | Well-Posed Median [95% CI] | Contraction-Valid Median [95% CI] |
|--------|----------------|----------------------------|-----------------------------------|
| $\lambda_{\max}(\epsilon=0.01)$ | 0.0051 | 0.1739 [0.0477, 1.0000] | 0.1221 [0.0023, 1.0000] |
| Log-View Recovery $R(\lambda_{\max})$ | -0.0368 | 0.2861 [0.0622, 1.0000] | 0.2046 [-0.0728, 1.0000] |
| View Recovery $R_V(\lambda_{\max})$ | -0.0206 | 0.2479 [0.0011, 1.0000] | 0.1521 [-0.0285, 1.0000] |
| Contraction Rate $k$ | 0.8524 | 0.3207 [0.0163, 0.9404] | 0.4942 [0.0231, 0.9749] |
| Price of Anarchy $\mathrm{PoA}$ | 2.8166 | 1.4665 [1.0006, 121.43] | 1.6502 [0.0292, 17952.82] |

---

## 6. Generated Figures

### Key Experimental Results

#### 1. Best-Response Dynamics Convergence
Convergence of 100 random initial profiles under synchronous best-response dynamics contracting to $x_{\mathrm{Nash}} = 0.0000$ at rate $k = 0.8524$.

![Best-Response Dynamics Convergence](plots/01_br_convergence.png)

#### 2. Social Welfare Trajectory & Price of Anarchy
Symmetric creator payoff trajectory across iterations, highlighting the $181.7\%$ social welfare coordination gap ($\text{PoA} = 2.8166$).

![Welfare Trajectory and Price of Anarchy Gap](plots/02_welfare_trajectory.png)

#### 3. Platform Mediator Recommendation Sweep
Unilateral deviation gain $G(\lambda)$ and welfare recovery $R(\lambda)$, illustrating the initial negative welfare dip before recovery.

![Platform Mediator Recommendation Sweep](plots/03_mediator_sweep.png)

#### 4. Payoff-Based Trial-and-Error Learning Convergence
Mean strategy trajectories across 500 rounds under low noise ($\sigma = 0.05$, converging) versus empirical volatility ($\sigma = 1.14$, dispersed).

![Payoff-Based Learning Convergence](plots/05_learning_convergence.png)

### Complete Figure Catalog

| File | Description |
|------|-------------|
| `01_br_convergence.png` | Log-scale $\|x_t - x_{\mathrm{Nash}}\|_\infty$ vs iteration. |
| `02_welfare_trajectory.png` | Symmetric welfare $\pi(\bar{x}_t, \bar{x}_t)$ vs iteration with PoA gap. |
| `03_mediator_sweep.png` | $G(\lambda)$ and $R(\lambda)$ vs $\lambda$. For Regime A, $R(\lambda)$ axis extends below 0, showing $R=0$ crossing and note on convex welfare. |
| `04_regime_comparison.png` | Grouped bar chart comparing equilibria, PoA, $\lambda_{\max}$, and iterations across regimes. |
| `05_learning_convergence.png` | Population mean $\bar{x}_t$ vs round $t$ for all regimes. |
| `06_learning_welfare.png` | Noise-free view welfare $W_V(t)$ vs round $t$. |
| `07_learning_dispersion.png` | Individual dispersion $\max_i |x_i - x_{\mathrm{Nash}}|$ vs round $t$ with 10-90% band. |
| `08_bootstrap_mediator.png` | Histograms of $\lambda_{\max}$, $R$, and $R_V$ across well-posed bootstrap draws. |
| `09_noise_sweep.png` | Sensitivity of dispersion $D_T$ and mean gap $\Delta_T$ across payoff noise $\sigma \in [0, 1.14]$. |
| `10_step_size_sweep.png` | Sensitivity of dispersion $D_T$ and mean gap $\Delta_T$ across base step sizes $\eta \in [0.01, 0.4]$. |
| `11_kappa_sensitivity.png` | Regime A sensitivity across assumed cost parameter $\kappa \in [4.5, 16.0]$ (equilibria, $k$, $\text{PoA}$, mediator, learning). |

---

## 7. How to Run

```bash
# Core pipeline + plots 01-04
python3 simulations/scripts/sim_regime_compare.py

# Complete pipeline with learning, bootstrap, and sweeps (all plots 01-11)
python3 simulations/scripts/sim_regime_compare.py --with-learning --with-bootstrap --with-sweeps

# Run standalone sweeps
python3 simulations/scripts/sim_sweeps.py

# Run test suite
python3 simulations/tests/test_simulations.py
```

---

## 8. Verification & Test Suite

```
Ran 41 tests in ~0.5s
OK
```

All 41 simulation tests and 14 game theory tests pass unconditionally.
