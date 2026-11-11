# Empirical Payoff Function Estimation

**CS 329: Foundations of AI: Multi-Agent Systems (2026)**  
*Indian Institute of Technology Gandhinagar*  
**Project:** Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube  
**Author:** Bhavya Shah (Roll: 24110324) — *Payoff Function Estimation, Regression Analysis & Statistical Evaluation*  
**Collaborators:**  
- Ankita Kushwaha (24110043) — Data Preprocessing & Click Appeal Score  
- Khushi Shah (24110326) — Game-Theoretic Formulation & Social Welfare Analysis  
- Harshita Singh (24110342) — Platform Mediation & Multi-Agent Learning Dynamics  

---

## 1. Executive Summary & Purpose

This component provides the empirical econometric foundation for modeling YouTube title/thumbnail competition as an exact potential game. Rather than assuming hypothetical or synthetic creator payoffs, we estimate the parameters of the game-theoretic payoff function directly from real-world multimodal video data ($N = 186$ videos across 38 creator channels).

The fitted parameters ($\beta_1, \beta_2, \beta_3, \beta_4, \beta_5$) and their cluster-robust standard errors are exported directly to machine-readable formats (`results/game_theory_coefficients.json` and `results/game_theory_coefficients.csv`) for handover to **Khushi Shah** for equilibrium derivation and Price of Anarchy (PoA) analysis, and **Harshita Singh** for trial-and-error multi-agent learning simulations.

---

## 2. Empirical Model Specification

Following Section 3.1 (Eq. 1) of the research proposal, the conditional expected log-view payoff of creator $i$ at time $t$ is estimated by:

$$\log(V_{it}) = \beta_0 + \beta_1 x_{it} + \beta_2 x_{it}^2 + \beta_3 \bar{x}_{-i,t} + \beta_5 x_{it}\bar{x}_{-i,t} + \beta_4 \log(s_i) + \gamma^\top Z_{it} + \varepsilon_{it}$$

where:
- $V_{it}$: Cumulative video view count. The log transform $\log(V_{it})$ stabilizes the heavy-tailed distribution of platform views.
- $x_{it} \in [0, 1]$: Multimodal **Click Appeal Score** of video $t$ produced by creator $i$ (0.4 Title Score + 0.6 Thumbnail Score).
- $x_{it}^2$: Quadratic own-score term capturing diminishing marginal returns and concavity.
- $\bar{x}_{-i,t}$: Leave-One-Out Competitor Average click appeal (strictly excluding creator $i$'s own videos).
- $x_{it}\bar{x}_{-i,t}$: Strategic interaction term allowing creator $i$'s best response to depend on competitor packaging.
- $s_i$: Channel total subscriber count ($\log(s_i)$ captures baseline audience scale).
- $Z_{it}$: Exogenous video-level control vector:
  - $\log(\text{duration}_{it})$: Natural log of video duration in seconds.
  - $\text{age}_{it}$: Elapsed days since publication (accounting for exposure time).
- $\varepsilon_{it}$: Disturbance term with channel-level clustering: $\mathbb{E}[\varepsilon_{it}\varepsilon_{is} \mid \text{channel}_i] \neq 0$.

### Reduced-Form Game-Theoretic Payoff
Collecting terms independent of $x_i$ into $c_i = \beta_0 + \beta_4 \log(s_i) + \gamma^\top Z_{it}$, the game-theoretic payoff reduces to:

$$\pi_i(x_i, \bar{x}_{-i}) = \beta_1 x_i + \beta_2 x_i^2 + \beta_3 \bar{x}_{-i} + \beta_5 x_i\bar{x}_{-i} + c_i$$

---

## 3. Data Integration & Feasibility Audit

The estimation pipeline ingests `Click appeal score/scores/video_scores.csv` without fabricating missing data or modifying upstream assets:

| Dimension | Empirical Finding | Methodological Implication |
| :--- | :--- | :--- |
| **Sample Size** | $N = 186$ videos | Sufficient for 5–7 parameter econometric estimation. |
| **Creator Panels** | 38 channels (37 with 5 videos, 1 with 1 video) | Unbalanced panel with creator-level clustering. Requires cluster-robust standard errors. |
| **Subscriber Counts** | 100% available ($1.18 \times 10^5 \le s_i \le 5.24 \times 10^8$) | Proposed full model with $\beta_4 \log(s_i)$ is fully estimable without imputation. |
| **Dependent Variable** | Min $= 1,132$, Max $= 215,604,900$, Median $= 1.38 \times 10^6$ | Zero non-positive views; $\log(V_{it})$ is smooth and approximately normally distributed. |
| **Temporal Span** | 2024-10-02 to 2026-10-01 (728 days / 24 calendar months) | Monthly clusters exist; necessitates rolling-window peer definitions rather than strict monthly groupings. |

---

## 4. Competitor Leave-One-Out (LOCO) Grouping Rule

To evaluate attention congestion ($\beta_3$) and strategic interaction ($\beta_5$), peer averages must exclude creator $i$'s own output ($\partial \bar{x}_{-i}/\partial x_i = 0$):

1. **Why Strict Calendar Months Fail:**
   The temporal distribution contains 182 creator-month groups. In 3 calendar months, only 1 video was published; in several others, only 1 channel was active. A strict monthly grouping results in undefined (NaN) peer averages for isolated videos and extreme small-sample noise.
2. **Primary Specification: Rolling 90-Day Window LOCO:**
   For each video $(i, t)$, the competitor pool $\mathcal{C}_{-i,t}$ consists of all videos published by competing creators $j \neq i$ in the window $[t - 45\text{ days}, t + 45\text{ days}]$:
   $$\bar{x}_{-i,t} = \frac{1}{|\mathcal{C}_{-i,t}|} \sum_{k \in \mathcal{C}_{-i,t}} x_k$$
   For 4 temporal boundary cases where $|\mathcal{C}_{-i,t}| < 3$, a smooth fallback to the Global Niche LOCO average is applied, preventing spurious outlier peer values.
3. **Robustness Check: Global Niche LOCO:**
   All 181 competing videos from the other 37 creators across the entire two-year dataset are averaged.

---

## 5. Econometric Estimation & Hypothesis Results

All models are estimated using Ordinary Least Squares with **Cluster-Robust Standard Errors** clustered at the channel level (38 clusters), guarding against deflated standard errors from repeated observations.

### Core Parameter Estimates (Full Proposed Model M4 vs. Quadratic Model M2)

$$\text{Full Model (M4): } \log(V) = -7.984 + 8.302 x - 9.168 x^2 + 3.802 \bar{x} - 0.782 (x\bar{x}) + 1.067 \log(s) + 0.274 \log(\text{dur}) + 0.0001 \text{age}$$

| Parameter | Theoretical Role | Expected Sign | Point Estimate (Cluster SE) | $t$-statistic | $p$-value (2-tailed) | $p$-value (1-tailed) | Empirical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| $\beta_1$ ($x_{it}$) | Linear own-appeal return | $+$ | **$+8.3020$** ($20.255$) | $+0.41$ | $0.6819$ | $0.3410$ | **Supported in sign** (+7.50 in M2, $p_{1\text{-tail}}=0.088$). |
| $\beta_2$ ($x_{it}^2$) | Diminishing returns (concavity) | $-$ | **$-9.1677$** ($8.438$) | $-1.09$ | $0.2773$ | $0.1386$ | **Supported in sign** (-8.57 in M2, $p_{1\text{-tail}}=0.150$). |
| $\beta_3$ ($\bar{x}_{-i,t}$) | Attention congestion | $-$ | **$+3.8021$** ($18.588$) | $+0.20$ | $0.8379$ | $0.4190$ | Inconclusive (imprecise in rolling window; negative in global LOCO). |
| $\beta_5$ ($x_{it}\bar{x}_{-i,t}$) | Strategic complements | $+$ | **$-0.7822$** ($46.475$) | $-0.02$ | $0.9866$ | $0.5067$ | Near-zero point estimate; cannot reject $\beta_5 = 0$. |
| $\beta_4$ ($\log(s_i)$) | Audience scale elasticity | $+$ | **$+1.0668$** ($0.114$) | $+9.35$ | $< 10^{-20}$ | $< 10^{-20}$ | **Statistically overwhelming**; primary driver of view scale. |

### Model Fit & Comparison Summary

| Model Specification | Features Included | $R^2$ | Adj. $R^2$ | AIC | BIC | RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline Linear** | $x, \log(s), \log(\text{dur}), \text{age}$ | $0.6446$ | $0.6367$ | $688.08$ | $704.21$ | $1.4285$ |
| **M2: Quadratic Own-Score** | $x, x^2, \log(s), \log(\text{dur}), \text{age}$ | $0.6473$ | $0.6375$ | $688.66$ | $708.01$ | $1.4230$ |
| **M3: Peer Average** | $x, x^2, \bar{x}, \log(s), \log(\text{dur}), \text{age}$ | $0.6479$ | $0.6360$ | $690.35$ | $712.92$ | $1.4219$ |
| **M4: Full Proposed (Raw)** | $x, x^2, \bar{x}, x\bar{x}, \log(s), \log(\text{dur}), \text{age}$ | **$0.6479$** | **$0.6340$** | **$692.35$** | **$718.14$** | **$1.4219$** |
| **M4-Centered: Mean-Centered** | $x_c, x_c^2, \bar{x}_c, x_c\bar{x}_c, \log(s), \log(\text{dur}), \text{age}$ | $0.6479$ | $0.6340$ | $692.35$ | $718.14$ | $1.4219$ |
| **R1: Global LOCO** | $x, x^2, \bar{x}^{glob}, x\bar{x}^{glob}, \log(s), \dots$ | $0.6496$ | $0.6358$ | $691.43$ | $717.22$ | $1.4184$ |
| **R2: Without Subscribers** | $x, x^2, \bar{x}, x\bar{x}, \log(\text{dur}), \text{age}$ | $0.0911$ | $0.0658$ | $857.73$ | $877.08$ | $2.3168$ |

*Note on R2 (No Subscribers):* Omitting $\log(s_i)$ causes $R^2$ to drop from $64.8\%$ to $9.1\%$, proving empirically that platform view scale is anchored in subscriber distribution.

---

## 6. Multicollinearity & Centering Resolution

In polynomial and interaction models, uncentered product terms ($x^2$ and $x\bar{x}$) naturally correlate with the linear terms, inflating Variance Inflation Factors (VIFs):
- **Raw Uncentered Model:** $\text{VIF}(x) = 824.1$, $\text{VIF}(x\bar{x}) = 770.9$.
- **Mean-Centered Model ($x_c = x - \bar{x}$):** All VIFs drop below **1.25** ($\text{VIF}(x_c) = 1.16$, $\text{VIF}(x_c^2) = 1.25$, $\text{VIF}(\bar{x}_c) = 1.07$, $\text{VIF}(x_c\bar{x}_c) = 1.11$).
- **Invariance:** Centering preserves identical estimates for the curvature parameter ($\beta_2 = -9.1677$) and interaction ($\beta_5 = -0.7822$).

---

## 7. Handover to Game-Theoretic Analysis (Khushi Shah)

All three theoretical well-posedness conditions stipulated in Eq. (3) of the proposal are **strictly satisfied**:

$$\begin{aligned}
\text{Condition 1 (Strict Concavity): } & \beta_2 = -9.1677 < 0 \quad &\textbf{[PASS]} \\
\text{Condition 2 (Best-Response Contraction): } & -18.335 < \beta_5 = -0.7822 < 9.1677 \quad &\textbf{[PASS]} \\
\text{Condition 3 (Social Welfare Concavity): } & \beta_2 + \beta_5 = -9.9499 < 0 \quad &\textbf{[PASS]}
\end{aligned}$$

### Derived Equilibria in Strategy Space $[0, 1]$

1. **Symmetric Nash Equilibrium (Eq. 5):**
   $$x_{Nash} = \text{clip}\left(\frac{\beta_1}{2|\beta_2| - \beta_5}, 0, 1\right) = \frac{8.3020}{2(9.1677) - (-0.7822)} = \frac{8.3020}{19.1176} = \mathbf{0.4343}$$
   - **Result:** $x_{Nash} = 0.4343$ is strictly interior ($0 < 0.4343 < 1$).
   - **Empirical Alignment:** Sits directly within the empirical click appeal distribution (mean $= 0.3958$, 75th percentile $= 0.4558$), perfectly capturing competitive click appeal escalation.

2. **Social Optimum (Eq. 6):**
   $$x_{SO} = \text{clip}\left(\frac{\beta_1 + \beta_3}{2(|\beta_2| - \beta_5)}, 0, 1\right) = \frac{8.3020 + 3.8021}{2(9.1677 - (-0.7822))} = \frac{12.1041}{19.8998} = \mathbf{0.6082}$$
   - **Result:** $x_{SO} = 0.6082$ is strictly interior.

---

## 8. Publication-Quality Diagnostic Visualizations

Generated plots in `plots/`:
1. `01_actual_vs_predicted.png`: Scatter plot of observed vs. fitted $\log(V_{it})$ with identity line, $R^2 = 0.648$, and cluster metrics.
2. `02_residual_diagnostics.png`: 4-panel diagnostic (Residuals vs. Fitted, Normal Q-Q, Density, and Channel-level boxplots).
3. `03_payoff_response_curves.png`: Empirical payoff function $\pi_i(x_i, \bar{x}_{-i})$ across low, median, and high competitor packaging, marking the interior peak ($x^* \approx 0.434$).
4. `04_peer_interaction_heatmap.png`: 2D contour map of $\pi_i(x_i, \bar{x}_{-i})$ illustrating the contraction best-response curve and symmetric Nash equilibrium.
5. `05_variable_importance_vif.png`: Multicollinearity resolution comparing uncentered vs. centered VIFs.
6. `06_subscribers_dominance.png`: Frisch-Waugh-Lovell partial regression showing subscriber base dominance vs. marginal click appeal incentives.

---

## 9. Research Limitations & Cautionary Framing

1. **Association vs. Causation:** This model reflects observational cross-sectional associations on YouTube. We do not claim exogenous random assignment or direct causal identification.
2. **Views vs. CTR:** Cumulative views are an aggregate platform outcome governed by algorithmic recommendations, subscribers, and watch time—not pure click-through rate (CTR).
3. **Sample Scope:** The dataset consists of 186 videos from 38 creators in the challenge/food entertainment niche; extrapolation to other genres should be done with caution.
4. **Dominance of Subscriber Scale:** Channel subscriber scale accounts for over $60\%$ of view variance. Title/thumbnail packaging operates as a fine-tuning marginal incentive ($x_{it}$), not a magic substitute for audience distribution.

---

## 10. Pipeline Execution & File Manifest

To reproduce all results end-to-end:
```bash
python payoff_estimation/payoff_estimation.py
```

### Component Structure
```text
payoff_estimation/
├── data/
│   ├── prepared_payoff_data.csv          <- 186 rows, 68 columns with LOCO averages & controls
│   └── data_dictionary.csv               <- Complete variable dictionary
├── scripts/
│   ├── prepare_data.py                   <- Data preparation & LOCO peer construction
│   ├── run_regressions.py                <- Econometric estimation & parameter export
│   └── generate_plots.py                 <- Generates publication diagnostic plots
├── results/
│   ├── regression_results.csv            <- Coefficients, Cluster SE, OLS SE, HC3 SE, CIs
│   ├── model_comparison.csv              <- Comparison across M1–M4, R1–R4 (R2, AIC, BIC)
│   ├── multicollinearity_diagnostics.csv <- VIF analysis (raw vs centered)
│   ├── hypothesis_tests.csv              <- Formal hypothesis test verdicts (H1–H4)
│   ├── game_theory_coefficients.json     <- Machine-readable handover for Khushi Shah
│   └── game_theory_coefficients.csv      <- CSV counterpart of game-theoretic parameters
├── plots/
│   ├── 01_actual_vs_predicted.png
│   ├── 02_residual_diagnostics.png
│   ├── 03_payoff_response_curves.png
│   ├── 04_peer_interaction_heatmap.png
│   ├── 05_variable_importance_vif.png
│   └── 06_subscribers_dominance.png
├── logs/
│   └── master_pipeline.log
├── payoff_estimation.py                  <- Master executable pipeline
└── README.md                             <- Comprehensive research documentation
```
