# Methodology and Results

This note walks through the models, the portfolio and execution layers, the evaluation protocol, and what I found in the recorded experiments. Each section links to the code that implements it.

---

## 1. Residual Models

I split each stock's daily excess return into a systematic part, which I hedge, and a residual, which I trade. Estimation runs on a trailing 252-session window, and I refit every 5 sessions. Inputs are cross-sectionally winsorized at the median ± 10 MAD, and a stock needs at least 90% observed returns in the window to be fitted.

### ETF Model

I assign each stock the single ETF (SPY or a SPDR sector fund) that explains it best:

$$r_{i,t} = a_i + \beta_i\, r_{\text{ETF}(i),t} + \epsilon_{i,t}.$$

Assignments are scored by in-window R². Since switching a stock's hedge at every refit creates unnecessary churn, I keep the previous assignment unless a challenger beats it by more than 0.02 R². SPY is the fallback. I remove the intercept before building the residual path.

### PCA Model

Here I take the top *k* eigenportfolios of trailing returns and get residuals by regressing on those factor returns. The eigenportfolio weights give me an explicit stock-level hedge.

Both models share one `fit(returns, regressors, previous) -> FittedModel` interface, and a fitted model can produce residuals for any later window. This is what lets the diagnostics apply an as-of fit to future returns without re-estimating.

→ [`models/residuals.py`](../src/statarb/models/residuals.py)

## 2. Ornstein–Uhlenbeck Fit and Signals

I sum the last 60 residual returns into a path $X_t$ and model it as

$$dX_t = \kappa(m - X_t)\,dt + \sigma\, dW_t .$$

I fit the exact AR(1) discretization $X_{t+1} = a + bX_t + \varepsilon$ and map the coefficients back to OU parameters:

$$\kappa = -\ln b / \Delta t,\qquad m = \frac{a}{1-b},\qquad \sigma_{eq} = \frac{\sigma}{\sqrt{2\kappa}}.$$

A fit is only kept if $0 < b < 1$ and $\kappa \ge 8.4$/yr, which keeps the half-life at roughly 30 days or less.

In practice, this filter screens out much less than it seems to. On a 60-observation window, the AR(1) slope is biased toward reversion (the Dickey–Fuller bias, $E[1-b] \approx 5.5/n$). On the synthetic market, pure random walks get a median $\hat\kappa \approx 25$/yr and pass the filter 78% of the time (see [`examples/demo.py`](../examples/demo.py)). So the discrimination between reverting and non-reverting names comes from the s-score, not from the κ threshold.

The signal itself is the s-score, $s = (X - m)/\sigma_{eq}$. Positions follow a hysteresis state machine, so a name sitting near a threshold does not flip in and out of the book:

| Transition  | Threshold        |
| ----------- | ---------------- |
| Open long   | s < −1.25        |
| Open short  | s > +1.25        |
| Close long  | s > −0.50        |
| Close short | s < +0.75        |

→ [`models/ou.py`](../src/statarb/models/ou.py),
[`models/signals.py`](../src/statarb/models/signals.py),
[`models/pipeline.py`](../src/statarb/models/pipeline.py)

The rolling loop lives in `run_signals`. At each rebalance, it hands a `RebalanceContext` to a pluggable portfolio step. The context holds the fit, the scores, the OU parameters, and the estimation window as the model saw it on that date. My tests check that the context never extends past its own date, and that truncating the panel after a cutoff leaves every earlier target, cost, and P&L bit-identical.

## 3. Portfolio Construction

### Rule-Based Sizing

Positions are sized either equally or by inverse volatility, with a 2% per-name cap and gross leverage of 2.0 including hedges. ETF hedges aggregate $-\beta_i w_i$ by assigned ETF, and the book is then made dollar-neutral.

→ [`portfolio/construction.py`](../src/statarb/portfolio/construction.py)

### Target Smoothing

To slow down trading, I blend each new target with the previous one using a half-life $h$ measured in rebalances:
$q_t = 2^{-1/h} q_{t-1} + (1-2^{-1/h})\, q^{raw}_t$.

→ [`portfolio/turnover.py`](../src/statarb/portfolio/turnover.py)

### OU Expected Return

Over a rebalance interval of *d* sessions, the expected residual return implied by the OU fit is

$$\alpha_i = -s_i\, \sigma_{eq,i}\,\bigl(1 - e^{-\kappa_i d/252}\bigr).$$

### Online Calibration

The raw OU forecasts turned out to be far too large, so I rescale them by a pooled, through-origin slope clipped to [0, 1]. The slope is fitted on the last 52 matured rebalances. A forecast only enters training after its full execution-lagged holding period has been realized, which keeps future information out of the calibration. Until 26 such records exist, the slope stays at zero.

### Receding-Horizon Optimizer

The optimizer (CVXPY with Clarabel) plans *H* stages ahead and only executes the first. Each stage maximizes

$$\alpha_h^\top w_h \;-\; \gamma\, w_h^\top \Sigma\, w_h \;-\; \text{spread}\cdot|w_h - w_{h-1}| \;-\; \text{impact}\cdot|w_h - w_{h-1}|^{3/2},$$

where alpha decays geometrically across stages and Σ is the diagonal residual risk. I use a power-cone formulation for impact. Per-name caps and gross limits hold at every stage, and weights are rescaled for numerical conditioning. If the solver fails, it raises an error instead of silently falling back to something else.

→ [`portfolio/optimizer.py`](../src/statarb/portfolio/optimizer.py)

## 4. Execution Simulation

A target decided on the close of day *t* trades at *t + lag − 1* and first earns on *t + lag*. The default lag is 2, which corresponds to a fill at the next close. Between rebalances, holdings drift with returns:

$$w^{drift}_{i,t} = \frac{w_{i,t}(1 + r_{i,t})}{1 + \text{P\&L}^{gross}_t}.$$

Trades are the target minus the drifted book. I skip trades smaller than a 0.002 no-trade band, except for full exits, which always trade. Costs per unit of traded weight *u*, with capital *C* and dollar volume *V*, are:

```text
spread  = u · half_spread_bps · 1e-4          (5 bp stocks / 1 bp ETFs; or a liquidity power law)
impact  = impact_coef · u · sqrt(u · C / V)
borrow  = short_weight · 50 bp / 252           (daily, on held shorts)
```

Reported annual return is the daily mean × 252, not CAGR. Turnover is Σ|trades| and is not halved.

→ [`backtest/engine.py`](../src/statarb/backtest/engine.py),
[`portfolio/costs.py`](../src/statarb/portfolio/costs.py)

## 5. Neural Policies

The neural track replaces the "OU forecast, then rule or optimizer" step with a network that maps a stock's residual history directly to a position preference. I train it end to end on cost-aware P&L.

### Feature Tape

Each stock's input is a 30 × 2 sequence of residual increments and their cumulative path, scaled by that stock's trailing residual volatility. The residuals come from ETF fits estimated on prior rows only. Later refits never rewrite earlier features, and there is no full-sample scaler.

### Architectures

All three networks share a `tanh`-bounded scalar head. The linear model works on the flattened history. The LSTM is a single layer of width 16. The causal conv–Transformer uses two left-padded width-2 convolutions, followed by sinusoidal positions and one causally masked attention layer with 4 heads.

### Differentiable Allocation

I project preferences onto $(1-\beta)^\top w = 0$ and then add ETF hedges $-\beta w$, so the full book is both dollar-neutral and hedged. Caps are enforced by a common rescaling.

### Objective

The loss is $\mathcal L = -\,\text{mean}(\text{P\&L}^{net}) + \lambda\,\text{Var}(\text{P\&L}^{gross})$. I differentiate it through allocation, execution lag, drift, spreads, and borrow, on contiguous 8-session chunks.

### Temporal Controls

I refit a fresh policy every 125 sessions on a rolling 1,000-session window. A sample is only eligible for training when its entire outcome span has matured before the cutoff, and each refit records an audit of the latest outcome it used. My tests also check that injecting a security that only lists in the future leaves past targets and training unchanged.

→ [`neural/`](../src/statarb/neural/): `dataset.py`, `networks.py`, `portfolio.py`,
`training.py`, `runner.py`

## 6. Evaluation

### Walk-Forward Selection

Folds are expanding, with 756 training sessions, 252-session test windows, and a 20-session embargo before each test window. I rank variants on training net return per unit of gross exposure, using gross Sharpe as the tie-break.

→ [`evaluation/walk_forward.py`](../src/statarb/evaluation/walk_forward.py)

### Inference

I use Newey–West (Bartlett) HAC t-statistics on daily P&L, the probabilistic Sharpe ratio, which adjusts for skew and kurtosis, and the deflated Sharpe ratio, which compares against the expected maximum Sharpe given the number of trials. Since the deflated Sharpe depends on how many things were tried, I kept a trial ledger throughout the project: 53 classical configurations, or 64 including the neural runs.

→ [`evaluation/stats.py`](../src/statarb/evaluation/stats.py)

### Signal Decay

I measure the rank IC of the as-of s-score against forward residual returns, holding the original fitted exposures fixed.

→ [`evaluation/decay.py`](../src/statarb/evaluation/decay.py)

## 7. Native Kernels

I wrote C++20 versions of batch OU fitting and the backtest loop, and exposed them through pybind11 with the GIL released. The Python versions remain the reference. Parity tests compare OU parameters to 1e-8, and holdings, P&L, costs, and turnover to 1e-12, across lag, drift, exits, NaNs, and no-trade-band edge cases. The native code is also tested with GoogleTest and built under ASan/UBSan.

| Kernel                 | Workload                          | Python    | C++      | Speedup    |
| ---------------------- | --------------------------------- | --------: | -------: | ---------: |
| Batch OU fit           | 1,500 paths × 60 sessions         | 49.44 ms  | 1.03 ms  | **47.8×**  |
| Portfolio simulation   | 1,000 sessions × 800 instruments  | 417.89 ms | 12.27 ms | **34.0×**  |

Exhibit A: Each figure is the best of three runs on synthetic data ([raw timings](benchmarks.csv)). These are kernel timings, not end-to-end speedups.

→ [`cpp/`](../cpp/), [`native/`](../src/statarb/native/)

---

## 8. Recorded Results

The studies ran on a survivorship-biased US equity panel with 16.5M clean daily observations across 4,186 securities. That data is not distributed. Free data only holds companies listed today, so delisted stocks are missing. Failed stocks and any bad-tick reversals the cleaning missed most likely flatter a buy-the-losers strategy, while acquisitions delisted at a premium would have hurt the short leg. The direction is likely, not proven. All figures are net of the assumed costs above and are arithmetic annual means. Since the studies cover different periods, they are not directly comparable to each other.

### 8.1 Classical Turnover and Optimization Comparison (2013-02 to 2026-10, 3,447 sessions)

| Variant                          | Net %/yr | Gross Sharpe | Cost %/yr | Turnover/day | Mean gross |
| -------------------------------- | -------: | -----------: | --------: | -----------: | ---------: |
| Baseline                         |   −0.83  |        0.42  |     3.67  |        0.329 |       1.91 |
| Baseline, no band                |   −0.93  |        0.48  |     4.05  |        0.359 |       1.96 |
| Smoothing h = 1, no band         |   −0.25  |        0.47  |     2.63  |        0.211 |       1.79 |
| Smoothing h = 4, no band         |   −0.13  |        0.34  |     1.16  |        0.079 |       1.33 |
| Calibrated optimizer, H = 1      |   −0.65  |        0.24  |     2.63  |        0.240 |       1.26 |
| Calibrated optimizer, H = 3      |   −0.30  |        0.33  |     3.07  |        0.279 |       1.41 |
| **Walk-forward selected**        | **−2.18**|      −0.01   |     2.14  |        0.186 |       1.22 |

Exhibit B: Classical variants net of assumed costs. Every fixed variant loses a small amount after costs, and the walk-forward selected book, the only one that picked its variant adaptively, is the worst of the group.

The residuals do mean-revert. The 5-day s-score IC is −0.013 with a HAC t of −5.4, and the negative sign is the contrarian direction I intended. It is a real effect, but a small one, and costs eat it.

Turnover control works, but with a catch. Smoothing with h = 4 cut turnover by about 78% against the band-free baseline, but it also cut exposure. Part of the cost saving is just holding less, not trading better.

The raw OU forecasts are badly miscalibrated. The median calibration slope was 0.025, meaning the raw forecasts (median ~175 bp per interval) were about 40× too large compared with realized residual returns.

Finally, adaptive selection overfit. The walk-forward selector did worse than every fixed variant, with a HAC t of −1.34 and a deflated Sharpe of 0.055.

### 8.2 Neural Architecture Comparison (2018-01 to 2026-10, 2,210 sessions)

| Book                         | Gross %/yr | Net %/yr | Gross Sharpe | Turnover/day | HAC t |
| ---------------------------- | ---------: | -------: | -----------: | -----------: | ----: |
| Linear                       |     −1.26  |   −6.83  |       −0.55  |        0.494 | −9.48 |
| LSTM                         |      2.10  |   −0.48  |        0.54  |        0.200 | −0.39 |
| Conv–Transformer             |      2.10  |   −0.12  |        0.46  |        0.168 | −0.09 |
| Classical, same dates        |      2.53  |   −1.06  |        0.33  |        0.326 | −0.43 |

Exhibit C: Neural books against the classical reference over the same dates. Each neural book averages three seeds, each traded and costed separately.

Architecture changed the trading policy, not just a forecast metric. The sequence models traded about 60–66% less than the linear policy at similar gross exposure, and gained 6.4 and 6.7 pp/yr net over it (paired HAC t ≈ 4.2–4.3).

However, there was no reliable edge over the classical reference. The paired t-statistics were only 0.25 and 0.36.

The seeds were also unstable. Under a 200-update training budget, LSTM seed nets ranged from −4.5% to +1.7%/yr, and mean cross-seed target correlations were only −0.26 to +0.19.

### 8.3 Robustness Checks

To test out-of-domain history, I ran the same engine on the French 49-industry portfolios over a held-out 1980–1999 period. These portfolios are built from CRSP, so they do not share the stock-list survivorship bias. It earned −0.89%/yr gross (HAC t = −3.5). The audit also found a problem in the development period. The Software industry's return is missing on 1971-03-11, the one day the strategy held it, and the engine treats a missing return while held as zero. That is a silent repair, and my registered rule said it makes the inference invalid. So I report the development period as descriptive only, rather than patching it and re-running.

To test cost sensitivity, I re-priced the saved trades with OHLC spread proxies (Abdi–Ranaldo, Corwin–Schultz) instead of a constant 5 bp. Under those proxies the Transformer book came out between −9.7% and −14.2%/yr. The results hinge on execution-cost assumptions that daily data cannot pin down.

## 9. Takeaways

The main lessons from this project were less about the signal itself and more about how a pipeline like this should be built and evaluated.

| Finding | Design implication |
| --- | --- |
| A causal data query doesn't make a pipeline causal | Test temporal contracts end to end: truncation invariance, label maturity, fit audits |
| Lower turnover often just means lower exposure | Compare gross, net, risk, and exposure jointly |
| OU forecast magnitudes need calibration | Keep raw predictions and matured outcomes so the scaling can be audited |
| Architecture and seed shape trading behavior | Judge models on portfolio stability and cost-aware P&L, not loss curves |
| A backtest can finish and still be invalid | Keep validity checks separate from "it ran" |
| Fast kernels need a slow oracle | Keep the Python reference and test the C++ against it |

All three studies failed their pre-registered positive-return gates, and I report the results as they came out.
