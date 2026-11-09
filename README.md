# Residual Mean-Reversion Statistical Arbitrage

> **Note:** This is a condensed version of a larger private research codebase, trimmed to the pieces that show the modeling and how I evaluated it. Some write-ups refer to files that are not included here. Feel free to reach out if you have questions or want more details.

This repository contains a market-neutral equity strategy that I built end to end, from the residual model through portfolio construction, execution, and evaluation. The idea is simple. I strip out each stock's factor exposure, model what is left over as an Ornstein–Uhlenbeck (OU) process, and trade its deviations from the mean. Most of my effort went into making the evaluation honest, so that the results reflect what the strategy would actually have done and not what a backtest made it look like.

The classical pipeline runs in the following order. I start from stock returns, compute ETF or PCA residuals, fit an OU process to each residual path, and turn that fit into an s-score signal. The signal feeds a hedged, dollar-neutral portfolio, built either by a simple rule or by a convex receding-horizon optimizer. From there I simulate execution with lag, drift, spreads, impact, and borrow costs, and evaluate everything walk-forward using HAC t-statistics, the probabilistic Sharpe ratio (PSR), and the deflated Sharpe ratio.

I added two pieces on top of the classical pipeline. The first is a set of neural policies. Linear, LSTM, and causal conv–Transformer networks learn positions directly from residual histories, and I train them end to end through a differentiable allocation and cost model. The second is a pair of C++20 kernels for the OU fit and the backtest. They run 34–48× faster than the Python reference, and I parity test them against it to 1e-8 and 1e-12.

The full method, equations, and results are in [docs/methodology.md](docs/methodology.md).

## Headline Findings

I ran the models on 16.5M daily observations across 4,186 US equities from 2010 to 2026, built from free Yahoo and SEC data. That data is not included in this repo. It only holds companies that are still listed today, so every stock-level number below is survivorship-biased. The missing failed stocks most likely flatter a buy-the-losers strategy, while acquisitions delisted at a premium would have hurt the short leg, so the direction is likely but not proven.

The signal is real, but small. The 5-day s-score IC is −0.013 (HAC t = −5.4), which is the contrarian direction I expected. However, my assumed costs consume it, and every classical variant lands between −0.1% and −0.9% per year net. I also found that the raw OU forecasts were about 40× too large. After calibrating them online, fitting only on matured outcomes, the median slope came out to 0.025. Turnover control cut trading by about 78%, but much of that saving came from carrying less exposure rather than trading better. Adaptive walk-forward selection overfit, doing worse than every fixed variant with a deflated Sharpe of 0.055.

On the neural side, the sequence models changed how the strategy traded. The LSTM and Transformer policies traded about 60–66% less than a linear policy and beat it by 6–7 pp per year. They did not beat the classical reference, and results were unstable from seed to seed.

As a cross-check without the stock-list bias, I ran the same rule, unchanged, on French's 49 industry portfolios over a 1980–1999 holdout. It lost money before costs (−0.89% per year gross, HAC t = −3.46), so on industry baskets the problem is not just costs.

None of the studies passed its pre-registered positive-return gate. This repo is meant to show the modeling and how I evaluated it, not to claim alpha.

## Code Tour

```text
src/statarb/
  models/          residuals.py   ETF (sticky R² assignment) and PCA residual models
                   ou.py          AR(1) → OU parameters (κ, m, σ_eq, half-life)
                   signals.py     s-score and hysteresis position rule
                   pipeline.py    rolling point-in-time refit loop → RebalanceContext
  portfolio/       construction.py  sizing, ETF/PCA hedges, dollar neutrality
                   optimizer.py     CVXPY receding-horizon optimizer + online alpha calibration
                   turnover.py, costs.py
  backtest/        engine.py      lagged execution, drift, no-trade band, cost accounting
                   runner.py      wires signals → portfolio → backtest
  evaluation/      walk_forward.py  embargoed folds, variant selection
                   stats.py         HAC t-stat, probabilistic and deflated Sharpe
                   decay.py         residual IC decay
  neural/          dataset.py (causal feature tape), networks.py, portfolio.py
                   (differentiable allocation and costs), training.py, runner.py
  native/          pybind11 wrappers for the C++ kernels and the benchmark
  synthetic.py     market with planted OU residuals (ground truth for tests and the demo)
cpp/               C++20 OU and backtest kernels, bindings, GoogleTest
tests/             look-ahead/truncation invariance, portfolio contracts, optimizer,
                   statistics, neural temporal audits, C++ parity
examples/demo.py   end-to-end run on synthetic data
```

For anyone reading the code for the first time, I suggest starting with these files in order:

1. [`models/pipeline.py`](src/statarb/models/pipeline.py), the core refit loop.
2. [`portfolio/optimizer.py`](src/statarb/portfolio/optimizer.py), the receding-horizon optimizer and calibration.
3. [`neural/training.py`](src/statarb/neural/training.py), the end-to-end policy training.
4. [`tests/test_pipeline.py`](tests/test_pipeline.py), the no-look-ahead tests.

## Running It

```sh
pip install -e ".[dev]"          # add ,neural for the PyTorch policies
python examples/demo.py          # classical pipeline on a synthetic market
python examples/demo.py --neural # plus a small conv-Transformer policy
pytest
```

The demo builds a synthetic market where I know the ground truth. About two-thirds of the stocks have OU residuals and the rest are random walks, and the demo checks how much of that structure the models recover:

```text
1. identification on clean residuals (150 stocks, 66% OU, rest random walk)
   median fitted kappa/yr: reverting 53 (true 38), random walk 25 (true 0: Dickey-Fuller small-sample bias)
   rank corr(fitted, true kappa) on reverting names: 0.84
   rebalances passing kappa >= 8.4: reverting 99%, random walk 78%

2. portfolio construction variants (net of 5 bp spreads + 50 bp borrow)
                          net %/yr  sharpe  HAC t    PSR  cost %/yr  turnover/day  gross
rule (baseline)             13.026   2.308  3.699  1.000      2.350         0.222  1.125
rule + smoothing h=4         2.570   0.914  1.336  0.917      0.670         0.047  0.752
calibrated optimizer H=3     9.040   2.058  3.268  0.999      1.735         0.162  0.859
```

One thing the synthetic run taught me is that a 60-day AR(1) fit makes even a random walk look mean-reverting (κ̂ ≈ 25/yr), so the usual κ filter lets most of the noise through. Since the filter cannot separate the two, the edge has to come from the s-score itself. When there is a planted edge, the pipeline finds it. Real data did not behave that way, as the findings above show.

The C++ kernels are optional and need CMake and a C++20 compiler:

```sh
cmake -S cpp -B build/cpp -DPython_EXECUTABLE=$(which python) \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build/cpp && ctest --test-dir build/cpp
pytest tests/parity                # Python ↔ C++ parity
python -m statarb.native.bench     # timing comparison
```

## Scope

This is a curated extract of a larger research codebase. The original also had a point-in-time data layer (DuckDB/Parquet, corporate actions, survivorship labels and provenance tracking), MLflow experiment tracking, trial ledgers, and a forward logger for frozen models. I left those out to keep the focus on the modeling.

## License

The code is MIT licensed (see [LICENSE](LICENSE)). The license covers the code only. Third-party market data (Yahoo Finance, SEC EDGAR, Tiingo, the Kenneth French data library) is not included and remains under its providers' terms.
