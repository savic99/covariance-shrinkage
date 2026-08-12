# Covariance Cleaning for Portfolio Risk: Raw vs. Ledoit–Wolf vs. Optimal RIE

This is a learning exercise comparing three estimators of the covariance matrix of daily
equity returns — the raw sample matrix, Ledoit–Wolf linear shrinkage, and the optimal
Rotationally Invariant Estimator (RIE) based on Random Matrix Theory — on ~300 US large
caps, judged by the out-of-sample risk of the minimum-variance portfolios they produce.

---

## The point

When the number of assets $N$ is comparable to the number of observations $T$, the sample
covariance matrix is dominated by estimation noise. The controlling parameter is $q = N/T$:
even if the true correlation matrix were the identity, the sample eigenvalues would spread
across the Marchenko–Pastur band $[(1-\sqrt q)^2, (1+\sqrt q)^2]$ instead of sitting at 1.
The smallest eigenvalues are the most corrupted, and they matter most, because a
minimum-variance optimizer computes $w \propto \Sigma^{-1}\mathbf{1}$ and therefore divides
by them.

All three estimators compared here are *rotationally invariant*: they keep the sample
eigenvectors and act only on the eigenvalues. (Ledoit–Wolf and the "optimal RIE" are both,
technically, rotationally invariant estimators — the "optimal" qualifier distinguishes the
nonlinear, RMT-derived shrinkage function from the affine one, since "RIE" on its own names
the whole class, not just this method.) So they differ only in the shrinkage function
$\lambda \mapsto \xi$ applied to the spectrum: the identity map (no cleaning), an affine map
(Ledoit–Wolf), and a nonlinear map estimated from the data (optimal RIE). Everything else in
the pipeline is held fixed, which is what makes the comparison interpretable.

## What we found

**1. Cleaning helps, substantially.** At $q = 0.6$ the raw sample covariance produces a GMV
portfolio realizing 14.65% annualized volatility against 11.32% for Ledoit–Wolf and 10.29%
for the optimal RIE. 

**2. The optimal RIE came out ahead of Ledoit–Wolf, by more than we expected.** The
1.03-point margin survives every robustness check we ran (see below). But published studies
separate these two methods by roughly 0.1 points on comparable data, and it is not obvious
why a 1000-day sample would resolve a larger effect than a 46-year one. We treat this as a
result to be explained rather than a finding to be believed.

**3. Part of it, at least, is not about the optimal RIE at all.** Sweeping the shrinkage
intensity by hand, linear shrinkage toward the identity at $\alpha_s = 0.65$ realizes 9.88%
— better than the optimal RIE. Ledoit–Wolf picks $\alpha_s = 0.938$, because its rule
minimizes $\|\Xi - C\|_F$, a loss dominated by the *largest* eigenvalues, while
minimum-variance risk is dominated by the smallest ones. So at least some of the gap
reflects the intensity rule being aimed at the wrong loss, rather than any advantage of
nonlinear shrinkage.

Making the comparison decisive would require an intensity rule targeting portfolio risk
rather than Frobenius distance, and a point-in-time universe (ours is selected using data
from the evaluation period).

---

## Data and universe

Current S&P 500 constituents from Wikipedia, ~5 years of adjusted daily closes and volumes
from Yahoo Finance via `yfinance`: 502 tickers, 2021-06-28 to 2026-07-27.
Among these stocks we have chosen the 300 most liquid names with at most 1% of the missing values and
we keep only one among the duplicate security pairs like GOOGL/GOOG.


Final panel: $N = 300$, $T = 1000$, 2022-07-27 to 2026-07-23, no gaps. Mean pairwise
correlation 0.232, with 25% of total variance in the market mode.

## The estimators

Volatilities and correlations are handled separately throughout: cleaning acts on the
correlation matrix $E$, and the covariance is rebuilt as
$\hat\Sigma_{ij} = \hat\sigma_i\hat\sigma_j\Xi_{ij}$. All three estimators use the same
volatility estimates, so differences in realized risk come from correlation cleaning alone.

**Ledoit–Wolf**, with the identity target:

$$\Xi = \alpha_s E + (1-\alpha_s)I_N \qquad\Longleftrightarrow\qquad \xi_i = 1 + \alpha_s(\lambda_i - 1),$$

a uniform contraction of the spectrum toward 1. The identity target is chosen over the
constant-correlation target of Ledoit–Wolf (2003) so the estimator stays rotationally
invariant and stays comparable to the optimal RIE. Implemented with
`sklearn.covariance.LedoitWolf` applied to z-scored returns, which makes sklearn's
$\mu I_N$ target collapse to exactly $I_N$.


**Optimal RIE**, which targets the oracle $\xi_i = \langle u_i, C u_i\rangle$ — the true
out-of-sample risk of the portfolio whose weights are the $i$-th sample eigenvector. That is
not computable, but RMT gives its large-$N$ limit in terms of the observed spectrum alone:

$$\hat\xi_i = \frac{Re(z_i)}{|1 - q + q z_i g(z_i)|^2},
\qquad g(z_i) = \frac{1}{N-1}\sum_{j \neq i}\frac{1}{z_i - \lambda_j},
\qquad z_i = \lambda_i - iN^{-1/2}.$$

The correction to each $\lambda_i$ depends on how densely the other eigenvalues crowd around
it: one buried in a dense bulk is probably noise and is shrunk hard, while an isolated one
is left alone. Finite-$\eta$ evaluation biases this at the left edge of the spectrum, which
the Inverse-Wishart step corrects by calibrating against a reference case where the exact
answer is known analytically. A trace rescale, a monotonicity sort, and a renormalization to
unit diagonal finish the job.

### One departure from the published algorithm

The published version calibrates the Inverse-Wishart shape parameter $\kappa$ on
$\lambda_N$, the single smallest sample eigenvalue. On this data that fails: genuine
industry duopolies legitimately produce small eigenvalues, $\lambda_N$ is the noisiest
statistic in the spectrum, and $\kappa$ collapses — inflating the bulk and, since the trace
is fixed, squeezing the market mode out of the budget.

| calibration | $\kappa$ | fraction capped | $\frac{\text{cleaned market mode}}{\text{raw market mode}}$ |
|---|---|---|---|
| $\lambda_N$ (published) | 0.225 | 65% | 0.747 |
| 15th percentile (ours) | 1.29 | 9.3% | 0.952 |

Calibrating on a low percentile instead estimates where the bulk edge actually is while
ignoring genuine low-end outliers. The result plateaus for any percentile above ~15, so this
is closer to a threshold than a tuned parameter. Worth noting that this is not a
data-quality problem — it reproduces on synthetic data with no duplicated assets, and an
earlier attempt to fix it by pruning the universe removed 30 economically meaningful names
and still failed.

## Walk-forward evaluation

Strict out-of-sample: at each rebalance, estimate on the preceding $T$ days only, form GMV
weights, hold, record the realized return, advance. The window is rolling and fixed-length
so that $q$ stays constant across the backtest. Main configuration: $T = 500$ ($q = 0.60$),
daily rebalancing, 500 out-of-sample days. Weights are unconstrained (a long-only constraint
would act as an implicit regularizer and mask the differences being measured) and computed
by Cholesky solve rather than explicit inversion.

All estimators see identical windows and identical evaluation days, so inference is on
paired per-day differences. Confidence intervals come from a stationary block bootstrap,
cross-checked against splitting the sample into non-overlapping 60-day blocks.

---

## Results

$N = 300$, $T = 500$ ($q = 0.60$), daily rebalancing, 500 out-of-sample days.
We define $\Psi$ as the ratio of the realized (i.e. out of sample) and predicted volatility.

| | realized vol | 95% CI | $\Psi$ | leverage | turnover |
|---|---|---|---|---|---|
| raw | 14.65% | [13.54, 16.05] | 3.07 | 9.24  | 0.795 |
| Ledoit–Wolf | 11.32% | [10.09, 12.91] | 1.98 | 5.5 | 0.346 |
| Optimal RIE (IWs) | 10.29% | [8.96, 12.06] | 1.60 | 4.09  | 0.232 |
| $\Xi = I$ | 12.82% | [9.71, 16.94] | 8.30 | 1.00 | 0.012 |

**Dependence on $q$.** The margin grows with $q$ up to a point, then the optimal RIE breaks:

| $T$ | $q$ | raw | LW | Optimal RIE | LW − Optimal RIE |
|---|---|---|---|---|---|
| 750 | 0.40 | 10.93 | 9.90 | 9.21 | +0.68 |
| 500 | 0.60 | 14.65 | 11.32 | 10.29 | +1.04 |
| 333 | 0.90 | 29.72 | 11.43 | 15.76 | −4.33 |

which is expected as $q \to 1$.

**Shrinkage intensity.** Sweeping $\alpha_s$ by hand at $T = 500$:

| $\alpha_s$ | 0.00 | 0.35 | 0.50 | 0.65 | 0.80 | 0.938 | 1.00 |
|---|---|---|---|---|---|---|---|
| realized vol | 12.82 | 10.05 | 9.89 | **9.88** | 10.11 | 11.23 | 14.65 |

Ledoit–Wolf's own rule selects 0.938. The plateau from 0.35 to 0.80 is wide, so the optimum
is not a knife-edge fit — but $\alpha_s = 0.65$ is tuned on the evaluation period and is not
an implementable competitor. It does locate a good part of the optimal-RIE-vs-LW gap in the
intensity rule rather than in the shape of the shrinkage curve.

**Risk calibration.** $\Psi$ is realized risk divided by the risk the estimator predicted
for its own portfolio. For the raw estimator the expected value is $1/(1-q) = 2.50$ at
$q = 0.6$ — two separate $\sqrt{1-q}$ effects that compound, since the in-sample forecast is
too low *and* the portfolio built from a noisy $E$ is genuinely worse. Simulating stationary
Gaussian panels with the same true correlation matrix and running the identical harness
gives $\Psi_{\text{raw}} = 2.49 \pm 0.08$, matching. Against that baseline the real-data
values are ~20% higher for raw and the optimal RIE, which could be explained by the
sample being non-stationary.


## Robustness checks

| check | result |
|---|---|
| non-overlapping 60-day blocks | LW − Optimal RIE = +1.13, sem 0.229, Optimal RIE wins 8/8 |
| five 100-day sub-periods | Optimal RIE wins 5/5, gap 0.44 to 2.12 |
| trimming the largest 1–10% of daily moves | gap +1.11 to +0.78 |
| rebalancing every 1 / 5 / 20 / 60 days | gap +1.04 to +1.17 |
| bootstrap block length 10 to 100 | interval stable within ±0.15 |


## Limitations

- **The universe is not point-in-time.** Completeness and liquidity filters are computed
  over a window that includes the evaluation period. Published studies select on liquidity
  *during the training period*. This should be fixed before drawing conclusions.
- **Survivorship bias.** Today's index membership applied retroactively; delisted and
  acquired names never appear. It is not obvious that this drives the optimal-RIE-vs-LW
  ordering — survivorship flatters portfolios holding large positions in surviving names,
  and the raw estimator holds by far the most extreme positions while performing worst —
  but we have not measured it.
- **Short sample, one regime.** The backtest covers 2024-07 to 2026-07. Published benchmarks
  span decades, and their evaluation protocol differs from ours (e.g. we use different
  regularization of the optimal RIE)
- **The $\alpha_s$ optimum is fitted in-sample** and is not a usable estimator.

## Repository

All code is in `comparison.ipynb`, in numbered cells so each step runs and can be inspected
in isolation: universe selection (1–5), covariance pipeline and spectrum diagnostics (6–9),
Ledoit–Wolf (10–12), Optimal RIE (13–15), and GMV plus the walk-forward comparison (16–18).

Standalone, commented `.py` versions of the individual pipeline steps are in `src/`, and
methodology write-ups (including a documented failure mode of the published RIE algorithm on
real equity data, and the fix) are in `notes/`.

`output/prices_full_history.parquet` and `output/volumes_full_history.parquet` are the raw
~5-year price/volume panel for the full downloaded universe (502 tickers) — the starting
point `comparison.ipynb` reads in its first cell, checked in so the notebook runs end to end
without a fresh Yahoo Finance download. Everything else the pipeline derives from them
(the filtered ~300-name universe, returns matrices, quality reports) is cheap to regenerate
via `src/01_download_data.py` and `src/02_universe_selection.py` and is not committed;
`data/selected_universe.txt` is kept as a plain-text summary of the final 300-ticker
universe for quick reference.

Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `yfinance`,
`pyarrow`.

## References

1. O. Ledoit and M. Wolf, "Improved estimation of the covariance matrix of stock returns
   with an application to portfolio selection," *Journal of Empirical Finance*, 10(5),
   603–621, 2003. — introduces shrinkage toward a constant-correlation target; the
   Appendix B $\beta/\gamma$ formulas were used here as an independent cross-check of the
   `sklearn` implementation.
2. O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance
   matrices," *Journal of Multivariate Analysis*, 88(2), 365–411, 2004. — the
   identity-target shrinkage estimator actually used in this project (matches
   `sklearn.covariance.LedoitWolf`), chosen over the 2003 target because it stays
   rotationally invariant and therefore directly comparable to the optimal RIE.
3. L. Laloux, P. Cizeau, J.-P. Bouchaud, and M. Potters, "Noise Dressing of Financial
   Correlation Matrices," *Physical Review Letters*, 83(7), 1467–1470, 1999. — the short
   seminal paper applying Marchenko–Pastur / Random Matrix Theory to financial correlation
   matrices for the first time, establishing that most of the spectrum of an empirical
   equity correlation matrix is consistent with pure noise.
4. J. Bun, J.-P. Bouchaud, and M. Potters, "Cleaning large correlation matrices: tools from
   random matrix theory," *Physics Reports*, 666, 1–109, 2017 (arXiv:1610.08104). — the
   full review this project follows for the optimal RIE (IWs) algorithm, its
   Inverse-Wishart regularization, and the numerical benchmarking methodology.
