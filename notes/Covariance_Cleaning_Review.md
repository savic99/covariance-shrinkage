# Comparing Covariance Cleaning Schemes on S&P 500 Equities

**Raw sample covariance vs. Ledoit–Wolf linear shrinkage vs. Rotationally Invariant Estimators**

*Methodology review — project working document*

---

## 1. Scope and objective

The aim is a controlled comparison of three estimators of the covariance matrix of daily
equity returns, evaluated by the out-of-sample risk of the global minimum-variance (GMV)
portfolios they produce:

1. the **raw sample covariance** matrix (no cleaning);
2. **Ledoit–Wolf linear shrinkage** toward the identity;
3. the **Rotationally Invariant Estimator** with Inverse-Wishart regularization and
   sorting, "RIE (IWs)" — the scheme advocated in the Bouchaud–Potters–Bun line of work.

All three are *rotationally invariant estimators* in the technical sense: they retain the
sample eigenvectors and act only on the eigenvalues. That is what makes the comparison
clean. They differ solely in the shrinkage function $\lambda \mapsto \xi$ applied to the
spectrum: the identity map (no cleaning), an affine map (Ledoit–Wolf), and a nonlinear map
estimated from the data (RIE). Everything else in the pipeline is held fixed.

The central parameter throughout is

$$q = \frac{N}{T},$$

the ratio of the number of assets to the number of return observations in the estimation
window. It controls how badly the sample covariance matrix is corrupted by estimation
noise, and therefore how much cleaning can help.

---

## 2. Universe construction

### 2.1 Source and raw download

The constituent list is scraped from the current Wikipedia S&P 500 page, yielding **503
tickers**. Ticker symbols are normalized from Wikipedia's dot convention to Yahoo's dash
convention (`BRK.B` → `BRK-B`) so that share-class names do not silently fail to download.

Prices are pulled from Yahoo Finance via `yfinance` in batches of 40 with exponential
retry/backoff (4 attempts, 5s doubling), covering approximately five years. **502 of 503
tickers downloaded successfully**; one (`AMCR`) failed and was dropped. The realized date
range is 2021-06-28 to 2026-07-27, comprising 1275 trading days.

Two download choices matter for the covariance estimates:

**Adjusted close, not raw close.** Splits and dividends produce discontinuities in the raw
price series that are not economic returns. Left uncorrected they enter the covariance
matrix as spurious volatility and spurious cross-sectional structure. `auto_adjust=True`
folds both adjustments into the price series.

**Five years downloaded, only ~1000 days used.** The extra history is deliberate headroom,
so that a planned sweep over $T$ (Section 7.2) can be run without re-downloading.

### 2.2 Quality filters

Two filters are applied over a window of the last $T_{\text{target}} + T_{\text{buffer}} =
1000 + 60 = 1060$ trading days.

**Completeness.** A ticker must have non-missing prices on at least 99.5% of that window.
This removes recent IPOs, spin-offs, and names with data gaps. In practice, the binding
cases are stocks that simply did not trade for part of the window — for example `CEG` and
`HOOD`, both of which have missing values concentrated in the first ~200 dates of the
downloaded history.

**Liquidity.** Among the tickers passing the completeness bar, the final universe is the
top $N = 300$ by average dollar volume, $\overline{P_t V_t}$, over the same window. The
justification is twofold. Thinly traded stocks have stale and noisy prices, which distorts
precisely the correlation structure under study; and restricting to liquid large caps is
standard in the empirical RMT literature, so results remain comparable to the published
benchmarks.

### 2.3 Final estimation panel

Returns are simple daily returns, $r_{it} = P_{it}/P_{i,t-1} - 1$.

The analysis panel is the block of **1000 calendar-contiguous trading days ending
2026-07-23** with no missing values anywhere. Contiguity is enforced explicitly rather than
by dropping NaN rows: a matrix assembled from non-adjacent dates would silently corrupt any
autocorrelation diagnostics and would misstate the effective sample size. The date
2026-07-24 is excluded because it carries missing values for part of the universe.

This gives $N = 300$, $T = 1000$, hence $q = 0.30$ for the full-sample diagnostics. Within
the backtest, $q$ varies with the estimation window length (Section 7.2).

Two extreme single-day returns survive and were inspected rather than removed: `HOOD`
$+50.4\%$ on 2021-08-04 and `CVNA` $+56.0\%$ on 2023-06-08. Both are genuine market moves,
not adjustment errors.

### 2.4 Known limitation: survivorship bias

The universe is today's index membership applied retroactively. Companies removed from the
index during the sample — through acquisition, bankruptcy, or relegation — never appear,
even though they were tradeable constituents at the time. This biases the sample toward
survivors and, in general, understates realized risk.

This is a deliberate, documented deferral rather than an oversight. It affects all three
estimators identically and is therefore largely neutral for the *relative* comparison,
which is the object of interest here. A point-in-time universe can be reconstructed from
Wikipedia's constituent-changes table if absolute risk levels later become important.

---

## 3. Notation and preliminaries

Let $R \in \mathbb{R}^{T \times N}$ denote the returns panel, rows indexed by date and
columns by asset.

Volatilities and correlations are handled separately, following standard practice in this
literature. Equity volatility is strongly heteroskedastic and non-stationary, while the
correlation structure is comparatively stable; mixing the two means a cleaning scheme is
partly judged on its ability to track volatility, which is not what it is for. Concretely,
each column is standardized,

$$z_{it} = \frac{r_{it} - \hat\mu_i}{\hat\sigma_i}, \qquad
\hat\sigma_i^2 = \frac{1}{T}\sum_t (r_{it} - \hat\mu_i)^2,$$

and the sample correlation matrix is

$$E = \frac{1}{T} Z^{\top} Z .$$

The $1/T$ normalization (rather than $1/(T-1)$) is used throughout, to match the
convention under which the Marchenko–Pastur results are stated.

Cleaning schemes act on $E$, producing $\Xi$. The covariance estimate used for portfolio
construction is recombined as

$$\hat\Sigma_{ij} = \hat\sigma_i \hat\sigma_j \, \Xi_{ij},$$

which presumes $\Xi_{ii} = 1$ exactly — a constraint that turns out to require explicit
enforcement for RIE (Section 5.7).

Write the eigendecomposition $E = \sum_{i=1}^{N} \lambda_i u_i u_i^{\top}$ with
$\lambda_1 \ge \cdots \ge \lambda_N$. Since $\operatorname{Tr} E = N$ for a correlation
matrix, $\sum_i \lambda_i = N$.

An EWMA-weighted estimator is also implemented, with weights $w_k \propto \lambda_{\text{ewma}}^k$
normalized to sum to one, reducing exactly to the equal-weight estimator at
$\lambda_{\text{ewma}} = 1$. The main comparison uses equal weighting; the EWMA variant is
retained as a robustness check.

---

## 4. Ledoit–Wolf linear shrinkage

### 4.1 Estimator and choice of target

Linear shrinkage forms a convex combination of the sample correlation matrix and a
structured target. We use the **identity target**:

$$\Xi^{\text{lin}} = \alpha_s E + (1 - \alpha_s) I_N, \qquad \alpha_s \in [0,1].$$

Equivalently, in the eigenbasis of $E$ — the eigenvectors are untouched and

$$\xi_i = 1 + \alpha_s(\lambda_i - 1).$$

The identity target is chosen over the constant-correlation target of Ledoit and Wolf
(2003) for a specific reason. The identity-target version is a genuine rotationally
invariant estimator: it preserves sample eigenvectors and rescales eigenvalues only. That
places it in the same family as RIE, so the three schemes differ *only* in their shrinkage
function and the comparison isolates exactly that. The constant-correlation target is not
rotationally invariant and would confound the comparison. It is also the "Linear LW"
variant most commonly benchmarked in this literature, keeping results directly comparable
to published tables.

Geometrically, the map $\xi_i = 1 + \alpha_s(\lambda_i - 1)$ is a contraction of the whole
spectrum toward 1, with the same contraction factor everywhere. Large eigenvalues come
down, small eigenvalues go up, and the total is preserved: $\sum_i \xi_i = N$ for any
$\alpha_s$. The estimator is positive definite for any $\alpha_s < 1$, since the smallest
eigenvalue is bounded below by $1 - \alpha_s > 0$. That is the property that makes it
"well-conditioned" and safe to invert.

### 4.2 Optimal shrinkage intensity

The intensity is not a free parameter; it is estimated from the data by minimizing the
expected squared Frobenius distance to the true correlation matrix. Ledoit and Wolf show
the optimizer depends on two estimable quantities: the dispersion of the sample matrix
about the target, and the estimation variance of the sample matrix's entries. Writing these
as $\beta$ (target misspecification) and $\gamma$ (estimation noise), the optimal intensity
takes the form $\alpha_s^\star = 1 - \beta/\gamma$, suitably clipped to $[0,1]$.

The qualitative behaviour is the important part:

- $\alpha_s \to 0$ (shrink hard to the identity) when the sample matrix is mostly noise —
  large $q$, or little genuine correlation structure;
- $\alpha_s \to 1$ (trust the sample) when $T \gg N$, or when the true correlation
  structure is strong relative to the noise.

**Implementation.** We use `sklearn.covariance.LedoitWolf`, which implements this
estimator. One detail matters: sklearn shrinks toward $\mu I_N$ with
$\mu = \operatorname{Tr}(S)/N$, the average sample variance, not toward $I_N$ itself. Fed
raw returns, $\mu$ would be the average daily return variance — a small number — and the
target would not be the identity. Feeding it the **standardized** returns $Z$ resolves this
exactly: the sample covariance of $Z$ *is* the correlation matrix $E$, and
$\mu = \operatorname{Tr}(E)/N = 1$ identically. Then sklearn's $\mu I_N$ is exactly $I_N$
and its output is precisely $\Xi^{\text{lin}}$ above, with
$\alpha_s = 1 - \texttt{shrinkage\_}$.

A note on provenance: a direct hand-coded transcription of the textbook $\beta/\gamma$
formula produced a degenerate $\hat\alpha_s \equiv 0$ in every regime tested, traced to an
inconsistency between the stated $y_k = z_k/\sqrt{T}$ scaling and the fluctuation term. The
sklearn route was cross-validated against an independent implementation of the Appendix B
formulas of Ledoit and Wolf (2003) and against the qualitative behaviour above ($\alpha_s$
decreasing in $q$; $\alpha_s \to 0$ under pure noise), and is used in preference.

### 4.3 Properties to keep in mind

Because the map is affine with a single slope, linear shrinkage cannot be optimal across
the whole spectrum. The market-mode eigenvalue and the small bulk eigenvalues require very
different corrections, and one slope must compromise between them. This is precisely the
deficiency the nonlinear estimator addresses, and it is visible directly in the shrinkage
curve (Section 5.8).

---

## 5. Rotationally Invariant Estimators

### 5.1 The oracle

Fix the sample eigenvectors and ask for the best possible eigenvalues. If the true
correlation matrix $C$ were known, the optimal choice — minimizing the Frobenius distance
to $C$ within the rotationally invariant class — is the **oracle estimator**

$$\xi_i^{\text{ora}} = \langle u_i,\, C\, u_i \rangle .$$

Read financially, this is the true out-of-sample variance of the portfolio whose weights
are the $i$-th sample eigenvector. It is not computable in practice, but it is the correct
target, and it is directly measurable on synthetic data — which is what makes it the right
validation instrument (Section 8).

### 5.2 The observable estimator

Random matrix theory supplies an asymptotic formula for the oracle in terms of observable
quantities only:

$$\hat\xi(\lambda) = \frac{\lambda}{\left| 1 - q + q\,\lambda\, g_E(\lambda - i0^{+}) \right|^{2}},
\qquad
g_E(z) = \frac{1}{N}\sum_{j} \frac{1}{z - \lambda_j},$$

where $g_E$ is the Stieltjes transform of the empirical spectral density — equivalently the
normalized resolvent $\frac{1}{N}\operatorname{Tr}(z - E)^{-1}$.

The mechanism is intuitive. The correction to $\lambda_i$ depends on how densely the
remaining eigenvalues crowd around it. An eigenvalue buried inside a dense bulk is likely
noise and is shrunk hard toward the mean; an isolated eigenvalue — the market mode — has
denominator close to 1 and is left essentially untouched. Isolation is evidence of signal.
This is what makes the shrinkage nonlinear, and it is the qualitative advantage over the
single-slope linear map.

### 5.3 Finite-$N$ evaluation and the role of $\eta$

The transform must be evaluated slightly off the real axis, $z_i = \lambda_i - i\eta$,
because at $z = \lambda_i$ exactly the $j = i$ term diverges. Two devices are used
together: the $j = i$ term is excluded from the sum, and the evaluation point is displaced
by $\eta$. The empirical resolvent is therefore

$$g(z_i) = \frac{1}{N-1}\sum_{j \ne i} \frac{1}{z_i - \lambda_j}.$$

The standard choice is $\eta = N^{-1/2}$, dictated by needing $N^{-1} \ll \eta \ll 1$: wide
enough to smooth over the typical inter-eigenvalue spacing of order $1/N$, narrow enough
not to blur genuine spectral features. It is the same Lorentzian-broadening prescription
familiar from Green's functions.

### 5.4 The small-eigenvalue bias

Finite $\eta$ biases the estimator, negligibly in the bulk but systematically at the **left
edge**. The reason is that the spectrum has a *hard* boundary — correlation eigenvalues
cannot be negative — and convolving a sharp edge with a Lorentzian of width $\eta$ drags
weight across it. Specializing to the null hypothesis $C = I$, one obtains

$$\hat\xi(\lambda_- - i\eta) = 1 - \sqrt{\frac{2\eta\sqrt{q}}{(1-\sqrt{q})^{2}}} + O(\eta),$$

so with $\eta = N^{-1/2}$ the leading error is $O(N^{-1/4})$ — a very slowly vanishing
finite-size effect. At $N = 300$, $N^{-1/4} \approx 0.24$, i.e. roughly a 24% underestimate
at the edge.

This matters disproportionately because the matrix is going to be **inverted**. Small
eigenvalues become large in $\hat\Sigma^{-1}$, and a minimum-variance optimizer allocates
aggressively along exactly those directions. Underestimating them causes the optimizer to
load up on combinations it wrongly believes are low risk.

### 5.5 The IWs correction

The bias cannot be measured directly on real data — the truth is unknown. The Inverse-Wishart
regularization is a **known-answer calibration**. There is one family of models where the
exact answer is available analytically: if $C$ is drawn from an Inverse-Wishart ensemble
with shape parameter $\kappa$, the optimal RIE is exactly linear shrinkage with
$\alpha_s = 1/(1+2q\kappa)$. Running the same biased finite-$\eta$ machinery on that
reference and taking the ratio measures the bias:

$$\Gamma_i = \frac{1 + \alpha_s(\lambda_i - 1)}{\text{rie}(z_i, q, g_{\text{iw}})},$$

where the reference transform is

$$g_{\text{iw}}(z) =
\frac{z(1+\kappa) - \kappa(1-q) - \sqrt{z - \lambda_+^{\text{iw}}}\sqrt{z - \lambda_-^{\text{iw}}}}
{z\,(z + 2q\kappa)},
\qquad
\lambda_{\pm}^{\text{iw}} = \frac{(1+q)\kappa + 1 \pm \sqrt{(2\kappa+1)(2q\kappa+1)}}{\kappa}.$$

The measured correction is then transferred to the real estimate, one-sidedly and only
where the bias is known to bite:

$$\hat\xi_i \leftarrow \Gamma_i \, \hat\xi_i \quad \text{if } \Gamma_i > 1 \text{ and } \lambda_i < 1.$$

The Inverse-Wishart form is not a claim that equity correlations are Inverse-Wishart
distributed. It is a smooth, analytically tractable reference spectrum used solely to
quantify a numerical bias.

### 5.6 Estimating $\kappa$

**The textbook prescription.** Choose $\kappa$ so that the reference spectrum's lower edge
coincides with the smallest observed eigenvalue, $\lambda_-^{\text{iw}} = \lambda_N$, which
inverts in closed form to

$$\kappa = \frac{2\lambda_N}{(1 - q - \lambda_N)^{2} - 4q\lambda_N}.$$

**Why this fails on equity data.** The entire regularization is then calibrated on a single
extreme order statistic — the most noise-sensitive number in the spectrum. Real equity
universes contain genuine industry duopolies (MA/V, HD/LOW, DAL/UAL, SO/DUK, MCO/SPGI,
TMO/DHR) correlated at 0.79–0.88. These are real economic structure, and they legitimately
produce eigenvalues far below the bulk edge. That drives $\kappa \to 0$, which makes the
reference spectrum enormously wide (its upper edge diverges as $2/\kappa$), which makes
$\Gamma$ explode and apply across most of the spectrum rather than a handful of edge
eigenvalues. The bulk is inflated; since the trace is pinned at $N$, the extra weight at
the bottom is taken out of the top; and the market mode collapses. In the limit the cleaned
matrix degenerates toward the identity.

On our data this produced a cleaned/raw market-mode ratio of 0.748 (largest eigenvalue
74.67 → 55.87), against an oracle expectation of essentially 1.0.

It is worth being explicit that this is *not* a data-quality problem. It reproduces on
synthetic data with **no** duplicated assets: 25 industry pairs at $\rho = 0.86$ are
sufficient. An attempt to fix it by pruning the universe removed 30 economically meaningful
names and still failed to converge — which is the diagnostic that the estimator, not the
data, needed correcting.

**Our prescription.** Calibrate $\kappa$ on a low **percentile** of the spectrum rather
than on the minimum:

$$\lambda_{\text{ref}} = \text{percentile}_{p}\left(\{\lambda_i\}\right), \quad p = 15,
\qquad
\kappa = \frac{2\lambda_{\text{ref}}}{(1 - q - \lambda_{\text{ref}})^{2} - 4q\lambda_{\text{ref}}},$$

with $\kappa$ capped at a large constant (numerically standing in for $\kappa \to \infty$,
the pure-noise limit) whenever the denominator is non-positive.

The rationale is that the reference spectrum should model the **bulk** edge, not the
extreme tail. A percentile estimates where the bulk actually begins while ignoring genuine
low-end outliers. This mirrors the logic RIE already applies at the *top* of the
spectrum, where the machinery operates on the "spikeless" matrix rather than the raw
one. Extending it to the bottom is the natural symmetric move.

Two safeguards accompany it. $\Gamma$ is capped at 2.0 — theory bounds the correction at
$1 + O(N^{-1/4}) \approx 1.3$ for $N = 300$, and on healthy universes the largest requested
value measures 1.2–1.6, so the cap never binds and costs nothing. And the health diagnostic
is the **fraction of eigenvalues requiring capping**, not $\max\Gamma$: once $\kappa$ is
calibrated on a percentile, a single near-degenerate direction can send $\Gamma$ at that
one point to $10^9$ while the estimate as a whole is fine. Measured cap fractions are 0%
on healthy universes, ~6% on a realistic messy one, and 53–74% when genuinely broken.

Evidence, on synthetic data with known $C$, reporting mean squared error against the oracle
(Section 8):

| Scenario | $\kappa$ from $\lambda_N$ | $\kappa$ from $p_{15}$ | Linear LW | Raw |
|---|---|---|---|---|
| Clean universe | 0.0197 | 0.0199 | 0.157 | 0.164 |
| 25 industry pairs | 1.9258 | **0.0120** | 0.127 | 0.128 |
| 40 industry pairs | 1.8098 | **0.0148** | 0.116 | 0.117 |
| $q = 0.88$ | 1.0576 | **0.0781** | 0.486 | 0.485 |
| Pure noise ($C=I$) | 0.0060 | 0.0060 | 0.000 | 0.299 |

The textbook calibration is roughly an order of magnitude *worse than no cleaning at all*
once realistic structure is present. The percentile calibration costs essentially nothing
on a clean universe and is 120–160× better on a realistic one, while restoring the
market-mode ratio to 0.981–0.984 against an oracle of 0.998–0.999.

### 5.7 Remaining steps

**Trace preservation.** Rescale $\xi_i \leftarrow s\,\xi_i$ with
$s = \sum_i \lambda_i / \sum_i \xi_i$. Cleaning should redistribute variance across
directions, not change the total; this enforces the sum rule $\sum_i \xi_i = N$.

**Sorting** (the "s" in IWs). The map $\lambda \mapsto \xi$ is monotone in the large-$N$
limit, so any inversions at finite $N$ are noise. Sorting the cleaned eigenvalues and
re-pairing them against the rank-ordered sample eigenvalues removes it, at negligible cost.

**Unit-diagonal renormalization.** This is an addition, not part of the textbook
algorithm, and the justification is worth stating carefully. Trace preservation forces the
diagonal of $\Xi$ to average 1, but not to equal 1 entrywise: since
$\Xi_{ii} = \sum_k \xi_k (u_k)_i^2$, each diagonal entry is a *differently* weighted average
of the cleaned eigenvalues, and moving the $\xi$ non-uniformly moves each one differently.
Measured drift is 0.878–1.087. A matrix with $\Xi_{ii} = 0.88$ is not a correlation matrix,
and it corrupts the recombination $\hat\Sigma_{ij} = \hat\sigma_i \hat\sigma_j \Xi_{ij}$,
understating that asset's variance by 12%. We therefore apply
$\Xi_{ij} \leftarrow \Xi_{ij} / \sqrt{\Xi_{ii}\Xi_{jj}}$.

Two honest caveats. Strictly, this congruence perturbs the eigenvectors and so leaves the
rotationally invariant class in which the optimality result is proved. And it makes the
preceding trace rescale redundant: scaling all $\xi$ by any constant leaves the
renormalized matrix unchanged to machine precision, so the printed "trace $= N$" check is
trivially true rather than an independent verification. Empirically it helps —
$\|\Xi - C\|_F$ measures 5.704 with renormalization, 6.085 without, against 8.715 for the
raw sample matrix — so it is retained.

### 5.8 Diagnostic: the shrinkage curve

The single most informative diagnostic is the plot of $\xi_i$ against $\lambda_i$ on log-log
axes, with the $45^\circ$ line marking "no cleaning". Linear shrinkage is a straight line in
linear coordinates by construction; RIE is genuinely curved. Degeneracy failures of the
kind described in Section 5.6 are immediately visible as the top-right point falling far
below the diagonal.

---

## 6. Global minimum-variance portfolio

The portfolio is the pure risk problem, with no expected-return input:

$$w = \frac{\hat\Sigma^{-1} \mathbf{1}}{\mathbf{1}^{\top} \hat\Sigma^{-1} \mathbf{1}},
\qquad \mathbf{1}^{\top} w = 1 .$$

GMV is the right test vehicle precisely because it contains no alpha model. Any difference
in realized risk between the three portfolios is attributable to the covariance estimate
alone. It is also the hardest test of the inverse: GMV weights load heavily on the
smallest-eigenvalue directions, which are exactly where estimation error is largest. Short
sales are unconstrained; imposing $w \ge 0$ would act as an implicit regularizer and mask
the differences we are trying to measure.

Weights are computed by solving $\hat\Sigma x = \mathbf{1}$ with a Cholesky-based solver
rather than forming $\hat\Sigma^{-1}$ explicitly, for numerical stability.

Alongside realized risk we plan to record three weight diagnostics per rebalance:

- **Gross leverage** $\sum_i |w_i|$. Uncleaned GMV weights are notoriously extreme; this
  quantifies it.
- **Effective number of positions** $1 / \sum_i w_i^2$. A concentration measure; for
  $N = 300$ equal-weight it equals 300, and it collapses when the optimizer chases spurious
  low-risk combinations.
- **Extreme-weight count**, the number of $|w_i|$ exceeding a fixed threshold.

These are not the headline metrics but they make the failure mode of the raw estimator
legible, and they are what a practitioner would look at first.

**Baselines.** Four reference portfolios bracket the comparison: equal weight ($1/N$);
total shrinkage ($\alpha_s = 0$, i.e. $\Xi = I$, giving the inverse-volatility portfolio);
no shrinkage ($\alpha_s = 1$, the raw sample matrix); and the three cleaning schemes.

> **Status: this section and Section 7 describe the planned protocol. The GMV portfolio
> construction and the walk-forward backtest have not been implemented yet — see the
> project README for current status.**

---

## 7. Walk-forward evaluation (planned)

### 7.1 Protocol

Strict out-of-sample, no look-ahead:

1. At rebalance date $t$, take the estimation window of returns from $t - T$ to $t - 1$
   **inclusive of neither $t$ nor anything after it**.
2. Estimate $\hat\sigma_i$ and $\Xi$ on that window; recombine into $\hat\Sigma$.
3. Form GMV weights $w_t$.
4. Hold over the out-of-sample period and record the realized portfolio return
   $r^{p}_{t+1} = w_t^{\top} r_{t+1}$.
5. Advance by $T_{\text{out}}$ and repeat.

The estimation window is **rolling with fixed length**, not expanding. This keeps
$q = N/T$ constant across the backtest, so results are not confounded by $q$ drifting over
time — which matters because $q$ is the very parameter governing how much cleaning helps.

We plan $T_{\text{out}} = 1$ (daily rebalancing). With a total sample of 1000 days this
maximizes the number of out-of-sample observations, $M = T_{\text{tot}} - T$, which is the
binding constraint on statistical power.

### 7.2 Choice of estimation window

$T$ is to be swept rather than fixed, since the dependence of the cleaning benefit on $q$ is
itself a result:

| $T$ | $q = 300/T$ | Out-of-sample days $M$ |
|---|---|---|
| 750 | 0.40 | 250 |
| 500 | 0.60 | 500 |
| 333 | 0.90 | 667 |

$q \ge 1$ is deliberately avoided. At $q > 1$ the sample correlation matrix is singular — it
has $N - T$ exact zero eigenvalues — so raw GMV is undefined and the comparison changes
character entirely. Both IWs regularization and QuEST-type methods are known to fail for
$q \ge 1$, and the finite-$\eta$ expansion underpinning the $\Gamma$ bound breaks down as
$q \to 1$ (the square-root argument exceeds unity).

### 7.3 Statistical treatment

Two features of the data forbid naive i.i.d. standard errors, and getting this wrong is the
easiest way to manufacture a spurious result.

**Overlapping windows.** Consecutive weight vectors $w_t$ and $w_{t+1}$ are built from
estimation windows sharing $T-1$ of $T$ observations, so they are almost identical and the
resulting return series is strongly serially dependent.

**Volatility clustering.** Squared returns $(r^p_t)^2$ are autocorrelated by construction of
the underlying process.

Plan: a **stationary block bootstrap** with expected block length on the order
of 20–60 trading days for all confidence intervals, rather than i.i.d. resampling or
Gaussian standard errors.

**Paired comparison.** All estimators will be evaluated on the *same* sequence of
out-of-sample dates, and inference conducted on the paired per-day differences

$$d_t = (r^{A}_t)^2 - (r^{B}_t)^2$$

rather than on independently averaged volatilities. Common market movements cancel in
$d_t$, which sharply increases power — expected to matter since the RIE-vs-LW margin in
this literature is typically small (on the order of 0.1 percentage points of annualized
volatility), a margin only resolvable by pairing.

---

## 8. Metrics (planned)

### 8.1 Realized out-of-sample volatility — the headline

$$\hat\sigma_{\text{oos}} = \sqrt{\frac{252}{M}\sum_{t=1}^{M} \left(w_t^{\top} r_{t+1}\right)^{2}}$$

Reported annualized, in percent, with a block-bootstrap confidence interval, and pairwise
differences against each competitor with their own intervals.

### 8.2 Risk calibration: predicted versus realized

For the GMV portfolio the predicted variance has a closed form,

$$\hat\sigma^{2}_{\text{pred}}(t) = w_t^{\top}\hat\Sigma_t w_t = \frac{1}{\mathbf{1}^{\top}\hat\Sigma_t^{-1}\mathbf{1}},$$

and we plan to report the **calibration ratio**

$$\Psi = \frac{\text{realized out-of-sample risk}}{\text{predicted in-sample risk}},$$

which should equal 1 for a well-calibrated estimator. The raw sample covariance is known to
understate the risk of its own optimized portfolio by a factor of approximately
$\sqrt{1-q}$ (Pafka–Kondor), so $\Psi \approx 1/\sqrt{1-q}$: about 1.58 at $q = 0.6$ and
3.16 at $q = 0.9$ — a large, unambiguous effect, and arguably the more fundamental point
cleaning is for. RIE is constructed so that predicted $\approx$ realized, so this metric
tests its defining property directly.

### 8.3 Turnover and implementability

$$\text{TO} = \frac{1}{M}\sum_{t} \sum_{i} \left| w_{t+1,i} - w_{t,i}^{\text{drift}} \right|,
\qquad
w_{t,i}^{\text{drift}} = \frac{w_{t,i}(1 + r_{i,t+1})}{1 + w_t^{\top} r_{t+1}},$$

comparing against the *drifted* weights so that passive price movement is not counted as
trading.

### 8.4 Conditioning diagnostics

Per rebalance: smallest eigenvalue and condition number of $\hat\Sigma_t$, to explain *why*
an estimator behaves as it does and catch pathologies such as a near-redundant pair leaving
a near-zero cleaned eigenvalue.

### 8.5 Validation against the oracle (done)

Separately from the backtest, each implementation is validated on **synthetic data with a
known** $C$, where the oracle $\xi^{\text{ora}}_i = \langle u_i, C u_i\rangle$ is directly
computable. This is the decisive correctness test, and it is what caught the $\kappa$
calibration failure documented in Section 5.6 and in `notes/RIE_failure_mode_note.md`. Two
weaker tests both passed while the estimator was badly broken: synthetic data with
mixed-sign factor loadings (which never produces a realistic dominant market mode), and
visual inspection of the output (a near-identity correlation matrix looks unremarkable
until compared against something). The synthetic generator therefore includes an
all-positive market factor, sector structure, and realistic industry pairs.

---

## 9. Summary of deviations from the textbook methods

| Component | Textbook | Here | Reason |
|---|---|---|---|
| LW target | Constant correlation (2003) | Identity | Keeps all three schemes rotationally invariant and mutually comparable |
| LW intensity | Hand-coded $\beta/\gamma$ | `sklearn` on standardized returns | Literal transcription degenerated to $\hat\alpha_s \equiv 0$; sklearn cross-validated against Appendix B |
| RIE $\kappa$ | Matched to $\lambda_N$ | Matched to $p_{15}$ of spectrum | $\lambda_N$ is the noisiest statistic in the spectrum; real industry structure collapses $\kappa$ and degenerates the estimator |
| $\Gamma$ | Uncapped | Capped at 2.0 | Theory bounds it at $\approx 1.3$; no-op on healthy data, prevents catastrophic blow-up |
| Health check | — | Fraction of eigenvalues capped | $\max\Gamma$ is dominated by single near-degenerate directions and gives false alarms |
| Diagonal | Trace preserved | Renormalized to unit diagonal | Trace preservation only fixes the *average* diagonal; required for $\hat\Sigma_{ij} = \hat\sigma_i\hat\sigma_j\Xi_{ij}$ |
| $q$ range | — | $q < 1$ enforced | Raw GMV undefined and IWs/QuEST both fail at $q \ge 1$ |

---

## 10. Open items

- **Survivorship bias.** Point-in-time membership reconstruction, deferred until the
  backtest is working.
- **Near-exact redundancy.** A pair correlated above 0.999 (a dual share class, say) leaves
  a genuinely near-zero cleaned eigenvalue. The oracle agrees it *should* be near zero, so
  this is real structure, not error — but it makes $\hat\Sigma$ dangerous to invert
  (condition numbers of order $10^{11}$). To be handled at the portfolio construction step.
- **Effective $q$.** Some of the literature finds that an effective ratio
  $q_{\text{eff}} > q$ fits real data better, attributed to return autocorrelation widening
  the spectrum. Calibrating $q_{\text{eff}}$ is left as a possible extension.
- **Choice of $p = 15$.** Validated as robust across every scenario tested, and identical to
  $p = 20$, but it remains a tuned constant. A principled bulk-edge estimator would be
  preferable.
- **Volatility estimator.** Currently the sample standard deviation over the estimation
  window. A cross-sectional volatility normalization, GARCH-type, or implied-volatility
  estimator is the natural refinement and composes independently with the correlation
  cleaning.
