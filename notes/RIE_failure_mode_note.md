# RIE (IWs): why the textbook κ calibration fails on real equity data

**Status:** diagnosed and fixed. This is a genuine limitation of the
published algorithm on realistic equity universes, and worth reading if you
want to see the debugging process rather than just the final code — note the
*first* diagnosis below was wrong, and the correction matters.

## Symptom

The RIE(IWs)-cleaned correlation matrix came out looking like the identity, and
in a milder form the market-mode eigenvalue took a ~25% haircut
(cleaned/raw = 0.748 on the real S&P data, λ₁ = 74.67 → 55.87).

## First diagnosis — WRONG, recorded as a warning

Initially attributed to near-duplicate assets (dual share classes) and "fixed"
by pruning the universe: iteratively drop whichever stock dominates the
smallest eigenvector until the calibration looks healthy.

**On the real data this removed 30 names** — MCO/SPGI, DAL/UAL, MA/V, KKR/BX,
HBAN/KEY, NXPI/ADI, EOG/FANG, MAR/HLT, HD/LOW, TMO/DHR, HAL/SLB, ADP/PAYX,
SO/DUK, RSG/WM, KMI/WMB, WDC/STX, PSX/MPC, … — at correlations of 0.79–0.88,
**and still failed to converge** (max Γ = 2.874 after hitting the 30-drop cap).

Those are real industry duopolies, i.e. precisely the correlation structure the
project exists to estimate. The arithmetic also never worked: a pair at ρ = 0.87
produces an eigenvalue near 0.13, nowhere near the observed λ_N = 0.0018. The
greedy search was chasing the wrong target.

**Lesson: when a "data cleaning" step starts deleting economically meaningful
observations, the model is wrong, not the data.**

## Correct diagnosis

The textbook algorithm calibrates the IW shape parameter κ by matching the
reference spectrum's lower edge to **λ_N, the single smallest
eigenvalue**:

    κ = 2λ_N / [(1−q−λ_N)² − 4qλ_N]

That is the most noise-sensitive statistic in the entire spectrum, and realistic
industry structure legitimately pushes it far below the bulk edge. κ collapses
toward 0, the IW reference spectrum becomes absurdly wide, the correction factor
Γ explodes and is applied to most of the spectrum, the bulk inflates, and — since
the trace is pinned at N — the market mode gets squeezed out of the fixed budget.

Decisively: **this happens with no bad data at all.** 25 genuine industry pairs
at ρ = 0.86 and zero duplicates already break it.

## Measurements (synthetic, known true C, N=300, T=1000, q=0.3)

MSE against the oracle estimator ξᵢ = ⟨uᵢ, C uᵢ⟩:

| scenario | κ from λ_N (textbook) | κ from p15 | LW | raw (no cleaning) |
|---|---|---|---|---|
| clean universe | 0.0197 | 0.0199 | 0.157 | 0.164 |
| 25 industry pairs | 1.9258 | **0.0120** | 0.127 | 0.128 |
| 40 industry pairs | 1.8098 | **0.0148** | 0.116 | 0.117 |
| q = 0.88 | 1.0576 | **0.0781** | 0.486 | 0.485 |
| pure noise (C=I) | 0.0060 | 0.0060 | 0.000 | 0.299 |

The textbook calibration is ~10–15× **worse than no cleaning at all** as soon as
realistic structure is present. The percentile calibration costs essentially
nothing on a clean universe (0.0199 vs 0.0197) and is 120–160× better on a
realistic one. Market-mode ratio tracks the oracle closely (0.981–0.984 vs
oracle 0.998–0.999) instead of collapsing to ~0.70.

## Fix adopted

1. **κ calibrated on the 15th percentile of the spectrum**, not λ_N
   (`KAPPA_PCTILE = 15.0`; set `kappa_pctile=None` to reproduce the textbook
   behaviour). This estimates where the bulk edge actually is while ignoring
   genuine low-end outliers — the same logic already applied at the top of
   the spectrum, where the machinery uses the "spikeless" matrix.
2. **Γ capped at 2.0** as a safety net. Verified a no-op on healthy universes
   (max Γ measured 1.2–1.6). Theory says the correction is 1 + O(N^(−1/4)) ≈
   1.3 for N = 300.
3. **Health metric = fraction of eigenvalues capped**, not max Γ. Once κ is
   calibrated on a percentile, max Γ is dominated by whatever single
   near-degenerate direction exists and reads ~1e9 while the estimate is
   perfectly fine. Cap fraction: healthy 0%, working-but-messy ~6%, broken
   53–74%. Warn above 20%.
4. **Market-mode assertion** `cleaned/raw > 0.90`, skipped when there is no
   dominant mode (λ_max < 5), since on pure noise every eigenvalue is
   legitimately pulled to 1. A threshold of 0.5 would be far too lenient and
   would print a false "market mode preserved" on the broken 0.748 result.
5. **Unit-diagonal renormalization** after reconstruction. Not in the textbook
   algorithm — strictly it perturbs the eigenvectors, so it leaves the
   rotationally-invariant class — but it enforces a constraint the truth
   satisfies and measurably helps: ‖Ξ−C‖_F = 5.704 with, 6.085 without, 8.715
   for raw E. It also makes the earlier global trace rescale redundant
   (verified: scaling ξ by any constant changes the final matrix by 5e-16),
   so the printed "trace = N" check is trivially true, not an independent
   verification.
6. **Pruning removed entirely.** `09a_universe_hygiene.py` is now
   diagnostics-only: low-end spectrum, eigenvector concentration (effective
   number of names carrying each near-zero mode — ~2 means one redundant
   pair, large means diffuse economic structure), and top pairwise
   correlations.

## Still open / caveats

- A near-exactly-redundant pair (ρ > 0.999, e.g. a dual share class) leaves a
  genuinely near-zero cleaned eigenvalue. The oracle agrees it *should* be near
  zero — it is real structure — but it makes the matrix dangerous to invert
  (condition number ~1e11). Handle at the portfolio step, not by deleting names.
- The finite-η edge expansion behind the Γ bound breaks down as q → 1 (the
  square-root argument exceeds 1). Both IWs and QuEST-type methods are known to
  fail for q ≥ 1. Relevant to the planned T = 250 sweep, where q ≈ 1.2.

## Methodological lesson

Validate any cleaning scheme against the **oracle** ξᵢ = ⟨uᵢ, C uᵢ⟩ on synthetic
data with known C — it is directly computable. Two weaker tests both passed
while the estimator was badly broken: mixed-sign factor loadings (never
produces a realistic market mode) and "does the output look plausible" (an
identity-ish correlation matrix looks fine until you compare it to something).
