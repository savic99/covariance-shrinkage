"""
Step 1b -- Universe Selection
==========================================================================
Covariance Shrinkage Project

Reproducible standalone version of the universe-selection logic. Expects
the output/ folder produced by 01_download_data.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# UNIVERSE SELECTION -- Cell 1 / 4 : load everything that was downloaded
# =============================================================================
# Step 1 of the data pipeline (`01_download_data.py`) already fetched ~5 years
# of split/dividend-adjusted daily closes and raw share volumes for every
# current S&P 500 constituent, and cached them as two wide parquet tables
# (rows = trading days, columns = tickers). Here we just load them back in and
# look at what we actually have, BEFORE any filtering -- that way every later
# filter can be reported as "X names in, Y names out".
#
# We deliberately load the FULL downloaded history (not just the window we
# intend to analyse) because the completeness filter in cell 2 is more
# informative over the long window: a stock that IPO'd or was added to the
# index two years ago looks perfectly clean in a 1000-day window but is
# clearly incomplete over 5 years, and those late arrivals are exactly the
# names whose data is most likely to have oddities.

OUTPUT_DIR = Path("output")          # created by 01_download_data.py

# --- configuration for the whole universe-selection procedure -----------------
ANALYSIS_END   = "2026-07-23"   # last trading day of the estimation sample.
                                # Chosen by hand: the raw download runs a few
                                # days past this, but those trailing rows are
                                # not trustworthy (see the NaN-per-row printout
                                # below), so we cut the sample here.
T_RETURNS      = 1000           # number of *return* observations we require,
                                # i.e. T in q = N/T. Needs T_RETURNS + 1
                                # consecutive price rows.
N_TARGET       = 300            # final universe size, i.e. N in q = N/T.
MIN_COMPLETENESS = 0.99         # cell 2: drop a ticker if more than 1% of its
                                # full-history prices are missing.
MAX_CORR       = 0.98           # cell 3: pairs above this are treated as the
                                # same security, not as economic structure.

# --- load ---------------------------------------------------------------------
prices_all  = pd.read_parquet(OUTPUT_DIR / "prices_full_history.parquet")
volumes_all = pd.read_parquet(OUTPUT_DIR / "volumes_full_history.parquet")

# Defensive check: the two tables must be aligned on both axes, otherwise the
# dollar-volume product in cell 2 would silently pair up the wrong columns.
prices_all, volumes_all = prices_all.align(volumes_all, join="inner")

print(f"Loaded prices : {prices_all.shape[0]} trading days x {prices_all.shape[1]} tickers")
print(f"Loaded volumes: {volumes_all.shape[0]} trading days x {volumes_all.shape[1]} tickers")
print(f"Date range    : {prices_all.index.min().date()} -> {prices_all.index.max().date()}")
print(f"Duplicate dates in index: {prices_all.index.duplicated().sum()}")
print(f"Total missing prices: {prices_all.isna().sum().sum():,} "
      f"({100 * prices_all.isna().mean().mean():.2f}% of all cells)")

# --- where are the missing values? --------------------------------------------
# Missing values come in two very different flavours and it matters which:
#   * missing down a COLUMN  -> that ticker has a short history (recent IPO,
#     recent index addition). Handled by the completeness filter in cell 2.
#   * missing across a ROW   -> that whole *date* is broken in the data feed.
#     One bad row would inject a fake NaN into every single stock and destroy
#     the "1000 consecutive clean days" requirement for the entire universe,
#     so these have to be spotted and cut, not filtered per-ticker.
nan_per_row = prices_all.isna().sum(axis=1)
bad_rows = nan_per_row[nan_per_row > 0.5 * prices_all.shape[1]]
print(f"\nDates where >50% of tickers are missing (broken feed rows): {len(bad_rows)}")
for d, n in bad_rows.items():
    print(f"   {d.date()}: {n} / {prices_all.shape[1]} tickers missing")

print(f"\nLast 6 rows, missing count per row "
      f"(sanity check on the ANALYSIS_END = {ANALYSIS_END} choice):")
print(nan_per_row.tail(6).to_string())

# --- worst offenders by column ------------------------------------------------
completeness_all = prices_all.notna().mean()
print("\n10 least complete tickers over the full history "
      "(fraction of days with a real price):")
print(completeness_all.sort_values().head(10).to_string(float_format="%.3f"))


# =============================================================================
# UNIVERSE SELECTION -- Cell 2 / 4 : quality filters (completeness, liquidity)
# =============================================================================
# Two filters, for two different reasons:
#
# COMPLETENESS. A covariance matrix needs the *same* T days for every stock:
# the whole point of q = N/T is that it counts one common sample size. Stocks
# with short or gappy histories force either dropping days (shrinking T for
# everyone) or filling gaps (inventing data, which biases correlations toward
# zero and quietly flatters every "cleaned" estimator). Cheaper to drop the
# stock. Threshold: >99% of the full downloaded history must be real prices.
#
# LIQUIDITY. Thinly traded stocks have stale prices: if a stock does not trade
# near the close, its recorded close is an old price, so its return appears on
# the wrong day. That mechanically *understates* its correlation with everything
# else (the "Epps effect") -- a real bias in the object we are trying to
# estimate, not just extra noise. Ranking by average dollar volume
# (price x shares traded) and keeping the top names is the standard fix, and is
# what the CFM/RMT papers do for the same reason. Dollar volume is also the
# right tie-breaker in cell 3, where we must choose which of two nearly
# identical stocks to keep.
#
# Note we RANK by liquidity here but do not yet cut to N_TARGET: cell 3 still
# has to remove duplicate securities, and cell 4 imposes the strict "no gaps in
# the estimation window" rule. Cutting to exactly 300 only makes sense once
# both of those have had their say.

# The liquidity ranking is computed over the estimation window we will actually
# use (not the full 5 years), because what matters is how tradeable / how
# well-priced each name is during the sample we estimate the covariance from.
liq_window = slice(None, ANALYSIS_END)
px_win  = prices_all.loc[liq_window].tail(T_RETURNS + 1)
vol_win = volumes_all.loc[liq_window].tail(T_RETURNS + 1)

# --- filter 1: completeness over the FULL downloaded history ------------------
completeness = prices_all.notna().mean()
passes_completeness = completeness >= MIN_COMPLETENESS

# --- filter 2 (ranking, not yet a cut): average dollar volume -----------------
# price * volume, averaged over the estimation window. In dollars/day.
avg_dollar_volume = (px_win * vol_win).mean()

quality = pd.DataFrame({
    "completeness": completeness,
    "avg_dollar_volume": avg_dollar_volume,
    "passes_completeness": passes_completeness,
})

n_before = len(quality)
dropped_incomplete = quality.index[~quality["passes_completeness"]].tolist()

# Survivors, sorted most liquid first. This ordering is carried through cells
# 3 and 4 -- "top N" always means "top N by average dollar volume".
filtered = (quality[quality["passes_completeness"]]
            .sort_values("avg_dollar_volume", ascending=False))

print(f"Completeness filter (>= {MIN_COMPLETENESS:.0%} of full history):")
print(f"   in : {n_before} tickers")
print(f"   out: {len(filtered)} tickers  ({len(dropped_incomplete)} dropped)")
print(f"   dropped: {dropped_incomplete}")

print(f"\nLiquidity ranking over {px_win.index[0].date()} -> {px_win.index[-1].date()} "
      f"({len(px_win)} price rows):")
print("\n   most liquid 10 ($bn/day):")
print((filtered["avg_dollar_volume"].head(10) / 1e9).to_string(float_format="%.2f"))
print("\n   least liquid 10 ($mn/day):")
print((filtered["avg_dollar_volume"].tail(10) / 1e6).to_string(float_format="%.1f"))

# Is the N_TARGET cut going to bite in a place where liquidity is still
# comfortable? If the 300th name trades a few hundred million a day we are fine;
# if it trades a few million we have a staleness problem regardless of ranking.
if len(filtered) >= N_TARGET:
    cutoff = filtered["avg_dollar_volume"].iloc[N_TARGET - 1]
    print(f"\nLiquidity at rank {N_TARGET} (provisional cutoff): "
          f"${cutoff/1e6:,.0f}mn/day")
else:
    print(f"\nWARNING: only {len(filtered)} tickers survive, fewer than "
          f"N_TARGET = {N_TARGET}. Loosen MIN_COMPLETENESS or lower N_TARGET.")


# =============================================================================
# UNIVERSE SELECTION -- Cell 3 / 4 : remove duplicated securities
# =============================================================================
# We now look for pairs of "different" stocks that are in fact the same asset.
# The classic cases are dual share classes (GOOGL / GOOG, FOXA / FOX, NWSA /
# NWS): same company, same cash flows, two listings, so their daily returns
# agree to within a rounding error.
#
# WHY THIS MATTERS FOR THIS PROJECT SPECIFICALLY. A pair with rho = 0.998
# creates an eigenvector -- the long/short spread between the two share
# classes -- with eigenvalue ~ 1 - rho ~ 0.002, i.e. essentially zero. Three
# things then break:
#   * the correlation matrix becomes near-singular, so Sigma^-1 in the GMV
#     weights w ~ Sigma^-1 1 blows up and the portfolio piles into a huge
#     offsetting long/short bet on one pair of share classes;
#   * that near-zero eigenvalue sits far below the Marchenko-Pastur left edge
#     (1 - sqrt(q))^2, so it is not noise the RMT machinery is designed to
#     clean -- it is a genuine (but useless) rank deficiency, and it drags the
#     RIE's kappa calibration around (see `notes/RIE_failure_mode_note.md`);
#   * the comparison between estimators ends up being decided by an artefact
#     rather than by how well each method handles noise.
#
# THRESHOLD CHOICE. 0.98 on daily returns is deliberately extreme. Genuine
# industry structure -- MA/V, HD/LOW, AMAT/LRCX/KLAC, the apartment REITs --
# lives at 0.80-0.93 and is exactly the correlation structure this project is
# meant to ESTIMATE, so it must be left alone. Only "this is literally the same
# security twice" clears 0.98. Do NOT lower this threshold to tidy up the
# spectrum: an earlier version of this project pruned 30 names that way and
# destroyed the signal.
#
# WHICH ONE TO KEEP. The more liquid share class, using the dollar-volume
# ranking from cell 2. For dual listings the liquid class is the one with the
# real price discovery; the other one is likelier to be stale.

candidates = filtered.index.tolist()          # already sorted by liquidity desc

# Returns over the estimation window. Simple returns (P_t / P_{t-1} - 1), for
# consistency with the rest of the pipeline; at daily frequency log and simple
# returns give correlations that agree to ~1e-4, so this choice is immaterial
# here. `.dropna(how="all")` just removes the first row, which is NaN by
# construction.
ret_win = px_win[candidates].pct_change().dropna(how="all")

# Pairwise correlation. pandas uses pairwise-complete observations, so the few
# remaining scattered NaNs (each ticker still has up to 1% of them) do not
# propagate; cell 4 removes them properly.
corr = ret_win.corr()

# Extract the upper triangle as a long list of pairs, sorted most correlated
# first. np.triu_indices(N, 1) gives (i, j) with i < j, i.e. each pair once and
# no diagonal.
i_idx, j_idx = np.triu_indices(len(candidates), k=1)
pair_corr = pd.DataFrame({
    "a": [candidates[i] for i in i_idx],
    "b": [candidates[j] for j in j_idx],
    "corr": corr.values[i_idx, j_idx],
}).sort_values("corr", ascending=False, ignore_index=True)

print(f"Pairwise correlations of daily returns over "
      f"{ret_win.index[0].date()} -> {ret_win.index[-1].date()} "
      f"({len(ret_win)} returns, {len(candidates)} tickers, "
      f"{len(pair_corr):,} pairs)\n")

print("Counts above each threshold:")
for th in (0.99, MAX_CORR, 0.95, 0.90, 0.85):
    print(f"   |rho| > {th:.2f}: {(pair_corr['corr'] > th).sum():>6,} pairs")

print("\nTop 15 pairs (for context -- only those above the threshold are cut):")
for _, row in pair_corr.head(15).iterrows():
    a, b = row["a"], row["b"]
    mark = "  <-- DUPLICATE, will cut" if row["corr"] > MAX_CORR else ""
    print(f"   {a:6s} / {b:6s}  rho = {row['corr']:+.4f}   "
          f"$/day {filtered.loc[a,'avg_dollar_volume']/1e6:>8,.0f}mn vs "
          f"{filtered.loc[b,'avg_dollar_volume']/1e6:>8,.0f}mn{mark}")

# --- do the elimination -------------------------------------------------------
# Walk the offending pairs from most to least correlated and drop the less
# liquid member. Walking a list of pairs (rather than doing it in one shot)
# handles clusters correctly: if A-B, A-C and B-C are all above the threshold,
# this keeps only the single most liquid of the three, because once B is
# dropped the B-C pair is already resolved.
offenders = pair_corr[pair_corr["corr"] > MAX_CORR]
dropped_duplicates = []
for _, row in offenders.iterrows():
    a, b = row["a"], row["b"]
    if a in dropped_duplicates or b in dropped_duplicates:
        continue                                    # cluster already resolved
    # `filtered` is sorted by liquidity, so the one appearing later is the
    # less liquid of the two.
    loser = b if filtered.index.get_loc(a) < filtered.index.get_loc(b) else a
    keeper = a if loser == b else b
    dropped_duplicates.append(loser)
    print(f"\n   rho({a}, {b}) = {row['corr']:.4f} > {MAX_CORR}: "
          f"keeping {keeper} (more liquid), dropping {loser}")

deduped = filtered.drop(index=dropped_duplicates)
print(f"\nDuplicate filter: {len(filtered)} -> {len(deduped)} tickers "
      f"({len(dropped_duplicates)} dropped: {dropped_duplicates})")

# Worth knowing about, even though we are not cutting them: pairs just under
# the threshold. If one of these turns out to be another share-class pair, it
# is a judgement call whether to raise or lower MAX_CORR -- but the decision
# should be made by looking at the names, not at the spectrum.
near_miss = pair_corr[(pair_corr["corr"] <= MAX_CORR) & (pair_corr["corr"] > 0.95)]
if len(near_miss):
    print(f"\nJust below the threshold (0.95 < rho <= {MAX_CORR}) -- "
          f"kept, but check whether these are share classes too:")
    for _, row in near_miss.iterrows():
        print(f"   {row['a']:6s} / {row['b']:6s}  rho = {row['corr']:+.4f}")


# =============================================================================
# UNIVERSE SELECTION -- Cell 4 / 4 : final cut to N x T with no gaps
# =============================================================================
# The estimators being compared (sample covariance, Ledoit-Wolf, RIE) all take
# a single rectangular T x N return matrix with no holes: they are functions of
# the eigenvalues of E = (1/T) Z^T Z, and that object is only defined if every
# stock has a return on every one of the same T days. So the last step is
# strict, not statistical:
#
#   1. take the block of T_RETURNS + 1 consecutive trading days ending at
#      ANALYSIS_END (T + 1 prices -> exactly T returns, so T in q = N/T is
#      exactly T_RETURNS -- easy to get off by one here);
#   2. keep only tickers with ZERO missing prices in that block (the ~1% of
#      missing values still tolerated by cell 2 is not tolerable here);
#   3. of those, take the top N_TARGET by the liquidity ranking from cell 2.
#
# Order matters: completeness is a hard constraint and liquidity is a
# preference, so the hard constraint is applied first and the ranking only
# breaks ties among the names that already qualify.

# --- step 1: the exact price block --------------------------------------------
px_block = prices_all.loc[:ANALYSIS_END].tail(T_RETURNS + 1)

if len(px_block) < T_RETURNS + 1:
    raise ValueError(
        f"Only {len(px_block)} trading days available up to {ANALYSIS_END}, "
        f"need {T_RETURNS + 1}. Download more history or lower T_RETURNS.")

print(f"Estimation block: {px_block.index[0].date()} -> {px_block.index[-1].date()}")
print(f"   {len(px_block)} price rows -> {len(px_block) - 1} return observations (T)")

# --- step 2: zero missing values in the block ---------------------------------
block = px_block[deduped.index]
fully_complete = block.columns[block.notna().all()]
print(f"\nZero-gap filter inside the block: "
      f"{len(deduped)} -> {len(fully_complete)} tickers")
dropped_gaps = [t for t in deduped.index if t not in set(fully_complete)]
if dropped_gaps:
    gap_counts = block[dropped_gaps].isna().sum().sort_values(ascending=False)
    print(f"   dropped (missing days in block): "
          f"{[f'{t}({n})' for t, n in gap_counts.items()]}")

# --- step 3: top N_TARGET by liquidity ----------------------------------------
eligible = deduped.loc[fully_complete]                # keeps liquidity order
if len(eligible) < N_TARGET:
    raise ValueError(
        f"Only {len(eligible)} eligible tickers, need {N_TARGET}. "
        f"Loosen MIN_COMPLETENESS / MAX_CORR, or lower N_TARGET.")

universe = eligible.head(N_TARGET).index.tolist()

print(f"\nFinal cut: top {N_TARGET} of {len(eligible)} eligible names by liquidity")
print(f"   liquidity range: ${eligible['avg_dollar_volume'].iloc[0]/1e9:,.1f}bn/day "
      f"({universe[0]}) down to "
      f"${eligible['avg_dollar_volume'].iloc[N_TARGET-1]/1e6:,.0f}mn/day "
      f"({universe[-1]})")

# --- build the final price / return matrices ----------------------------------
prices  = px_block[universe]
returns = prices.pct_change().dropna(how="all")      # T x N, no NaNs

N, T = returns.shape[1], returns.shape[0]
print(f"\nReturn matrix R: T = {T} days x N = {N} stocks,  q = N/T = {N/T:.4f}")

# --- final assertions ---------------------------------------------------------
# Cheap, but these are exactly the failure modes that would otherwise show up
# 200 lines later as a mysterious negative eigenvalue.
assert returns.isna().sum().sum() == 0, "NaNs survived into the return matrix"
assert not returns.index.duplicated().any(), "duplicate dates"
assert returns.shape == (T_RETURNS, N_TARGET), \
    f"expected ({T_RETURNS}, {N_TARGET}), got {returns.shape}"
assert (returns.std() > 0).all(), "a stock has zero variance over the window"
print("Assertions passed: no NaNs, no duplicate dates, correct shape, no zero-variance names.")

# Extreme daily moves are worth a look but NOT auto-removed: a -40% day is
# usually a real earnings shock, occasionally a botched split adjustment. Only
# the second kind should be fixed, and only by looking.
extreme = returns.abs() > 0.35
if extreme.any().any():
    print(f"\n{int(extreme.sum().sum())} single-day |return| > 35% "
          f"(flagged for inspection, not removed):")
    for (date, ticker), val in returns.where(extreme).stack().dropna().items():
        print(f"   {ticker:6s} {date.date()}  {val:+.1%}")

# --- persist ------------------------------------------------------------------
# Saved under new filenames so the step-1 outputs stay untouched, and so every
# later step can be re-run from disk without redoing this selection.
prices.to_parquet(OUTPUT_DIR / "prices_universe.parquet")
returns.to_parquet(OUTPUT_DIR / "returns_universe.parquet")
(OUTPUT_DIR / "universe.txt").write_text("\n".join(universe))
selection_report = quality.assign(
    dropped_incomplete=quality.index.isin(dropped_incomplete),
    dropped_duplicate=quality.index.isin(dropped_duplicates),
    dropped_gap_in_block=quality.index.isin(dropped_gaps),
    in_final_universe=quality.index.isin(universe),
).sort_values("avg_dollar_volume", ascending=False)
selection_report.to_csv(OUTPUT_DIR / "universe_selection_report.csv")

print(f"\nSaved -> {OUTPUT_DIR}/prices_universe.parquet, returns_universe.parquet, "
      f"universe.txt, universe_selection_report.csv")
print(f"\nFirst 20 of the final universe: {universe[:20]}")

# `returns` (T x N, no NaNs) is the object every later step consumes -- the same
# `R` used by the covariance / Ledoit-Wolf / RIE cells.
R = returns
