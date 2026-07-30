"""
Step 1 -- Data Acquisition
==========================================================================
Covariance Shrinkage Project

WHAT THIS SCRIPT DOES
----------------------
1. Scrapes the *current* S&P 500 constituent list from Wikipedia.
2. Downloads ~5 years of daily adjusted-close prices and volumes for every
   constituent from Yahoo Finance (via yfinance), in batches, with
   retry/backoff so transient rate-limiting doesn't kill the whole run.
3. Filters down to a "high quality" subset of ~300 stocks: tickers that
   (a) have a complete price history over the analysis window (no
   IPO/gap issues) and (b) are among the most liquid names by average
   dollar volume.
4. Computes simple daily returns for the final universe.
5. Saves everything to local files (prices, volumes, returns, and a
   per-ticker quality report).

WHY THESE CHOICES
-------------------------------------------------------------------------
- Adjusted close (not raw close): stock splits and dividends create
  artificial price jumps that have nothing to do with real returns. If we
  used raw close, those jumps would show up as fake, huge volatility and
  contaminate the covariance matrix we build in later steps. yfinance's
  `auto_adjust=True` handles this for us automatically.

- Liquidity filter (average dollar volume = price * volume): thinly
  traded stocks have noisier and sometimes stale prices, which distorts
  exactly the covariance structure this project studies. Filtering to the
  most liquid ~300 names is both a data-quality fix and closer to what a
  real portfolio could actually trade -- it's a standard move in the
  empirical covariance/RMT literature (e.g. the CFM papers restrict to
  liquid US large caps for the same reason).

- We download more history (~5 years) than we strictly need for the
  current T ~ 1000 trading-day window. That's deliberate: the project
  plan calls for testing a few different values of q = N/T (the
  asset-count / sample-size ratio that governs how noisy the sample
  covariance matrix is), so having a longer cached history lets us try
  longer T later without re-downloading.

- Survivorship bias (known, deliberately deferred limitation): this
  pulls TODAY's S&P 500 list, not the S&P 500 as it existed at each
  historical date. Names removed from the index in the last few years
  won't appear even though they were real constituents at the time.
  Flagged in the project plan as something to revisit after the first
  backtest works, not before.

HOW TO RUN
----------
1. Install dependencies (one-time), ideally in a virtual environment:
     pip install -r requirements.txt

2. Run:
     python src/01_download_data.py

3. It creates an `output/` folder next to this script containing:
     - sp500_universe_wikipedia.csv   raw scraped ticker/sector list
     - prices_full_history.parquet    adjusted close, ALL tickers that downloaded OK, full ~5yr window
     - volumes_full_history.parquet   raw share volume, same tickers/dates
     - universe_quality_report.csv    per-ticker completeness/liquidity stats + included flag
     - selected_universe.txt          final ~300 tickers, one per line
     - prices_selected.parquet        adjusted close, final universe only
     - returns_selected.parquet       simple daily returns, final universe only
     - download_log.txt               full run log, including any tickers that failed

TROUBLESHOOTING
----------------
- If most/all chunks fail with HTTP 429 or "JSON decode" style errors,
  Yahoo is rate-limiting you. The retry/backoff below usually recovers,
  but if it doesn't: wait a few minutes and re-run, or reduce CHUNK_SIZE.
- If yfinance itself throws odd errors on import/download, it's usually
  a version mismatch with Yahoo's undocumented API -- try:
    pip install --upgrade yfinance
==========================================================================
"""

import time
import warnings
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")  # yfinance is chatty about deprecations we don't care about here

# ---------------------------------------------------------------------------
# CONFIGURATION -- tweak these if you want a different universe size,
# window, or quality bar. Everything downstream reads from these constants.
# ---------------------------------------------------------------------------
N_TARGET = 300              # final number of stocks we want
T_TARGET = 1000             # trading days needed for the *current* analysis
T_BUFFER = 60                # extra trading days of margin for the completeness check
YEARS_DOWNLOAD = 5           # how much history to actually fetch (more than T_TARGET
                              # needs, so later steps can try longer windows for free)
MIN_COMPLETENESS = 0.995     # a ticker must have real (non-NaN) prices for at least
                              # this fraction of the required window to qualify
CHUNK_SIZE = 40              # tickers per yfinance batch call
MAX_RETRIES = 4              # retries per chunk before giving up on it
BASE_BACKOFF_SECONDS = 5     # doubles each retry: 5s, 10s, 20s, 40s...

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=int(YEARS_DOWNLOAD * 365.25) + 30)

try:
    # Normal case: running as a .py script -- __file__ points at this file,
    # so "output/" is created right next to it.
    OUTPUT_DIR = Path(__file__).parent / "output"
except NameError:
    # __file__ isn't defined when this code runs inside a Jupyter/notebook
    # cell (e.g. pasted into a .ipynb) -- fall back to the current working
    # directory instead.
    OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUTPUT_DIR / "download_log.txt"


def log(msg: str) -> None:
    """Print AND append to a log file, so you have a record after the run."""
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# STEP A -- get the current S&P 500 ticker list from Wikipedia
# ---------------------------------------------------------------------------
def get_sp500_tickers() -> pd.DataFrame:
    """
    Scrapes https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

    Returns a DataFrame with columns: ticker, security, sector, date_added.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # pd.read_html(url) would fetch this itself, but it sends a generic
    # User-Agent that Wikipedia's servers now reject with a 403 Forbidden.
    # Fetching the page ourselves with a normal browser-like User-Agent
    # header (via `requests`) sidesteps that, and we hand the resulting
    # HTML text to pd.read_html instead of a URL.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; covariance-shrinkage-project/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    df = tables[0].rename(columns={
        "Symbol": "ticker",
        "Security": "security",
        "GICS Sector": "sector",
        "Date added": "date_added",
    })[["ticker", "security", "sector", "date_added"]]

    # yfinance/Yahoo use a dash where Wikipedia uses a dot for share
    # classes, e.g. Berkshire Hathaway class B is "BRK.B" on Wikipedia but
    # "BRK-B" on Yahoo Finance. Fix that up so downloads don't silently
    # fail on these names.
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    return df.drop_duplicates(subset="ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP B -- batched download with retry/backoff
# ---------------------------------------------------------------------------
def chunked(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def download_chunk(tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Downloads one chunk of tickers. Returns a wide DataFrame (2-level
    column index: (ticker, field), field in {"Close", "Volume"}) or None
    if the chunk failed outright after all retries.

    auto_adjust=True means "Close" here is *adjusted* close (splits and
    dividends already folded in) -- see the module docstring for why that
    matters for a covariance study.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            if data is None or data.empty:
                raise ValueError("empty response")
            return data
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            log(f"  chunk attempt {attempt}/{MAX_RETRIES} failed ({e}); "
                f"retrying in {wait}s")
            time.sleep(wait)
    log(f"  chunk permanently failed after {MAX_RETRIES} attempts: {tickers}")
    return None


def download_all(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Downloads all tickers in chunks and stitches the results into two wide
    DataFrames: adjusted close prices and volumes (rows = dates, columns =
    tickers). Returns (prices, volumes, tickers_that_failed).
    """
    price_series, volume_series, failed = [], [], []

    chunks = list(chunked(tickers, CHUNK_SIZE))
    for i, chunk in enumerate(chunks, 1):
        log(f"Downloading chunk {i}/{len(chunks)} ({len(chunk)} tickers)...")
        data = download_chunk(chunk)
        if data is None:
            failed.extend(chunk)
            continue

        for t in chunk:
            try:
                # yfinance returns a single-level column index if only one
                # ticker was requested, and a 2-level (ticker, field) index
                # for multiple tickers -- handle both.
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.get_level_values(0):
                        raise KeyError(t)
                    close = data[t]["Close"]
                    vol = data[t]["Volume"]
                else:
                    close = data["Close"]
                    vol = data["Volume"]
                if close.dropna().empty:
                    raise ValueError("all-NaN")
                price_series.append(close.rename(t))
                volume_series.append(vol.rename(t))
            except Exception:
                failed.append(t)

        # Small pause between chunks -- polite to Yahoo's servers and
        # avoids tripping rate limits that would otherwise fail the *next*
        # chunk too.
        time.sleep(1.5)

    prices = pd.concat(price_series, axis=1).sort_index()
    volumes = pd.concat(volume_series, axis=1).sort_index()
    return prices, volumes, failed


# ---------------------------------------------------------------------------
# STEP C -- quality filtering: completeness + liquidity
# ---------------------------------------------------------------------------
def build_quality_report(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, computes:
      - completeness: fraction of the last (T_TARGET + T_BUFFER) trading
        days that have a real (non-NaN) price. A recently-IPO'd stock, or
        one with data gaps, scores low here.
      - avg_dollar_volume: mean(price * volume) over that same window --
        a standard liquidity proxy. Bigger = easier to trade in size
        without moving the price, and empirically, cleaner data.
    """
    window = prices.tail(T_TARGET + T_BUFFER)
    vol_window = volumes.tail(T_TARGET + T_BUFFER)

    completeness = window.notna().mean()
    dollar_volume = (window * vol_window).mean()

    report = pd.DataFrame({
        "completeness": completeness,
        "avg_dollar_volume": dollar_volume,
    })
    report["qualifies_on_completeness"] = report["completeness"] >= MIN_COMPLETENESS
    return report.sort_values("avg_dollar_volume", ascending=False)


def select_universe(report: pd.DataFrame) -> List[str]:
    """Top N_TARGET tickers by liquidity, among those passing the
    completeness bar."""
    eligible = report[report["qualifies_on_completeness"]]
    if len(eligible) < N_TARGET:
        log(f"WARNING: only {len(eligible)} tickers passed the completeness "
            f"filter, fewer than the target of {N_TARGET}. Selecting all of "
            f"them -- consider lowering MIN_COMPLETENESS or T_TARGET.")
    return eligible.head(N_TARGET).index.tolist()


# ---------------------------------------------------------------------------
# STEP D -- sanity checks
# ---------------------------------------------------------------------------
def sanity_check(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    log("--- Sanity checks ---")
    log(f"Duplicate dates in index: {prices.index.duplicated().sum()}")
    log(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    log(f"Total trading days: {len(prices)}")

    nan_counts = prices.isna().sum()
    log(f"Tickers with any remaining NaNs: {(nan_counts > 0).sum()} / {prices.shape[1]}")

    # Extreme single-day returns can indicate a bad split/dividend
    # adjustment -- or a genuine real crash/spike (earnings surprise,
    # M&A news, etc). We just flag them here for a manual look, we don't
    # auto-remove anything.
    extreme = returns.abs() > 0.5
    n_extreme = int(extreme.sum().sum())
    log(f"Single-day |return| > 50%: {n_extreme} instances")
    if n_extreme > 0:
        flagged = returns.where(extreme).stack()
        log("  Examples (ticker, date, return):")
        for (date, ticker), val in flagged.dropna().head(10).items():
            log(f"    {ticker} on {date.date()}: {val:+.1%}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log("=== Step 1: Data Acquisition ===")
    log(f"Window: {START_DATE:%Y-%m-%d} to {END_DATE:%Y-%m-%d} "
        f"({YEARS_DOWNLOAD} years buffer for future flexibility)")

    log("Fetching S&P 500 ticker list from Wikipedia...")
    universe = get_sp500_tickers()
    log(f"Found {len(universe)} tickers.")
    universe.to_csv(OUTPUT_DIR / "sp500_universe_wikipedia.csv", index=False)

    log("Downloading price history from Yahoo Finance (this can take a "
        "few minutes)...")
    prices, volumes, failed = download_all(universe["ticker"].tolist())
    log(f"Downloaded {prices.shape[1]} tickers successfully, "
        f"{len(failed)} failed: {failed}")

    prices.to_parquet(OUTPUT_DIR / "prices_full_history.parquet")
    volumes.to_parquet(OUTPUT_DIR / "volumes_full_history.parquet")

    log("Building quality report (completeness + liquidity)...")
    report = build_quality_report(prices, volumes)
    report.to_csv(OUTPUT_DIR / "universe_quality_report.csv")

    selected = select_universe(report)
    log(f"Selected {len(selected)} tickers for the final universe.")
    with open(OUTPUT_DIR / "selected_universe.txt", "w") as f:
        f.write("\n".join(selected))

    selected_prices = prices[selected]
    returns = selected_prices.pct_change().dropna(how="all")
    returns.to_parquet(OUTPUT_DIR / "returns_selected.parquet")
    selected_prices.to_parquet(OUTPUT_DIR / "prices_selected.parquet")

    sanity_check(selected_prices, returns)

    log("=== Done. ===")


if __name__ == "__main__":
    main()
