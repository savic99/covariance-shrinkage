# =============================================================================
# RIE (Rotationally Invariant Estimator) with IWs regularization  [REVISED]
# =============================================================================
#
# Unlike Ledoit-Wolf linear shrinkage (one global affine rescaling
# xi_i = 1 + alpha_s*(lambda_i - 1) applied to every eigenvalue), RIE computes
# a DIFFERENT shrinkage factor per eigenvalue, using the full empirical
# spectrum to estimate how much noise contaminates each one. Still rotationally
# invariant (sample eigenvectors untouched, only eigenvalues change) -- same
# family as LW, just a nonlinear shrinkage function instead of a straight line.
#
# This is the "RIE (IWs)" method (Inverse-Wishart regularization + sorting),
# the RMT-based cleaning scheme advocated in the Bouchaud-Potters-Bun line of
# work (CFM). See notes/Covariance_Cleaning_Review.md for the full derivation.
#
# REVISION NOTE (what changed vs. the textbook algorithm, and why):
# The published algorithm calibrates its shape parameter kappa by matching the
# Inverse-Wishart reference spectrum's lower edge to lambda_N -- the single
# SMALLEST sample eigenvalue. On real equity data that is far too fragile.
# Genuine industry duopolies (MA/V, HD/LOW, DAL/UAL, SO/DUK) are real economic
# structure and they legitimately produce small eigenvalues. Those drag
# lambda_N far below the bulk edge, collapse kappa toward 0, and make the
# small-eigenvalue correction factor Gamma explode -- inflating the bulk,
# squeezing the market mode out of the fixed trace budget, and degenerating
# the result toward the identity.
#
# Measured on synthetic data with known true C (N=300, T=1000, q=0.3), where
# MSE is against the oracle estimator xi_i = <u_i, C u_i>:
#
#   scenario            kappa from lambda_N   kappa from p15    LW      raw
#   clean universe             0.0197             0.0199      0.157   0.164
#   25 industry pairs          1.9258             0.0120      0.127   0.128
#   40 industry pairs          1.8098             0.0148      0.116   0.117
#   q = 0.88                   1.0576             0.0781      0.486   0.485
#
# So the literal calibration is ~10-15x WORSE than no cleaning as soon as
# realistic industry structure is present, while the percentile calibration
# costs nothing on a clean universe and is 120-160x better on a realistic one.
# Note this is NOT a data-quality problem, so the fix is NOT to delete stocks:
# an earlier attempt at that pruned 30 economically meaningful names and still
# failed. The estimator needed fixing, not the universe. See
# notes/RIE_failure_mode_note.md for the full failure-mode writeup.
#
# Changes vs the textbook algorithm:
#   1. kappa calibrated on the KAPPA_PCTILE-th percentile of the spectrum
#      instead of lambda_N (set kappa_pctile=None to recover the literal
#      published behaviour).
#   2. Gamma capped at GAMMA_CAP as a safety net. Verified a no-op on healthy
#      universes (max Gamma measured 1.2-1.6, cap never binding).
#   3. Diagnostics returned in an `info` dict, with a loud warning when the
#      calibration looks untrustworthy.

import numpy as np

KAPPA_CAP = 1e6   # numerical stand-in for kappa -> infinity (pure-noise limit)
GAMMA_CAP = 2.0   # safety net: theory says the correction is ~1+O(N^{-1/4})
# Health metric. max(Gamma) is a BAD indicator once we calibrate on a
# percentile: a single near-degenerate pair sends Gamma at that one point to
# ~1e9 while the estimate as a whole stays fine. What actually tracks breakage
# is the FRACTION of the spectrum that needs capping. Measured: healthy 0%,
# working-but-messy universe ~6%, genuinely broken 53-74%.
CAP_FRAC_WARN = 0.20
KAPPA_PCTILE = 15.0  # calibrate kappa on this percentile of the spectrum, NOT
                     # on lambda_N -- see the note above. None = paper's literal
                     # lambda_N behaviour (kept so you can reproduce it).
# Oracle test: on healthy data the largest eigenvalue should barely move
# (measured oracle ratio 0.96-1.02). A 25% haircut is NOT normal -- it means
# the small eigenvalues were over-inflated and ate the trace budget.
MARKET_MODE_MIN_RATIO = 0.90


def _g_iw(z, q, kappa):
    """
    Limiting Stieltjes transform of the sample-correlation eigenvalue density
    when the TRUE correlation matrix is Inverse-Wishart with shape kappa.
    Used only as a smooth, analytically-known reference curve near the noisy
    left edge -- not a claim that stock correlations are really IW-distributed.

    kappa -> infinity reduces to the null-hypothesis (C = I) Marchenko-Pastur
    case; kappa -> 0 is an infinitely wide, uninformative prior.
    """
    root = np.sqrt((2 * kappa + 1) * (2 * q * kappa + 1))
    lam_plus = ((1 + q) * kappa + 1 + root) / kappa
    lam_minus = ((1 + q) * kappa + 1 - root) / kappa
    num = z * (1 + kappa) - kappa * (1 - q) - np.sqrt(z - lam_plus) * np.sqrt(z - lam_minus)
    return num / (z * (z + 2 * q * kappa))


def _rie_shrinkage(z, q, g):
    """
    Core nonlinear-shrinkage map: a Stieltjes transform g evaluated just off
    the real axis near a sample eigenvalue -> a cleaned eigenvalue. Used both
    for the empirical estimate and for the IW reference curve; only g differs.
    """
    return np.real(z) / np.abs(1 - q + q * z * g) ** 2


def rie_iws(eigvals, q, gamma_cap=GAMMA_CAP, kappa_pctile=KAPPA_PCTILE, verbose=True):
    """
    Clean sample correlation eigenvalues via RIE (IWs): nonlinear shrinkage
    + Inverse-Wishart small-eigenvalue regularization + trace-preserving
    rescale + monotonicity sort.

    Parameters
    ----------
    eigvals : ndarray, shape (N,)   Sample correlation eigenvalues, any order.
    q : float                       N / T.
    gamma_cap : float or None       Cap on the correction factor (None = off,
                                    i.e. the literal published algorithm).
    verbose : bool                  Print/warn on suspicious calibration.

    Returns
    -------
    xi_final : ndarray, shape (N,)
        Cleaned eigenvalues, in the SAME order as the input `eigvals`.
    info : dict
        Diagnostics: kappa, alpha_s, max_gamma_raw (BEFORE capping),
        n_corrected, n_capped, trustworthy (bool).
    """
    eigvals = np.asarray(eigvals, dtype=float)
    N = len(eigvals)

    # Sort descending (lambda_1 >= ... >= lambda_N), a common convention.
    # `order` lets us restore the caller's ordering at the end.
    order = np.argsort(eigvals)[::-1]
    lam = eigvals[order]

    # --- Step 1: calibrate kappa (robustly) --------------------------------
    lam_N = lam[-1]
    # The textbook algorithm matches the IW reference spectrum's lower edge to
    # lambda_N, the single smallest eigenvalue. On real equity data that is
    # far too fragile: genuine industry duopolies (MA/V, HD/LOW, DAL/UAL)
    # create real low eigenvalues, which drag lambda_N far below the bulk edge
    # and collapse kappa. Instead, match to a low PERCENTILE of the spectrum,
    # which estimates where the bulk edge actually is while ignoring genuine
    # low-end outliers. This mirrors the logic RIE already applies at the top
    # of the spectrum, where the machinery works with the "spikeless" matrix
    # rather than the raw one.
    lam_ref = lam[-1] if kappa_pctile is None else float(np.percentile(lam, kappa_pctile))
    denom = (1 - q - lam_ref) ** 2 - 4 * q * lam_ref
    if denom <= 0:
        # lam_ref sits at/beyond the null-hypothesis MP edge. Stand in for
        # kappa -> infinity (the pure-noise reference) rather than let the
        # formula return a negative, unphysical kappa.
        kappa = KAPPA_CAP
    else:
        kappa = min(2 * lam_ref / denom, KAPPA_CAP)

    alpha_s = 1.0 / (1.0 + 2 * q * kappa)

    # --- Step 2: raw nonlinear RIE from the empirical resolvent ------------
    # eta = N^{-1/2}: far enough off the real axis to stay well-conditioned,
    # close enough to resolve individual eigenvalues.
    eta = N ** -0.5
    z = lam - 1j * eta

    # g(z_i) = (1/(N-1)) * sum_{j != i} 1/(z_i - lambda_j), vectorized:
    # diff[i, j] = z_i - lambda_j, then drop the j == i diagonal term.
    diff = z[:, None] - lam[None, :]
    inv = 1.0 / diff
    g_empirical = (inv.sum(axis=1) - np.diag(inv)) / (N - 1)

    xi_raw = _rie_shrinkage(z, q, g_empirical)

    # --- Step 3: IW bias correction for small eigenvalues ------------------
    # Gamma = (known exact answer if C were IW) / (what the finite-eta formula
    # gives on that same reference). So Gamma measures the finite-eta bias and
    # should be ~1 except right at the hard left edge.
    xi_reference = _rie_shrinkage(z, q, _g_iw(z, q, kappa))
    Gamma = (1 + alpha_s * (lam - 1)) / xi_reference

    correct_mask = (Gamma > 1) & (lam < 1)
    max_gamma_raw = float(Gamma[correct_mask].max()) if correct_mask.any() else 1.0

    n_capped = 0
    if gamma_cap is not None:
        n_capped = int((Gamma[correct_mask] > gamma_cap).sum())
        Gamma = np.minimum(Gamma, gamma_cap)

    xi = xi_raw.copy()
    xi[correct_mask] = Gamma[correct_mask] * xi_raw[correct_mask]

    # --- Step 4: trace-preserving rescale ---------------------------------
    xi = xi * (lam.sum() / xi.sum())

    # --- Step 5: sort (the "s" in IWs) ------------------------------------
    # The cleaning map should be monotonic in the sample eigenvalues as
    # N -> infinity; non-monotonicity at finite N is noise. Sorting the
    # cleaned values and re-pairing against the descending lam removes it.
    xi_sorted = np.sort(xi)[::-1]
    xi_final = np.empty(N)
    xi_final[order] = xi_sorted

    cap_frac = n_capped / N
    trustworthy = cap_frac <= CAP_FRAC_WARN
    info = {"kappa": float(kappa), "alpha_s": float(alpha_s), "lam_N": float(lam_N),
            "lam_ref": float(lam_ref), "kappa_pctile": kappa_pctile,
            "max_gamma_raw": max_gamma_raw, "n_corrected": int(correct_mask.sum()),
            "n_capped": n_capped, "cap_frac": float(cap_frac),
            "trustworthy": bool(trustworthy)}

    if verbose and not trustworthy:
        print("*** WARNING: RIE calibration looks untrustworthy ***")
        print(f"    smallest eigenvalue lambda_N = {lam_N:.3e}, kappa = {kappa:.3e}")
        print(f"    {n_capped} of {N} eigenvalues ({100*cap_frac:.0f}%) needed capping at "
              f"{gamma_cap} -- healthy is 0%, and >20% means the calibration has gone bad.")
        print(f"    lam_ref (p{kappa_pctile}) = {lam_ref:.4f}. Try raising kappa_pctile,")
        print("    and inspect the low end of the spectrum (09a_universe_hygiene.py).")

    return xi_final, info


def rie_iws_clean_correlation(R, gamma_cap=GAMMA_CAP, kappa_pctile=KAPPA_PCTILE, verbose=True):
    """
    Clean a sample correlation matrix via RIE (IWs).

    Parameters
    ----------
    R : pandas.DataFrame or ndarray, shape (T, N)   Returns matrix.
    gamma_cap : float or None                       See rie_iws.
    verbose : bool                                  See rie_iws.

    Returns
    -------
    Xi_rie : ndarray, shape (N, N)   Cleaned correlation matrix.
    info : dict                      Diagnostics from rie_iws.
    """
    A = np.asarray(R, dtype=float)
    T, N = A.shape
    q = N / T

    # Sample correlation matrix (z-score columns; ddof=0 for the 1/T
    # convention used throughout this project's RMT-facing code).
    Z = (A - A.mean(axis=0)) / A.std(axis=0, ddof=0)
    E = (Z.T @ Z) / T

    # eigh: symmetric solver, eigenvalues ASCENDING, eigenvectors as columns.
    eigvals, eigvecs = np.linalg.eigh(E)

    xi, info = rie_iws(eigvals, q, gamma_cap=gamma_cap,
                       kappa_pctile=kappa_pctile, verbose=verbose)

    # Reconstruct Xi = V diag(xi) V^T, then symmetrize away rounding noise.
    Xi_rie = eigvecs @ np.diag(xi) @ eigvecs.T
    Xi_rie = (Xi_rie + Xi_rie.T) / 2

    # The trace rescale only pins the AVERAGE diagonal to 1; individual
    # diagonal entries still drift (0.73-1.15 observed), because cleaning acts
    # in the eigenbasis, not asset-by-asset. Renormalize to a true unit-
    # diagonal correlation matrix -- matters because we will later recombine
    # this with separately-estimated vols as Sigma_ij = sigma_i sigma_j Xi_ij,
    # where diagonal drift would silently distort each asset's variance.
    d = np.sqrt(np.diag(Xi_rie))
    Xi_rie = Xi_rie / np.outer(d, d)

    info["q"] = q
    return Xi_rie, info


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    R = pd.read_parquet(Path("output") / "returns_universe.parquet")

    Xi_rie, rie_info = rie_iws_clean_correlation(R)

    alpha_s_rie = rie_info["alpha_s"]
    kappa_hat = rie_info["kappa"]

    print(f"q = {rie_info['q']:.4f}")
    print(f"lambda_N = {rie_info['lam_N']:.6f}   (single smallest eigenvalue -- the textbook "
          f"algorithm calibrates on this; too fragile)")
    print(f"lam_ref  = {rie_info['lam_ref']:.6f}   (p{rie_info['kappa_pctile']} of the spectrum -- what we "
          f"actually calibrate on)")
    print(f"kappa = {kappa_hat:.4g},  alpha_s = {alpha_s_rie:.4f}")
    print(f"capped: {rie_info['n_capped']} of {Xi_rie.shape[0]} eigenvalues "
          f"({100*rie_info['cap_frac']:.0f}%)  -- healthy 0%, broken >20%")
    print(f"calibration trustworthy: {rie_info['trustworthy']}")

    # Sanity checks
    assert np.allclose(Xi_rie, Xi_rie.T), "Xi_rie must be symmetric"
    assert np.allclose(np.diag(Xi_rie), 1.0), "diagonal must be exactly 1"
    ev = np.linalg.eigvalsh(Xi_rie)
    assert ev.min() > 0, "Xi_rie must be positive definite"
    print(f"\nCleaned eigenvalues: min {ev.min():.4g}, max {ev.max():.2f}  (positive definite)")
    print(f"Condition number: {ev.max()/ev.min():.3g}")
    if ev.min() < 1e-3:
        print("  *** A cleaned eigenvalue is ~0: some assets are near-exactly redundant.")
        print("      That is REAL structure (the oracle agrees), but it makes the matrix")
        print("      dangerous to invert. Handle it at the portfolio step, not here. ***")

    # The check that catches the degenerate-to-identity failure: the market mode
    # must survive. Only meaningful when there IS a market mode -- on data with no
    # dominant factor (e.g. pure noise) every eigenvalue is legitimately pulled
    # toward 1, so the ratio is meaningless and the test is skipped.
    _A = np.asarray(R, float)
    _Z = (_A - _A.mean(0)) / _A.std(0, ddof=0)
    lam_raw = np.linalg.eigvalsh((_Z.T @ _Z) / _A.shape[0])
    ratio = ev.max() / lam_raw.max()
    print(f"Market mode: raw {lam_raw.max():.2f} -> cleaned {ev.max():.2f}  (ratio {ratio:.3f})")
    if lam_raw.max() < 5:
        print("  (no dominant market mode in this data -- ratio check skipped)")
    else:
        assert ratio > MARKET_MODE_MIN_RATIO, (
            f"Market mode shrunk to {ratio:.3f} of raw. On healthy data the oracle ratio is "
            f"0.96-1.02, so below {MARKET_MODE_MIN_RATIO} means the small eigenvalues were "
            "over-inflated and squeezed it out of the fixed trace budget. Raise kappa_pctile "
            "and inspect the low end of the spectrum (09a_universe_hygiene.py).")
        print("  Market mode preserved (oracle ratio on healthy data is 0.96-1.02).")
