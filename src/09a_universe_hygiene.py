# =============================================================================
# Spectrum diagnostics -- LOOK, don't delete
# =============================================================================
#
# This replaces an earlier "prune the universe" cell, which was a mistake.
# That version deleted whichever stock dominated the smallest eigenvector,
# iteratively. On real S&P data it removed 30 names -- MA/V, HD/LOW, DAL/UAL,
# SO/DUK, MCO/SPGI, TMO/DHR, SLB/HAL... -- and then still failed to converge.
# Those pairs are not data errors. They are real industry duopolies
# correlated at 0.79-0.88, i.e. exactly the correlation structure this whole
# project is meant to ESTIMATE. Deleting them destroys the signal.
#
# The right conclusion was that the estimator was wrong, not the universe:
# the RIE algorithm calibrates its shape parameter kappa on the single
# smallest eigenvalue, which realistic industry structure pushes far below
# the bulk edge. That is now fixed inside rie_iws (kappa is calibrated on a
# percentile of the spectrum instead, see 09_rie_iws.py), so no stock needs
# to be removed.
#
# This cell is therefore purely diagnostic. Run it to understand the low end
# of your spectrum -- in particular whether a very small eigenvalue comes from
# one near-exactly-redundant pair (a data issue worth investigating) or from a
# broad continuum (genuine economic structure, nothing to fix).

import numpy as np
import pandas as pd


def spectrum_diagnostics(R, n_show=12, n_loadings=5):
    """
    Report the low end of the correlation spectrum and what drives it.

    Parameters
    ----------
    R : pandas.DataFrame or ndarray, shape (T, N)   Returns matrix.
    n_show : int      How many of the smallest eigenvalues to list.
    n_loadings : int  How many top loadings to show per eigenvector.
    """
    names = list(R.columns) if hasattr(R, "columns") else [str(i) for i in range(R.shape[1])]
    A = np.asarray(R, dtype=float)
    T, N = A.shape
    q = N / T

    Z = (A - A.mean(axis=0)) / A.std(axis=0, ddof=0)
    E = (Z.T @ Z) / T
    eigvals, eigvecs = np.linalg.eigh(E)          # ascending

    mp_lo = (1 - np.sqrt(q)) ** 2
    print(f"N = {N}, T = {T}, q = {q:.4f}")
    print(f"Marchenko-Pastur left edge for pure noise, (1-sqrt(q))^2 = {mp_lo:.4f}")
    print(f"Largest eigenvalue (market mode): {eigvals[-1]:.2f} "
          f"({100*eigvals[-1]/N:.1f}% of total variance)")
    print(f"Condition number: {eigvals[-1]/eigvals[0]:.3g}")

    print(f"\nSmallest {n_show} eigenvalues:")
    print("   " + "  ".join(f"{v:.5f}" for v in eigvals[:n_show]))
    # A clean *gap* between the first few and the rest means isolated
    # near-redundancies (possible data problem). A smooth continuum means
    # ordinary economic structure and there is nothing to fix.
    gaps = np.diff(eigvals[:n_show])
    big = int(np.argmax(gaps))
    print(f"   largest gap is between #{big+1} and #{big+2} "
          f"({eigvals[big]:.5f} -> {eigvals[big+1]:.5f}, ratio {eigvals[big+1]/max(eigvals[big],1e-12):.1f}x)")

    print(f"\nComposition of the 3 smallest eigenvectors "
          f"(these are the near-zero-variance portfolios):")
    for k in range(3):
        v = eigvecs[:, k]
        top = np.argsort(-np.abs(v))[:n_loadings]
        # concentration: 1/sum(w^4) ~ effective number of names carrying the mode
        eff = 1.0 / np.sum(v ** 4)
        parts = ", ".join(f"{names[i]} {v[i]:+.3f}" for i in top)
        print(f"   #{k+1}  lambda={eigvals[k]:.5f}  eff. #names={eff:.1f}")
        print(f"        {parts}")
    print("   (eff. #names ~2 => one redundant pair; large => diffuse structure)")

    iu = np.triu_indices(N, 1)
    rho = E[iu]
    srt = np.argsort(-np.abs(rho))[:10]
    print(f"\nTop 10 pairwise correlations:")
    for s in srt:
        print(f"   {names[iu[0][s]]:6s} / {names[iu[1][s]]:6s}   {rho[s]:+.4f}")

    n_ge = [(t, int((np.abs(rho) > t).sum())) for t in (0.99, 0.95, 0.90, 0.85)]
    print("   pairs above |rho|:  " + ",  ".join(f"{t}: {c}" for t, c in n_ge))
    print("\n   |rho| > 0.99 usually means a genuine data problem (same security")
    print("   twice, dual share class). 0.80-0.90 is normal industry structure --")
    print("   leave it alone, the percentile calibration handles it.")

    return eigvals, eigvecs


if __name__ == "__main__":
    from pathlib import Path
    R = pd.read_parquet(Path("output") / "returns_universe.parquet")
    eigvals_diag, eigvecs_diag = spectrum_diagnostics(R)
