# =============================================================================
# Ledoit-Wolf linear shrinkage of the correlation matrix towards identity
# (Xi^lin = alpha_s * E + (1-alpha_s) * I_N)
# =============================================================================
#
# scikit-learn already ships this estimator: sklearn.covariance.LedoitWolf.
# It implements exactly the Ledoit & Wolf (2004) "well-conditioned estimator"
# paper: given data X (T observations x N features), it returns
#       Sigma_hat = (1 - delta) * S  +  delta * mu * I_N
# where S is the plain sample covariance of X, mu = Tr(S)/N (the average
# variance -- this is what "identity target" means in practice: not I_N
# itself, but the closest multiple of I_N, mu*I_N), and delta in [0,1] is
# the data-driven shrinkage intensity estimated with the same
# pi_hat / gamma_hat / kappa_hat machinery as in the original Ledoit-Wolf
# paper.
#
# delta plays the role of (1 - alpha_s) in this project's notation:
#   alpha_s = 1 - delta  (weight on the sample estimator E)
#   delta   = weight on the identity target
#
# TRICK to make the target *exactly* I_N (not just mu*I_N with some mu != 1):
# standardize each column of the returns to unit variance first. Then the
# sample covariance of the standardized data Z is exactly the correlation
# matrix E of the original returns R (correlation = covariance of z-scored
# variables), and mu = Tr(S)/N = 1 exactly (average of N ones). So sklearn's
# mu*I_N becomes I_N, and its output is precisely Xi^lin = alpha_s*E + (1-alpha_s)*I_N.
#
# (A hand-coded transcription of the textbook beta/gamma formula hit a
# scaling bug that made alpha_s collapse to 0 in every test. Cross-checking
# against sklearn's implementation -- which is battle-tested and matches the
# primary Ledoit-Wolf (2004) source line for line -- is the right fix: same
# underlying math, no need to re-derive it by hand.)

from sklearn.covariance import LedoitWolf
import numpy as np


def lw_shrink_correlation(R):
    """
    Clean a sample correlation matrix via Ledoit-Wolf linear shrinkage
    towards the identity target.

    Parameters
    ----------
    R : pandas.DataFrame or ndarray, shape (T, N)
        Returns matrix (T dates x N stocks). Does not need to be pre-
        standardized -- this function does that internally.

    Returns
    -------
    Xi_lin : ndarray, shape (N, N)
        The shrunk correlation matrix: alpha_s * E + (1 - alpha_s) * I_N,
        where E is the sample correlation matrix of R.
    alpha_s : float
        Estimated weight on the sample correlation matrix E (in [0, 1]).
        alpha_s -> 0 means "shrink almost fully to identity" (little trust
        in the sample estimate, e.g. N/T close to 1). alpha_s -> 1 means
        "keep almost all of the sample estimate" (lots of data relative to
        N, and/or strong real correlation structure worth keeping).
    """
    R = np.asarray(R, dtype=float)

    # Standardize each column to zero mean, unit variance (ddof=0, to match
    # the population-style 1/T normalization used throughout this project).
    # After this, cov(Z) = corr(R) exactly, and Tr(cov(Z))/N = 1 exactly,
    # so sklearn's "mu * I_N" target becomes exactly I_N.
    mu_R = R.mean(axis=0)
    sd_R = R.std(axis=0, ddof=0)
    Z = (R - mu_R) / sd_R

    # Fit sklearn's Ledoit-Wolf estimator on the standardized data.
    # assume_centered=False: it will re-center Z (harmless, Z is already
    # centered to numerical precision).
    lw = LedoitWolf(assume_centered=False).fit(Z)

    Xi_lin = lw.covariance_          # = alpha_s * E + (1 - alpha_s) * I_N
    alpha_s = 1.0 - lw.shrinkage_    # sklearn's shrinkage_ = weight on the identity target

    return Xi_lin, alpha_s


if __name__ == "__main__":
    # --- Example usage -----------------------------------------------------
    # Expects `R` (a T x N returns DataFrame/ndarray) to be available, e.g.
    # loaded from output/returns_universe.parquet produced by
    # 02_universe_selection.py.
    import pandas as pd
    from pathlib import Path

    R = pd.read_parquet(Path("output") / "returns_universe.parquet")

    Xi_lin, alpha_s = lw_shrink_correlation(R)

    print(f"Estimated alpha_s (weight kept on sample correlation E): {alpha_s:.4f}")
    print(f"Shrinkage intensity towards identity (1 - alpha_s):      {1 - alpha_s:.4f}")

    # Sanity checks
    assert np.allclose(Xi_lin, Xi_lin.T), "Xi_lin should be symmetric"
    assert np.allclose(np.diag(Xi_lin), 1.0), "diagonal should stay exactly 1 (both E and I have diag 1)"
    eigvals_lw = np.linalg.eigvalsh(Xi_lin)
    print(f"Smallest eigenvalue of Xi_lin: {eigvals_lw.min():.4f} (should be > 0: shrinkage to I always yields a positive-definite matrix)")
