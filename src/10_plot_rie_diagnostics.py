# =============================================================================
# Diagnostics for the RIE(IWs)-cleaned correlation matrix  [REVISED]
# =============================================================================
# Three panels:
#   (B) heatmap of the cleaned correlation matrix
#   (C) cleaned bulk eigenvalue histogram, with the raw bulk as a step outline
#   (D) the shrinkage curve lambda_i -> xi_i -- the single most informative
#       diagnostic. I added it because it is the plot that makes a
#       degenerate-to-identity failure obvious in one glance: if the market
#       mode has been crushed, the top-right point falls far below the
#       diagonal and you see it immediately. The LW line is overlaid for
#       comparison -- LW is a straight line in linear space by construction,
#       RIE is genuinely curved, which is the whole point of the method.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rie_diagnostics(Corr_raw, Xi_rie, info, alpha_s_lw=None,
                         n_top_eigs_excluded: int = 1):
    """
    Three-panel diagnostic view of an RIE(IWs)-cleaned correlation matrix.

    Parameters
    ----------
    Corr_raw : pd.DataFrame or ndarray, shape (N, N)
        Raw sample correlation matrix (comparison baseline).
    Xi_rie : ndarray, shape (N, N)
        RIE(IWs)-cleaned correlation matrix from rie_iws_clean_correlation.
    info : dict
        The diagnostics dict returned alongside Xi_rie.
    alpha_s_lw : float, optional
        LW shrinkage intensity, to overlay LW's linear curve in panel D.
    n_top_eigs_excluded : int, default 1
        Largest eigenvalue(s) held out of the panel-C histogram.
    """
    Corr_raw = np.asarray(Corr_raw)
    Xi_rie = np.asarray(Xi_rie)
    n = Xi_rie.shape[0]

    eigvals_raw = np.linalg.eigvalsh(Corr_raw)[::-1]
    eigvals_clean = np.linalg.eigvalsh(Xi_rie)[::-1]

    top_clean = eigvals_clean[:n_top_eigs_excluded]
    bulk_clean = eigvals_clean[n_top_eigs_excluded:]
    bulk_raw = eigvals_raw[n_top_eigs_excluded:]

    kappa = info["kappa"]
    kappa_str = f"{kappa:.3g}" if kappa < 1e4 else r"$\to\infty$"

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # --- (B) heatmap ---
    ax = axes[0]
    im = ax.imshow(Xi_rie, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"RIE(IWs)-cleaned correlation ({n} x {n})\n"
                 + r"$\kappa$" + f" = {kappa_str}, " + r"$\alpha_s$"
                 + f" = {info['alpha_s']:.3f}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Correlation")

    # --- (C) eigenvalue histogram ---
    ax = axes[1]
    bins = np.linspace(min(bulk_raw.min(), bulk_clean.min()),
                       max(bulk_raw.max(), bulk_clean.max()), 41)
    ax.hist(bulk_raw, bins=bins, histtype="step", linewidth=1.5,
            color="#7f7f7f", label="Raw (bulk)")
    ax.hist(bulk_clean, bins=bins, color="#2E8B57", alpha=0.85,
            label="RIE-cleaned (bulk)")
    ax.set_title(f"Eigenvalue spectrum\n(top {n_top_eigs_excluded} excluded — see box)")
    ax.set_xlabel("Eigenvalue"); ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.02, 0.95,
            "Excluded top cleaned eig.:\n" + ", ".join(f"{e:.2f}" for e in top_clean)
            + f"\n(raw top: {eigvals_raw[0]:.1f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    # --- (D) shrinkage curve ---
    ax = axes[2]
    lo = min(eigvals_raw.min(), eigvals_clean.min()) * 0.7
    hi = max(eigvals_raw.max(), eigvals_clean.max()) * 1.4
    diag = np.array([lo, hi])
    ax.plot(diag, diag, color="#999999", lw=1.2, ls="--", label="no cleaning")
    if alpha_s_lw is not None:
        # LW: xi = 1 + alpha_s * (lambda - 1), a straight line in LINEAR space
        # (it looks curved here only because the axes are logarithmic).
        ax.plot(eigvals_raw, 1 + alpha_s_lw * (eigvals_raw - 1),
                color="#2E5EAA", lw=1.6, label=f"LW linear ($\\alpha_s$={alpha_s_lw:.3f})")
    ax.plot(eigvals_raw, eigvals_clean, "o", ms=3.5, color="#2E8B57",
            alpha=0.75, label="RIE (IWs)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sample eigenvalue  $\\lambda_i$")
    ax.set_ylabel("Cleaned eigenvalue  $\\xi_i$")
    ax.set_title("Shrinkage curve\n(below diagonal = shrunk, above = lifted)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.show()

    # --- printed diagnostics ---
    ratio = eigvals_clean[0] / eigvals_raw[0]
    print(f"kappa = {kappa_str},  alpha_s = {info['alpha_s']:.4f},  q = {info.get('q', float('nan')):.4f}")
    print(f"max Gamma requested = {info['max_gamma_raw']:.3g}  "
          f"(theory ~1.3; healthy universes ~1.5),  hit cap: {info['n_capped']}")
    print(f"calibration trustworthy: {info['trustworthy']}")
    print(f"Largest  eigenvalue: raw {eigvals_raw[0]:8.2f} -> cleaned {eigvals_clean[0]:8.2f}  "
          f"(ratio {ratio:.3f})")
    print(f"Smallest eigenvalue: raw {eigvals_raw[-1]:8.4f} -> cleaned {eigvals_clean[-1]:8.4f}")
    print(f"Trace: {eigvals_clean.sum():.2f} (= N = {n} by construction, unit diagonal)")

    tri = np.triu_indices(n, 1)
    print(f"Mean off-diagonal correlation: raw {Corr_raw[tri].mean():.4f} -> "
          f"cleaned {Xi_rie[tri].mean():.4f}")
    if ratio < 0.5:
        print("\n*** The market mode has collapsed -- the cleaned matrix has degenerated")
        print("    toward the identity. Check for near-duplicate stocks (09a_universe_hygiene.py). ***")


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from importlib import import_module

    rie_mod = import_module("09_rie_iws".replace("-", "_"))  # placeholder if renamed
    lw_mod = import_module("07_lw_shrinkage".replace("-", "_"))

    R = pd.read_parquet(Path("output") / "returns_universe.parquet")
    A = np.asarray(R, float)
    Z = (A - A.mean(0)) / A.std(0, ddof=0)
    Corr_raw = (Z.T @ Z) / A.shape[0]

    Xi_rie, rie_info = rie_mod.rie_iws_clean_correlation(R)
    _, alpha_s = lw_mod.lw_shrink_correlation(R)

    plot_rie_diagnostics(Corr_raw, Xi_rie, rie_info, alpha_s_lw=alpha_s)
