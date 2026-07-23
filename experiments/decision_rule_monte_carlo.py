#!/usr/bin/env python3
"""Decision Rule 1: regret comparison E[l(argmax)] - E[l(soup)] (exact).

Uses the identity

  E[l(argmax)] - E[l(soup)] = kappa - g(delta),

where g(delta) = E[p_selected] - pbar depends ONLY on delta.  The whole
2-D (delta, kappa) surface follows from the 1-D function g, eliminating the
sampling noise that a direct 2-D Monte Carlo would scatter into the contours.

g(delta) is computed EXACTLY via Gauss-Legendre quadrature of the
tie-breaking integral (uniform random tie-breaking).  No Monte Carlo noise.

Model: m candidates with true accuracies evenly spaced on
       [pbar - delta/2, pbar + delta/2], each scored on N i.i.d. examples.

Also computes:
  - Worst-case epsilon (max over bang-bang configurations, j optimised)
  - Empirical epsilon for Qwen2.5-7B seed 42 actual checkpoint distribution

Usage:
    python3 decision_rule_monte_carlo.py [outdir]
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

# =============================================================================
# Constants
# =============================================================================
M = 10                               # number of candidates
N = 200                              # eval set size
PBAR = 0.50                          # mean accuracy (synthetic)
Z = 1.96                             # z_{0.025}
SE1 = np.sqrt(PBAR * (1 - PBAR) / N) * 100.0   # 3.5355 pp
DELTA_MIN = Z * np.sqrt(2 * PBAR * (1 - PBAR) / N) * 100   # 9.80 pp, eq.(6)
SUFF_KAPPA = (M - 1) / M * DELTA_MIN                        # 8.82 pp
D_MAX = 24.0                                                 # extended range
K_LO, K_HI = -6.0, 10.0
N_GL = 8                             # GL nodes.  The integrand is a degree-
                                     # (m-1) polynomial in t, and GL(n) is
                                     # exact to degree 2n-1, so n=5 already
                                     # gives machine-exact values; 8 is margin.

# Qwen2.5-7B seed 42 empirical (§8.1 B1 protocol: 49.5%–62.0%, mean 56.0%)
QWEN_DELTA = 12.5                               # pp (62.0 − 49.5)
QWEN_KAPPA = -4.5                               # pp (soup 51.5% − mean 56.0%)

print(f"SE1 = {SE1:.4f} pp")
print(f"Delta_min = {DELTA_MIN:.2f} pp")
print(f"Sufficient condition: kappa >= {SUFF_KAPPA:.2f} pp")
print(f"Grid up to delta = {D_MAX} pp, {D_MAX / SE1:.1f} x SE1")
print(f"Qwen: delta={QWEN_DELTA:.1f} pp, kappa={QWEN_KAPPA:.1f} pp")

# =============================================================================
# Core: selection gain for an arbitrary accuracy vector
# =============================================================================
def selection_gain(p, n=N, n_gl=N_GL):
    """E[p_selected] - pbar (pp) for candidate accuracies p (shape (m,)).

    Uses the integral identity 1/(1+B) = int_0^1 t^B dt to convert the
    tie-breaking combinatorics into a smooth 1-D integral done by
    Gauss-Legendre quadrature.
    """
    m = len(p)
    pbar = p.mean()
    if np.ptp(p) == 0.0:
        return 0.0
    k = np.arange(n + 1)

    pmf = binom.pmf(k[None, :], n, p[:, None])               # (m, n+1)
    cdf = binom.cdf(k[None, :], n, p[:, None])               # (m, n+1)
    cdf_lt = np.clip(cdf - pmf, 0.0, 1.0)                     # P(X < k)

    t, w = np.polynomial.legendre.leggauss(n_gl)
    t = 0.5 * (t + 1.0)
    w = 0.5 * w

    A = cdf_lt[:, :, None] + t[None, None, :] * pmf[:, :, None]

    prob_sel = np.empty(m)
    for i in range(m):
        others = np.delete(A, i, axis=0)
        # A = P(X<k) + t*P(X=k) <= P(X<=k) <= 1, so the product cannot
        # overflow and NaN cannot arise; underflow to 0 costs < 1e-26 of
        # total weight.  A plain product is exact here and skips a log/exp
        # round trip.
        prod_others = others.prod(axis=0)
        inner = (prod_others * w[None, :]).sum(axis=1)
        prob_sel[i] = (pmf[i] * inner).sum()

    assert abs(prob_sel.sum() - 1.0) < 1e-8
    return (prob_sel @ p - pbar) * 100.0


def g_uniform(delta_pp, m=M, n=N, pbar=PBAR, n_gl=N_GL):
    """E[p_selected] - pbar (pp) for uniformly spaced candidates."""
    if delta_pp <= 0.0:
        return 0.0
    d = delta_pp / 100.0
    p = np.linspace(pbar - d / 2, pbar + d / 2, m)
    return selection_gain(p, n, n_gl)


def g_worst(delta_pp, m=M, n=N, pbar=PBAR, n_gl=N_GL):
    """Maximum over bang-bang configurations of range delta of E[p_selected]-pbar (pp).

    Restrict to bang-bang configurations: j candidates at the top, (m-j) at the
    bottom, mean pinned to pbar.  j=1 maximises the LEVEL p_max - pbar =
    (m-1)/m*delta and is therefore optimal as N->inf (main.tex L547-550).
    At finite N it is NOT the maximiser: raising j increases the probability
    that argmax lands on a top candidate, and while delta is small relative
    to SE_1 that gain outweighs the lower level.  We maximise over j.
    Nelder-Mead checks on 4 delta values confirm that the bang-bang family
    contains the global optimum, but this is not a theorem.
    """
    if delta_pp <= 0.0:
        return 0.0
    d = delta_pp / 100.0
    best = 0.0
    for j in range(1, m):
        u = np.zeros(m)
        u[:j] = 1.0
        best = max(best, selection_gain(pbar - d * u.mean() + d * u, n, n_gl))
    return best


def g_worst_argj(delta_pp, m=M, n=N, pbar=PBAR, n_gl=N_GL):
    """Number of top candidates j attaining the bang-bang maximum (for reporting)."""
    if delta_pp <= 0.0:
        return 0
    d = delta_pp / 100.0
    vals = []
    for j in range(1, m):
        u = np.zeros(m)
        u[:j] = 1.0
        vals.append(selection_gain(pbar - d * u.mean() + d * u, n, n_gl))
    return int(np.argmax(vals)) + 1


# =============================================================================
# Validation against MC
# =============================================================================
def g_mc(delta_pp, n_trials=400_000, seed=0, m=M, n=N, pbar=PBAR):
    """Monte-Carlo cross-check of g_uniform."""
    d = delta_pp / 100.0
    p = np.linspace(pbar - d / 2, pbar + d / 2, m)
    rng = np.random.default_rng(seed)
    obs = rng.binomial(n, p, size=(n_trials, m)).astype(float)
    obs += rng.random(obs.shape) * 1e-6
    idx = obs.argmax(axis=1)
    return (p[idx].mean() - pbar) * 100.0


print("\nValidating g_uniform against MC (400k trials each point):")
print(f"{'delta(pp)':>10} {'exact':>9} {'MC':>9} {'diff':>8}")
for dv in [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]:
    ge, gm = g_uniform(dv), g_mc(dv)
    print(f"{dv:>10.1f} {ge:>9.4f} {gm:>9.4f} {ge - gm:>8.4f}")

# ---- Qwen empirical ----
g_unif_at_qwen = g_uniform(QWEN_DELTA)
print(f"\nQwen empirical (B1 protocol, §8.1):")
print(f"  delta      = {QWEN_DELTA:.1f} pp")
print(f"  kappa      = {QWEN_KAPPA:.1f} pp")
print(f"  eps_unif   = {g_unif_at_qwen:.4f} pp  (uniform spacing at δ={QWEN_DELTA:.1f} pp)")
print(f"  diff_unif  = {QWEN_KAPPA - g_unif_at_qwen:.4f} pp")

# =============================================================================
# Construct the exact (delta, kappa) surface
# =============================================================================
deltas = np.linspace(0.0, D_MAX, 401)
print(f"\nComputing g_uniform over [{deltas[0]:.1f}, {deltas[-1]:.1f}]...")
g_unif = np.array([g_uniform(dv) for dv in deltas])
print("Computing g_worst...")
g_wc = np.array([g_worst(dv) for dv in deltas])

kappas = np.linspace(K_LO, K_HI, 401)
DD, KK = np.meshgrid(deltas, kappas)
diff = KK - g_unif[None, :]          # exact, sampling-noise-free

# Crossing of constant sufficient-condition line with break-even
if g_unif[-1] < SUFF_KAPPA:
    raise RuntimeError(
        f"Cannot find crossing: g_unif(D_MAX={D_MAX:.1f}) = {g_unif[-1]:.3f} "
        f"< SUFF_KAPPA = {SUFF_KAPPA:.3f}. Increase D_MAX."
    )
cross_delta = np.interp(SUFF_KAPPA, g_unif, deltas)

# Key values for annotation
g_at_dmin = np.interp(DELTA_MIN, deltas, g_unif)
wc_at_dmin = np.interp(DELTA_MIN, deltas, g_wc)

config_gap = wc_at_dmin - g_at_dmin                # 4.19 pp
finite_gap = SUFF_KAPPA - wc_at_dmin                # 1.30 pp
total_gap = SUFF_KAPPA - g_at_dmin                  # 5.49 pp

print(f"\nAt Delta_min = {DELTA_MIN:.2f} pp:")
print(f"  g_uniform   = {g_at_dmin:.3f} pp")
print(f"  g_worst     = {wc_at_dmin:.3f} pp")
print(f"  theorem bnd = {SUFF_KAPPA:.3f} pp")
print(f"  gap: {total_gap:.2f} = {config_gap:.2f} (config.) "
      f"+ {finite_gap:.2f} (finite-sample)")
print(f"Crossing delta = {cross_delta:.2f} pp")
print("\nExtremal configuration attaining the bang-bang maximum (j = #candidates at top):")
for dv in [1.2, 2.45, 4.90, 7.35, DELTA_MIN, 15.0, 24.0]:
    j = g_worst_argj(dv)
    one = selection_gain(np.array([PBAR + (M - 1) * dv / 100 / M]
                                  + [PBAR - dv / 100 / M] * (M - 1)))
    print(f"  delta={dv:>6.2f} pp -> j={j}, bang-bang max={g_worst(dv):>7.4f} pp"
          f"   (j=1 alone gives {one:>7.4f} pp)")

# =============================================================================
# Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(11.0, 7.6))

# Background colour contours
levels = np.arange(-3.0, 3.001, 0.5)
cf = ax.contourf(DD, KK, diff, levels=levels, cmap="RdBu_r", extend="both")
cb = fig.colorbar(cf, ax=ax, pad=0.02, ticks=levels[::2])
cb.set_label(r"$\mathbb{E}[\ell(\mathrm{argmax})] - \mathbb{E}[\ell(\mathrm{soup})]$  (pp)")

# P_Delta region shading
ax.axvspan(0, DELTA_MIN, alpha=0.06, color="gray", zorder=1)

# Break-even curve: uniform spacing
ax.plot(deltas, g_unif, color="k", lw=2.6, zorder=5,
        label=r"Break-even (uniform spacing)  $\kappa = g_{\mathrm{even}}(\delta)$")

# Worst-case epsilon curve
ax.plot(deltas, g_wc, color="tab:orange", ls="--", lw=2.0, zorder=4,
        label=r"Bang-bang envelope  $\max_{j}\,\varepsilon$ ($j$ optimised)")

# Sufficient condition: horizontal line
ax.axhline(SUFF_KAPPA, color="tab:green", ls="--", lw=2.4, zorder=5,
           label=rf"Sufficient condition $\kappa \geq \frac{{{M-1}}}{{{M}}}\Delta_{{\min}}$"
                 rf" $= {SUFF_KAPPA:.2f}$ pp")

# P_Delta boundary
ax.axvline(DELTA_MIN, color="0.35", ls=":", lw=1.5, zorder=4,
           label=rf"$\mathcal{{P}}_{{\Delta}}$ boundary ($\delta = \Delta_{{\min}}$"
                 rf" $= {DELTA_MIN:.1f}$ pp)")

# Crossing of sufficient-condition line with break-even
ax.plot([cross_delta, cross_delta], [K_LO, SUFF_KAPPA],
        color="0.35", ls=":", lw=1.0, zorder=3)
ax.plot(cross_delta, SUFF_KAPPA, marker="o", ms=7, mfc="none", mec="0.2",
        mew=1.6, zorder=6)
ax.annotate(rf"$\delta \approx {cross_delta:.1f}$ pp",
            xy=(cross_delta, SUFF_KAPPA),
            xytext=(cross_delta - 9.5, SUFF_KAPPA + 0.7),
            fontsize=9.5, color="0.2",
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.1))

# Empirical point (Qwen2.5-7B seed 42, §8.1 B1 protocol)
ax.plot(QWEN_DELTA, QWEN_KAPPA, marker="X", ms=17, mfc="red", mec="k",
        mew=1.4, ls="none", zorder=7)
ax.annotate(
    rf"$\delta \approx {QWEN_DELTA:.1f}$ pp,  $\kappa = {QWEN_KAPPA:.1f}$ pp"
    rf"  (Qwen2.5-7B seed 42)",
    xy=(QWEN_DELTA, QWEN_KAPPA),
    xytext=(QWEN_DELTA + 0.7, QWEN_KAPPA - 1.15), ha="left",
    fontsize=9.5, color="0.15",
    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))

# Region labels
ax.text(0.035, 0.955, "Soup wins", transform=ax.transAxes, fontsize=13,
        color="0.12", va="top")
ax.text(0.80, 0.46, "Argmax wins", transform=ax.transAxes, fontsize=13,
        color="0.92", va="bottom")

# Conservatism decomposition annotation
ax.annotate(
    rf"$\mathcal{{P}}_\Delta$ ($\delta \leq \Delta_{{\min}}$):"
    rf"  bound $\varepsilon \leq {SUFF_KAPPA:.2f}$,"
    rf" worst ${wc_at_dmin:.2f}$, uniform ${g_at_dmin:.2f}$ pp"
    "\n"
    rf"gap ${total_gap:.2f} = {config_gap:.2f}$ (config.)"
    rf" $+\ {finite_gap:.2f}$ (finite-sample)",
    xy=(DELTA_MIN, (g_at_dmin + wc_at_dmin) / 2),
    xytext=(0.60, -4.4), ha="left", va="center",
    fontsize=8.5, color="0.2",
    bbox=dict(boxstyle="round,pad=0.35", fc="wheat", alpha=0.85),
    arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))

# Axes
ax.set_xlabel(r"True accuracy range  $\delta$  (pp)")
ax.set_ylabel(r"Soup gain  $\kappa = p_{\mathrm{soup}} - \bar p_{\mathrm{cand}}$  (pp)")
ax.set_title(rf"Decision Rule 1: regret comparison "
             rf"($m={M}$, $N={N}$, $\bar p={PBAR}$)")
ax.set_xlim(0, D_MAX)
ax.set_ylim(K_LO, K_HI)
ax.grid(alpha=0.25, lw=0.6)

# Normalised top axis (makes the read-off transferable across N)
secax = ax.secondary_xaxis("top",
                           functions=(lambda x: x / SE1, lambda z: z * SE1))
secax.set_xlabel(rf"$\delta\ /\ \mathrm{{SE}}_1$"
                 rf"   (SE$_1={SE1:.2f}$ pp)")
secax.set_xticks(np.arange(0, int(D_MAX / SE1) + 1, 1))

# Legend below plot (prevents occlusion)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=2, frameon=False, fontsize=10)

fig.tight_layout()

outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
os.makedirs(outdir, exist_ok=True)
fig.savefig(f"{outdir}/fig_decision_rule_contour.pdf", dpi=200,
            bbox_inches="tight")
fig.savefig(f"{outdir}/fig_decision_rule_contour.png", dpi=200,
            bbox_inches="tight")
print(f"\nSaved fig_decision_rule_contour.pdf / .png  -->  {outdir}/")
