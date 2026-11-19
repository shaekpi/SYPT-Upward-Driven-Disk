"""reproduce.py -- regenerate the numerical figures of Maas (2025) with the stated inputs.

    python reproduce.py        # ~3 min; writes figures/fig09 ... fig16 and regimes.md

Where the paper states an input (drive speed, equation, parameter) it is used.  Where it does not
(initial state of a driven run, time window, transient), the choice is made here and printed on the
figure.  Chaotic trajectories cannot be reproduced curve-for-curve by anyone, Maas included; the
reproduction targets the regimes, the orbit shapes and the stability results.
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maas_model import (P0, angular_momentum, charpoly_roots, dle_eigenvalues, k_of, lyapunov_full,
                        run_dle, run_full)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.0, "legend.frameon": False})

DRIVES = [0.014, 0.039, 0.055, 0.071, 0.088]          # cm/s values used in the paper (Figs 5-14, Video 3)
# Candidate starts.  Maas: driven runs "from rest as an inverted pendulum, Y(0) ~ 0, Z(0) = 4 cm";
# an exact Y(0) = 0 never flips, so a stated small offset is added.  A hanging start is the fallback.
STARTS = [([0.005, 0.04, 0.0], "inverted rest, Y0 = 5 mm, Z0 = +4 cm"),
          ([0.001, 0.04, 0.0], "inverted rest, Y0 = 1 mm, Z0 = +4 cm"),
          ([0.01, 0.04, 0.0], "inverted rest, Y0 = 10 mm, Z0 = +4 cm"),
          ([0.02, -0.04, 0.0], "hanging, Y0 = 2 cm, Z0 = -4 cm")]


def first_surviving(W, t_end):
    """Run the candidate starts in order; return the first run that keeps the disc for t_end."""
    for s0, label in STARTS:
        t, Y, Z, Om, J, lost = run_full(s0, W, t_end, dt=2e-3)
        if not lost:
            return t, Y, Z, Om, J, label
    return t, Y, Z, Om, J, label + " (disc lost at %.1f s)" % t[-1]


def fig09():
    th = np.radians(127.0)                       # theta from the UPWARD vertical (Sec. II B)
    s0 = [0.058 * np.sin(th), 0.058 * np.cos(th), 0.0]
    t, Y, Z, Om, J, lost = run_full(s0, 0.0, 8.0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(t, Y, color=C2); ax[0].set_xlabel("T (s)"); ax[0].set_ylabel("Y (m)"); ax[0].set_title("(a) model, W = 0", loc="left")
    ax[1].plot(t, Z, color=C2); ax[1].set_xlabel("T (s)"); ax[1].set_ylabel("Z (m)"); ax[1].set_title("(b)", loc="left")
    om = np.linspace(-15, 15, 601)
    ax[2].plot(om, k_of(om), color=C1); ax[2].set_xlabel("Ω (rad/s)"); ax[2].set_ylabel("k (m²/s)"); ax[2].set_title("(c) friction law (18)", loc="left")
    fig.suptitle("Fig. 9 equivalent: zero drive from R(0) = 0.058 m, θ(0) = 127° (from the upward vertical), Ω(0) = 0; "
                 "the paper's measured trace (blue) is not available as data", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig09_zero_drive.png")); plt.close(fig)
    # swings before rest (amplitude threshold 1 degree of swing ~ 1 mm at l = 5.8 cm)
    big = np.abs(Y) > 1e-3
    print("[fig09] last time |Y| > 1 mm: %.2f s" % t[big][-1])


def fig10():
    t, Y, Z, Om, J, lost = run_full([0.005, 0.04, 0.0], 0.014, 100.0, dt=2e-3)
    td, om, y, z = run_dle([0.0, -0.175, 1.0], 0.1, 150.0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    ax[0].plot(Y, Z, color=C1, lw=0.7); ax[0].plot(0, 0, "ko", ms=3); ax[0].set_aspect("equal")
    ax[0].add_patch(plt.Circle((0, 0), P0.a, fill=False, ls="--", color=MUTED, lw=0.7))
    ax[0].set_xlabel("Y (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title("(a) full model (8),(11),(18), W = 1.4 cm/s, 0–100 s\nstart: inverted rest, Y0 = 5 mm, Z0 = +4 cm" + (" — lost at %.0f s" % t[-1] if lost else ""), loc="left", fontsize=8.5)
    ax[1].plot(y, z, color=C1, lw=0.7); ax[1].plot(0, 0, "ko", ms=3); ax[1].set_aspect("equal")
    ax[1].set_xlabel("y"); ax[1].set_ylabel("z")
    ax[1].set_title("(b) DLE (14), w = 0.1, t = 0–150\nfrom (y, z) = (−0.175, 1); ω(0) = 0 (not stated in the paper)", loc="left", fontsize=8.5)
    fig.suptitle("Fig. 10 equivalent: small drive", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig10_small_drive.png")); plt.close(fig)
    print("[fig10] full model run lost:", lost)


def fig11_14(runs):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for k, (W, (t, Y, Z, Om, J, label)) in enumerate(runs.items()):
        sel = (t >= 40) & (t <= 100)
        ax[k].plot(Y[sel], Z[sel], color=C1, lw=0.6); ax[k].plot(0, 0, "ko", ms=3); ax[k].set_aspect("equal")
        ax[k].add_patch(plt.Circle((0, 0), P0.a, fill=False, ls="--", color=MUTED, lw=0.7))
        ax[k].set_xlim(-0.085, 0.085); ax[k].set_ylim(-0.085, 0.085)
        ax[k].set_xlabel("Y (m)"); ax[k].set_ylabel("Z (m)")
        ax[k].set_title("(%s) W = %.1f cm/s, t = 40–100 s\n%s" % ("abc"[k], W * 100, label), loc="left", fontsize=8.5)
    fig.suptitle("Fig. 11 equivalent: centre orbits, full model", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig11_full_orbits.png")); plt.close(fig)

    fig = plt.figure(figsize=(13, 4.6))
    for k, (W, (t, Y, Z, Om, J, label)) in enumerate(runs.items()):
        axk = fig.add_subplot(1, 3, k + 1, projection="3d")
        sel = (t >= 40) & (t <= 100)
        axk.plot(J[sel], Y[sel], Z[sel], color=C1, lw=0.5)
        axk.set_xlabel("𝒥 (m²/s)"); axk.set_ylabel("Y (m)"); axk.set_zlabel("Z (m)")
        axk.set_title("(%s) W = %.1f cm/s" % ("abc"[k], W * 100), loc="left", fontsize=9)
    fig.suptitle("Fig. 14 equivalent: 𝒥–Y–Z phase-space orbits, full model, t = 40–100 s", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig14_phase_space.png")); plt.close(fig)


def fig12():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for k, w in enumerate([3.9, 4.5, 7.1]):
        t, om, y, z = run_dle([0.0, 1.0, -2.0], w, 400.0)
        sel = t >= 200
        ax[k].plot(y[sel], z[sel], color=C1, lw=0.5); ax[k].plot(0, 0, "ko", ms=3); ax[k].set_aspect("equal")
        ax[k].set_xlabel("y"); ax[k].set_ylabel("z")
        ax[k].set_title("(%s) w = %.1f  (W ≈ %.1f cm/s), t = 200–400\nstart (ω, y, z) = (0, 1, −2) (not stated)" % ("abc"[k], w, P0.W_of_w(w) * 100), loc="left", fontsize=8.5)
    fig.suptitle("Fig. 12 equivalent: DLE (14) orbits", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig12_dle_orbits.png")); plt.close(fig)


def fig15_16():
    ws = np.linspace(0.05, 10, 400)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for w in ws:
        r = np.sort_complex(charpoly_roots(w))
        ax[0].plot([w] * 3, r.real, ".", color=C1, ms=1.5); ax[0].plot([w] * 3, r.imag, ".", color=C2, ms=1.5)
        e = np.sort_complex(dle_eigenvalues(w, 0.2))
        ax[1].plot([w] * 3, e.real, ".", color=C1, ms=1.5); ax[1].plot([w] * 3, e.imag, ".", color=C2, ms=1.5)
    for a_ in ax[:2]:
        a_.axhline(0, color=MUTED, lw=0.6); a_.set_xlabel("w")
    ax[0].set_ylabel("eigenvalue"); ax[0].set_title("Fig. 15 equivalent: roots of (20), DLE; blue = real, orange = imaginary", loc="left", fontsize=8.5)
    ax[1].set_title("Fig. 16a equivalent: with the −Bwz term (21), B = 0.2", loc="left", fontsize=8.5)
    w, B = 3.7, 0.2
    r = np.sqrt(w)
    for z0, col, lab in [(-3.2, INK, "z(0) = −3.2: winds inward (stable)"), (-3.5, C2, "z(0) = −3.5: chaotic")]:
        t, om, y, z = run_dle([r, r, z0], w, 300.0, B=B)
        ax[2].plot(y, z, color=col, lw=0.6, label=lab)
    ax[2].plot(r, 0, "o", color=C3, ms=5, label="rotating point (√w, 0)")
    ax[2].set_xlabel("y"); ax[2].set_ylabel("z"); ax[2].legend(fontsize=7.5, loc="lower left")
    ax[2].set_title("Fig. 16b equivalent: (21), w = 3.7, B = 0.2, ω(0) = y(0) = √w", loc="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig15_16_stability.png")); plt.close(fig)
    wc = ws[np.argmax([dle_eigenvalues(w, 0.2).real.max() < 0 for w in ws])]
    print("[fig15/16] (20) unstable for every w; with B = 0.2 the rotating point becomes stable above w ≈ %.2f" % wc)


def _lyap(args):
    W, s0 = args
    return W, s0, lyapunov_full(s0, W, t_skip=30, t_total=120)


def poincare_spread(W):
    """Poincare section Z at crossings of Y = 0, from the longest-surviving start.
    Returns (number of distinct Z values at 1 mm resolution, std of Z [m], number of crossings,
    time the disc was lost or None)."""
    best = None
    for s0, label in STARTS:
        t, Y, Z, Om, J, lost = run_full(s0, W, 150.0, dt=2e-3)
        if best is None or t[-1] > best[0][-1]:
            best = (t, Y, Z, lost)
        if not lost:
            break
    t, Y, Z, lost = best
    idx = np.where(np.sign(Y[1:]) != np.sign(Y[:-1]))[0]
    idx = idx[t[idx] > min(60.0, 0.4 * t[-1])]
    if len(idx) < 3:
        return 0, float("nan"), len(idx), (t[-1] if lost else None)
    distinct = len(np.unique(np.round(Z[idx] * 1e3)))
    return distinct, float(np.std(Z[idx])), len(idx), (t[-1] if lost else None)


def regimes():
    jobs = [(W, s0) for W in DRIVES for s0, _ in STARTS[:3]]
    with Pool(2) as pool:
        res = pool.map(_lyap, jobs)
        spreads = pool.map(poincare_spread, DRIVES)
    rows = []
    for W, (ndist, sp, ncross, tlost) in zip(DRIVES, spreads):
        lams = [lam for (Wj, s0, lam) in res if Wj == W]
        ok = [l for l in lams if not np.isnan(l)]
        rows.append((W, len(lams) - len(ok), np.mean(ok) if ok else float("nan"), ndist, sp, ncross, tlost))
    paper = {0.014: "chaotic (Sec. III B)", 0.039: "chaotic", 0.055: "chaos suppressed", 0.071: "periodic (bent figure-eight)", 0.088: "Video 3 only"}
    with open(os.path.join(OUT, "regimes.md"), "w") as f:
        f.write("| W (cm/s) | w = τW/L | Lyapunov starts ejected (of 3) | mean λ (1/s) | Poincaré: distinct Z values (1 mm bins) / spread | model regime | paper (experiment) |\n|---|---|---|---|---|---|---|\n")
        for W, nlost, lm, ndist, sp, ncross, tlost in rows:
            if ncross < 3:
                reg = "ejected before any cycle"
            elif ndist <= 4:
                reg = "periodic"
            else:
                reg = "chaotic"
            if tlost is not None:
                reg += " (longest start ejected at %.0f s)" % tlost
            f.write("| %.1f | %.2f | %d | %s | %s | %s | %s |\n" % (W * 100, P0.w_of_W(W), nlost, "%.2f" % lm if not np.isnan(lm) else "–",
                    "%d / %.1f mm (%d crossings)" % (ndist, sp * 1e3, ncross) if ncross >= 3 else "–", reg, paper[W]))
    print(open(os.path.join(OUT, "regimes.md")).read())


if __name__ == "__main__":
    fig09()
    fig10()
    runs = {W: first_surviving(W, 100.0) for W in (0.039, 0.055, 0.071)}
    for W, r in runs.items():
        print("[fig11/14] W = %.1f cm/s: start used = %s" % (W * 100, r[5]))
    fig11_14(runs)
    fig12()
    fig15_16()
    regimes()
    print("figures written to", OUT)
