"""
Noise mechanism breakdown: baseline vs optimum B-spline propeller.

Calls each BPM/Amiet mechanism independently to show what fraction of the
total SPL reduction comes from: TBL-TE, LBL-VS, Amiet LETI, BVI tonal.

Usage
-----
  python results/noise_breakdown.py                # uses hardened-model 71.77 dBA optimum
  python results/noise_breakdown.py --rpm 6052 \
    --dtwist -0.75  2.981  0.397  4.999  4.776 \
    --dchord  0.022 -0.015 -0.015 -0.015 -0.015 \
    --sweep   0.0   0.0    0.1135 0.1135 0.1185 \
    --dtc    -0.0064 -0.0264 -0.0153 0.0047 0.0247
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_apc7x5e
from aerodynamics.ccblade_component import bem_solve
from acoustics.bpm_component import bpm_noise, THIRD_OCT_FREQS, A_WEIGHT
from scipy.interpolate import CubicSpline

N_STATIONS = 20
N_CP       = 5
CRUISE_VINF = 3.32   # m/s axial inflow component at cruise pitch


def _build_blade(rpm, dt_cp, dc_cp, sw_cp, dtc_cp):
    blade = baseline_apc7x5e()
    r_cp  = np.linspace(blade.r_R[0], blade.r_R[-1], N_CP)
    r_def = blade.r_R

    def spl(cp): return CubicSpline(r_cp, cp)(r_def)

    dc  = spl(dc_cp)
    dt  = spl(dt_cp)
    dtc = spl(dtc_cp)
    sw  = np.clip(spl(sw_cp), 0.0, 0.12)

    return (blade
            .perturb_twist(dt)
            .perturb_chord(dc)
            .perturb_tc(dtc)
            .set_sweep(sw)
            .set_blade_angles([0.0, 120.0, 240.0]))


def _run_case(blade, rpm, v_inf, label):
    aero = bem_solve(blade, rpm=rpm, v_inf=v_inf, rho=1.225, n_stations=N_STATIONS)
    _, chord_m, _ = blade.get_stations(N_STATIONS)
    res = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=rpm, num_blades=blade.num_blades, radius_m=blade.radius_m,
        rho=1.225, r_obs=1.0,
        x_tr_c=aero["x_tr_c"],
        blade_angles_deg=blade.blade_angles_deg,
        sweep_m=aero.get("sweep_m", None),
    )
    res["thrust"] = aero["thrust"]
    res["rpm"]    = rpm
    res["label"]  = label
    return res


def plot_breakdown(baseline_hover, opt_hover, baseline_cruise, opt_cruise, out_dir):
    BG    = "#1a1a2e"
    PANEL = "#16213e"
    GRID  = "#444466"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white")
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # --- Panel 1: Bar chart — mechanism breakdown (hover + cruise) ---
    # Only show the mechanisms that are above -100 dBA so TBL-TE (-188) is skipped
    mechs     = ["LBL-VS", "Amiet LETI", "BVI Tonal", "Total"]
    keys      = ["SPL_lbl_vs_dBA", "SPL_leti_dBA", "SPL_bvi_dBA", "SPL_total"]
    x         = np.arange(len(mechs))
    w         = 0.2

    bh = [baseline_hover[k]  for k in keys]
    oh = [opt_hover[k]       for k in keys]
    bc = [baseline_cruise[k] for k in keys]
    oc = [opt_cruise[k]      for k in keys]

    ax = axes[0]
    ax.bar(x - 1.5*w, bh, w, label="Baseline hover",  color="#4a90d9", alpha=0.90)
    ax.bar(x - 0.5*w, oh, w, label="Optimum hover",   color="#50fa7b", alpha=0.90)
    ax.bar(x + 0.5*w, bc, w, label="Baseline cruise", color="#4a90d9", alpha=0.50)
    ax.bar(x + 1.5*w, oc, w, label="Optimum cruise",  color="#50fa7b", alpha=0.50)

    for xi, (bv, ov) in enumerate(zip(bh, oh)):
        if bv > -100:
            ax.text(xi - 1.5*w, bv + 0.4, f"{bv:.1f}", ha="center", va="bottom",
                    fontsize=7, color="white")
            ax.text(xi - 0.5*w, ov + 0.4, f"{ov:.1f}", ha="center", va="bottom",
                    fontsize=7, color="#50fa7b")

    # delta annotation on Total bar
    delta_h = bh[-1] - oh[-1]
    delta_c = bc[-1] - oc[-1]
    ax.annotate(f"−{delta_h:.1f} dBA\n(hover)",
                xy=(x[-1] - 0.5*w, oh[-1]),
                xytext=(x[-1] - 0.5*w - 0.7, oh[-1] + 4),
                color="#50fa7b", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#50fa7b", lw=1.2))

    all_vals = [v for v in bh + oh + bc + oc if v > -100]
    ax.set_xticks(x)
    ax.set_xticklabels(mechs, color="white", fontsize=9)
    ax.set_ylabel("A-weighted SPL [dBA]", color="white")
    ax.set_title("Noise Mechanism Breakdown", color="white", fontsize=11)
    ax.set_ylim(max(0, min(all_vals) - 8), max(all_vals) + 10)
    ax.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRID, fontsize=7.5)
    ax.grid(axis="y", color=GRID, alpha=0.4)

    # --- Panel 2: A-weighted 1/3-octave spectrum at hover ---
    freq = THIRD_OCT_FREQS

    def dba(spec): return spec + A_WEIGHT

    ax2 = axes[1]
    ax2.semilogx(freq, dba(baseline_hover["SPL_spectrum"]),
                 "o-", color="#4a90d9", lw=2.0, ms=4, label="Baseline total")
    ax2.semilogx(freq, dba(opt_hover["SPL_spectrum"]),
                 "s-", color="#50fa7b", lw=2.0, ms=4, label="Optimum total")
    ax2.semilogx(freq, dba(baseline_hover["spec_leti"]),
                 "--", color="#4a90d9", lw=1.2, alpha=0.6, label="Baseline LETI")
    ax2.semilogx(freq, dba(opt_hover["spec_leti"]),
                 "--", color="#50fa7b", lw=1.2, alpha=0.6, label="Optimum LETI")
    ax2.semilogx(freq, dba(baseline_hover["spec_lbl"]),
                 ":", color="#e9c46a", lw=1.2, alpha=0.7, label="Baseline LBL-VS")
    ax2.semilogx(freq, dba(opt_hover["spec_lbl"]),
                 ":", color="#f4a261", lw=1.2, alpha=0.7, label="Optimum LBL-VS")

    ax2.set_xlabel("Frequency [Hz]", color="white")
    ax2.set_ylabel("SPL [dBA]", color="white")
    ax2.set_title("1/3-Octave Spectrum — Hover @ 1 m", color="white", fontsize=11)
    ax2.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRID, fontsize=7.5, ncol=2)
    ax2.grid(color=GRID, alpha=0.4, which="both")
    ax2.set_xlim(50, 20000)

    # clip y-axis to the meaningful range (drop extreme negatives from muted bands)
    all_spec = np.concatenate([
        dba(baseline_hover["SPL_spectrum"]), dba(opt_hover["SPL_spectrum"]),
        dba(baseline_hover["spec_leti"]),    dba(opt_hover["spec_leti"]),
    ])
    meaningful = all_spec[all_spec > -80]
    if len(meaningful):
        ax2.set_ylim(min(meaningful) - 5, max(meaningful) + 5)

    fig.suptitle(
        "Aeroacoustic Noise Breakdown — APC 7×5E Optimised vs Baseline\n"
        f"Hover: {baseline_hover['SPL_total']:.2f} → {opt_hover['SPL_total']:.2f} dBA  |  "
        f"Cruise: {baseline_cruise['SPL_total']:.2f} → {opt_cruise['SPL_total']:.2f} dBA  |  "
        f"RPM {opt_hover['rpm']:.0f}",
        color="white", fontsize=10, y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, "noise_breakdown.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[VIZ] Saved -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    # Defaults: hardened-model 71.77 dBA optimum (8-start, 2026-04-22)
    parser.add_argument("--rpm",     type=float, default=6052.0)
    parser.add_argument("--dtwist",  type=float, nargs=N_CP,
                        default=[-0.75, 2.981, 0.397, 4.999, 4.776])
    parser.add_argument("--dchord",  type=float, nargs=N_CP,
                        default=[0.022, -0.015, -0.015, -0.015, -0.015])
    parser.add_argument("--sweep",   type=float, nargs=N_CP,
                        default=[0.0, 0.0, 0.1135, 0.1135, 0.1185])
    parser.add_argument("--dtc",     type=float, nargs=N_CP,
                        default=[-0.0064, -0.0264, -0.0153, 0.0047, 0.0247])
    args = parser.parse_args()

    blade_base = baseline_apc7x5e()
    blade_opt  = _build_blade(
        args.rpm,
        np.array(args.dtwist), np.array(args.dchord),
        np.array(args.sweep),  np.array(args.dtc),
    )

    RPM_BASELINE = 7000.0
    print("Running BEM + BPM for baseline and optimum...")
    base_hover   = _run_case(blade_base, RPM_BASELINE, 0.0,         "Baseline hover")
    base_cruise  = _run_case(blade_base, RPM_BASELINE, CRUISE_VINF, "Baseline cruise")
    opt_hover    = _run_case(blade_opt,  args.rpm,               0.0,    "Optimum hover")
    opt_cruise   = _run_case(blade_opt,  args.rpm,               CRUISE_VINF, "Optimum cruise")

    print("\n=== Noise Mechanism Breakdown (Hover @ 1 m) ===")
    print(f"{'Mechanism':<18} {'Baseline':>10} {'Optimum':>10} {'Delta':>8}")
    print("-" * 50)
    rows = [
        ("TBL-TE",     "SPL_tbl_te_dBA"),
        ("LBL-VS",     "SPL_lbl_vs_dBA"),
        ("Amiet LETI", "SPL_leti_dBA"),
        ("BVI Tonal",  "SPL_bvi_dBA"),
        ("TOTAL",      "SPL_total"),
    ]
    for name, key in rows:
        b = base_hover[key]
        o = opt_hover[key]
        print(f"  {name:<16} {b:>10.2f} {o:>10.2f} {o-b:>+8.2f} dBA")

    print("\n=== Noise Mechanism Breakdown (Cruise @ 1 m) ===")
    print(f"{'Mechanism':<18} {'Baseline':>10} {'Optimum':>10} {'Delta':>8}")
    print("-" * 50)
    for name, key in rows:
        b = base_cruise[key]
        o = opt_cruise[key]
        print(f"  {name:<16} {b:>10.2f} {o:>10.2f} {o-b:>+8.2f} dBA")

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_breakdown(base_hover, opt_hover, base_cruise, opt_cruise, out_dir)


if __name__ == "__main__":
    main()
