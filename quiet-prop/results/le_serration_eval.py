"""
LE serration / tubercle parametric evaluation.

Sweeps serration amplitude h_LE from 0 to 6 mm (uniform across span) at:
  - APC 7x5E baseline geometry, 7000 RPM hover
  - Hardened-model optimum geometry (71.77 dBA, 6052 RPM)

Models (Amiet LETI is the dominant source at UAV Re; Re_c < 62k -> fully laminar):
  Sawtooth LE  -- Lyu et al. (2016) compact limit: G = sinc^2(f*h/v_rel)
  Tubercle LE  -- Chaitanya et al. (2017) compact approx: G = J0^2(pi*f*h/v_rel)

Usage
-----
  python results/le_serration_eval.py           # runs all cases, saves plot
  python results/le_serration_eval.py --no-plot # print table only
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_apc7x5e
from aerodynamics.ccblade_component import bem_solve
from acoustics.bpm_component import bpm_noise, THIRD_OCT_FREQS, A_WEIGHT

N_STATIONS = 20
N_CP       = 5

# ---------------------------------------------------------------------------
# Hardened-model optimum DVs (71.77 dBA, 8-start 2026-04-22)
# ---------------------------------------------------------------------------
OPT_RPM = 6052.0
OPT_DVS = {
    "delta_twist_cp":  np.array([-0.75,  2.981,  0.397,  4.999,  4.776]),
    "delta_chord_cp":  np.array([ 0.022, -0.015, -0.015, -0.015, -0.015]),
    "sweep_cp":        np.array([ 0.0,    0.0,    0.1135, 0.1135, 0.1185]),
    "delta_tc_cp":     np.array([-0.0064,-0.0264,-0.0153, 0.0047, 0.0247]),
    "delta_camber_cp": np.array([ 0.0041,-0.0159,-0.0184,-0.0089, 0.0094]),
}


def _build_blade(dvs):
    blade = baseline_apc7x5e()
    r_cp  = np.linspace(blade.r_R[0], blade.r_R[-1], N_CP)

    def spl(cp): return CubicSpline(r_cp, cp)(blade.r_R)

    return (blade
            .perturb_twist(spl(dvs["delta_twist_cp"]))
            .perturb_chord(spl(dvs["delta_chord_cp"]))
            .perturb_tc(spl(dvs["delta_tc_cp"]))
            .set_sweep(np.clip(spl(dvs["sweep_cp"]), 0.0, 0.12))
            .set_camber(blade.camber_dist + spl(dvs["delta_camber_cp"]))
            .set_blade_angles([0.0, 120.0, 240.0]))


def _eval(blade, rpm, h_LE_m, le_type="sawtooth"):
    """BEM + BPM for a given blade, RPM, and uniform LE serration amplitude."""
    aero = bem_solve(blade, rpm=rpm, v_inf=0.0, rho=1.225, n_stations=N_STATIONS)
    _, chord_m, _ = blade.get_stations(N_STATIONS)
    h_LE = np.full(N_STATIONS, h_LE_m) if h_LE_m > 0 else None
    return bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=rpm, num_blades=blade.num_blades, radius_m=blade.radius_m,
        rho=1.225, r_obs=1.0,
        x_tr_c=aero["x_tr_c"],
        blade_angles_deg=blade.blade_angles_deg,
        dT_dr=aero.get("dT_dr"), dQ_dr=aero.get("dQ_dr"),
        sweep_m=aero.get("sweep_m", np.zeros(N_STATIONS)),
        h_LE=h_LE, le_type=le_type,
    )


def sweep_h_LE(blade, rpm, h_vals_mm, le_type="sawtooth"):
    spls, letis = [], []
    for h_mm in h_vals_mm:
        res = _eval(blade, rpm, h_mm * 1e-3, le_type)
        spls.append(res["SPL_total"])
        letis.append(res["SPL_leti_dBA"])
    return np.array(spls), np.array(letis)


def plot_results(h_vals, cases, out_dir):
    """
    cases: list of (label, blade, rpm, le_type, spls, letis)
    """
    BG     = "#1a1a2e"
    PANEL  = "#16213e"
    GRID   = "#444466"
    COLORS = ["#4a90d9", "#50fa7b", "#f39c12"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white")
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, alpha=0.4)

    ax = axes[0]
    for (label, blade, rpm, le_type, spls, letis), col in zip(cases, COLORS):
        ax.plot(h_vals, spls,  "o-", color=col, lw=2.0, label=f"{label} total")
        ax.plot(h_vals, letis, "--", color=col, lw=1.2, alpha=0.55,
                label=f"{label} LETI")
    ax.axhline(cases[0][4][0], color="white", lw=0.8, linestyle=":",
               alpha=0.5, label="No LE modification")
    ax.set_xlabel("LE serration amplitude h_LE [mm]", color="white")
    ax.set_ylabel("A-weighted SPL [dBA]", color="white")
    ax.set_title("Noise vs LE Serration Amplitude", color="white", fontsize=11)
    ax.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRID, fontsize=7.5)

    ax2 = axes[1]
    for (label, blade, rpm, le_type, spls, letis), col in zip(cases, COLORS):
        best_idx  = int(np.argmin(spls))
        best_h_mm = h_vals[best_idx]
        res_base = _eval(blade, rpm, 0.0)
        res_opt  = _eval(blade, rpm, best_h_mm * 1e-3, le_type=le_type)
        ax2.semilogx(THIRD_OCT_FREQS, res_base["SPL_spectrum"] + A_WEIGHT,
                     "-", color=col, lw=1.5, alpha=0.45,
                     label=f"{label} baseline")
        ax2.semilogx(THIRD_OCT_FREQS, res_opt["SPL_spectrum"] + A_WEIGHT,
                     "--", color=col, lw=2.0,
                     label=f"{label} h={best_h_mm:.1f}mm -> {spls[best_idx]:.1f} dBA")
    ax2.set_xlabel("Frequency [Hz]", color="white")
    ax2.set_ylabel("SPL [dBA]", color="white")
    ax2.set_title("Spectrum: Baseline vs Best LE Serration", color="white", fontsize=11)
    ax2.set_xlim(200, 20000)
    ax2.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRID, fontsize=7.5)

    fig.suptitle("LE Serration / Tubercle Noise Reduction  APC 7x5E @ 1 m",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "le_serration_sweep.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n[VIZ] Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    blade_base = baseline_apc7x5e()
    blade_opt  = _build_blade(OPT_DVS)

    h_vals = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])

    run_cases = [
        ("Baseline 7000 RPM sawtooth", blade_base, 7000.0, "sawtooth"),
        ("Baseline 7000 RPM tubercle", blade_base, 7000.0, "tubercle"),
        ("Optimum  6052 RPM sawtooth", blade_opt,  OPT_RPM, "sawtooth"),
    ]

    print("\n=== LE Serration Parametric Sweep ===\n")
    cases_for_plot = []
    for label, blade, rpm, le_type in run_cases:
        print(f"  {label}")
        spls, letis = sweep_h_LE(blade, rpm, h_vals, le_type)
        best_idx = int(np.argmin(spls))
        print(f"    h=0 mm : {spls[0]:.2f} dBA")
        print(f"    Best   : {spls[best_idx]:.2f} dBA  at h={h_vals[best_idx]:.1f} mm"
              f"  (delta = {spls[best_idx]-spls[0]:.2f} dBA)\n")
        cases_for_plot.append((label, blade, rpm, le_type, spls, letis))

    if not args.no_plot:
        out_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(out_dir, exist_ok=True)
        plot_results(h_vals, cases_for_plot, out_dir)


if __name__ == "__main__":
    main()
