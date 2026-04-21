"""
APC pitch-variant comparison — all geometries from Brandt & Selig (2011), AIAA 2011-1255.

Compares APC 7x4E / 7x5E (baseline) / 7x6E as 3-blade rotors across:
  1. Thrust vs RPM (static, V=0)
  2. Shaft power vs RPM
  3. SPL total vs RPM (r_obs = 1 m)
  4. T / sqrt(P) efficiency proxy vs RPM
  5. Chord distribution
  6. Twist distribution
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from geometry.blade_generator import baseline_apc7x5e
from geometry.blade_importer import load_prop
from aerodynamics.ccblade_component import bem_solve
from acoustics.bpm_component import bpm_noise

BG     = "#0d1117"
BLUE   = "#58a6ff"   # APC 7x5E baseline
ORANGE = "#f0883e"   # APC 7x4E (lower pitch)
GREEN  = "#3fb950"   # APC 7x6E (higher pitch)
SUBTLE = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

N_STATIONS = 20
RHO        = 1.225
V_INF      = 0.0


def _sweep_rpm(blade, rpms):
    thrust, power, spl = [], [], []
    _, chord_m, _ = blade.get_stations(N_STATIONS)
    for rpm in rpms:
        res   = bem_solve(blade, rpm=rpm, v_inf=V_INF, rho=RHO)
        noise = bpm_noise(
            r_m=res["r"], chord_m=chord_m,
            v_rel=res["v_rel"], aoa_deg=res["aoa_deg"], cl=res["cl"],
            thrust=res["thrust"], torque=res["torque"],
            rpm=rpm, num_blades=blade.num_blades, radius_m=blade.radius_m,
            x_tr_c=res["x_tr_c"],
        )
        thrust.append(res["thrust"])
        power.append(res["power"])
        spl.append(noise["SPL_total"])
    return np.array(thrust), np.array(power), np.array(spl)


def _setup_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(SUBTLE)
    ax.grid(True, color=SUBTLE, linewidth=0.4, alpha=0.5)


def plot_comparison(save_path=None, show=False):
    props = [
        (load_prop("APC_7x4E", num_blades_override=3), ORANGE, "APC 7x4E (low pitch)"),
        (baseline_apc7x5e(),                            BLUE,   "APC 7x5E (baseline)"),
        (load_prop("APC_7x6E", num_blades_override=3), GREEN,  "APC 7x6E (high pitch)"),
    ]

    rpms = np.linspace(3500, 10000, 28)
    results = []
    for blade, col, label in props:
        print(f"Sweeping {label}…")
        t, p, s = _sweep_rpm(blade, rpms)
        results.append((blade, col, label, t, p, s))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        "APC 7-inch Pitch Comparison — 3-blade, static hover\n"
        "Source: Brandt & Selig (2011), AIAA 2011-1255  |  ISA sea level, V=0, r_obs=1 m",
        color=TEXT, fontsize=12, y=0.99,
    )

    ax_T, ax_P, ax_S = axes[0]
    ax_FM, ax_C, ax_TW = axes[1]

    THRUST_TARGET = 5.69   # N — TWR 2.5 for 928 g AUW, 4 rotors

    # 1 — Thrust
    _setup_ax(ax_T, "Thrust vs RPM", "RPM", "Thrust (N)")
    ax_T.axhline(THRUST_TARGET, color=TEXT, lw=1, ls="--", alpha=0.6)
    ax_T.annotate(f"TWR 2.5 target ({THRUST_TARGET} N)",
                  xy=(3600, THRUST_TARGET + 0.15), color=TEXT, fontsize=7)
    for blade, col, label, t, p, s in results:
        ax_T.plot(rpms, t, color=col, lw=2, label=label)
    ax_T.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 2 — Power
    _setup_ax(ax_P, "Shaft Power vs RPM", "RPM", "Power (W)")
    for blade, col, label, t, p, s in results:
        ax_P.plot(rpms, p, color=col, lw=2, label=label)
    ax_P.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 3 — SPL
    _setup_ax(ax_S, "SPL total vs RPM (r=1 m)", "RPM", "SPL (dBA)")
    for blade, col, label, t, p, s in results:
        ax_S.plot(rpms, s, color=col, lw=2, label=label)
    ax_S.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 4 — T / sqrt(P)
    _setup_ax(ax_FM, "T / √P  (efficiency proxy)", "RPM", "T / √P  (N / √W)")
    for blade, col, label, t, p, s in results:
        with np.errstate(divide="ignore", invalid="ignore"):
            fm = np.where(p > 0, t / np.sqrt(p), np.nan)
        ax_FM.plot(rpms, fm, color=col, lw=2, label=label)
    ax_FM.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 5 — Chord
    _setup_ax(ax_C, "Chord Distribution", "r/R", "Chord (mm)")
    for blade, col, label, *_ in results:
        ax_C.plot(blade.r_R, blade.chord_m * 1000, color=col, lw=2, label=label)
    ax_C.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 6 — Twist
    _setup_ax(ax_TW, "Twist Distribution", "r/R", "Twist (deg)")
    for blade, col, label, *_ in results:
        ax_TW.plot(blade.r_R, blade.twist_deg, color=col, lw=2, label=label)
    ax_TW.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"[VIZ] Saved -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_comparison(save_path=os.path.join(out_dir, "apc_pitch_comparison.png"))
