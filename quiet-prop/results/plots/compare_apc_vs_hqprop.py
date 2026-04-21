"""
APC 7x5E (3-blade, baseline) vs HQProp 7x4x3 (reference) — performance comparison.

Panels:
  1. Thrust vs RPM (static, V=0)
  2. Power vs RPM (static, V=0)
  3. SPL total vs RPM (static, V=0, r_obs=1 m)
  4. Figure-of-merit proxy: Thrust / sqrt(Power) vs RPM
  5. Chord distribution comparison
  6. Twist distribution comparison
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from geometry.blade_generator import baseline_apc7x5e, baseline_hqprop
from aerodynamics.ccblade_component import bem_solve
from acoustics.bpm_component import bpm_noise

BG     = "#0d1117"
BLUE   = "#58a6ff"
ORANGE = "#f0883e"
SUBTLE = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
GREEN  = "#3fb950"

N_STATIONS = 20
RHO        = 1.225   # kg/m³  ISA sea level
V_INF      = 0.0    # static hover


def _sweep_rpm(blade, rpms):
    thrust, power, spl = [], [], []
    _, chord_m, _ = blade.get_stations(N_STATIONS)
    for rpm in rpms:
        res = bem_solve(blade, rpm=rpm, v_inf=V_INF, rho=RHO)
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
    apc    = baseline_apc7x5e()
    hqprop = baseline_hqprop()

    rpms = np.linspace(3500, 10000, 28)
    print("Sweeping APC 7x5E (3-blade)…")
    t_apc, p_apc, s_apc = _sweep_rpm(apc, rpms)
    print("Sweeping HQProp 7x4x3…")
    t_hq, p_hq, s_hq    = _sweep_rpm(hqprop, rpms)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
    fig.suptitle("APC 7x5E (3-blade, baseline) vs HQProp 7x4x3 (reference)\n"
                 "Static hover performance — ISA sea level, V=0, r_obs=1 m",
                 color=TEXT, fontsize=12, y=0.99)

    ax_T, ax_P, ax_S = axes[0]
    ax_FM, ax_C, ax_TW = axes[1]

    # 1 — Thrust vs RPM
    _setup_ax(ax_T, "Thrust vs RPM", "RPM", "Thrust (N)")
    ax_T.plot(rpms, t_apc, color=BLUE,   lw=2, label="APC 7x5E 3-blade")
    ax_T.plot(rpms, t_hq,  color=ORANGE, lw=2, label="HQProp 7x4x3")
    ax_T.axhline(5.69, color=GREEN, lw=1, ls="--", alpha=0.8)
    ax_T.annotate("TWR 2.5 target (5.69 N)", xy=(3600, 5.85),
                  color=GREEN, fontsize=7)
    ax_T.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 2 — Power vs RPM
    _setup_ax(ax_P, "Shaft Power vs RPM", "RPM", "Power (W)")
    ax_P.plot(rpms, p_apc, color=BLUE,   lw=2, label="APC 7x5E 3-blade")
    ax_P.plot(rpms, p_hq,  color=ORANGE, lw=2, label="HQProp 7x4x3")
    ax_P.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 3 — SPL vs RPM
    _setup_ax(ax_S, "SPL total vs RPM (r=1 m)", "RPM", "SPL (dBA)")
    ax_S.plot(rpms, s_apc, color=BLUE,   lw=2, label="APC 7x5E 3-blade")
    ax_S.plot(rpms, s_hq,  color=ORANGE, lw=2, label="HQProp 7x4x3")
    ax_S.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 4 — Thrust/sqrt(Power) figure of merit proxy
    _setup_ax(ax_FM, "T / √P  (efficiency proxy)", "RPM", "T / √P  (N / √W)")
    with np.errstate(divide="ignore", invalid="ignore"):
        fm_apc = np.where(p_apc > 0, t_apc / np.sqrt(p_apc), np.nan)
        fm_hq  = np.where(p_hq  > 0, t_hq  / np.sqrt(p_hq),  np.nan)
    ax_FM.plot(rpms, fm_apc, color=BLUE,   lw=2, label="APC 7x5E 3-blade")
    ax_FM.plot(rpms, fm_hq,  color=ORANGE, lw=2, label="HQProp 7x4x3")
    ax_FM.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 5 — Chord distribution
    _setup_ax(ax_C, "Chord Distribution", "r/R", "Chord (mm)")
    ax_C.plot(apc.r_R,    apc.chord_m    * 1000, color=BLUE,   lw=2, label="APC 7x5E 3-blade")
    ax_C.plot(hqprop.r_R, hqprop.chord_m * 1000, color=ORANGE, lw=2, label="HQProp 7x4x3")
    ax_C.fill_between(apc.r_R,
                      apc.chord_m * 1000,
                      np.interp(apc.r_R, hqprop.r_R, hqprop.chord_m * 1000),
                      alpha=0.12, color=BLUE)
    ax_C.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # 6 — Twist distribution
    _setup_ax(ax_TW, "Twist Distribution", "r/R", "Twist (deg)")
    ax_TW.plot(apc.r_R,    apc.twist_deg,    color=BLUE,   lw=2, label="APC 7x5E 3-blade")
    ax_TW.plot(hqprop.r_R, hqprop.twist_deg, color=ORANGE, lw=2, label="HQProp 7x4x3")
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
    plot_comparison(save_path=os.path.join(out_dir, "apc_vs_hqprop_performance.png"))
