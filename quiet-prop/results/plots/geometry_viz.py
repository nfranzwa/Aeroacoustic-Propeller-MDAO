"""
Comprehensive blade geometry visualization – Phase 2.

6-panel figure:
  1. Top-down rotor planform (3 blades, sweep shown)
  2. 3D blade wireframe (cross-sections + LE/TE spars)
  3. Chord distribution
  4. Twist distribution
  5. t/c ratio + physical wall thickness
  6. Aft-sweep and dihedral offsets
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from geometry.blade_generator import baseline_hqprop

# Phase-1 optimised deltas
DELTA_TWIST_P1 = np.array([
    -0.643, -3.712, -4.403, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
    -5.0, -4.727, -4.276, -3.298, -2.644, -1.856, -1.153, 1.004, 0.106])
DELTA_CHORD_P1 = np.array([
    -0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
     0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
     0.03, 0.03, 0.03, 0.0292])

BG = "#0d1117"; BLUE = "#58a6ff"; GREEN = "#3fb950"
ORANGE = "#f0883e"; PURPLE = "#d2a8ff"; SUBTLE = "#30363d"
TEXT = "#e6edf3"; MUTED = "#8b949e"


def _naca44xx(tc=0.12, n=35):
    beta = np.linspace(0, np.pi, n)
    x = 0.5*(1 - np.cos(beta))
    m, p, t = 0.04, 0.4, tc
    yt = 5*t*(0.2969*np.sqrt(np.clip(x,0,1)) - 0.1260*x
              - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x<p, m/p**2*(2*p*x-x**2), m/(1-p)**2*((1-2*p)+2*p*x-x**2))
    dydx = np.where(x<p, 2*m/p**2*(p-x), 2*m/(1-p)**2*(p-x))
    th = np.arctan(dydx)
    xu = x - yt*np.sin(th); yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th); yl = yc - yt*np.cos(th)
    return (np.concatenate([xu, xl[-2:0:-1]]),
            np.concatenate([yu, yl[-2:0:-1]]))


def _section_3d(chord, twist_deg, sweep_x, z_off, radius, tc, n=30):
    x_af, y_af = _naca44xx(tc, n)
    x = (x_af - 0.25)*chord
    z = y_af*chord
    tw = np.deg2rad(twist_deg)
    xt =  x*np.cos(tw) + z*np.sin(tw)
    zt = -x*np.sin(tw) + z*np.cos(tw)
    return xt + sweep_x, np.full_like(xt, radius), zt + z_off


def _planform(blade, n=60, angle_deg=0.0):
    r_R = np.linspace(blade.r_R[0], blade.r_R[-1], n)
    r_m = r_R * blade.radius_m
    c_m = np.interp(r_R, blade.r_R, blade.chord_R) * blade.radius_m
    sw_m = np.interp(r_R, blade.r_R, blade.sweep_R) * blade.radius_m
    tw  = np.deg2rad(np.interp(r_R, blade.r_R, blade.twist_deg))
    qc_x = sw_m; qc_y = r_m
    x_le = qc_x - 0.25*c_m*np.cos(tw)
    x_te = qc_x + 0.75*c_m*np.cos(tw)
    a = np.deg2rad(angle_deg)
    def rot(x, y):
        return x*np.cos(a) - y*np.sin(a), x*np.sin(a) + y*np.cos(a)
    xl, yl = rot(x_le, qc_y)
    xt, yt = rot(x_te, qc_y)
    return xl, yl, xt, yt


def _setup_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(SUBTLE)
    ax.grid(True, color=SUBTLE, linewidth=0.4, alpha=0.5)


def plot_geometry(blade_base, blade_opt=None, save_path=None, show=False):
    blades  = [blade_base] + ([blade_opt] if blade_opt else [])
    labels  = ["Baseline"] + (["Optimised"] if blade_opt else [])
    colours = [BLUE]       + ([GREEN]       if blade_opt else [])

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle("Propeller Blade Geometry – Aeroacoustic MDAO",
                 color=TEXT, fontsize=13, y=0.99)

    ax_plan  = fig.add_subplot(2, 3, 1, facecolor=BG)
    ax_3d    = fig.add_subplot(2, 3, 2, projection="3d")
    ax_3d.set_facecolor(BG)
    ax_chord = fig.add_subplot(2, 3, 3, facecolor=BG)
    ax_twist = fig.add_subplot(2, 3, 4, facecolor=BG)
    ax_tc    = fig.add_subplot(2, 3, 5, facecolor=BG)
    ax_dihed = fig.add_subplot(2, 3, 6, facecolor=BG)

    # ---- Planform ----------------------------------------------------------
    ax_plan.set_aspect("equal")
    _setup_ax(ax_plan, "Rotor Planform (top view)", "X (m)", "Y (m)")
    r_hub = blade_base.r_R[0] * blade_base.radius_m
    th_h  = np.linspace(0, 2*np.pi, 100)
    ax_plan.fill(r_hub*np.cos(th_h), r_hub*np.sin(th_h),
                 color=SUBTLE, alpha=0.7, zorder=0)

    for blade, col, lbl in zip(blades, colours, labels):
        for b_i, ang in enumerate(blade.blade_angles_deg):
            xl, yl, xt, yt = _planform(blade, angle_deg=ang)
            xs = np.concatenate([xl, xt[::-1], [xl[0]]])
            ys = np.concatenate([yl, yt[::-1], [yl[0]]])
            ax_plan.fill(xs, ys, alpha=0.22, color=col)
            ax_plan.plot(xs, ys, color=col, lw=0.9,
                         label=lbl if b_i == 0 else None)
        R = blade.radius_m
        for ang in blade.blade_angles_deg:
            a = np.deg2rad(ang)
            ax_plan.annotate(f"{ang:.0f}deg",
                             xy=(R*0.78*np.sin(a), R*0.78*np.cos(a)),
                             color=col, fontsize=7, ha="center", va="center")
    ax_plan.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE,
                   labelcolor=TEXT, loc="lower right")

    # ---- 3D wireframe ------------------------------------------------------
    ax_3d.set_title("3D Blade Wireframe", color=TEXT, fontsize=9, pad=2)
    ax_3d.tick_params(colors=MUTED, labelsize=6)
    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(SUBTLE)
    ax_3d.grid(True, color=SUBTLE, linewidth=0.3)

    n_sec = 14
    for blade, col in zip(blades, colours):
        r_m, c_m, tw, tc, sw, zoff = blade.get_full_stations(n_sec)
        le_x, le_y, le_z = [], [], []
        te_x, te_y, te_z = [], [], []
        for i in range(n_sec):
            xs, ys, zs = _section_3d(c_m[i], tw[i], sw[i], zoff[i], r_m[i], tc[i], 22)
            xc = np.append(xs, xs[0]); yc = np.append(ys, ys[0]); zc = np.append(zs, zs[0])
            ax_3d.plot(xc, yc, zc, color=col, lw=0.5, alpha=0.65)
            mid = len(xs)//2
            le_x.append(xs[0]);   le_y.append(ys[0]);   le_z.append(zs[0])
            te_x.append(xs[mid]); te_y.append(ys[mid]); te_z.append(zs[mid])
        ax_3d.plot(le_x, le_y, le_z, color=col, lw=1.5, alpha=0.9, label=f"{col} LE")
        ax_3d.plot(te_x, te_y, te_z, color=col, lw=0.9, alpha=0.5, ls="--")
        ax_3d.plot(sw.tolist(), r_m.tolist(), zoff.tolist(),
                   color=ORANGE, lw=1.0, ls=":", alpha=0.8)

    ax_3d.set_xlabel("X (m)", color=MUTED, fontsize=7, labelpad=1)
    ax_3d.set_ylabel("Radius (m)", color=MUTED, fontsize=7, labelpad=1)
    ax_3d.set_zlabel("Z (m)", color=MUTED, fontsize=7, labelpad=1)
    ax_3d.view_init(elev=22, azim=-55)

    # ---- Chord -------------------------------------------------------------
    _setup_ax(ax_chord, "Chord Distribution", "r/R", "Chord (mm)")
    for blade, col, lbl in zip(blades, colours, labels):
        ax_chord.plot(blade.r_R, blade.chord_m*1000, color=col, lw=2, label=lbl)
    if len(blades) > 1:
        c_opt_i = np.interp(blades[0].r_R, blades[1].r_R, blades[1].chord_m*1000)
        ax_chord.fill_between(blades[0].r_R, blades[0].chord_m*1000, c_opt_i,
                              alpha=0.15, color=GREEN)
    ax_chord.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # ---- Twist -------------------------------------------------------------
    _setup_ax(ax_twist, "Twist Distribution", "r/R", "Twist (deg)")
    for blade, col, lbl in zip(blades, colours, labels):
        ax_twist.plot(blade.r_R, blade.twist_deg, color=col, lw=2, label=lbl)
    if len(blades) > 1:
        t_opt_i = np.interp(blades[0].r_R, blades[1].r_R, blades[1].twist_deg)
        ax_twist.fill_between(blades[0].r_R, blades[0].twist_deg, t_opt_i,
                              alpha=0.15, color=GREEN)
    ax_twist.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # ---- t/c ratio ---------------------------------------------------------
    _setup_ax(ax_tc, "Thickness-to-Chord Ratio", "r/R", "t/c")
    for blade, col, lbl in zip(blades, colours, labels):
        ax_tc.plot(blade.r_R, blade.tc_ratio, color=col, lw=2, label=lbl)
        ax_tc2 = ax_tc.twinx()
        ax_tc2.plot(blade.r_R, blade.chord_m*blade.tc_ratio*1000,
                    color=col, lw=1, ls=":", alpha=0.5)
        ax_tc2.set_ylabel("Wall thickness (mm)", color=MUTED, fontsize=7)
        ax_tc2.tick_params(colors=MUTED, labelsize=6)
        ax_tc2.axhline(0.5, color=ORANGE, lw=0.8, ls="--", alpha=0.7)
        ax_tc2.annotate("0.5 mm min", xy=(0.85, 0.5),
                        xycoords=("axes fraction", "data"),
                        color=ORANGE, fontsize=7)
    ax_tc.axhline(0.06, color=ORANGE, lw=0.5, ls=":", alpha=0.4)
    ax_tc.legend(fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT)

    # ---- Sweep & dihedral --------------------------------------------------
    _setup_ax(ax_dihed, "Aft-Sweep & Dihedral", "r/R", "Offset (mm)")
    for blade, col, lbl in zip(blades, colours, labels):
        ax_dihed.plot(blade.r_R, blade.sweep_m*1000, color=col,
                      lw=2, ls="-", label=f"{lbl} sweep")
        ax_dihed.plot(blade.r_R, blade.z_offset_m*1000, color=col,
                      lw=1.5, ls="--", label=f"{lbl} dihedral")
    ax_dihed.axhline(0, color=SUBTLE, lw=0.6)
    ax_dihed.legend(fontsize=7, facecolor=BG, edgecolor=SUBTLE,
                    labelcolor=TEXT, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"[VIZ] Saved -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    blade_base = baseline_hqprop()
    blade_p1   = (blade_base
                  .perturb_twist(DELTA_TWIST_P1)
                  .perturb_chord(DELTA_CHORD_P1))

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Baseline vs Phase-1 optimised
    plot_geometry(blade_base, blade_p1,
                  save_path=os.path.join(out_dir, "blade_geometry.png"))

    # Baseline vs unequal-spacing example (Phase-2 preview)
    blade_unequal = blade_base.set_blade_angles([0.0, 115.0, 235.0])
    plot_geometry(blade_base, blade_unequal,
                  save_path=os.path.join(out_dir, "blade_geometry_unequal.png"))

    print("\nBaseline summary:")
    blade_base.summary()
