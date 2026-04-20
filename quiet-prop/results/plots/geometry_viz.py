"""
Propeller geometry visualization — baseline vs Phase 1 optimized.

Generates:
  results/plots/blade_geometry.png   (4-panel figure for GitHub README)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from geometry.blade_generator import baseline_hqprop

# ---------------------------------------------------------------------------
# Optimized deltas  (from Phase 1 run)
# ---------------------------------------------------------------------------

DELTA_TWIST = np.array([-0.643, -3.712, -4.403, -5.0,   -5.0,   -5.0,   -5.0,
                         -5.0,   -5.0,   -5.0,   -4.727, -4.276, -3.298, -2.644,
                         -1.856, -1.153,  1.004,  0.106])

DELTA_CHORD_R = np.array([-0.03,   0.03,   0.03,   0.03,   0.03,   0.03,   0.03,
                            0.03,   0.03,   0.03,   0.03,   0.03,   0.03,   0.03,
                            0.03,   0.03,   0.03,   0.0292])

# ---------------------------------------------------------------------------
# Build geometries
# ---------------------------------------------------------------------------

blade_base = baseline_hqprop()
blade_opt  = blade_base.perturb_twist(DELTA_TWIST).perturb_chord(DELTA_CHORD_R)

N = 200
r_R_fine   = np.linspace(blade_base.r_R[0], blade_base.r_R[-1], N)
chord_R_b  = np.interp(r_R_fine, blade_base.r_R, blade_base.chord_R)
twist_b    = np.interp(r_R_fine, blade_base.r_R, blade_base.twist_deg)
chord_R_o  = np.interp(r_R_fine, blade_opt.r_R,  blade_opt.chord_R)
twist_o    = np.interp(r_R_fine, blade_opt.r_R,   blade_opt.twist_deg)

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")  # GitHub dark background

ax_plan   = fig.add_subplot(2, 2, (1, 2))   # top row — planform
ax_chord  = fig.add_subplot(2, 2, 3)
ax_twist  = fig.add_subplot(2, 2, 4)

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#30363d"
BASE_COL = "#58a6ff"   # blue
OPT_COL  = "#3fb950"   # green

for ax in [ax_plan, ax_chord, ax_twist]:
    ax.set_facecolor(PANEL_BG)
    ax.spines[["top", "right", "left", "bottom"]].set_color(GRID_COL)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(color=GRID_COL, linewidth=0.5, linestyle="--")

# ---------------------------------------------------------------------------
# Panel 1: blade planform (top-down, both blades shown as filled shapes)
# ---------------------------------------------------------------------------

def blade_outline(r_R, chord_R, radius_m, angle_offset_deg=0):
    """Return (x_fill, y_fill) for filled planform polygon."""
    r_m   = r_R * radius_m
    c_m   = chord_R * radius_m
    # Leading edge at +c/4, trailing edge at -3c/4 (chord measured aft of LE)
    le =  0.25 * c_m
    te = -0.75 * c_m
    ang = np.deg2rad(angle_offset_deg)
    ca, sa = np.cos(ang), np.sin(ang)

    # Upper edge (LE side), lower edge (TE side)
    x_top = ca * r_m - sa * le
    y_top = sa * r_m + ca * le
    x_bot = ca * r_m - sa * te
    y_bot = sa * r_m + ca * te

    # Closed polygon: LE outboard → TE outboard → TE inboard → LE inboard
    x = np.concatenate([x_top, x_bot[::-1]])
    y = np.concatenate([y_top, y_bot[::-1]])
    return x, y

R = blade_base.radius_m
for angle in [0, 120, 240]:
    xb, yb = blade_outline(r_R_fine, chord_R_b, R, angle)
    xo, yo = blade_outline(r_R_fine, chord_R_o, R, angle)
    ax_plan.fill(xb, yb, color=BASE_COL, alpha=0.35)
    ax_plan.fill(xo, yo, color=OPT_COL,  alpha=0.35)
    ax_plan.plot(xb, yb, color=BASE_COL, linewidth=0.8)
    ax_plan.plot(xo, yo, color=OPT_COL,  linewidth=0.8)

# Hub circle
hub_r = blade_base.r_R[0] * R
hub = plt.Circle((0, 0), hub_r, color="#888", alpha=0.6)
ax_plan.add_patch(hub)

ax_plan.set_aspect("equal")
ax_plan.set_xlim(-R * 1.15, R * 1.15)
ax_plan.set_ylim(-R * 1.15, R * 1.15)
ax_plan.set_title("HQProp 7x4x3 — Blade Planform (3-blade, top view)", fontsize=11, pad=8)
ax_plan.set_xlabel("x (m)")
ax_plan.set_ylabel("y (m)")

patch_b = mpatches.Patch(color=BASE_COL, alpha=0.6, label="Baseline (5000 RPM, 28.7 dBA)")
patch_o = mpatches.Patch(color=OPT_COL,  alpha=0.6, label="Phase 1 Optimized (21.2 dBA, -7.5 dBA)")
ax_plan.legend(handles=[patch_b, patch_o], loc="upper right",
               facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor="white", fontsize=9)

# ---------------------------------------------------------------------------
# Panel 2: chord distribution
# ---------------------------------------------------------------------------

ax_chord.plot(r_R_fine, chord_R_b * R * 1000, color=BASE_COL, linewidth=1.8,
              label="Baseline")
ax_chord.plot(r_R_fine, chord_R_o * R * 1000, color=OPT_COL,  linewidth=1.8,
              linestyle="--", label="Optimized")
ax_chord.fill_between(r_R_fine,
                      chord_R_b * R * 1000,
                      chord_R_o * R * 1000,
                      alpha=0.15, color=OPT_COL)
ax_chord.set_xlabel("r/R")
ax_chord.set_ylabel("Chord (mm)")
ax_chord.set_title("Chord Distribution")
ax_chord.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor="white", fontsize=9)
ax_chord.set_xlim(r_R_fine[0], r_R_fine[-1])

# ---------------------------------------------------------------------------
# Panel 3: twist distribution
# ---------------------------------------------------------------------------

ax_twist.plot(r_R_fine, twist_b, color=BASE_COL, linewidth=1.8, label="Baseline")
ax_twist.plot(r_R_fine, twist_o, color=OPT_COL,  linewidth=1.8,
              linestyle="--", label="Optimized")
ax_twist.fill_between(r_R_fine, twist_b, twist_o,
                      alpha=0.15, color=OPT_COL)
ax_twist.set_xlabel("r/R")
ax_twist.set_ylabel("Twist (deg)")
ax_twist.set_title("Twist Distribution")
ax_twist.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor="white", fontsize=9)
ax_twist.set_xlim(r_R_fine[0], r_R_fine[-1])

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

fig.text(0.5, 0.01,
         "Phase 1 optimization: SLSQP, 37 DVs (RPM + 18x twist + 18x chord), "
         "thrust constraint >= 1.0 N",
         ha="center", va="bottom", color="#8b949e", fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 1])

out_path = os.path.join(os.path.dirname(__file__), "blade_geometry.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
