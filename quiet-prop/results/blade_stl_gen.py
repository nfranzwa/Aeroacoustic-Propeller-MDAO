"""
Propeller blade STL generator with LE serrations.

Generates and exports:
  1. Baseline APC 7x5E blade STL (no serrations)
  2. LE-serrated blade STL (sawtooth, Elegoo Saturn 4 Ultra 16K bounds)
  3. Side-by-side 3D rendered comparison image

LE serration optimum (71.32 dBA, baseline geometry):
  h_LE_cp = [2.51, 3.67, 3.39, 2.37, 0.80] mm  (per-CP, 15% chord limit)
  lambda_LE = 2 * h_LE  (Lyu 2016 optimal aspect ratio)
  RPM = 6911

Usage
-----
  python results/blade_stl_gen.py
  python results/blade_stl_gen.py --no-stl   # visualisation only
"""

import sys, os, argparse
import numpy as np
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry.blade_generator import baseline_apc7x5e
from optimization.mdao_problem import N_CP

# ---------------------------------------------------------------------------
# LE serration optimum from printer-constrained multistart
# ---------------------------------------------------------------------------
H_LE_CP_MM = np.array([2.51, 3.67, 3.39, 2.37, 0.80])   # mm, at 5 B-spline CPs
H_LE_CP_M  = H_LE_CP_MM * 1e-3


# ---------------------------------------------------------------------------
# NACA 4412 airfoil profile (c = 1, LE at x=0, TE at x=1)
# ---------------------------------------------------------------------------

def naca4412(n=80):
    """
    Return (x, z_upper, z_lower) for NACA4412 at unit chord.
    Uses cosine spacing for LE resolution.
    """
    beta   = np.linspace(0, np.pi, n)
    x      = 0.5 * (1 - np.cos(beta))   # cosine-spaced [0,1]

    # Thickness (NACA symmetric formula)
    t = 0.12
    yt = 5*t * (0.2969*np.sqrt(x) - 0.1260*x
                - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    # Camber (NACA4412: m=0.04, p=0.4)
    m, p = 0.04, 0.40
    yc      = np.where(x < p,
                       m/p**2     * (2*p*x - x**2),
                       m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
    dyc_dx  = np.where(x < p,
                       2*m/p**2     * (p - x),
                       2*m/(1-p)**2 * (p - x))
    theta   = np.arctan(dyc_dx)

    z_upper = yc + yt * np.cos(theta)
    z_lower = yc - yt * np.cos(theta)

    return x, z_upper, z_lower


def _section_3d(r_m, chord_m, twist_deg, sweep_m, z_off_m,
                x_foil, z_upper, z_lower, le_shift_m=0.0):
    """
    Transform a 2D NACA4412 section to 3D blade coordinates.

    Coordinate system:
      X  -- streamwise (positive forward / into oncoming flow)
      Y  -- spanwise   (positive outward)
      Z  -- axial      (positive upstream / lift direction)

    le_shift_m: forward shift of the LE point for sawtooth serration [m].
    """
    n = len(x_foil)

    # Scale chord; shift so LE at x=0, TE at x=chord
    xc = x_foil * chord_m          # [0 .. chord]
    zu = z_upper * chord_m
    zl = z_lower * chord_m

    # Apply LE sawtooth shift: taper the shift over 20% chord
    blend = np.maximum(1.0 - xc / (0.2 * chord_m), 0.0)
    xc_u  = xc - le_shift_m * blend
    xc_l  = xc - le_shift_m * blend

    # Rotate by twist (pitch) about spanwise axis (Y)
    tw   = np.deg2rad(twist_deg)
    cos_tw, sin_tw = np.cos(tw), np.sin(tw)

    def rotate(xc_arr, z_arr):
        X =  xc_arr * cos_tw + z_arr * sin_tw
        Z = -xc_arr * sin_tw + z_arr * cos_tw
        return X, Z

    Xu, Zu = rotate(xc_u, zu)
    Xl, Zl = rotate(xc_l, zl)

    # Add sweep (x-offset) and z-dihedral
    Xu += sweep_m;  Xl += sweep_m
    Zu += z_off_m;  Zl += z_off_m

    # Y = span radius
    Yu = np.full(n, r_m)
    Yl = np.full(n, r_m)

    # upper: LE→TE,  lower: TE→LE  (consistent winding for surface normals)
    pts_upper = np.column_stack([Xu, Yu, Zu])
    pts_lower = np.column_stack([Xl[::-1], Yl[::-1], Zl[::-1]])

    return pts_upper, pts_lower


def build_surface(blade, h_LE_m_func=None, n_span=200, n_chord=60):
    """
    Build upper and lower surface point arrays.

    h_LE_m_func(r_R) -> forward LE shift in metres (or None for baseline).
    Returns (surf_upper, surf_lower) each (n_span, n_chord, 3).
    """
    x_foil, z_u, z_l = naca4412(n_chord)

    r_R_arr = np.linspace(blade.r_R[0], blade.r_R[-1], n_span)
    R       = blade.radius_m

    surf_u = np.zeros((n_span, n_chord, 3))
    surf_l = np.zeros((n_span, n_chord, 3))

    for i, r_R in enumerate(r_R_arr):
        r_m   = r_R * R
        chord = float(np.interp(r_R, blade.r_R, blade.chord_R)) * R
        twist = float(np.interp(r_R, blade.r_R, blade.twist_deg))
        sweep = float(np.interp(r_R, blade.r_R, blade.sweep_R)) * R
        z_off = float(np.interp(r_R, blade.r_R, blade.z_offset_R)) * R

        if h_LE_m_func is not None:
            le_shift = float(h_LE_m_func(r_R)) * _sawtooth_phase(r_R, blade, h_LE_m_func)
        else:
            le_shift = 0.0

        pu, pl = _section_3d(r_m, chord, twist, sweep, z_off,
                             x_foil, z_u, z_l, le_shift_m=le_shift)
        surf_u[i] = pu
        surf_l[i] = pl

    return surf_u, surf_l


# pre-integrate the sawtooth phase for consistent tooth count along span
_phase_cache = {}

def _sawtooth_phase(r_R, blade, h_func, n_steps=1000):
    """
    Returns sawtooth value in [-1, 1] at r_R based on accumulated wavelength.
    Wavelength lambda = 2 * h_LE (Lyu 2016 optimal h/lambda = 0.5).
    """
    key = id(h_func)
    if key not in _phase_cache:
        r_arr  = np.linspace(blade.r_R[0], blade.r_R[-1], n_steps)
        h_arr  = np.array([max(h_func(r), 1e-6) for r in r_arr])
        lam    = 2.0 * h_arr
        dphase = np.gradient(r_arr) / (lam * blade.radius_m)
        phase  = np.cumsum(dphase)
        _phase_cache[key] = (r_arr, phase)
    r_arr, phase = _phase_cache[key]
    ph = float(np.interp(r_R, r_arr, phase))
    # Triangular wave: 0 at valleys, 1 at peaks
    frac = ph - np.floor(ph)
    return 1.0 - 2.0 * abs(frac - 0.5)   # in [-1, 1]; we use abs for [0,1] teeth
    # Use absolute value so teeth always extend FORWARD (not backward)


def _sawtooth_phase(r_R, blade, h_func, n_steps=1000):
    key = id(h_func)
    if key not in _phase_cache:
        r_arr  = np.linspace(blade.r_R[0], blade.r_R[-1], n_steps)
        h_arr  = np.array([max(h_func(r), 1e-6) for r in r_arr])
        lam    = 2.0 * h_arr
        dphase = np.gradient(r_arr) / (lam * blade.radius_m)
        phase  = np.cumsum(dphase)
        _phase_cache[key] = (r_arr, phase)
    r_arr, phase = _phase_cache[key]
    ph = float(np.interp(r_R, r_arr, phase))
    frac = ph - np.floor(ph)
    # Triangular wave in [0,1]: 1 at tips, 0 at valleys
    return 1.0 - 2.0 * abs(frac - 0.5)


def make_h_LE_func(blade, h_LE_cp):
    """Return a function r_R -> h_LE(r_R) [m] using B-spline interpolation."""
    r_cp = np.linspace(blade.r_R[0], blade.r_R[-1], N_CP)
    cs   = CubicSpline(r_cp, h_LE_cp)
    def h_func(r_R):
        return float(np.clip(cs(r_R), 0.0, None))
    return h_func


# ---------------------------------------------------------------------------
# STL export (binary format)
# ---------------------------------------------------------------------------

def _triangulate(surf_u, surf_l):
    """Quad-mesh triangulation of upper + lower surfaces + LE + TE caps."""
    tris = []

    def add_quad(p00, p10, p01, p11):
        n = np.cross(p10 - p00, p01 - p00)
        if np.linalg.norm(n) > 0:
            n /= np.linalg.norm(n)
        tris.append((n, p00, p10, p01))
        tris.append((n, p10, p11, p01))

    ns, nc = surf_u.shape[:2]
    for i in range(ns - 1):
        for j in range(nc - 1):
            add_quad(surf_u[i, j], surf_u[i+1, j],
                     surf_u[i, j+1], surf_u[i+1, j+1])
            add_quad(surf_l[i, j], surf_l[i, j+1],
                     surf_l[i+1, j], surf_l[i+1, j+1])

    # LE cap (j=0 of upper = LE; lower reversed, so j=nc-1 of lower = LE)
    for i in range(ns - 1):
        p1 = surf_u[i,   0];  p2 = surf_u[i+1, 0]
        p3 = surf_l[i,  -1];  p4 = surf_l[i+1, -1]
        add_quad(p1, p2, p3, p4)

    # TE cap (j=nc-1 upper, j=0 lower)
    for i in range(ns - 1):
        p1 = surf_u[i,  -1];  p2 = surf_u[i+1, -1]
        p3 = surf_l[i,   0];  p4 = surf_l[i+1,  0]
        add_quad(p3, p4, p1, p2)

    # Root and tip caps (flat)
    for j in range(nc - 1):
        add_quad(surf_u[0, j], surf_u[0, j+1],
                 surf_l[0, -j-1], surf_l[0, -j-2])
        add_quad(surf_u[-1, j+1], surf_u[-1, j],
                 surf_l[-1, -j-2], surf_l[-1, -j-1])

    return tris


def write_stl(tris, path):
    """Write binary STL."""
    with open(path, "wb") as f:
        f.write(b" " * 80)                     # header
        f.write(struct.pack("<I", len(tris)))   # triangle count
        for (n, v1, v2, v3) in tris:
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<fff", *v3))
            f.write(struct.pack("<H", 0))       # attribute byte count
    print(f"[STL] {len(tris)} triangles -> {path}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _fake_lighting(verts_list, light_dir=None):
    if light_dir is None:
        light_dir = np.array([0.5, -0.3, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shades = []
    for quad in verts_list:
        v = np.array(quad)
        n = np.cross(v[1]-v[0], v[2]-v[0])
        nn = np.linalg.norm(n)
        if nn > 0:
            n /= nn
        shade = 0.35 + 0.65 * max(0.0, float(np.dot(n, light_dir)))
        shades.append(shade)
    return shades


def render_comparison(surf_u_base, surf_l_base, surf_u_ser, surf_l_ser, out_path):
    BG    = "#0d0d1a"
    PANEL = "#12122a"

    fig = plt.figure(figsize=(18, 8), facecolor=BG)
    titles    = ["Baseline APC 7x5E  (73.45 dBA)", "LE Serrated  (71.32 dBA,  -3.12 dBA)"]
    surfs     = [(surf_u_base, surf_l_base), (surf_u_ser, surf_l_ser)]
    base_cols = [np.array([0.29, 0.56, 0.85]), np.array([0.31, 0.98, 0.48])]

    for col, (title, (su, sl), bc) in enumerate(zip(titles, surfs, base_cols)):
        ax = fig.add_subplot(1, 2, col+1, projection="3d")
        ax.set_facecolor(PANEL)
        ax.set_title(title, color="white", fontsize=12, pad=10, fontweight="bold")

        step_s = max(1, su.shape[0] // 80)
        step_c = max(1, su.shape[1] // 40)
        su_d, sl_d = su[::step_s, ::step_c], sl[::step_s, ::step_c]
        ns, nc = su_d.shape[:2]

        verts_u, verts_l = [], []
        for i in range(ns - 1):
            for j in range(nc - 1):
                verts_u.append([su_d[i,j], su_d[i+1,j], su_d[i+1,j+1], su_d[i,j+1]])
                verts_l.append([sl_d[i,j], sl_d[i+1,j], sl_d[i+1,j+1], sl_d[i,j+1]])

        light = np.array([0.4, 0.2, 1.0])
        fc_u = [(*np.clip(bc * s, 0, 1), 0.95) for s in _fake_lighting(verts_u, light)]
        fc_l = [(*np.clip(bc * s * 0.7, 0, 1), 0.80) for s in _fake_lighting(verts_l, light)]

        ax.add_collection3d(Poly3DCollection(verts_u, linewidth=0, facecolor=fc_u))
        ax.add_collection3d(Poly3DCollection(verts_l, linewidth=0, facecolor=fc_l))

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("#1a1a3a")
        ax.tick_params(colors="#888899", labelsize=7)
        ax.set_xlabel("streamwise [m]", color="#aaaacc", fontsize=8, labelpad=4)
        ax.set_ylabel("span [m]",       color="#aaaacc", fontsize=8, labelpad=4)
        ax.set_zlabel("axial [m]",      color="#aaaacc", fontsize=8, labelpad=4)

        all_pts = np.concatenate([su.reshape(-1,3), sl.reshape(-1,3)])
        for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            lo, hi = all_pts[:,dim].min(), all_pts[:,dim].max()
            pad = (hi - lo) * 0.05
            setter(lo - pad, hi + pad)
        ax.view_init(elev=28, azim=-50)

    # LE profile inset
    blade  = baseline_apc7x5e()
    h_func = make_h_LE_func(blade, H_LE_CP_M)
    r_fine = np.linspace(blade.r_R[0], blade.r_R[-1], 600)
    R      = blade.radius_m
    h_spl  = np.array([h_func(r) for r in r_fine])
    h_arr  = np.maximum(h_spl, 1e-6)
    dphase = np.gradient(r_fine) / (2.0 * h_arr * R)
    phase  = np.cumsum(dphase)
    frac   = phase - np.floor(phase)
    tooth  = 1.0 - 2.0 * np.abs(frac - 0.5)
    le_off = h_spl * tooth * 1000

    ax2 = fig.add_axes([0.395, 0.56, 0.21, 0.30], facecolor=PANEL)
    ax2.fill_between(r_fine * R * 1000, 0, le_off, color="#50fa7b", alpha=0.30)
    ax2.plot(r_fine * R * 1000, le_off,           color="#50fa7b", lw=1.0)
    ax2.plot(r_fine * R * 1000, h_spl * 1000, "--", color="#f39c12", lw=1.0,
             label="h_LE envelope")
    ax2.set_xlabel("Span [mm]",         color="white", fontsize=7)
    ax2.set_ylabel("LE offset [mm]",    color="white", fontsize=7)
    ax2.set_title("Sawtooth LE profile", color="white", fontsize=8)
    ax2.tick_params(colors="white", labelsize=6)
    ax2.legend(facecolor=PANEL, labelcolor="white", edgecolor="#333355",
               fontsize=6, loc="upper right")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333355")
    n_teeth = int(round(phase[-1]))
    ax2.text(0.05, 0.88, f"~{n_teeth} teeth total", transform=ax2.transAxes,
             color="#f39c12", fontsize=7)

    fig.suptitle("APC 7x5E  |  LE Sawtooth Serrations  |  Elegoo Saturn 4 Ultra 16K",
                 color="white", fontsize=10, y=0.99)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[VIZ] Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stl",   action="store_true", help="Skip STL export")
    parser.add_argument("--n-span",   type=int, default=250, help="Spanwise resolution")
    parser.add_argument("--n-chord",  type=int, default=80,  help="Chordwise resolution")
    args = parser.parse_args()

    blade  = baseline_apc7x5e()
    h_func = make_h_LE_func(blade, H_LE_CP_M)

    print(f"Building baseline surface ({args.n_span} span x {args.n_chord} chord)...")
    surf_u_base, surf_l_base = build_surface(blade, None,   args.n_span, args.n_chord)

    print("Building LE-serrated surface...")
    surf_u_ser,  surf_l_ser  = build_surface(blade, h_func, args.n_span, args.n_chord)

    out_dir = os.path.join(os.path.dirname(__file__), "stl")
    os.makedirs(out_dir, exist_ok=True)

    if not args.no_stl:
        print("Triangulating...")
        tris_base = _triangulate(surf_u_base, surf_l_base)
        tris_ser  = _triangulate(surf_u_ser,  surf_l_ser)
        write_stl(tris_base, os.path.join(out_dir, "blade_baseline_new.stl"))
        write_stl(tris_ser,  os.path.join(out_dir, "blade_le_serrated.stl"))

    print("Rendering comparison...")
    plot_path = os.path.join(os.path.dirname(__file__), "plots", "blade_le_serrated.png")
    render_comparison(surf_u_base, surf_l_base, surf_u_ser, surf_l_ser, plot_path)


if __name__ == "__main__":
    main()
