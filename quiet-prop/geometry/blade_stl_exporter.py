"""
3D blade STL exporter using CadQuery.

Generates a parametric 3D propeller blade with:
  - NACA 44XX airfoil cross-sections (t/c varies spanwise)
  - Chord, twist, sweep, and dihedral distribution from BladeGeometry
  - 3-blade assembly with arbitrary azimuthal spacing
  - STL export (single blade or full rotor)

Usage
-----
    from geometry.blade_stl_exporter import export_blade_stl, export_rotor_stl
    blade = baseline_apc7x5e()
    export_rotor_stl(blade, "rotor.stl", n_sections=18, n_airfoil=40)
"""

import numpy as np
import os


# ---------------------------------------------------------------------------
# NACA 4-digit airfoil coordinates
# ---------------------------------------------------------------------------

def naca44xx_coords(tc_ratio=0.12, n=50, close_te=True):
    """
    Generate NACA 44XX airfoil (4% camber at 40% chord, variable thickness).

    Parameters
    ----------
    tc_ratio : float  Thickness-to-chord ratio (e.g. 0.12 for NACA 4412)
    n        : int    Number of points per surface (total ~2n points)
    close_te : bool   Force zero thickness at TE

    Returns
    -------
    x, y : arrays of shape (2n-1,) — full closed contour, LE to TE upper,
           then TE to LE lower.  x in [0,1], y centred on camber line.
    """
    m, p, t = 0.04, 0.4, float(tc_ratio)

    # Cosine clustering for LE resolution
    beta = np.linspace(0.0, np.pi, n)
    x    = 0.5 * (1.0 - np.cos(beta))

    # Thickness distribution
    coef = [0.2969, -0.1260, -0.3516, 0.2843, -0.1036 if close_te else -0.1015]
    yt   = 5.0 * t * (coef[0]*np.sqrt(np.clip(x, 0, 1))
                      + coef[1]*x + coef[2]*x**2 + coef[3]*x**3 + coef[4]*x**4)

    # Camber line
    yc = np.where(x < p,
                  m / p**2 * (2*p*x - x**2),
                  m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2))

    # Camber slope -> surface normal angle
    dycdx = np.where(x < p,
                     2*m/p**2 * (p - x),
                     2*m/(1-p)**2 * (p - x))
    theta = np.arctan(dycdx)

    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine: upper LE->TE, then lower TE->LE (skip duplicate endpoints)
    x_full = np.concatenate([xu,       xl[-2:0:-1]])
    y_full = np.concatenate([yu,       yl[-2:0:-1]])

    return x_full, y_full


# ---------------------------------------------------------------------------
# 3D section construction
# ---------------------------------------------------------------------------

def section_3d_points(chord, twist_deg, sweep_x, z_offset, radius, tc_ratio,
                      n_airfoil=40):
    """
    Build 3D point cloud for one blade cross-section.

    Coordinate system
    -----------------
    - Radial direction: Y axis (hub at Y=0, tip at Y=R)
    - Chordwise:        X axis (positive = aft, negative = leading edge)
    - Thickness:        Z axis (positive = suction side up)

    Parameters
    ----------
    chord     : float  Chord length (m)
    twist_deg : float  Geometric twist, positive nose-up (deg)
    sweep_x   : float  Aft sweep offset of quarter-chord from Y axis (m)
    z_offset  : float  Dihedral z-offset (m)
    radius    : float  Radial position of this section (m)
    tc_ratio  : float  Thickness-to-chord ratio
    n_airfoil : int    Number of points per surface

    Returns
    -------
    pts : list of (x, y, z) tuples, length = 2*n_airfoil - 1
    """
    x_af, y_af = naca44xx_coords(tc_ratio=tc_ratio, n=n_airfoil)

    # Scale to physical chord; quarter-chord at x=0
    x = (x_af - 0.25) * chord   # chordwise (m), QC-centred
    z = y_af * chord             # thickness direction (m)

    # Apply twist: rotate around the radial (Y) axis
    tw  = np.deg2rad(twist_deg)
    x_t =  x * np.cos(tw) + z * np.sin(tw)
    z_t = -x * np.sin(tw) + z * np.cos(tw)

    # Add sweep (aft offset of QC) and dihedral
    x_3d = x_t + sweep_x
    y_3d = np.full_like(x_t, radius)
    z_3d = z_t + z_offset

    return list(zip(x_3d.tolist(), y_3d.tolist(), z_3d.tolist()))


# ---------------------------------------------------------------------------
# CadQuery loft
# ---------------------------------------------------------------------------

def _build_blade_solid(blade, n_sections=18, n_airfoil=40):
    """
    Build a CadQuery Solid for a single blade via loft.
    Returns the solid or None if CadQuery is unavailable.
    """
    try:
        import cadquery as cq
    except ImportError:
        return None

    r_m, chord_m, twist_deg, tc, sweep_m, z_off = blade.get_full_stations(n_sections)

    wires = []
    for i in range(n_sections):
        pts = section_3d_points(
            chord     = chord_m[i],
            twist_deg = twist_deg[i],
            sweep_x   = sweep_m[i],
            z_offset  = z_off[i],
            radius    = r_m[i],
            tc_ratio  = tc[i],
            n_airfoil = n_airfoil,
        )
        # Close the contour
        pts_closed = pts + [pts[0]]
        vectors    = [cq.Vector(*p) for p in pts_closed]
        wire       = cq.Wire.makePolygon(vectors)
        wires.append(wire)

    try:
        solid = cq.Solid.makeLoft(wires, ruled=False)
    except Exception:
        # Fall back to ruled loft if smooth loft fails
        solid = cq.Solid.makeLoft(wires, ruled=True)

    return solid


# ---------------------------------------------------------------------------
# Pure-Python STL writer (no CadQuery dependency)
# ---------------------------------------------------------------------------

def _tri_normal(p1, p2, p3):
    """Outward face normal via cross product."""
    a = np.array(p2) - np.array(p1)
    b = np.array(p3) - np.array(p1)
    n = np.cross(a, b)
    mag = np.linalg.norm(n)
    if mag < 1e-15:
        return np.zeros(3)
    return n / mag


def _write_stl_triangles(triangles, filepath, solid_name="blade"):
    """Write list of (p1,p2,p3) triangles to ASCII STL."""
    lines = [f"solid {solid_name}"]
    for p1, p2, p3 in triangles:
        n = _tri_normal(p1, p2, p3)
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for p in (p1, p2, p3):
            lines.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def _triangulate_surface(sections):
    """
    Triangulate a lofted surface from a list of cross-section point arrays.

    Each section is an array of shape (n_af, 3). Adjacent sections are
    connected by a strip of quads, each split into 2 triangles.
    Returns list of (p1, p2, p3) tuples.
    """
    triangles = []
    n_sec = len(sections)
    n_af  = len(sections[0])

    for i in range(n_sec - 1):
        sec0 = sections[i]
        sec1 = sections[i + 1]
        for j in range(n_af):
            j1 = (j + 1) % n_af
            p00 = sec0[j]
            p01 = sec0[j1]
            p10 = sec1[j]
            p11 = sec1[j1]
            # Two triangles per quad
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))

    # Root cap (flat fill using fan triangulation)
    root = sections[0]
    c    = root.mean(axis=0)
    for j in range(n_af):
        triangles.append((c, root[(j+1) % n_af], root[j]))

    # Tip cap
    tip = sections[-1]
    c   = tip.mean(axis=0)
    for j in range(n_af):
        triangles.append((c, tip[j], tip[(j+1) % n_af]))

    return triangles


def _to_drone_frame(pts):
    """
    Convert from blade-local frame to drone-prop world frame.

    Blade-local:  x=chordwise, y=radial (hub→tip), z=thickness (suction up)
    Drone-world:  X=radial (blade 0 at azimuth 0°),
                  Y=tangential/chord, Z=thrust (up)

    Swap x↔y so the blade lies flat in the XY (rotor disk) plane with
    thrust pointing up (+Z). Blade 0 extends along +X.
    """
    p = np.asarray(pts)
    return np.column_stack([p[:, 1],   # X = radial
                             p[:, 0],   # Y = chordwise (tangential)
                             p[:, 2]])  # Z = thickness (thrust axis)


def _rotate_pts_z(pts, angle_deg):
    """Rotate (N,3) point array around the Z axis (drone thrust axis)."""
    a = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    R = np.array([[ cos_a, -sin_a, 0],
                  [ sin_a,  cos_a, 0],
                  [ 0,      0,     1]])
    return np.asarray(pts) @ R.T


def _hub_cylinder(radius_m, hub_r_R, height_m=0.004, n_sides=32):
    """
    Generate triangles for a thin hub cylinder (cosmetic, not structural).
    radius_m  : blade tip radius (m)
    hub_r_R   : hub radius as fraction of tip radius
    height_m  : cylinder half-height (total height = 2*height)
    """
    r  = hub_r_R * radius_m
    h  = height_m
    th = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    top    = np.column_stack([r*np.cos(th), r*np.sin(th), np.full(n_sides,  h)])
    bottom = np.column_stack([r*np.cos(th), r*np.sin(th), np.full(n_sides, -h)])
    c_top  = np.array([0., 0.,  h])
    c_bot  = np.array([0., 0., -h])
    tris = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        # Side quad
        tris.append((bottom[i], top[i],    top[j]))
        tris.append((bottom[i], top[j],    bottom[j]))
        # Top cap
        tris.append((c_top, top[i],    top[j]))
        # Bottom cap
        tris.append((c_bot, bottom[j], bottom[i]))
    return tris


def _build_blade_sections(blade, n_sections=18, n_airfoil=40):
    """
    Return list of (n_af, 3) arrays in DRONE FRAME (XY=disk, Z=thrust).
    Blade 0 extends along +X at azimuth 0 deg.
    """
    r_m, chord_m, twist_deg, tc, sweep_m, z_off = blade.get_full_stations(n_sections)
    sections = []
    for i in range(n_sections):
        pts = np.array(section_3d_points(chord_m[i], twist_deg[i], sweep_m[i],
                                         z_off[i], r_m[i], tc[i], n_airfoil))
        sections.append(_to_drone_frame(pts))
    return sections


def export_blade_stl(blade, output_path, n_sections=18, n_airfoil=40,
                     add_hub=True):
    """
    Export a single drone-propeller blade STL.

    Coordinate system: rotor disk = XY plane, thrust = +Z.
    Blade extends along +X at azimuth 0 deg.
    """
    print(f"[STL] Building blade ({n_sections} sections x {n_airfoil} pts)...")
    sections  = _build_blade_sections(blade, n_sections, n_airfoil)
    triangles = _triangulate_surface(sections)
    if add_hub:
        triangles += _hub_cylinder(blade.radius_m, blade.r_R[0])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _write_stl_triangles(triangles, output_path, "blade")
    print(f"[STL] Single blade -> {output_path}  ({len(triangles)} triangles)")
    return True


def export_rotor_stl(blade, output_path, n_sections=18, n_airfoil=40,
                     add_hub=True):
    """
    Export a full drone-propeller rotor STL (all blades + hub).

    Coordinate system: rotor disk = XY plane, thrust = +Z.
    Blades are placed at azimuthal angles in blade.blade_angles_deg by
    rotating around Z (thrust axis).
    """
    print(f"[STL] Building rotor ({blade.num_blades} blades, {n_sections} sections)...")
    base_sections = _build_blade_sections(blade, n_sections, n_airfoil)

    all_triangles = []
    for angle in blade.blade_angles_deg:
        if abs(angle) < 0.01:
            secs = base_sections
        else:
            secs = [_rotate_pts_z(sec, angle) for sec in base_sections]
        all_triangles.extend(_triangulate_surface(secs))

    if add_hub:
        all_triangles += _hub_cylinder(blade.radius_m, blade.r_R[0])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _write_stl_triangles(all_triangles, output_path, "rotor")

    imb = blade.imbalance_factor()
    print(f"[STL] Rotor -> {output_path}  ({len(all_triangles)} triangles)")
    print(f"      Blade angles : {np.round(blade.blade_angles_deg, 1)} deg")
    print(f"      Imbalance    : {imb:.4f} (0=balanced)")
    return True


# ---------------------------------------------------------------------------
# CSV fallback (if CadQuery not available)
# ---------------------------------------------------------------------------

def _export_fallback_csv(blade, stl_path, n_sections, n_airfoil):
    """
    When CadQuery is unavailable, export blade station cross-sections as CSV.
    One row per point: section_idx, blade_idx, x_m, y_m (radius), z_m
    """
    csv_path = stl_path.replace(".stl", "_sections.csv")
    r_m, chord_m, twist_deg, tc, sweep_m, z_off = blade.get_full_stations(n_sections)
    rows = ["section,blade,x_m,radius_m,z_m"]
    for b_idx, angle in enumerate(blade.blade_angles_deg):
        ang_rad = np.deg2rad(angle)
        for i in range(n_sections):
            pts = section_3d_points(chord_m[i], twist_deg[i], sweep_m[i],
                                    z_off[i], r_m[i], tc[i], n_airfoil)
            for x, y_rad, z in pts:
                # Rotate blade by its azimuthal angle around Y axis
                x_rot = x * np.cos(ang_rad) + y_rad * np.sin(ang_rad)
                y_rot = -x * np.sin(ang_rad) + y_rad * np.cos(ang_rad)
                rows.append(f"{i},{b_idx},{x_rot:.6f},{y_rot:.6f},{z:.6f}")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("\n".join(rows))
    print(f"[STL] Fallback CSV (no CadQuery) -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from geometry.blade_generator import baseline_apc7x5e
    from geometry.blade_importer import load_prop

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "stl")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Remove stale STL files from previous runs ----
    stale = [
        "rotor_HQProp_7x4x3_3blade.stl",
        "rotor_baseline.stl",
        "rotor_baseline_equal.stl",
        "rotor_optimised.stl",
        "rotor_optimised_equal.stl",
        "blade_optimised_single.stl",
        "rotor_apc7x5e_2blade.stl",
        "rotor_APC_7x5E_2blade.stl",
        "rotor_APC_7x4E_2blade.stl",
        "rotor_APC_7x6E_2blade.stl",
        "rotor_GWS_7x3.5_2blade.stl",
        "blade_APC_7x5E_single.stl",
    ]
    for fname in stale:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            os.remove(path)
            print(f"[STL] Removed stale: {fname}")

    blade_base = baseline_apc7x5e()

    # ---- Baseline: APC 7x5E (3-blade, equal spacing) ----
    print("\n--- Baseline APC 7x5E (3-blade) ---")
    export_blade_stl(blade_base,
                     os.path.join(out_dir, "blade_baseline_single.stl"))
    export_rotor_stl(blade_base,
                     os.path.join(out_dir, "rotor_APC_7x5E_3blade.stl"))

    # ---- Pitch variants (3-blade, Brandt & Selig 2011) ----
    for name in ["APC_7x4E", "APC_7x6E"]:
        blade_v = load_prop(name, num_blades_override=3)
        print(f"\n--- {name} (3-blade) ---")
        export_rotor_stl(blade_v,
                         os.path.join(out_dir, f"rotor_{name}_3blade.stl"))

    # ---- Phase 2 optimised (9454 RPM, TWR 2.5, SPL_weighted 45.82 dBA) ----
    # DVs from successful SLSQP run (Exit mode 0)
    dtwist = np.array([-1.046, -5., -5., -4.684, -4.939, -4.936, -4.997,
                       -3.131, -1.126, 0.784, 2.638, 4.501, 5., 5., 5., 5., 5., -4.986])
    dchord = np.array([-0.03, -0.03, -0.03, -0.03, -0.03, 0.025, 0.025,
                       -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03,
                       -0.0125, -0.0091, -0.0014, -0.0123])
    dtc    = np.array([0.035]*17 + [0.04])

    blade_opt = (blade_base
                 .perturb_twist(dtwist)
                 .perturb_chord(dchord)
                 .perturb_tc(dtc)
                 .set_blade_angles([0.0, 105.0, 253.1]))

    print("\n--- Phase 2 optimised (0/105/253 deg) ---")
    export_blade_stl(blade_opt,
                     os.path.join(out_dir, "blade_optimised_phase2_single.stl"))
    export_rotor_stl(blade_opt,
                     os.path.join(out_dir, "rotor_optimised_phase2.stl"))

    print("\nAll STL files written to:", os.path.abspath(out_dir))
    print("Open any .stl in Windows 3D Viewer (double-click) to inspect.")
