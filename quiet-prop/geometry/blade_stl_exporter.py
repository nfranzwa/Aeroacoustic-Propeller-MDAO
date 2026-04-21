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
    blade = baseline_hqprop()
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


def _rotate_pts_y(pts, angle_deg):
    """Rotate (N,3) point array around the Y axis (radial axis = blade rotation plane)."""
    a   = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    R = np.array([[cos_a, 0, sin_a],
                  [0,     1, 0     ],
                  [-sin_a,0, cos_a ]])
    return pts @ R.T


def _build_blade_sections(blade, n_sections=18, n_airfoil=40):
    """Return list of (n_af, 3) arrays, one per spanwise section."""
    r_m, chord_m, twist_deg, tc, sweep_m, z_off = blade.get_full_stations(n_sections)
    sections = []
    for i in range(n_sections):
        pts = section_3d_points(chord_m[i], twist_deg[i], sweep_m[i],
                                z_off[i], r_m[i], tc[i], n_airfoil)
        sections.append(np.array(pts))
    return sections


def export_blade_stl(blade, output_path, n_sections=18, n_airfoil=40):
    """
    Generate and export a single-blade STL file (pure Python, no CadQuery).

    Parameters
    ----------
    blade       : BladeGeometry
    output_path : str   Path to output .stl file
    n_sections  : int   Spanwise cross-sections
    n_airfoil   : int   Points per airfoil surface

    Returns True always (pure-Python fallback).
    """
    print(f"[STL] Building blade ({n_sections} sections x {n_airfoil} pts)...")
    sections   = _build_blade_sections(blade, n_sections, n_airfoil)
    triangles  = _triangulate_surface(sections)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _write_stl_triangles(triangles, output_path, "blade")
    print(f"[STL] Single blade -> {output_path}  ({len(triangles)} triangles)")
    return True


def export_rotor_stl(blade, output_path, n_sections=18, n_airfoil=40):
    """
    Generate and export a full 3-blade rotor assembly STL (pure Python).

    Each blade is replicated and rotated around the Y axis (radial axis)
    by its azimuthal angle. All blades are written into one STL file as
    separate solids.

    Parameters
    ----------
    blade       : BladeGeometry
    output_path : str   Path to output .stl file
    """
    print(f"[STL] Building rotor ({blade.num_blades} blades, {n_sections} sections)...")
    base_sections = _build_blade_sections(blade, n_sections, n_airfoil)

    all_triangles = []
    for b_idx, angle in enumerate(blade.blade_angles_deg):
        if abs(angle) < 0.01:
            rotated_sections = base_sections
        else:
            rotated_sections = [_rotate_pts_y(sec, angle) for sec in base_sections]
        tris = _triangulate_surface(rotated_sections)
        all_triangles.extend(tris)

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
    from geometry.blade_generator import baseline_hqprop

    blade = baseline_hqprop()

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "results", "stl")

    # Single blade
    export_blade_stl(blade,
                     os.path.join(out_dir, "blade_baseline_single.stl"))

    # Full rotor – equal spacing baseline
    export_rotor_stl(blade,
                     os.path.join(out_dir, "rotor_baseline_equal.stl"))

    # Full rotor – unequal spacing example
    blade_unequal = blade.set_blade_angles([0.0, 115.0, 235.0])
    export_rotor_stl(blade_unequal,
                     os.path.join(out_dir, "rotor_baseline_unequal.stl"))
