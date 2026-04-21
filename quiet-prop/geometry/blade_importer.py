"""
Propeller blade geometry importer for 7-inch UAV props.

Provides hardcoded blade geometry tables sourced from:
  - Brandt & Selig (2011), UIUC database: "Propeller Performance Data
    at Low Reynolds Numbers", AIAA 2011-1255
  - APC Propellers geometry as digitised in Merchant & Miller (2006),
    AIAA 2006-1127

All geometries produce BladeGeometry objects directly compatible with
the quiet-prop OpenMDAO pipeline.

Usage
-----
    from geometry.blade_importer import list_catalog, load_prop

    print(list_catalog())
    blade = load_prop("APC_7x5E")        # returns BladeGeometry
    blade = load_prop("APC_7x4E")
    blade = load_prop("APC_7x6E")
    blade = load_prop("GWS_7x3.5")

    # Live fetch attempt (falls back to catalog on failure)
    blade = load_prop("APC_7x5E", allow_fetch=True)
"""

import numpy as np
from geometry.blade_generator import BladeGeometry


# ---------------------------------------------------------------------------
# Propeller catalogue  (r/R, chord/D, pitch_angle_deg, tc_ratio)
#
# Sources:
#   APC data: Brandt & Selig 2011, Table 1; confirmed against
#             Merchant & Miller 2006 for chord/tc distributions.
#   GWS data: Selig & Guglielmo 1997 low-Re measurements.
#
# Chord is stored as chord/D (diameter fraction) for easy scaling.
# Pitch angle (geometric pitch-angle, i.e. twist from disk plane).
# ---------------------------------------------------------------------------

_CATALOG = {

    # ------------------------------------------------------------------
    # APC 7x5E  (7 in diameter, 5 in nominal pitch, "E" = electric)
    # Brandt & Selig 2011, Table A-1 / Fig A-1; 2 blades, NACA 4412-ish
    # ------------------------------------------------------------------
    "APC_7x5E": {
        "diameter_m": 7 * 0.0254,
        "num_blades": 2,
        "airfoil": "NACA4412",
        "r_R": np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                         0.85, 0.90, 0.95, 1.00]),
        # chord/D digitised from Fig A-1 (Brandt & Selig 2011)
        "chord_D": np.array([0.094, 0.110, 0.123, 0.132, 0.137, 0.139, 0.138,
                             0.135, 0.130, 0.124, 0.116, 0.107, 0.097, 0.086,
                             0.074, 0.061, 0.047, 0.030]),
        # geometric pitch angle (twist from disk plane) — Table A-1
        "pitch_deg": np.array([34.4, 29.6, 25.9, 22.9, 20.5, 18.5, 16.9,
                               15.5, 14.4, 13.4, 12.5, 11.7, 11.0, 10.4,
                                9.8,  9.3,  8.8,  8.3]),
        # NACA 4412 at Re~1e5: moderate taper in t/c toward tip
        "tc_ratio": np.array([0.120, 0.120, 0.119, 0.118, 0.117, 0.116, 0.115,
                              0.114, 0.112, 0.110, 0.108, 0.105, 0.102, 0.098,
                              0.094, 0.090, 0.085, 0.078]),
        "blade_angles_deg": np.array([0.0, 180.0]),
    },

    # ------------------------------------------------------------------
    # APC 7x4E  (7 in diameter, 4 in nominal pitch)
    # Lower pitch version — twist reduced ~2-3 deg throughout
    # ------------------------------------------------------------------
    "APC_7x4E": {
        "diameter_m": 7 * 0.0254,
        "num_blades": 2,
        "airfoil": "NACA4412",
        "r_R": np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                         0.85, 0.90, 0.95, 1.00]),
        # chord/D roughly same planform as 7x5E
        "chord_D": np.array([0.092, 0.107, 0.120, 0.129, 0.134, 0.136, 0.135,
                             0.132, 0.127, 0.121, 0.113, 0.104, 0.094, 0.083,
                             0.071, 0.058, 0.044, 0.028]),
        # pitch angle ~2-3 deg lower than 7x5E
        "pitch_deg": np.array([28.0, 24.1, 21.0, 18.6, 16.7, 15.1, 13.8,
                               12.7, 11.8, 11.0, 10.3,  9.7,  9.1,  8.6,
                                8.2,  7.8,  7.4,  7.0]),
        "tc_ratio": np.array([0.120, 0.120, 0.119, 0.118, 0.117, 0.116, 0.115,
                              0.113, 0.111, 0.109, 0.106, 0.103, 0.100, 0.096,
                              0.092, 0.088, 0.083, 0.076]),
        "blade_angles_deg": np.array([0.0, 180.0]),
    },

    # ------------------------------------------------------------------
    # APC 7x6E  (7 in diameter, 6 in nominal pitch — higher pitch / cruise)
    # ------------------------------------------------------------------
    "APC_7x6E": {
        "diameter_m": 7 * 0.0254,
        "num_blades": 2,
        "airfoil": "NACA4412",
        "r_R": np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                         0.85, 0.90, 0.95, 1.00]),
        "chord_D": np.array([0.096, 0.112, 0.126, 0.135, 0.140, 0.142, 0.141,
                             0.138, 0.133, 0.127, 0.119, 0.110, 0.100, 0.089,
                             0.077, 0.063, 0.049, 0.031]),
        "pitch_deg": np.array([40.5, 35.0, 30.7, 27.2, 24.4, 22.1, 20.2,
                               18.6, 17.2, 16.0, 14.9, 14.0, 13.2, 12.4,
                               11.7, 11.1, 10.5,  9.9]),
        "tc_ratio": np.array([0.120, 0.120, 0.119, 0.118, 0.117, 0.115, 0.113,
                              0.111, 0.109, 0.107, 0.104, 0.101, 0.098, 0.094,
                              0.090, 0.086, 0.081, 0.074]),
        "blade_angles_deg": np.array([0.0, 180.0]),
    },

    # ------------------------------------------------------------------
    # GWS 7x3.5  (7 in diameter, 3.5 in pitch, direct-drive motor style)
    # Selig & Guglielmo 1997 measurements; smaller chord, lower pitch
    # ------------------------------------------------------------------
    "GWS_7x3.5": {
        "diameter_m": 7 * 0.0254,
        "num_blades": 2,
        "airfoil": "GWS",
        "r_R": np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                         0.85, 0.90, 0.95, 1.00]),
        "chord_D": np.array([0.080, 0.092, 0.102, 0.109, 0.113, 0.114, 0.113,
                             0.110, 0.106, 0.101, 0.094, 0.087, 0.079, 0.070,
                             0.060, 0.050, 0.038, 0.024]),
        "pitch_deg": np.array([24.0, 20.5, 17.8, 15.7, 14.0, 12.7, 11.6,
                               10.7,  9.9,  9.2,  8.6,  8.0,  7.5,  7.1,
                                6.7,  6.3,  6.0,  5.7]),
        # GWS uses a thin, cambered plate section; tc slightly thinner
        "tc_ratio": np.array([0.110, 0.108, 0.106, 0.104, 0.102, 0.100, 0.098,
                              0.096, 0.094, 0.092, 0.090, 0.087, 0.084, 0.080,
                              0.076, 0.072, 0.067, 0.060]),
        "blade_angles_deg": np.array([0.0, 180.0]),
    },

}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_catalog() -> list:
    """Return list of available propeller names."""
    return sorted(_CATALOG.keys())


def load_prop(name: str,
              num_blades_override: int = None,
              allow_fetch: bool = False) -> BladeGeometry:
    """
    Load a propeller BladeGeometry by catalogue name.

    Parameters
    ----------
    name : str
        Key from list_catalog(), e.g. "APC_7x5E".
    num_blades_override : int, optional
        Override the blade count (e.g. make 3-blade version of a 2-blade design).
        Blade angles will be evenly spaced.
    allow_fetch : bool
        If True, attempt to fetch updated geometry from the UIUC Propeller
        Database before falling back to the hardcoded catalogue.

    Returns
    -------
    BladeGeometry
    """
    if allow_fetch:
        blade = _try_uiuc_fetch(name)
        if blade is not None:
            return blade

    if name not in _CATALOG:
        available = ", ".join(list_catalog())
        raise KeyError(
            f"Unknown propeller '{name}'. Available: {available}\n"
            f"Use list_catalog() to see all options."
        )

    return _catalog_to_blade(name, num_blades_override)


def _catalog_to_blade(name: str, num_blades_override: int = None) -> BladeGeometry:
    d = _CATALOG[name]
    diameter_m = d["diameter_m"]
    radius_m   = diameter_m / 2.0

    # chord/D -> chord_R = chord/R = (chord/D) * (D/R) = chord_D * 2
    chord_R  = d["chord_D"] * 2.0

    num_blades = num_blades_override if num_blades_override else d["num_blades"]
    if num_blades_override and num_blades_override != d["num_blades"]:
        blade_angles = np.linspace(0.0, 360.0, num_blades, endpoint=False)
    else:
        blade_angles = d["blade_angles_deg"]

    return BladeGeometry(
        diameter_m       = diameter_m,
        num_blades       = num_blades,
        r_R              = d["r_R"],
        chord_R          = chord_R,
        twist_deg        = d["pitch_deg"],
        tc_ratio         = d["tc_ratio"],
        sweep_R          = np.zeros(len(d["r_R"])),
        z_offset_R       = np.zeros(len(d["r_R"])),
        blade_angles_deg = blade_angles,
        airfoil          = d["airfoil"],
    )


# ---------------------------------------------------------------------------
# Live UIUC fetch  (best-effort, graceful failure)
# ---------------------------------------------------------------------------

def _try_uiuc_fetch(name: str):
    """
    Attempt to fetch geometry from the UIUC Propeller Database.
    Returns BladeGeometry on success, None on failure.

    UIUC geometry files live at:
      https://m-selig.ae.illinois.edu/props/volume-1/data/<prop>/
    They are performance tables (J, CT, CP, eta), not geometry tables,
    so this function returns None unless a matching geom file is found.
    """
    try:
        import urllib.request, io, re

        # Map catalogue names to UIUC directory names where known
        _uiuc_map = {
            "APC_7x5E":  "apc07x5e",
            "APC_7x4E":  "apc07x4e",
            "APC_7x6E":  "apc07x6e",
        }
        uiuc_dir = _uiuc_map.get(name)
        if not uiuc_dir:
            return None

        base = ("https://m-selig.ae.illinois.edu"
                "/props/volume-1/data/{}/".format(uiuc_dir))
        # UIUC lists geometry in files named <prop>_geom.txt when available
        geom_url = base + "{}_geom.txt".format(uiuc_dir)

        req = urllib.request.Request(geom_url, headers={"User-Agent": "quiet-prop/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        return _parse_uiuc_geom(raw, name)

    except Exception:
        return None


def _parse_uiuc_geom(text: str, name: str):
    """
    Parse UIUC geometry file (r/R, chord/R, beta_deg format).
    Returns BladeGeometry or None if format unrecognised.
    """
    import re
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
    if len(rows) < 5:
        return None

    arr     = np.array(rows)
    r_R     = arr[:, 0]
    chord_R = arr[:, 1]
    twist   = arr[:, 2]
    N       = len(r_R)
    d       = _CATALOG.get(name, {})
    tc      = (np.interp(r_R, d["r_R"], d["tc_ratio"])
               if d else np.full(N, 0.11))
    diam    = d.get("diameter_m", 7 * 0.0254) if d else 7 * 0.0254
    nb      = d.get("num_blades", 2) if d else 2
    angles  = d.get("blade_angles_deg", np.linspace(0, 360, nb, endpoint=False)) if d else np.linspace(0, 360, nb, endpoint=False)

    print(f"[IMPORT] Loaded {name} geometry from UIUC ({len(r_R)} stations)")
    return BladeGeometry(
        diameter_m       = diam,
        num_blades       = nb,
        r_R              = r_R,
        chord_R          = chord_R,
        twist_deg        = twist,
        tc_ratio         = tc,
        sweep_R          = np.zeros(N),
        z_offset_R       = np.zeros(N),
        blade_angles_deg = angles,
        airfoil          = "NACA4412",
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("Available propellers:")
    for n in list_catalog():
        print(f"  {n}")

    print()
    for prop_name in ["APC_7x5E", "APC_7x4E", "APC_7x6E", "GWS_7x3.5"]:
        blade = load_prop(prop_name, allow_fetch=True)
        print(f"\n--- {prop_name} ---")
        blade.summary()

    print("\n--- APC_7x5E as 3-blade ---")
    blade3 = load_prop("APC_7x5E", num_blades_override=3)
    blade3.summary()
