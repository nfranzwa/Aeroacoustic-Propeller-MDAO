"""
Parametric blade geometry for quiet-prop.
Defines blade stations (r/R, chord, twist, sweep, t/c, dihedral) for the
HQProp 7x4x3 baseline. Supports 3-blade assemblies with arbitrary azimuthal
spacing (unequal blade indexing for BPF coherence reduction).
"""

import numpy as np


# ---------------------------------------------------------------------------
# HQProp 7x4x3 baseline geometry
# ---------------------------------------------------------------------------

HQPROP_7x4x3 = {
    "name": "HQProp 7x4x3",
    "diameter_m": 7 * 0.0254,       # 0.1778 m
    "num_blades": 3,
    "airfoil": "NACA4412",
    "r_R": np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                     0.85, 0.90, 0.95, 1.00]),
    "chord_R": np.array([0.130, 0.149, 0.163, 0.172, 0.178, 0.180, 0.179,
                         0.175, 0.169, 0.161, 0.152, 0.141, 0.129, 0.116,
                         0.102, 0.087, 0.067, 0.040]),
    "twist_deg": np.array([34.0, 30.2, 27.0, 24.3, 22.0, 20.0, 18.3,
                           16.8, 15.5, 14.3, 13.2, 12.2, 11.3, 10.5,
                            9.7,  9.0,  8.3,  7.8]),
    # NACA 4412 baseline: 12% from root, thinning toward tip
    "tc_ratio": np.array([0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.119,
                          0.118, 0.117, 0.115, 0.113, 0.111, 0.108, 0.105,
                          0.100, 0.095, 0.090, 0.080]),
    # No sweep in baseline (HQProp is straight)
    "sweep_R": np.zeros(18),
    # No dihedral in baseline
    "z_offset_R": np.zeros(18),
    # Equal azimuthal spacing for 3 blades
    "blade_angles_deg": np.array([0.0, 120.0, 240.0]),
}


class BladeGeometry:
    """
    Parametric blade geometry with full 3D parameterization.

    Parameters
    ----------
    diameter_m       : float   Tip-to-tip diameter (m).
    num_blades       : int     Number of blades.
    r_R              : (N,)    Non-dimensional radial stations r/R.
    chord_R          : (N,)    Chord length / radius.
    twist_deg        : (N,)    Geometric twist (deg, nose-up positive).
    tc_ratio         : (N,)    Thickness-to-chord ratio at each station.
    sweep_R          : (N,)    Aft-sweep x-offset / radius (positive = aft).
    z_offset_R       : (N,)    Dihedral/anhedral z-offset / radius.
    blade_angles_deg : (B,)    Azimuthal position of each blade (deg).
    airfoil          : str     Airfoil family identifier.
    """

    def __init__(self, diameter_m, num_blades, r_R, chord_R, twist_deg,
                 tc_ratio=None, sweep_R=None, z_offset_R=None,
                 blade_angles_deg=None, airfoil="NACA4412"):
        self.diameter_m = float(diameter_m)
        self.radius_m   = diameter_m / 2.0
        self.num_blades = int(num_blades)
        self.r_R        = np.asarray(r_R,       dtype=float)
        self.chord_R    = np.asarray(chord_R,   dtype=float)
        self.twist_deg  = np.asarray(twist_deg, dtype=float)
        self.airfoil    = airfoil
        N = len(self.r_R)
        self.tc_ratio       = np.asarray(tc_ratio,         dtype=float) if tc_ratio         is not None else np.full(N, 0.12)
        self.sweep_R        = np.asarray(sweep_R,          dtype=float) if sweep_R          is not None else np.zeros(N)
        self.z_offset_R     = np.asarray(z_offset_R,       dtype=float) if z_offset_R       is not None else np.zeros(N)
        self.blade_angles_deg = (np.asarray(blade_angles_deg, dtype=float)
                                 if blade_angles_deg is not None
                                 else np.linspace(0.0, 360.0, num_blades, endpoint=False))

    # ---- Derived properties ------------------------------------------------

    @property
    def r_m(self):
        return self.r_R * self.radius_m

    @property
    def chord_m(self):
        return self.chord_R * self.radius_m

    @property
    def twist_rad(self):
        return np.deg2rad(self.twist_deg)

    @property
    def sweep_m(self):
        return self.sweep_R * self.radius_m

    @property
    def z_offset_m(self):
        return self.z_offset_R * self.radius_m

    # ---- Resampling ---------------------------------------------------------

    def get_stations(self, n_stations=None):
        """Return (r_m, chord_m, twist_deg) at n_stations resolution."""
        if n_stations is None:
            return self.r_m.copy(), self.chord_m.copy(), self.twist_deg.copy()
        r_R_new   = np.linspace(self.r_R[0], self.r_R[-1], n_stations)
        chord_R_n = np.interp(r_R_new, self.r_R, self.chord_R)
        twist_n   = np.interp(r_R_new, self.r_R, self.twist_deg)
        return (r_R_new * self.radius_m,
                chord_R_n * self.radius_m,
                twist_n)

    def get_full_stations(self, n_stations=None):
        """Return (r_m, chord_m, twist_deg, tc_ratio, sweep_m, z_offset_m)."""
        r_m, chord_m, twist_deg = self.get_stations(n_stations)
        if n_stations is None:
            return (r_m, chord_m, twist_deg,
                    self.tc_ratio.copy(), self.sweep_m.copy(), self.z_offset_m.copy())
        r_R_new = np.linspace(self.r_R[0], self.r_R[-1], n_stations)
        tc  = np.interp(r_R_new, self.r_R, self.tc_ratio)
        sw  = np.interp(r_R_new, self.r_R, self.sweep_R) * self.radius_m
        zof = np.interp(r_R_new, self.r_R, self.z_offset_R) * self.radius_m
        return r_m, chord_m, twist_deg, tc, sw, zof

    # ---- Perturbation -------------------------------------------------------

    def _copy_with(self, **kw):
        return BladeGeometry(
            diameter_m       = kw.get("diameter_m",       self.diameter_m),
            num_blades       = kw.get("num_blades",       self.num_blades),
            r_R              = kw.get("r_R",              self.r_R),
            chord_R          = kw.get("chord_R",          self.chord_R),
            twist_deg        = kw.get("twist_deg",        self.twist_deg),
            tc_ratio         = kw.get("tc_ratio",         self.tc_ratio),
            sweep_R          = kw.get("sweep_R",          self.sweep_R),
            z_offset_R       = kw.get("z_offset_R",       self.z_offset_R),
            blade_angles_deg = kw.get("blade_angles_deg", self.blade_angles_deg),
            airfoil          = self.airfoil,
        )

    def perturb_chord(self, delta_chord_R):
        c = np.clip(self.chord_R + np.asarray(delta_chord_R), 0.01, 0.30)
        return self._copy_with(chord_R=c)

    def perturb_twist(self, delta_twist_deg):
        return self._copy_with(twist_deg=self.twist_deg + np.asarray(delta_twist_deg))

    def perturb_tc(self, delta_tc):
        tc = np.clip(self.tc_ratio + np.asarray(delta_tc), 0.06, 0.20)
        return self._copy_with(tc_ratio=tc)

    def set_sweep(self, sweep_R_abs):
        """Set absolute aft-sweep distribution (0 = straight, positive = aft)."""
        return self._copy_with(sweep_R=np.clip(np.asarray(sweep_R_abs), 0.0, 0.15))

    def set_z_offset(self, z_offset_R_abs):
        """Set absolute dihedral/anhedral (positive = anhedral for tractor prop)."""
        return self._copy_with(z_offset_R=np.clip(np.asarray(z_offset_R_abs), -0.08, 0.12))

    def set_blade_angles(self, blade_angles_deg):
        return self._copy_with(blade_angles_deg=np.asarray(blade_angles_deg, dtype=float))

    # ---- Rotor balance ------------------------------------------------------

    def imbalance_factor(self):
        """
        Static imbalance of rotor assuming equal blade masses.
        Returns |Σ exp(j*theta_k)| / B.  0 = perfectly balanced, 1 = one blade only.
        """
        th = np.deg2rad(self.blade_angles_deg)
        vec = np.sum(np.exp(1j * th))
        return float(np.abs(vec)) / self.num_blades

    # ---- Diagnostics --------------------------------------------------------

    def summary(self):
        print(f"Blade: {self.airfoil}  D={self.diameter_m*100:.1f} cm  B={self.num_blades}")
        print(f"  Blade angles : {np.round(self.blade_angles_deg, 1)} deg")
        print(f"  Imbalance    : {self.imbalance_factor():.4f} (0=balanced, 1=single blade)")
        hdr = f"  {'r/R':>5}  {'c(mm)':>7}  {'b(deg)':>7}  {'t/c':>5}  {'sw(mm)':>7}  {'z(mm)':>6}"
        print(hdr)
        for r, c, t, tc, sw, z in zip(self.r_R,
                                       self.chord_m * 1000,
                                       self.twist_deg,
                                       self.tc_ratio,
                                       self.sweep_m * 1000,
                                       self.z_offset_m * 1000):
            print(f"  {r:5.2f}  {c:7.2f}  {t:7.2f}  {tc:5.3f}  {sw:7.2f}  {z:6.2f}")


def baseline_hqprop():
    """Return a BladeGeometry for the HQProp 7x4x3 baseline."""
    g = HQPROP_7x4x3
    return BladeGeometry(
        diameter_m       = g["diameter_m"],
        num_blades       = g["num_blades"],
        r_R              = g["r_R"],
        chord_R          = g["chord_R"],
        twist_deg        = g["twist_deg"],
        tc_ratio         = g["tc_ratio"],
        sweep_R          = g["sweep_R"],
        z_offset_R       = g["z_offset_R"],
        blade_angles_deg = g["blade_angles_deg"],
        airfoil          = g["airfoil"],
    )


if __name__ == "__main__":
    blade = baseline_hqprop()
    blade.summary()
    print(f"\nUnequal spacing example [0, 115, 235]:")
    unequal = blade.set_blade_angles([0.0, 115.0, 235.0])
    print(f"  Imbalance: {unequal.imbalance_factor():.4f}")
