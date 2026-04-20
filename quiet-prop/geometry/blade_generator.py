"""
Parametric blade geometry for quiet-prop.
Defines blade stations (r/R, chord, twist) for the HQProp 7x4x3 baseline
and provides a class for perturbing geometry during optimization.
"""

import numpy as np


# ---------------------------------------------------------------------------
# HQProp 7x4x3 baseline geometry
# Diameter = 7 in (0.1778 m), pitch = 4 in, 3 blades
# Station data: r/R, chord (m), twist (deg)
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
}


class BladeGeometry:
    """
    Parametric blade geometry.

    Parameters
    ----------
    diameter_m : float
        Tip-to-tip diameter in metres.
    num_blades : int
        Number of blades.
    r_R : array (N,)
        Non-dimensional radial stations r/R.
    chord_R : array (N,)
        Chord length normalised by radius (c/R).
    twist_deg : array (N,)
        Geometric twist at each station (degrees, nose-up positive).
    airfoil : str
        Airfoil identifier.
    """

    def __init__(self, diameter_m, num_blades, r_R, chord_R, twist_deg,
                 airfoil="NACA4412"):
        self.diameter_m = float(diameter_m)
        self.radius_m = diameter_m / 2.0
        self.num_blades = int(num_blades)
        self.r_R = np.asarray(r_R, dtype=float)
        self.chord_R = np.asarray(chord_R, dtype=float)
        self.twist_deg = np.asarray(twist_deg, dtype=float)
        self.airfoil = airfoil

    @property
    def r_m(self):
        return self.r_R * self.radius_m

    @property
    def chord_m(self):
        return self.chord_R * self.radius_m

    @property
    def twist_rad(self):
        return np.deg2rad(self.twist_deg)

    def get_stations(self, n_stations=None):
        """Return (r_m, chord_m, twist_deg), optionally resampled."""
        if n_stations is None:
            return self.r_m.copy(), self.chord_m.copy(), self.twist_deg.copy()
        r_R_new = np.linspace(self.r_R[0], self.r_R[-1], n_stations)
        chord_R_new = np.interp(r_R_new, self.r_R, self.chord_R)
        twist_new = np.interp(r_R_new, self.r_R, self.twist_deg)
        return (r_R_new * self.radius_m,
                chord_R_new * self.radius_m,
                twist_new)

    def perturb_chord(self, delta_chord_R):
        return BladeGeometry(self.diameter_m, self.num_blades, self.r_R,
                             self.chord_R + np.asarray(delta_chord_R),
                             self.twist_deg, self.airfoil)

    def perturb_twist(self, delta_twist_deg):
        return BladeGeometry(self.diameter_m, self.num_blades, self.r_R,
                             self.chord_R,
                             self.twist_deg + np.asarray(delta_twist_deg),
                             self.airfoil)

    def summary(self):
        print(f"Blade: {self.airfoil}  D={self.diameter_m*100:.1f} cm  "
              f"B={self.num_blades}")
        print(f"  {'r/R':>6}  {'chord(mm)':>10}  {'twist(deg)':>10}")
        for r, c, t in zip(self.r_R, self.chord_m * 1000, self.twist_deg):
            print(f"  {r:6.3f}  {c:10.2f}  {t:10.2f}")


def baseline_hqprop():
    """Return a BladeGeometry for the HQProp 7x4x3 baseline."""
    g = HQPROP_7x4x3
    return BladeGeometry(
        diameter_m=g["diameter_m"],
        num_blades=g["num_blades"],
        r_R=g["r_R"],
        chord_R=g["chord_R"],
        twist_deg=g["twist_deg"],
        airfoil=g["airfoil"],
    )


if __name__ == "__main__":
    blade = baseline_hqprop()
    blade.summary()
