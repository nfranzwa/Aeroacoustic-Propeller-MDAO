"""
Structural stress component for propeller blade.

Computes centrifugal root stress and bending root stress from thrust,
constraining against resin allowable stress.

Physics
-------
Centrifugal force at root:
    F_c = rho_mat * omega^2 * integral(r * A(r) dr)
    where A(r) = chord(r) * tc(r) * chord(r) = chord(r)^2 * tc(r)
    sigma_c = F_c / A_root

Bending moment at root (one blade, thrust load):
    M_b = T_blade * r_cg  where r_cg ~ 0.7*R
    Z_root = chord_root^3 * tc_root^2 / 6   (rectangular approx)
    sigma_b = M_b / Z_root

Max stress: sigma_max = sigma_c + sigma_b
Min physical thickness: min(chord(r) * tc(r)) across span

Material: Siraya Tech Blu Tough (MSLA consumer resin)
-------------------------------------------------------
  UTS            : 50 MPa  (manufacturer TDS)
  Elongation     : 32 %    (sufficient for bending without brittle fracture)
  Flexural mod.  : 1.75 GPa
  HDT            : 70 °C   (adequate for motor-heated air in hover)
  Density        : 1100 kg/m³
  Available      : Amazon / siraya.tech (~$35-46/kg)

Factor of safety = 3.5 (cyclic/fatigue per FAA AC 35.37-1B guidance for
rotating propeller blades under combined centrifugal + bending loads).
Allowable = 50 / 3.5 = 14.3 MPa.
"""

import numpy as np
import openmdao.api as om

# Material: Siraya Tech Blu Tough (consumer MSLA resin)
RHO_RESIN        = 1100.0   # kg/m³  (cured density)
UTS              = 50e6     # Pa     (tensile ultimate strength, manufacturer TDS)
SAFETY_FACTOR    = 3.5      # fatigue/cyclic loading, FAA AC 35.37-1B
ALLOWABLE_STRESS = UTS / SAFETY_FACTOR   # 14.3 MPa

# Printer minimum wall: 0.5 mm
MIN_PRINT_THICKNESS = 0.7e-3   # m  (raised from 0.5 mm — thin walls under-extrude and delaminate)


def compute_stress(r_m, chord_m, tc_ratio, thrust, rpm, num_blades):
    """
    Compute blade root stress and minimum wall thickness.

    Parameters
    ----------
    r_m        : (N,)  Radial stations (m)
    chord_m    : (N,)  Chord at each station (m)
    tc_ratio   : (N,)  Thickness-to-chord ratio
    thrust     : float Total thrust (N)
    rpm        : float Rotational speed (rpm)
    num_blades : int   Number of blades

    Returns
    -------
    dict with max_stress (Pa), min_thickness_m (m)
    """
    omega     = rpm * 2 * np.pi / 60.0
    r_root    = r_m[0]
    chord_root = chord_m[0]
    tc_root   = tc_ratio[0]

    # Cross-sectional area A(r) = chord * thickness = chord^2 * tc
    A = chord_m ** 2 * tc_ratio

    # Centrifugal force at root (integrate r * A(r) dr from root to tip)
    integrand = r_m * A
    F_c = RHO_RESIN * omega ** 2 * np.trapezoid(integrand, r_m)
    A_root = max(A[0], 1e-10)
    sigma_c = F_c / A_root

    # Bending moment per blade from thrust
    T_blade = thrust / max(num_blades, 1)
    # Centroid of thrust distribution ~ 0.7 R
    r_cg = 0.7 * r_m[-1]
    M_b  = T_blade * (r_cg - r_root)

    # Section modulus at root (rectangular cross-section approximation)
    h_root = chord_root * tc_root             # thickness at root
    Z_root = chord_root * h_root ** 2 / 6.0  # = chord * (tc*chord)^2 / 6
    Z_root = max(Z_root, 1e-15)
    sigma_b = M_b / Z_root

    sigma_max = sigma_c + sigma_b

    # Minimum physical wall thickness across span
    thickness_m = chord_m * tc_ratio
    min_thickness = float(np.min(thickness_m))

    return {
        "max_stress":    float(sigma_max),
        "min_thickness": min_thickness,
        "sigma_c":       float(sigma_c),
        "sigma_b":       float(sigma_b),
    }


class StressComponent(om.ExplicitComponent):
    """
    OpenMDAO component: computes centrifugal + bending root stress.

    Outputs
    -------
    max_stress    : Pa   Root stress (centrifugal + bending); constrain <= 14.3 MPa
    min_thickness : m    Minimum wall thickness; constrain >= 0.5 mm
    """

    def initialize(self):
        self.options.declare("blade",      default=None)
        self.options.declare("n_stations", default=20)
        self.options.declare("num_blades", default=3)
        self.options.declare("fd_step",    default=1e-5)

    def setup(self):
        from geometry.blade_generator import baseline_apc7x5e
        blade = self.options["blade"] or baseline_apc7x5e()
        self._blade = blade
        N = self.options["n_stations"]
        _, chord0, _ = blade.get_stations(N)
        _, _, _, tc0, _, _ = blade.get_full_stations(N)

        self.add_input("chord_m",  val=chord0, units="m")
        self.add_input("tc_ratio", val=tc0)
        self.add_input("r_m",      val=np.zeros(N), units="m")
        self.add_input("thrust",   val=0.0,    units="N")
        self.add_input("rpm",      val=5000.0, units="rpm")

        self.add_output("max_stress",    val=0.0, units="Pa")
        self.add_output("min_thickness", val=0.0, units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=self.options["fd_step"])

    def compute(self, inputs, outputs):
        res = compute_stress(
            r_m=inputs["r_m"],
            chord_m=inputs["chord_m"],
            tc_ratio=inputs["tc_ratio"],
            thrust=float(inputs["thrust"][0]),
            rpm=float(inputs["rpm"][0]),
            num_blades=self.options["num_blades"],
        )
        outputs["max_stress"]    = res["max_stress"]
        outputs["min_thickness"] = res["min_thickness"]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from geometry.blade_generator import baseline_apc7x5e

    blade = baseline_apc7x5e()
    r_m, chord_m, _ = blade.get_stations(20)
    _, _, _, tc, _, _ = blade.get_full_stations(20)

    res = compute_stress(r_m, chord_m, tc,
                         thrust=1.12, rpm=5000, num_blades=3)
    print(f"sigma_centrifugal : {res['sigma_c']/1e6:.2f} MPa")
    print(f"sigma_bending     : {res['sigma_b']/1e6:.2f} MPa")
    print(f"sigma_max         : {res['max_stress']/1e6:.2f} MPa")
    print(f"min_thickness     : {res['min_thickness']*1e3:.2f} mm")
    print(f"allowable         : {ALLOWABLE_STRESS/1e6:.1f} MPa")
    print(f"margin            : {(ALLOWABLE_STRESS - res['max_stress'])/1e6:.2f} MPa")
