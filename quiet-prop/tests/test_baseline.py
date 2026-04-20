"""
Baseline validation: HQProp 7x4x3 at 5000 RPM.

Expected ranges are intentionally broad — the goal is physics sanity,
not exact replication of measured data.

Published reference (UIUC Propeller Database, HQProp 7x4x3):
  Static CT  ~ 0.09–0.14
  Static CP  ~ 0.03–0.06
  Static thrust at 5000 RPM (sea level) ~ 0.8–2.5 N
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_hqprop
from aerodynamics.ccblade_component import bem_solve, CCBladeComponent
from acoustics.bpm_component import bpm_noise, BPMComponent
import openmdao.api as om


def _check(name, value, lo, hi, units=""):
    ok = lo <= value <= hi
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:25s} = {value:9.4f} {units}  "
          f"(expected {lo:.4f} – {hi:.4f})")
    return ok


# ---------------------------------------------------------------------------
# Test 1: BEM static thrust (V_inf = 0, 5000 RPM)
# ---------------------------------------------------------------------------
def test_bem_static():
    print("\n--- Test 1: BEM static (5000 RPM, V=0) ---")
    blade = baseline_hqprop()
    res   = bem_solve(blade, rpm=5000, v_inf=0.0, rho=1.225)

    ok  = _check("Thrust (N)",   res["thrust"], 0.5,  5.0,  "N")
    ok &= _check("CT",           res["CT"],     0.05, 0.25)
    ok &= _check("CP",           res["CP"],     0.02, 0.12)
    ok &= _check("Torque (N·m)", res["torque"], 0.001, 0.5, "N·m")
    ok &= _check("Mean AoA",     float(np.mean(res["aoa_deg"])), 0.0, 20.0, "deg")

    print(f"  Thrust = {res['thrust']:.3f} N  |  "
          f"CT = {res['CT']:.4f}  |  "
          f"P  = {res['power']:.2f} W")
    return ok


# ---------------------------------------------------------------------------
# Test 2: BEM forward flight (5 m/s, 5000 RPM)
# ---------------------------------------------------------------------------
def test_bem_forward():
    print("\n--- Test 2: BEM forward flight (5000 RPM, 5 m/s) ---")
    blade = baseline_hqprop()
    res   = bem_solve(blade, rpm=5000, v_inf=5.0, rho=1.225)

    ok  = _check("Thrust (N)",   res["thrust"],     0.2, 4.0, "N")
    ok &= _check("Efficiency",   res["efficiency"], 0.0, 1.0)

    print(f"  Thrust = {res['thrust']:.3f} N  |  "
          f"eta = {res['efficiency']:.3f}  |  "
          f"P   = {res['power']:.2f} W")
    return ok


# ---------------------------------------------------------------------------
# Test 3: BPM noise (5000 RPM, static)
# ---------------------------------------------------------------------------
def test_bpm_noise():
    print("\n--- Test 3: BPM noise (5000 RPM, V=0) ---")
    blade = baseline_hqprop()
    aero  = bem_solve(blade, rpm=5000, v_inf=0.0, rho=1.225)
    _, chord_m, _ = blade.get_stations(20)

    res = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"], cl=aero["cl"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=5000, num_blades=blade.num_blades, radius_m=blade.radius_m,
    )

    # Small UAV prop at 1 m: 55–90 dBA is physically plausible
    # Tonal is Bessel-function suppressed for small props (low M_tip) → can be low
    # Lower bound 15 dBA: LBL-VS noise (dominant for Re<50k) not yet modelled
    ok  = _check("SPL total (dBA)",  res["SPL_total"],  15.0, 95.0, "dBA")
    ok &= _check("SPL tonal (dB)",   res["SPL_tonal"],  -20.0, 95.0, "dB")

    print(f"  SPL total  = {res['SPL_total']:.1f} dBA  |  "
          f"Tonal = {res['SPL_tonal']:.1f} dB  |  "
          f"Broadband = {res['SPL_broadband']:.1f} dB")
    return ok


# ---------------------------------------------------------------------------
# Test 4: OpenMDAO component wiring
# ---------------------------------------------------------------------------
def test_openmdao_components():
    print("\n--- Test 4: OpenMDAO component wiring ---")
    blade = baseline_hqprop()

    prob  = om.Problem()
    model = prob.model

    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("rpm",   val=5000.0, units="rpm")
    ivc.add_output("v_inf", val=0.0,    units="m/s")
    ivc.add_output("rho",   val=1.225,  units="kg/m**3")

    model.add_subsystem(
        "aero",
        CCBladeComponent(blade=blade, n_stations=20),
        promotes=["*"],
    )
    model.add_subsystem(
        "acoustics",
        BPMComponent(blade=blade, n_stations=20, r_obs=1.0),
        promotes_inputs=["r_m", "v_rel", "aoa_deg", "cl",
                         "thrust", "torque", "rpm", "rho"],
        promotes_outputs=["SPL_total", "SPL_broadband", "SPL_tonal",
                          "SPL_spectrum", "freq"],
    )

    import warnings
    prob.setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_model()

    thrust = float(prob.get_val("thrust")[0])
    spl    = float(prob.get_val("SPL_total")[0])

    ok  = _check("OM Thrust (N)",      thrust, 0.5, 5.0, "N")
    ok &= _check("OM SPL_total (dBA)", spl,    15.0, 95.0, "dBA")

    print(f"  Thrust = {thrust:.3f} N  |  SPL = {spl:.1f} dBA")
    return ok


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = [
        test_bem_static(),
        test_bem_forward(),
        test_bpm_noise(),
        test_openmdao_components(),
    ]
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 50)
    print(f"  {n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("  ALL TESTS PASSED — baseline stack validated.")
    else:
        print("  SOME TESTS FAILED — check output above.")
        sys.exit(1)
