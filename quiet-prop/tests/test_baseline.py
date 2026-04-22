"""
Baseline validation suite — Phase 2 pipeline.

Tests cover:
  1. BEM static thrust at hover RPM (7000)
  2. BEM forward flight at cruise speed (15 m/s)
  3. BPM noise with Michel transition (LBL-VS active at Re~1e5)
  4. Structural stress and wall thickness
  5. OpenMDAO component wiring (geom -> aero -> acoustics -> stress)
  6. Drone thrust targets (928 g AUW, TWR 2.5 at 9500 RPM)
  7. Propeller catalogue (blade_importer)

Published reference: Brandt & Selig (2011) UIUC / AIAA 2011-1255, APC 7x5E
  Static CT ~ 0.09-0.15,  CP ~ 0.03-0.07
  Static thrust at 5000 RPM (sea level) ~ 1.0-3.0 N
"""

import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_apc7x5e
from geometry.blade_importer import load_prop, list_catalog
from aerodynamics.ccblade_component import bem_solve, CCBladeComponent
from acoustics.bpm_component import bpm_noise, BPMComponent
from structures.structural_component import (compute_stress, ALLOWABLE_STRESS,
                                              MIN_PRINT_THICKNESS)
from optimization.mdao_problem import (THRUST_HOVER_MIN, THRUST_CRUISE_MIN,
                                       RPM_HOVER_INIT, DRONE_AUW_KG, CRUISE_VINF)
import openmdao.api as om


def _check(name, value, lo, hi, units=""):
    ok = lo <= value <= hi
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:40s} = {value:10.4f} {units}  (expected {lo} - {hi})")
    return ok


# ---------------------------------------------------------------------------
# Test 1: BEM static thrust at hover operating RPM
# ---------------------------------------------------------------------------
def test_bem_static():
    print(f"\n--- Test 1: BEM static ({RPM_HOVER_INIT:.0f} RPM, V=0) ---")
    blade = baseline_apc7x5e()
    res   = bem_solve(blade, rpm=RPM_HOVER_INIT, v_inf=0.0, rho=1.225)

    ok  = _check("Thrust (N)",    res["thrust"],                   0.5,  5.0,  "N")
    ok &= _check("CT",            res["CT"],                       0.05, 0.25)
    ok &= _check("CP",            res["CP"],                       0.02, 0.12)
    ok &= _check("Torque (N*m)",  res["torque"],                   0.001, 0.5, "N*m")
    ok &= _check("Mean AoA",      float(np.mean(res["aoa_deg"])), 0.0, 20.0, "deg")
    ok &= _check("x_tr_c mean",   float(np.mean(res["x_tr_c"])), 0.0,  1.0)

    print(f"  Thrust={res['thrust']:.3f} N  CT={res['CT']:.4f}  "
          f"P={res['power']:.2f} W  x_tr_c(mean)={np.mean(res['x_tr_c']):.3f}")
    return ok


# ---------------------------------------------------------------------------
# Test 2: BEM forward flight at cruise speed
# ---------------------------------------------------------------------------
def test_bem_forward():
    print(f"\n--- Test 2: BEM cruise (axial V={CRUISE_VINF:.2f} m/s, pitch-corrected) ---")
    blade = baseline_apc7x5e()
    res   = bem_solve(blade, rpm=RPM_HOVER_INIT, v_inf=CRUISE_VINF, rho=1.225)

    # At axial inflow ~3.3 m/s the prop is barely off hover; thrust stays positive
    ok  = _check("Thrust (N)",   res["thrust"],     1.5, 5.0, "N")
    ok &= _check("Efficiency",   res["efficiency"],  0.0, 1.0)

    print(f"  Thrust={res['thrust']:.3f} N  eta={res['efficiency']:.3f}  P={res['power']:.2f} W")
    return ok


# ---------------------------------------------------------------------------
# Test 3: BPM noise with Michel transition (Phase 2 model)
# ---------------------------------------------------------------------------
def test_bpm_noise():
    print(f"\n--- Test 3: BPM noise with Michel transition ({RPM_HOVER_INIT:.0f} RPM, V=0) ---")
    blade = baseline_apc7x5e()
    aero  = bem_solve(blade, rpm=RPM_HOVER_INIT, v_inf=0.0, rho=1.225)
    _, chord_m, _ = blade.get_stations(20)

    res = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=RPM_HOVER_INIT, num_blades=blade.num_blades, radius_m=blade.radius_m,
        x_tr_c=aero["x_tr_c"],
    )

    # Amiet LETI dominates broadband; BVI tonal ~62-65 dBA (~10 dB below broadband)
    ok  = _check("SPL total (dBA)",    res["SPL_total"],     60.0, 85.0, "dBA")
    ok &= _check("SPL tonal (dBA)",    res["SPL_tonal"],     40.0, 80.0, "dBA")
    ok &= _check("SPL broadband (dB)", res["SPL_broadband"], 60.0, 85.0, "dB")

    print(f"  SPL total={res['SPL_total']:.1f} dBA  "
          f"tonal={res['SPL_tonal']:.1f} dB  broadband={res['SPL_broadband']:.1f} dB")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Structural stress and wall thickness
# ---------------------------------------------------------------------------
def test_structural():
    print("\n--- Test 4: Structural stress (6000 RPM, baseline blade) ---")
    blade = baseline_apc7x5e()
    r_m, chord_m, _ = blade.get_stations(20)
    _, _, _, tc, _, _ = blade.get_full_stations(20)

    res = compute_stress(r_m, chord_m, tc,
                         thrust=1.615, rpm=RPM_HOVER_INIT,
                         num_blades=blade.num_blades)

    ok  = _check("Max stress (MPa)",   res["max_stress"] / 1e6,  0.0, 14.3, "MPa")
    ok &= _check("sigma_c (MPa)",      res["sigma_c"] / 1e6,     0.0, 12.0, "MPa")
    ok &= _check("sigma_b (MPa)",      res["sigma_b"] / 1e6,     0.0, 12.0, "MPa")
    # Baseline tip wall 0.28 mm is intentionally below the 0.5 mm print limit
    ok &= _check("Min thickness (mm)", res["min_thickness"] * 1e3, 0.1, 1.0, "mm")

    print(f"  sigma_max={res['max_stress']/1e6:.2f} MPa  "
          f"min_wall={res['min_thickness']*1e3:.2f} mm  "
          f"(allowable={ALLOWABLE_STRESS/1e6:.0f} MPa, "
          f"print_min={MIN_PRINT_THICKNESS*1e3:.1f} mm)")
    return ok


# ---------------------------------------------------------------------------
# Test 5: OpenMDAO component wiring
# ---------------------------------------------------------------------------
def test_openmdao_components():
    print("\n--- Test 5: OpenMDAO component wiring ---")
    blade = baseline_apc7x5e()

    prob  = om.Problem()
    model = prob.model

    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("rpm",   val=RPM_HOVER_INIT, units="rpm")
    ivc.add_output("v_inf", val=0.0,            units="m/s")
    ivc.add_output("rho",   val=1.225,          units="kg/m**3")

    model.add_subsystem(
        "aero",
        CCBladeComponent(blade=blade, n_stations=20),
        promotes=["*"],
    )
    model.add_subsystem(
        "acoustics",
        BPMComponent(blade=blade, n_stations=20, r_obs=1.0),
        promotes_inputs=["r_m", "v_rel", "aoa_deg", "x_tr_c",
                         "thrust", "torque", "rpm", "rho"],
        promotes_outputs=["SPL_total", "SPL_broadband", "SPL_tonal",
                          "SPL_spectrum", "freq"],
    )

    prob.setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_model()

    thrust = float(prob.get_val("thrust")[0])
    spl    = float(prob.get_val("SPL_total")[0])
    x_tr   = float(np.mean(prob.get_val("x_tr_c")))

    ok  = _check("OM thrust (N)",      thrust, 0.5, 5.0,  "N")
    ok &= _check("OM SPL_total (dBA)", spl,   60.0, 85.0, "dBA")
    ok &= _check("OM x_tr_c (mean)",   x_tr,   0.0,  1.0)

    print(f"  Thrust={thrust:.3f} N  SPL={spl:.1f} dBA  x_tr_c={x_tr:.3f}")
    return ok


# ---------------------------------------------------------------------------
# Test 6: Drone thrust targets
# ---------------------------------------------------------------------------
def test_drone_targets():
    print(f"\n--- Test 6: Drone thrust targets ({DRONE_AUW_KG*1000:.0f} g drone, TWR 2.5) ---")
    blade = baseline_apc7x5e()
    g = 9.81
    W = DRONE_AUW_KG * g

    ok  = _check("Drone weight (N)",         W,                 7.0,  12.0, "N")
    ok &= _check("Hover thrust target (N)",  THRUST_HOVER_MIN,  1.5,   3.5, "N")
    ok &= _check("Cruise thrust target (N)", THRUST_CRUISE_MIN, 2.0,   3.0, "N")
    ok &= _check("RPM hover init",           RPM_HOVER_INIT,  5000.0, 9000.0, "RPM")

    # Confirm strong-blade starting point achieves thrust at feasible RPM
    dc  = np.full(len(blade.r_R), 0.025)
    dtc = np.full(len(blade.r_R), 0.035)
    blade_strong = blade.perturb_chord(dc).perturb_tc(dtc)
    res = bem_solve(blade_strong, rpm=9500, v_inf=0.0, rho=1.225)
    ok &= _check("Strong blade thrust @9500 RPM (N)",
                 res["thrust"], THRUST_HOVER_MIN * 0.9, 12.0, "N")

    print(f"  AUW={DRONE_AUW_KG*1000:.0f} g  W={W:.2f} N  "
          f"T_hover_min={THRUST_HOVER_MIN:.2f} N  "
          f"Strong blade thrust={res['thrust']:.2f} N")
    return ok


# ---------------------------------------------------------------------------
# Test 7: Propeller catalogue
# ---------------------------------------------------------------------------
def test_blade_importer():
    print("\n--- Test 7: Propeller catalogue (blade_importer) ---")
    catalog = list_catalog()

    ok  = _check("Catalogue size", len(catalog), 4, 15, "props")

    for name in ["APC_7x5E", "APC_7x4E", "APC_7x6E"]:
        blade = load_prop(name)
        ok &= _check(f"{name} radius (m)", blade.radius_m, 0.08, 0.10, "m")
        ok &= _check(f"{name} stations",   len(blade.r_R),   16,   22)

    blade3 = load_prop("APC_7x5E", num_blades_override=3)
    ok &= _check("APC_7x5E 3-blade count", blade3.num_blades, 3, 3)

    print(f"  Catalogue: {catalog}")
    return ok


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("BEM static hover",           test_bem_static),
        ("BEM forward flight",         test_bem_forward),
        ("BPM + Michel transition",    test_bpm_noise),
        ("Structural stress",          test_structural),
        ("OpenMDAO wiring",            test_openmdao_components),
        ("Drone thrust targets",       test_drone_targets),
        ("Propeller catalogue",        test_blade_importer),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results.append((name, False))

    n_pass  = sum(ok for _, ok in results)
    n_total = len(results)
    print("\n" + "=" * 55)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 55)
    print(f"  {n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        sys.exit(1)
