"""
OpenMDAO MDAO problem – Phase 2.

Fixes applied
-------------
1. Boundary-layer transition: Michel's criterion feeds x_tr_c into BPM,
   activating LBL-VS penalty when flow is laminar (forces optimizer to
   thin blade or modify twist to trip transition).

2. Structural constraint: centrifugal + bending root stress <= 22 MPa;
   minimum wall thickness >= 0.5 mm.

3. Hybrid optimizer: GA global search (pop=50, gen=30) -> SLSQP local
   refinement to escape local minima in the 57-variable space.

4. Multipoint evaluation: hover (V=0 m/s) + cruise (V=15 m/s) with
   weighted objective 0.7·SPL_hover + 0.3·SPL_cruise.

New design variables (Phase 2)
-------------------------------
  sweep_R         18  [0.00, 0.12]  R   aft-sweep distribution
  delta_tc        18  [-0.03, 0.03]     t/c perturbation from baseline
  z_offset_R_tip  10  [-0.05, 0.10] R   dihedral/anhedral (outer 55–100% span)
  theta2           1  [105, 135]   deg  azimuthal position of blade 2
  theta3           1  [225, 255]   deg  azimuthal position of blade 3

Total design variables: 1 + 18 + 18 + 18 + 18 + 10 + 1 + 1 = 85
"""

import sys
import os
import numpy as np
import openmdao.api as om

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_hqprop, BladeGeometry
from aerodynamics.ccblade_component import CCBladeComponent, bem_solve
from acoustics.bpm_component import BPMComponent, bpm_noise
from structures.structural_component import (StressComponent, ALLOWABLE_STRESS,
                                              MIN_PRINT_THICKNESS)

N_STATIONS   = 20
N_BLADE_STAT = 18          # blade definition stations (matches HQProp baseline)
N_TIP_ZONES  = 10          # dihedral design var stations (outer 55–100% span)
CRUISE_VINF  = 15.0        # m/s forward cruise speed
W_HOVER      = 0.7         # acoustic weight for hover
W_CRUISE     = 0.3         # acoustic weight for cruise


# ---------------------------------------------------------------------------
# Geometry component – full parameterisation
# ---------------------------------------------------------------------------

class GeometryFullComponent(om.ExplicitComponent):
    """
    Applies all perturbations to the baseline blade and outputs geometry
    arrays at N_STATIONS resolution.

    Inputs
    ------
    delta_twist_deg : (18,)   twist perturbation (deg)
    delta_chord_R   : (18,)   chord perturbation (R)
    sweep_R         : (18,)   absolute aft-sweep (R)
    delta_tc        : (18,)   t/c perturbation from baseline
    z_offset_R_tip  : (10,)   dihedral z-offset for outer 10 stations (R)
    theta2          : scalar  blade-2 azimuthal angle (deg)
    theta3          : scalar  blade-3 azimuthal angle (deg)

    Outputs
    -------
    chord_m, twist_deg, tc_ratio, sweep_m, z_offset_m  : (N_STATIONS,)
    blade_angles_deg                                     : (3,)
    """

    def initialize(self):
        self.options.declare("blade",      default=None)
        self.options.declare("n_stations", default=N_STATIONS)

    def setup(self):
        blade = self.options["blade"] or baseline_hqprop()
        self._blade = blade
        N       = self.options["n_stations"]
        n_def   = N_BLADE_STAT
        _, chord0, twist0 = blade.get_stations(N)
        _, _, _, tc0, sw0, z0 = blade.get_full_stations(N)

        self.add_input("delta_twist_deg", val=np.zeros(n_def))
        self.add_input("delta_chord_R",   val=np.zeros(n_def))
        self.add_input("sweep_R",         val=np.zeros(n_def))
        self.add_input("delta_tc",        val=np.zeros(n_def))
        self.add_input("z_offset_R_tip",  val=np.zeros(N_TIP_ZONES))
        self.add_input("theta2",          val=120.0)
        self.add_input("theta3",          val=240.0)

        self.add_output("chord_m",         val=chord0, units="m")
        self.add_output("twist_deg",       val=twist0)
        self.add_output("tc_ratio",        val=tc0)
        self.add_output("sweep_m",         val=np.zeros(N), units="m")
        self.add_output("z_offset_m",      val=np.zeros(N), units="m")
        self.add_output("blade_angles_deg",val=np.array([0.0, 120.0, 240.0]))

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=1e-4)

    def compute(self, inputs, outputs):
        blade = self._blade
        N     = self.options["n_stations"]

        perturbed = (blade
                     .perturb_twist(inputs["delta_twist_deg"])
                     .perturb_chord(inputs["delta_chord_R"])
                     .perturb_tc(inputs["delta_tc"])
                     .set_sweep(inputs["sweep_R"]))

        # Dihedral: applies to outer N_TIP_ZONES stations of the 18-station definition
        n_def   = N_BLADE_STAT
        z_full  = np.zeros(n_def)
        z_full[-N_TIP_ZONES:] = inputs["z_offset_R_tip"]
        perturbed = perturbed.set_z_offset(z_full)

        theta2 = float(inputs["theta2"][0])
        theta3 = float(inputs["theta3"][0])
        perturbed = perturbed.set_blade_angles([0.0, theta2, theta3])

        r_m, chord_m, twist_deg, tc, sw, zof = perturbed.get_full_stations(N)

        outputs["chord_m"]          = chord_m
        outputs["twist_deg"]        = twist_deg
        outputs["tc_ratio"]         = tc
        outputs["sweep_m"]          = sw
        outputs["z_offset_m"]       = zof
        outputs["blade_angles_deg"] = perturbed.blade_angles_deg


# ---------------------------------------------------------------------------
# Weighted-SPL objective component
# ---------------------------------------------------------------------------

class WeightedSPLComponent(om.ExplicitComponent):
    """Combines hover and cruise SPL into single weighted objective."""

    def setup(self):
        self.add_input("SPL_hover",  val=0.0)
        self.add_input("SPL_cruise", val=0.0)
        self.add_output("SPL_weighted", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        outputs["SPL_weighted"] = (W_HOVER * inputs["SPL_hover"][0] +
                                   W_CRUISE * inputs["SPL_cruise"][0])


# ---------------------------------------------------------------------------
# Balance constraint component
# ---------------------------------------------------------------------------

class BalanceComponent(om.ExplicitComponent):
    """Computes rotor static imbalance factor from blade azimuthal positions."""

    def setup(self):
        self.add_input("blade_angles_deg", val=np.array([0.0, 120.0, 240.0]))
        # imbalance_factor = |Σ exp(j*theta_k)| / B  ∈ [0, 1]; want <= max_imbalance
        self.add_output("imbalance_factor", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        th  = np.deg2rad(inputs["blade_angles_deg"])
        vec = np.sum(np.exp(1j * th))
        outputs["imbalance_factor"] = float(np.abs(vec)) / len(th)


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

def build_problem(thrust_min_hover=1.0, thrust_min_cruise=0.5,
                  rpm_init=5000.0, rho=1.225, max_imbalance=0.15):
    """
    Assemble and return the full Phase-2 OpenMDAO Problem.

    Model structure
    ---------------
    ivc -> geom -> hover_aero -> hover_acoustics
                -> cruise_aero -> cruise_acoustics
                -> stress
                -> balance
          -> obj (WeightedSPLComponent)
    """
    blade   = baseline_hqprop()
    n_def   = N_BLADE_STAT

    prob  = om.Problem()
    model = prob.model

    # ---- Independent variables -------------------------------------------
    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("rpm",             val=rpm_init,        units="rpm")
    ivc.add_output("rho",             val=rho,             units="kg/m**3")
    ivc.add_output("v_hover",         val=0.0,             units="m/s")
    ivc.add_output("v_cruise",        val=CRUISE_VINF,     units="m/s")
    ivc.add_output("delta_twist_deg", val=np.zeros(n_def))
    ivc.add_output("delta_chord_R",   val=np.zeros(n_def))
    ivc.add_output("sweep_R",         val=np.zeros(n_def))
    ivc.add_output("delta_tc",        val=np.zeros(n_def))
    ivc.add_output("z_offset_R_tip",  val=np.zeros(N_TIP_ZONES))
    ivc.add_output("theta2",          val=120.0)
    ivc.add_output("theta3",          val=240.0)

    # ---- Geometry perturbation -------------------------------------------
    model.add_subsystem(
        "geom",
        GeometryFullComponent(blade=blade, n_stations=N_STATIONS),
        promotes_inputs=["delta_twist_deg", "delta_chord_R",
                         "sweep_R", "delta_tc", "z_offset_R_tip",
                         "theta2", "theta3"],
        promotes_outputs=["chord_m", "twist_deg", "tc_ratio",
                          "sweep_m", "z_offset_m", "blade_angles_deg"],
    )

    # ---- Hover aerodynamics ----------------------------------------------
    model.add_subsystem(
        "hover_aero",
        CCBladeComponent(blade=blade, n_stations=N_STATIONS),
        promotes_inputs=[("rpm", "rpm"), ("v_inf", "v_hover"),
                         ("rho", "rho"), "chord_m", "twist_deg"],
        promotes_outputs=[
            ("thrust",     "thrust_hover"),
            ("torque",     "torque_hover"),
            ("power",      "power_hover"),
            ("r_m",        "r_m_hover"),
            ("v_rel",      "v_rel_hover"),
            ("aoa_deg",    "aoa_deg_hover"),
            ("cl",         "cl_hover"),
            ("cd",         "cd_hover"),
            ("x_tr_c",     "x_tr_c_hover"),
            ("CT",         "CT_hover"),
            ("CP",         "CP_hover"),
            ("efficiency", "eff_hover"),
        ],
    )

    # ---- Hover acoustics -------------------------------------------------
    model.add_subsystem(
        "hover_acoustics",
        BPMComponent(blade=blade, n_stations=N_STATIONS, r_obs=1.0),
        promotes_inputs=[
            ("r_m",              "r_m_hover"),
            "chord_m",
            ("v_rel",            "v_rel_hover"),
            ("aoa_deg",          "aoa_deg_hover"),
            ("cl",               "cl_hover"),
            ("x_tr_c",           "x_tr_c_hover"),
            ("thrust",           "thrust_hover"),
            ("torque",           "torque_hover"),
            ("rpm",              "rpm"),
            ("rho",              "rho"),
            "blade_angles_deg",
        ],
        promotes_outputs=[
            ("SPL_total",     "SPL_hover"),
            ("SPL_broadband", "SPL_broad_hover"),
            ("SPL_tonal",     "SPL_tonal_hover"),
        ],
    )

    # ---- Cruise aerodynamics ---------------------------------------------
    model.add_subsystem(
        "cruise_aero",
        CCBladeComponent(blade=blade, n_stations=N_STATIONS),
        promotes_inputs=[("rpm", "rpm"), ("v_inf", "v_cruise"),
                         ("rho", "rho"), "chord_m", "twist_deg"],
        promotes_outputs=[
            ("thrust",     "thrust_cruise"),
            ("torque",     "torque_cruise"),
            ("r_m",        "r_m_cruise"),
            ("v_rel",      "v_rel_cruise"),
            ("aoa_deg",    "aoa_deg_cruise"),
            ("cl",         "cl_cruise"),
            ("cd",         "cd_cruise"),
            ("x_tr_c",     "x_tr_c_cruise"),
        ],
    )

    # ---- Cruise acoustics ------------------------------------------------
    model.add_subsystem(
        "cruise_acoustics",
        BPMComponent(blade=blade, n_stations=N_STATIONS, r_obs=1.0),
        promotes_inputs=[
            ("r_m",              "r_m_cruise"),
            "chord_m",
            ("v_rel",            "v_rel_cruise"),
            ("aoa_deg",          "aoa_deg_cruise"),
            ("cl",               "cl_cruise"),
            ("x_tr_c",           "x_tr_c_cruise"),
            ("thrust",           "thrust_cruise"),
            ("torque",           "torque_cruise"),
            ("rpm",              "rpm"),
            ("rho",              "rho"),
            "blade_angles_deg",
        ],
        promotes_outputs=[
            ("SPL_total",     "SPL_cruise"),
            ("SPL_broadband", "SPL_broad_cruise"),
            ("SPL_tonal",     "SPL_tonal_cruise"),
        ],
    )

    # ---- Structural stress -----------------------------------------------
    model.add_subsystem(
        "stress",
        StressComponent(blade=blade, n_stations=N_STATIONS, num_blades=blade.num_blades),
        promotes_inputs=[("chord_m", "chord_m"), ("tc_ratio", "tc_ratio"),
                         ("r_m",     "r_m_hover"), ("thrust",  "thrust_hover"),
                         ("rpm",     "rpm")],
        promotes_outputs=["max_stress", "min_thickness"],
    )

    # ---- Rotor balance ---------------------------------------------------
    model.add_subsystem(
        "balance",
        BalanceComponent(),
        promotes_inputs=["blade_angles_deg"],
        promotes_outputs=["imbalance_factor"],
    )

    # ---- Weighted objective ----------------------------------------------
    model.add_subsystem(
        "obj",
        WeightedSPLComponent(),
        promotes_inputs=["SPL_hover", "SPL_cruise"],
        promotes_outputs=["SPL_weighted"],
    )

    # ---- Driver (will be replaced for hybrid optimisation) ---------------
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"]       = 1e-4
    prob.driver.options["maxiter"]   = 500

    # ---- Design variables ------------------------------------------------
    prob.model.add_design_var("rpm",             lower=2000.0,  upper=8000.0,  units="rpm")
    prob.model.add_design_var("delta_twist_deg", lower=-5.0,    upper=5.0)
    prob.model.add_design_var("delta_chord_R",   lower=-0.03,   upper=0.03)
    prob.model.add_design_var("sweep_R",         lower=0.0,     upper=0.12)
    prob.model.add_design_var("delta_tc",        lower=-0.03,   upper=0.04)
    prob.model.add_design_var("z_offset_R_tip",  lower=-0.05,   upper=0.10)
    prob.model.add_design_var("theta2",          lower=105.0,   upper=135.0)
    prob.model.add_design_var("theta3",          lower=225.0,   upper=255.0)

    # ---- Objective -------------------------------------------------------
    prob.model.add_objective("SPL_weighted")

    # ---- Constraints -----------------------------------------------------
    prob.model.add_constraint("thrust_hover",     lower=thrust_min_hover)
    prob.model.add_constraint("thrust_cruise",    lower=thrust_min_cruise)
    prob.model.add_constraint("max_stress",       upper=ALLOWABLE_STRESS)
    prob.model.add_constraint("min_thickness",    lower=MIN_PRINT_THICKNESS)
    prob.model.add_constraint("imbalance_factor", upper=max_imbalance)

    return prob


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_baseline(rpm=5000.0, rho=1.225, verbose=True):
    """Run baseline analysis (no optimisation)."""
    prob = build_problem(rpm_init=rpm, rho=rho)
    prob.setup()
    prob.run_model()
    if verbose:
        print("\n=== Baseline Analysis ===")
        _print_results(prob)
    return prob


def run_optimization(thrust_min_hover=1.0, thrust_min_cruise=0.5,
                     rpm_init=5000.0, rho=1.225, verbose=True,
                     use_hybrid=True):
    """
    Run the full MDAO optimisation.

    Hybrid strategy (use_hybrid=True):
      Phase 1: SimpleGADriver (pop=50, gen=30) for global search
      Phase 2: ScipyOptimizeDriver SLSQP starting from GA best
    """
    if use_hybrid:
        return _run_hybrid(thrust_min_hover, thrust_min_cruise,
                           rpm_init, rho, verbose)
    else:
        return _run_slsqp(thrust_min_hover, thrust_min_cruise,
                          rpm_init, rho, verbose)


def _run_slsqp(thrust_min_hover, thrust_min_cruise, rpm_init, rho, verbose):
    prob = build_problem(thrust_min_hover=thrust_min_hover,
                         thrust_min_cruise=thrust_min_cruise,
                         rpm_init=rpm_init, rho=rho)
    prob.setup()
    if verbose:
        print("\n=== Baseline (before optimisation) ===")
        prob.run_model()
        _print_results(prob)
        print("\nRunning SLSQP optimiser…")
    prob.run_driver()
    if verbose:
        print("\n=== Optimised Result ===")
        _print_results(prob)
    return prob


def _run_hybrid(thrust_min_hover, thrust_min_cruise, rpm_init, rho, verbose):
    """GA global search -> SLSQP local refinement."""

    # ---- Phase 1: GA global search ----------------------------------------
    if verbose:
        print("\n=== Phase 1: GA global search (pop=50, gen=30) ===")

    prob_ga = build_problem(thrust_min_hover=thrust_min_hover,
                            thrust_min_cruise=thrust_min_cruise,
                            rpm_init=rpm_init, rho=rho)
    prob_ga.driver = om.SimpleGADriver()
    prob_ga.driver.options["pop_size"] = 50
    prob_ga.driver.options["max_gen"]  = 30
    prob_ga.driver.options["Pc"]       = 0.5    # crossover probability
    prob_ga.driver.options["Pm"]       = 0.01   # mutation probability
    prob_ga.setup()
    prob_ga.run_driver()

    # Capture GA result
    ga_rpm   = float(prob_ga.get_val("rpm")[0])
    ga_dt    = prob_ga.get_val("delta_twist_deg").copy()
    ga_dc    = prob_ga.get_val("delta_chord_R").copy()
    ga_sw    = prob_ga.get_val("sweep_R").copy()
    ga_dtc   = prob_ga.get_val("delta_tc").copy()
    ga_z     = prob_ga.get_val("z_offset_R_tip").copy()
    ga_th2   = float(prob_ga.get_val("theta2")[0])
    ga_th3   = float(prob_ga.get_val("theta3")[0])

    if verbose:
        print("GA result:")
        _print_results(prob_ga)
        print("\n=== Phase 2: SLSQP refinement ===")

    # ---- Phase 2: SLSQP local refinement ----------------------------------
    prob = build_problem(thrust_min_hover=thrust_min_hover,
                         thrust_min_cruise=thrust_min_cruise,
                         rpm_init=rpm_init, rho=rho)
    prob.setup()

    # Warm-start from GA best
    prob.set_val("rpm",             ga_rpm)
    prob.set_val("delta_twist_deg", ga_dt)
    prob.set_val("delta_chord_R",   ga_dc)
    prob.set_val("sweep_R",         ga_sw)
    prob.set_val("delta_tc",        ga_dtc)
    prob.set_val("z_offset_R_tip",  ga_z)
    prob.set_val("theta2",          ga_th2)
    prob.set_val("theta3",          ga_th3)

    prob.run_driver()

    if verbose:
        print("\n=== Final Optimised Result ===")
        _print_results(prob)
        _print_design_vars(prob)

    return prob


def _print_results(prob):
    def _g(name):
        try:
            return prob.get_val(name)[0]
        except Exception:
            return float("nan")

    print(f"  RPM          : {_g('rpm'):.0f}")
    print(f"  Thrust hover : {_g('thrust_hover'):.3f} N")
    print(f"  Thrust cruise: {_g('thrust_cruise'):.3f} N")
    print(f"  Power hover  : {_g('power_hover'):.2f} W")
    print(f"  SPL hover    : {_g('SPL_hover'):.2f} dBA")
    print(f"  SPL cruise   : {_g('SPL_cruise'):.2f} dBA")
    print(f"  SPL weighted : {_g('SPL_weighted'):.2f} dBA")
    print(f"  Max stress   : {_g('max_stress')/1e6:.2f} MPa  (<=22 MPa)")
    print(f"  Min thickness: {_g('min_thickness')*1e3:.2f} mm  (>=0.5 mm)")
    print(f"  Imbalance    : {_g('imbalance_factor'):.4f}  (<=0.15)")
    th2 = _g('theta2')
    th3 = _g('theta3')
    print(f"  Blade angles : 0 / {th2:.1f} / {th3:.1f} deg")


def _print_design_vars(prob):
    def _g(name):
        try:
            return prob.get_val(name)
        except Exception:
            return np.array([float("nan")])

    print(f"\n  dtwist (deg)    : {np.round(_g('delta_twist_deg'), 3)}")
    print(f"  dchord (R)      : {np.round(_g('delta_chord_R'), 4)}")
    print(f"  sweep (R)       : {np.round(_g('sweep_R'), 4)}")
    print(f"  delta_tc        : {np.round(_g('delta_tc'), 4)}")
    print(f"  z_offset_R_tip  : {np.round(_g('z_offset_R_tip'), 4)}")


if __name__ == "__main__":
    prob = run_optimization(
        thrust_min_hover=1.0,
        thrust_min_cruise=0.5,
        rpm_init=5000.0,
        use_hybrid=True,
    )
