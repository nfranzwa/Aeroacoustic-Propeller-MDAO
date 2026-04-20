"""
Main OpenMDAO MDAO problem definition.

Data flow:
  IndepVarComp → GeometryPerturbComponent → CCBladeComponent → BPMComponent
                       (chord_m, twist_deg)    (aero outputs)    (SPL outputs)

Design variables
----------------
  rpm            scalar      [2000, 8000] RPM
  delta_twist    18 values   [−5, +5] deg per blade station
  delta_chord    18 values   [−0.03, +0.03] R per blade station

Objective    : minimise SPL_total (dBA)
Constraint   : thrust ≥ thrust_min (N)
"""

import sys
import os
import numpy as np
import openmdao.api as om

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_hqprop, BladeGeometry
from aerodynamics.ccblade_component import CCBladeComponent, bem_solve
from acoustics.bpm_component import BPMComponent, bpm_noise

N_STATIONS = 20


class GeometryPerturbComponent(om.ExplicitComponent):
    """
    Applies delta_twist and delta_chord perturbations to the baseline blade
    and outputs chord_m and twist_deg at N_STATIONS resolution.
    """

    def initialize(self):
        self.options.declare("blade",      default=None)
        self.options.declare("n_stations", default=N_STATIONS)

    def setup(self):
        blade = self.options["blade"] or baseline_hqprop()
        self._blade = blade
        N       = self.options["n_stations"]
        n_blade = len(blade.r_R)
        _, chord0, twist0 = blade.get_stations(N)

        self.add_input("delta_twist_deg", val=np.zeros(n_blade))
        self.add_input("delta_chord_R",   val=np.zeros(n_blade))
        self.add_output("chord_m",        val=chord0, units="m")
        self.add_output("twist_deg",      val=twist0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=1e-4)

    def compute(self, inputs, outputs):
        blade = self._blade
        N     = self.options["n_stations"]
        perturbed = blade.perturb_twist(inputs["delta_twist_deg"]) \
                        .perturb_chord(inputs["delta_chord_R"])
        _, chord_m, twist_deg = perturbed.get_stations(N)
        outputs["chord_m"]  = chord_m
        outputs["twist_deg"] = twist_deg


def build_problem(thrust_min_N=1.0, rpm_init=5000.0, v_inf=0.0, rho=1.225):
    """Assemble and return the OpenMDAO Problem."""
    blade   = baseline_hqprop()
    n_blade = len(blade.r_R)

    prob  = om.Problem()
    model = prob.model

    # ---- Independent variables ----
    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("rpm",             val=rpm_init,          units="rpm")
    ivc.add_output("v_inf",           val=v_inf,             units="m/s")
    ivc.add_output("rho",             val=rho,               units="kg/m**3")
    ivc.add_output("delta_twist_deg", val=np.zeros(n_blade))
    ivc.add_output("delta_chord_R",   val=np.zeros(n_blade))

    # ---- Geometry perturbation ----
    model.add_subsystem(
        "geom",
        GeometryPerturbComponent(blade=blade, n_stations=N_STATIONS),
        promotes_inputs=["delta_twist_deg", "delta_chord_R"],
        promotes_outputs=["chord_m", "twist_deg"],
    )

    # ---- BEM aerodynamics ----
    model.add_subsystem(
        "aero",
        CCBladeComponent(blade=blade, n_stations=N_STATIONS),
        promotes_inputs=["rpm", "v_inf", "rho", "chord_m", "twist_deg"],
        promotes_outputs=["thrust", "torque", "power", "efficiency",
                          "CT", "CP", "r_m", "v_rel", "aoa_deg", "cl", "cd"],
    )

    # ---- BPM acoustics ----
    model.add_subsystem(
        "acoustics",
        BPMComponent(blade=blade, n_stations=N_STATIONS, r_obs=1.0),
        promotes_inputs=["r_m", "chord_m", "v_rel", "aoa_deg", "cl",
                         "thrust", "torque", "rpm", "rho"],
        promotes_outputs=["SPL_total", "SPL_broadband", "SPL_tonal",
                          "SPL_spectrum", "freq"],
    )

    # ---- Optimizer ----
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"]       = 1e-4
    prob.driver.options["maxiter"]   = 500

    # ---- Design variables ----
    prob.model.add_design_var("rpm",             lower=2000.0, upper=8000.0, units="rpm")
    prob.model.add_design_var("delta_twist_deg", lower=-5.0,   upper=5.0)
    prob.model.add_design_var("delta_chord_R",   lower=-0.03,  upper=0.03)

    # ---- Objective ----
    prob.model.add_objective("SPL_total")

    # ---- Constraint ----
    prob.model.add_constraint("thrust", lower=thrust_min_N)

    return prob


def run_baseline(rpm=5000.0, v_inf=0.0, rho=1.225, verbose=True):
    """Run baseline analysis (no optimisation)."""
    prob = build_problem(rpm_init=rpm, v_inf=v_inf, rho=rho)
    prob.setup()
    prob.run_model()

    if verbose:
        print("\n=== Baseline Analysis ===")
        _print_results(prob)

    return prob


def run_optimization(thrust_min_N=1.0, rpm_init=5000.0, v_inf=0.0,
                     rho=1.225, verbose=True):
    """Run the full MDAO optimisation."""
    prob = build_problem(thrust_min_N=thrust_min_N,
                         rpm_init=rpm_init, v_inf=v_inf, rho=rho)
    prob.setup()

    if verbose:
        print("\n=== Baseline (before optimisation) ===")
        prob.run_model()
        _print_results(prob)
        print("\nRunning optimiser…")

    prob.run_driver()

    if verbose:
        print("\n=== Optimised Result ===")
        _print_results(prob)
        blade   = baseline_hqprop()
        dt = prob.get_val("delta_twist_deg")
        dc = prob.get_val("delta_chord_R")
        print(f"\n  dtwist (deg): {np.round(dt, 3)}")
        print(f"  dchord (R)  : {np.round(dc, 4)}")

    return prob


def _print_results(prob):
    print(f"  RPM        : {prob.get_val('rpm')[0]:.0f}")
    print(f"  Thrust     : {prob.get_val('thrust')[0]:.3f} N")
    print(f"  Power      : {prob.get_val('power')[0]:.2f} W")
    print(f"  CT         : {prob.get_val('CT')[0]:.4f}")
    print(f"  CP         : {prob.get_val('CP')[0]:.4f}")
    print(f"  SPL total  : {prob.get_val('SPL_total')[0]:.2f} dBA")
    print(f"  SPL broad  : {prob.get_val('SPL_broadband')[0]:.2f} dB")
    print(f"  SPL tonal  : {prob.get_val('SPL_tonal')[0]:.2f} dB")


if __name__ == "__main__":
    prob = run_optimization(thrust_min_N=1.0, rpm_init=5000.0)
