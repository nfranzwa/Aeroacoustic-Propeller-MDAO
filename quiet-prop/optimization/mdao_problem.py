"""
Main OpenMDAO MDAO problem definition.

Connects CCBladeComponent → BPMComponent and sets up an optimizer
to minimise SPL_total subject to a minimum thrust constraint.

Design variables
----------------
  rpm            : rotational speed (RPM)
  delta_twist    : twist perturbation at each blade station (deg)
  delta_chord    : chord perturbation at each blade station (normalised by R)

Objective
---------
  Minimise  SPL_total  (dBA)

Constraint
----------
  thrust  >= thrust_min  (N)

Usage
-----
  python mdao_problem.py
"""

import sys
import os
import numpy as np
import openmdao.api as om

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_hqprop, BladeGeometry
from aerodynamics.ccblade_component import CCBladeComponent, bem_solve
from acoustics.bpm_component import BPMComponent

N_STATIONS = 20


class GeometryPerturbComponent(om.ExplicitComponent):
    """
    Applies delta_twist and delta_chord to the baseline blade geometry
    and exposes the perturbed chord array for downstream use.
    (Twist and chord perturbations are passed directly to CCBladeComponent
    via a modified blade object at setup time during optimisation.)
    This component is a lightweight passthrough for scalar bookkeeping.
    """

    def initialize(self):
        self.options.declare("blade", default=None)
        self.options.declare("n_stations", default=N_STATIONS)

    def setup(self):
        blade = self.options["blade"] or baseline_hqprop()
        self._blade = blade
        N = self.options["n_stations"]
        n_blade = len(blade.r_R)

        self.add_input("delta_twist_deg", val=np.zeros(n_blade), units="deg")
        self.add_input("delta_chord_R",   val=np.zeros(n_blade))
        self.add_output("chord_m",        val=np.zeros(N), units="m")
        self.add_output("twist_deg_out",  val=np.zeros(N), units="deg")

    def compute(self, inputs, outputs):
        blade = self._blade
        N = self.options["n_stations"]
        perturbed = blade.perturb_twist(inputs["delta_twist_deg"]) \
                        .perturb_chord(inputs["delta_chord_R"])
        _, chord_m, twist_deg = perturbed.get_stations(N)
        outputs["chord_m"]       = chord_m
        outputs["twist_deg_out"] = twist_deg


def build_problem(thrust_min_N=1.0, rpm_init=5000.0, v_inf=0.0, rho=1.225):
    """
    Assemble and return the OpenMDAO Problem.

    Parameters
    ----------
    thrust_min_N : float  Minimum thrust constraint (N)
    rpm_init     : float  Initial RPM
    v_inf        : float  Freestream velocity (m/s)
    rho          : float  Air density (kg/m³)
    """
    blade = baseline_hqprop()
    n_blade = len(blade.r_R)

    prob = om.Problem()
    model = prob.model

    # Independent variables (design vars + fixed inputs)
    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("rpm",             val=rpm_init,          units="rpm")
    ivc.add_output("v_inf",           val=v_inf,             units="m/s")
    ivc.add_output("rho",             val=rho,               units="kg/m**3")
    ivc.add_output("delta_twist_deg", val=np.zeros(n_blade), units="deg")
    ivc.add_output("delta_chord_R",   val=np.zeros(n_blade))

    # Geometry perturbation
    model.add_subsystem(
        "geom",
        GeometryPerturbComponent(blade=blade, n_stations=N_STATIONS),
        promotes_inputs=["delta_twist_deg", "delta_chord_R"],
        promotes_outputs=["chord_m", "twist_deg_out"],
    )

    # BEM aerodynamics
    model.add_subsystem(
        "aero",
        CCBladeComponent(blade=blade, n_stations=N_STATIONS),
        promotes_inputs=["rpm", "v_inf", "rho"],
        promotes_outputs=["thrust", "torque", "power", "efficiency",
                          "CT", "CP", "r_m", "v_rel", "aoa_deg", "cl", "cd"],
    )

    # BPM acoustics
    model.add_subsystem(
        "aero_acoustics",
        BPMComponent(blade=blade, n_stations=N_STATIONS, r_obs=1.0),
        promotes_inputs=["r_m", "v_rel", "aoa_deg", "cl",
                         "thrust", "torque", "rpm", "rho"],
        promotes_outputs=["SPL_total", "SPL_broadband", "SPL_tonal",
                          "SPL_spectrum", "freq"],
    )

    # Driver / optimizer (SLSQP via scipy — no external install needed)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-6
    prob.driver.options["maxiter"] = 200

    # Design variables
    prob.model.add_design_var("rpm",
                              lower=2000.0, upper=8000.0, units="rpm")
    prob.model.add_design_var("delta_twist_deg",
                              lower=-5.0, upper=5.0,
                              indices=list(range(n_blade)))
    prob.model.add_design_var("delta_chord_R",
                              lower=-0.03, upper=0.03,
                              indices=list(range(n_blade)))

    # Objective
    prob.model.add_objective("SPL_total")

    # Constraint
    prob.model.add_constraint("thrust", lower=thrust_min_N)

    return prob


def run_baseline(rpm=5000.0, v_inf=0.0, rho=1.225, verbose=True):
    """Run the baseline (no optimisation) and print results."""
    prob = build_problem(rpm_init=rpm, v_inf=v_inf, rho=rho)
    prob.setup()
    prob.run_model()

    if verbose:
        print("\n=== Baseline Analysis ===")
        print(f"  RPM       : {prob.get_val('rpm')[0]:.0f}")
        print(f"  Thrust    : {prob.get_val('thrust')[0]:.3f} N")
        print(f"  Torque    : {prob.get_val('torque')[0]:.4f} N·m")
        print(f"  Power     : {prob.get_val('power')[0]:.2f} W")
        print(f"  CT        : {prob.get_val('CT')[0]:.4f}")
        print(f"  CP        : {prob.get_val('CP')[0]:.4f}")
        print(f"  SPL total : {prob.get_val('SPL_total')[0]:.1f} dBA")
        print(f"  SPL broad : {prob.get_val('SPL_broadband')[0]:.1f} dB")
        print(f"  SPL tonal : {prob.get_val('SPL_tonal')[0]:.1f} dB")

    return prob


def run_optimization(thrust_min_N=1.0, rpm_init=5000.0, verbose=True):
    """Run the full MDAO optimisation."""
    prob = build_problem(thrust_min_N=thrust_min_N, rpm_init=rpm_init)
    prob.setup()
    prob.run_driver()

    if verbose:
        print("\n=== Optimisation Result ===")
        print(f"  RPM       : {prob.get_val('rpm')[0]:.0f}")
        print(f"  Thrust    : {prob.get_val('thrust')[0]:.3f} N")
        print(f"  SPL total : {prob.get_val('SPL_total')[0]:.1f} dBA")
        print(f"  SPL broad : {prob.get_val('SPL_broadband')[0]:.1f} dB")
        delta_t = prob.get_val("delta_twist_deg")
        delta_c = prob.get_val("delta_chord_R")
        print(f"  Δtwist (deg) : {delta_t}")
        print(f"  Δchord (R)   : {delta_c}")

    return prob


if __name__ == "__main__":
    run_baseline()
