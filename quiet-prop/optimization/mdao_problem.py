"""
OpenMDAO MDAO problem – Phase 3.

Drone sizing basis (7-inch UAV, 4-rotor)
-----------------------------------------
  Frame (ImpulseRC Apex 7 carbon)  : 120 g
  Motors x4 (2814 900KV)           : 272 g   (4 × 68 g, iFlight XING-E 2814)
  ESC 4-in-1 45A                   :  30 g
  Flight controller                :  20 g
  Battery 4S 3300 mAh LiPo         : 335 g   (realistic measured weight)
  Props x4                         :  48 g
  FPV / DJI O3 Air Unit            :  55 g
  GPS + receiver + wiring          :  48 g
  ---------------------------------- -----
  AUW                              : 928 g   -> W = 9.10 N
  Per-prop weight at hover         :   2.28 N   (W/4)

Thrust targets (BEM-derived, iFlight XING-E 2814 900KV on 4S):
  900 KV × 14.8 V nominal = 13,320 RPM no-load; ~75% under load -> ~10,000 RPM max
  Hover    (1g, ~53% throttle) :  2.28 N @ ~7,000 RPM   <- hover constraint
  Maneuver (TWR 2.5, ~85% thr) :  5.69 N @ ~9,500 RPM
  Cruise   (level flight 15m/s):  2.33 N                 <- cruise constraint (see below)
  Full thr (TWR ~3.1, 100%)    :  7.96 N @ ~10,000 RPM

Cruise constraint derivation (Wagter et al. 2014; ICAS 2020-0781):
  A quadrotor at 15 m/s pitches forward at angle θ, not 90°.
  Body drag at 15 m/s: F_drag = 0.5 × ρ × V² × CdA
    CdA_DRONE = 0.015 m²  (Cd ≈ 0.30 for a clean 7-inch LR airframe)
    F_drag = 0.5 × 1.225 × 225 × 0.015 = 2.07 N
  Cruise pitch: θ = arctan(F_drag / W) = arctan(2.07 / 9.10) = 12.8°
  Per-rotor cruise thrust magnitude: T = W / (4·cos θ) = 2.33 N
  Axial inflow component for BEM: V_axial = V_cruise × sin θ = 3.32 m/s
  (Edgewise component V·cos θ ≈ 14.6 m/s is a second-order effect at this pitch angle.)

RPM design-variable bounds: [3500, 10000]
RPM initial point: 7000 (estimated hover RPM for 928 g AUW)

Fixes applied
-------------
1. Boundary-layer transition: Michel's criterion feeds x_tr_c into BPM,
   activating LBL-VS penalty when flow is laminar (forces optimizer to
   thin blade or modify twist to trip transition).

2. Structural constraint: centrifugal + bending root stress <= 14.3 MPa
   (Siraya Tech Blu Tough UTS=50 MPa, FoS=3.5 fatigue/cyclic);
   minimum wall thickness >= 0.5 mm.

3. Hybrid optimizer: GA global search (pop=200, gen=50) -> SLSQP local
   refinement to escape local minima in the 75-variable space.

4. Multipoint evaluation: hover (V=0 m/s) + cruise (V_axial=3.32 m/s) with
   weighted objective 0.7·SPL_hover + 0.3·SPL_cruise.
   Cruise inflow is the axial component at the equilibrium pitch angle (~12.8°),
   not the full forward speed (which would overstate thrust loss by ~4.5×).

New design variables (Phase 2)
-------------------------------
  sweep_R         18  [0.00, 0.12]  R   aft-sweep distribution
  delta_tc        18  [-0.03, 0.03]     t/c perturbation from baseline
  z_offset_R_tip  10  [-0.05, 0.10] R   dihedral/anhedral (outer 55–100% span)
  theta2           1  [105, 135]   deg  azimuthal position of blade 2
  theta3           1  [225, 255]   deg  azimuthal position of blade 3

Active design variables: 1 + 18 + 18 + 18 + 18 + 1 + 1 = 75  (dihedral excluded)
"""

import sys
import os
import numpy as np
import openmdao.api as om
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_apc7x5e, BladeGeometry
from aerodynamics.ccblade_component import CCBladeComponent, bem_solve
from acoustics.bpm_component import BPMComponent, bpm_noise
from structures.structural_component import (StressComponent, ALLOWABLE_STRESS,
                                              MIN_PRINT_THICKNESS)

N_STATIONS   = 20
N_BLADE_STAT = 18          # blade definition stations (matches APC 7x5E baseline)
N_TIP_ZONES  = 10          # dihedral design var stations (outer 55–100% span)
N_CP         = 5           # B-spline control points per DV — guarantees C2 smooth distributions
W_HOVER      = 0.7         # acoustic weight for hover
W_CRUISE     = 0.3         # acoustic weight for cruise

# ---------------------------------------------------------------------------
# Drone sizing constants (7-inch 4-rotor UAV, 928 g AUW)
# ---------------------------------------------------------------------------
DRONE_AUW_KG        = 0.928    # kg  all-up weight
DRONE_N_ROTORS      = 4        # number of rotors
DRONE_G             = 9.81     # m/s^2

# Thrust per rotor required to sustain 1g hover (equals W/4).
# NOTE: TWR 2.5 maneuver capability is a motor/ESC selection requirement,
# not an acoustic operating point.  Optimising at TWR 2.5 forces RPM to
# ~9 500 RPM, inflating hover noise by ~14 dB — an apples-to-oranges error.
THRUST_HOVER_MIN    = DRONE_AUW_KG * DRONE_G / DRONE_N_ROTORS   # 2.28 N

# ---------------------------------------------------------------------------
# Cruise operating point — pitched-forward quadrotor physics
# ---------------------------------------------------------------------------
# At V_cruise, the drone pitches at θ = arctan(F_drag / W).
# The BEM axial inflow is V_cruise·sin(θ), NOT the full forward speed.
# Source: Wagter et al. (2014); ICAS 2020-0781 multirotor forward-flight model.
CRUISE_SPEED        = 15.0     # m/s  true airspeed
CDA_DRONE           = 0.015    # m²   drag area (Cd≈0.30, A≈0.05 m² for 7" LR quad)
_W                  = DRONE_AUW_KG * DRONE_G                         # 9.10 N
_F_drag             = 0.5 * 1.225 * CRUISE_SPEED**2 * CDA_DRONE      # 2.07 N
_theta_cruise       = np.arctan(_F_drag / _W)                         # ~12.8 deg
CRUISE_VINF         = CRUISE_SPEED * np.sin(_theta_cruise)            # 3.32 m/s axial
THRUST_CRUISE_MIN   = _W / (DRONE_N_ROTORS * np.cos(_theta_cruise))  # 2.33 N per rotor

# RPM operating envelope (900 KV motor on 4S, 7" prop)
RPM_HOVER_INIT      = 7000.0   # RPM  estimated hover point for 928 g AUW
RPM_LOWER           = 3500.0   # RPM  idle / low-throttle floor
RPM_UPPER           = 10000.0  # RPM  full-throttle ceiling (900 KV on 4S under load)


# ---------------------------------------------------------------------------
# Geometry component – full parameterisation
# ---------------------------------------------------------------------------

class GeometryFullComponent(om.ExplicitComponent):
    """
    B-spline blade geometry — N_CP control points per parameter, evaluated
    at the 18 blade-definition stations via CubicSpline (C2 continuous).

    Inputs  (N_CP control points each)
    ------
    delta_twist_cp : twist perturbation control points (deg)
    delta_chord_cp : chord perturbation control points (R)
    sweep_cp       : absolute aft-sweep control points (R)
    delta_tc_cp    : t/c perturbation control points
    theta2 / theta3: locked at 120° / 240° (not DVs)

    Outputs  (N_STATIONS interpolated values)
    -------
    chord_m, twist_deg, tc_ratio, sweep_m, z_offset_m, blade_angles_deg
    """

    def initialize(self):
        self.options.declare("blade",      default=None)
        self.options.declare("n_stations", default=N_STATIONS)
        self.options.declare("fd_step",    default=3e-4)

    def setup(self):
        blade = self.options["blade"] or baseline_apc7x5e()
        self._blade = blade
        N = self.options["n_stations"]
        # Control-point radii: evenly spaced across the blade span
        self._r_cp = np.linspace(blade.r_R[0], blade.r_R[-1], N_CP)

        _, chord0, twist0 = blade.get_stations(N)
        _, _, _, tc0, _, _ = blade.get_full_stations(N)

        self.add_input("delta_twist_cp", val=np.zeros(N_CP))
        self.add_input("delta_chord_cp", val=np.zeros(N_CP))
        self.add_input("sweep_cp",       val=np.zeros(N_CP))
        self.add_input("delta_tc_cp",    val=np.zeros(N_CP))
        self.add_input("theta2",         val=120.0)
        self.add_input("theta3",         val=240.0)

        self.add_output("chord_m",          val=chord0, units="m")
        self.add_output("twist_deg",        val=twist0)
        self.add_output("tc_ratio",         val=tc0)
        self.add_output("sweep_m",          val=np.zeros(N), units="m")
        self.add_output("z_offset_m",       val=np.zeros(N), units="m")
        self.add_output("blade_angles_deg", val=np.array([0.0, 120.0, 240.0]))

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=self.options["fd_step"])

    def _spline(self, cp_vals, r_def):
        """Evaluate cubic spline defined by N_CP control points at r_def stations."""
        return CubicSpline(self._r_cp, cp_vals)(r_def)

    def compute(self, inputs, outputs):
        blade  = self._blade
        N      = self.options["n_stations"]
        r_def  = blade.r_R

        dc_def  = self._spline(inputs["delta_chord_cp"], r_def)
        dt_def  = self._spline(inputs["delta_twist_cp"], r_def)
        dtc_def = self._spline(inputs["delta_tc_cp"],    r_def)
        sw_def  = np.clip(self._spline(inputs["sweep_cp"], r_def), 0.0, 0.12)

        perturbed = (blade
                     .perturb_twist(dt_def)
                     .perturb_chord(dc_def)
                     .perturb_tc(dtc_def)
                     .set_sweep(sw_def)
                     .set_blade_angles([0.0,
                                        float(inputs["theta2"][0]),
                                        float(inputs["theta3"][0])]))

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
# Twist monotonicity constraint component
# ---------------------------------------------------------------------------

class TwistMonotonicityComponent(om.ExplicitComponent):
    """
    Enforces aerodynamic wash-out at the 18-station DEFINITION level.

    Works on delta_twist_deg + stored baseline so the constraint fires before
    interpolation to N_STATIONS — prevents rollercoaster distributions that
    pass the resampled check but are non-physical at the definition nodes.

    Outputs twist_def_diff[i] = (baseline+delta)[i] - (baseline+delta)[i+1].
    Constraining twist_def_diff >= 0 enforces monotonically decreasing twist.
    """

    def initialize(self):
        self.options.declare("baseline_twist", recordable=False)
        self.options.declare("fd_step", default=3e-4)

    def setup(self):
        n = len(self.options["baseline_twist"])
        self.add_input("delta_twist_deg", val=np.zeros(n))
        self.add_output("twist_def_diff",  val=np.zeros(n - 1))

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=self.options["fd_step"])

    def compute(self, inputs, outputs):
        total = self.options["baseline_twist"] + inputs["delta_twist_deg"]
        outputs["twist_def_diff"] = total[:-1] - total[1:]


# ---------------------------------------------------------------------------
# Sweep monotonicity constraint component
# ---------------------------------------------------------------------------

class SweepMonotonicityComponent(om.ExplicitComponent):
    """
    Enforces monotonically non-decreasing aft-sweep from root to tip.

    Outputs sweep_def_diff[i] = sweep_R[i+1] - sweep_R[i].
    Constraining sweep_def_diff >= 0 prevents the saw-tooth exploit where
    the optimizer oscillates sweep 0/max/0/max to maximise |dSweep/dr|
    at every station and falsely inflate the cos⁴(Λ) noise reduction.
    """

    def initialize(self):
        self.options.declare("n_def", default=N_BLADE_STAT)
        self.options.declare("fd_step", default=3e-4)

    def setup(self):
        n = self.options["n_def"]
        self.add_input("sweep_R",        val=np.zeros(n))
        self.add_output("sweep_def_diff", val=np.zeros(n - 1))

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=self.options["fd_step"])

    def compute(self, inputs, outputs):
        sw = inputs["sweep_R"]
        outputs["sweep_def_diff"] = sw[1:] - sw[:-1]


# ---------------------------------------------------------------------------
# Smoothness constraint component (chord and twist deltas)
# ---------------------------------------------------------------------------

class SmoothnessComponent(om.ExplicitComponent):
    """
    Computes adjacent differences of a DV array to enforce spanwise smoothness.

    Outputs diff[i] = x[i+1] - x[i].
    Constraining -max_step <= diff <= max_step prevents the optimizer from
    creating large station-to-station jumps that BEM strip theory cannot
    accurately resolve (each strip is aerodynamically independent in BEM,
    so rapid spanwise variations look "free" to the optimizer but are
    physically unrealisable on a rigid blade).
    """

    def initialize(self):
        self.options.declare("input_name",  default="x")
        self.options.declare("output_name", default="x_diff")
        self.options.declare("n_def",       default=N_BLADE_STAT)
        self.options.declare("fd_step",     default=3e-4)

    def setup(self):
        n = self.options["n_def"]
        self.add_input(self.options["input_name"],   val=np.zeros(n))
        self.add_output(self.options["output_name"], val=np.zeros(n - 1))

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=self.options["fd_step"])

    def compute(self, inputs, outputs):
        x = inputs[self.options["input_name"]]
        outputs[self.options["output_name"]] = x[1:] - x[:-1]


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

def build_problem(thrust_min_hover=THRUST_HOVER_MIN,
                  thrust_min_cruise=THRUST_CRUISE_MIN,
                  rpm_init=RPM_HOVER_INIT, rho=1.225, max_imbalance=0.05,
                  fd_step=3e-4, optimizer="SLSQP",
                  twist_smooth_max=2.0, chord_smooth_max=0.02):
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
    blade   = baseline_apc7x5e()
    n_def   = N_BLADE_STAT

    prob  = om.Problem()
    model = prob.model

    # ---- Independent variables -------------------------------------------
    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("rpm",             val=rpm_init,        units="rpm")
    ivc.add_output("rho",             val=rho,             units="kg/m**3")
    ivc.add_output("v_hover",         val=0.0,             units="m/s")
    ivc.add_output("v_cruise",        val=CRUISE_VINF,     units="m/s")
    ivc.add_output("delta_twist_cp",  val=np.zeros(N_CP))
    ivc.add_output("delta_chord_cp",  val=np.zeros(N_CP))
    ivc.add_output("sweep_cp",        val=np.zeros(N_CP))
    ivc.add_output("delta_tc_cp",     val=np.zeros(N_CP))
    ivc.add_output("theta2",          val=120.0)
    ivc.add_output("theta3",          val=240.0)

    # ---- Geometry perturbation (B-spline) --------------------------------
    model.add_subsystem(
        "geom",
        GeometryFullComponent(blade=blade, n_stations=N_STATIONS, fd_step=fd_step),
        promotes_inputs=["delta_twist_cp", "delta_chord_cp",
                         "sweep_cp", "delta_tc_cp", "theta2", "theta3"],
        promotes_outputs=["chord_m", "twist_deg", "tc_ratio",
                          "sweep_m", "z_offset_m", "blade_angles_deg"],
    )

    # ---- Hover aerodynamics ----------------------------------------------
    model.add_subsystem(
        "hover_aero",
        CCBladeComponent(blade=blade, n_stations=N_STATIONS, fd_step=fd_step),
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
        BPMComponent(blade=blade, n_stations=N_STATIONS, r_obs=1.0, fd_step=fd_step),
        promotes_inputs=[
            ("r_m",              "r_m_hover"),
            "chord_m",
            ("v_rel",            "v_rel_hover"),
            ("aoa_deg",          "aoa_deg_hover"),
            ("x_tr_c",           "x_tr_c_hover"),
            ("thrust",           "thrust_hover"),
            ("torque",           "torque_hover"),
            ("rpm",              "rpm"),
            ("rho",              "rho"),
            "blade_angles_deg",
            "sweep_m",
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
        CCBladeComponent(blade=blade, n_stations=N_STATIONS, fd_step=fd_step),
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
        BPMComponent(blade=blade, n_stations=N_STATIONS, r_obs=1.0, fd_step=fd_step),
        promotes_inputs=[
            ("r_m",              "r_m_cruise"),
            "chord_m",
            ("v_rel",            "v_rel_cruise"),
            ("aoa_deg",          "aoa_deg_cruise"),
            ("x_tr_c",           "x_tr_c_cruise"),
            ("thrust",           "thrust_cruise"),
            ("torque",           "torque_cruise"),
            ("rpm",              "rpm"),
            ("rho",              "rho"),
            "blade_angles_deg",
            "sweep_m",
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
        StressComponent(blade=blade, n_stations=N_STATIONS, num_blades=blade.num_blades,
                        fd_step=fd_step),
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

    # ---- Control-point monotonicity (sweep non-decreasing, twist wash-out) -
    # Monotonicity on the N_CP control points is sufficient — the CubicSpline
    # is not guaranteed monotone by monotone CPs, but with N_CP=5 the
    # overshoot is small and the thickness/thrust constraints catch violations.
    blade_r_cp = np.linspace(blade.r_R[0], blade.r_R[-1], N_CP)
    baseline_twist_at_cp = np.interp(blade_r_cp, blade.r_R, blade.twist_deg)

    model.add_subsystem(
        "sweep_cp_mono",
        SmoothnessComponent(input_name="sweep_cp", output_name="sweep_cp_diff",
                            n_def=N_CP, fd_step=fd_step),
        promotes_inputs=["sweep_cp"],
        promotes_outputs=["sweep_cp_diff"],
    )
    model.add_subsystem(
        "twist_cp_mono",
        TwistMonotonicityComponent(baseline_twist=baseline_twist_at_cp,
                                   fd_step=fd_step),
        promotes_inputs=[("delta_twist_deg", "delta_twist_cp")],
        promotes_outputs=[("twist_def_diff", "twist_cp_diff")],
    )

    # ---- Driver ----------------------------------------------------------
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    if optimizer == "trust-constr":
        prob.driver.options["tol"]     = 1e-6
        prob.driver.options["maxiter"] = 1000
    else:
        prob.driver.options["tol"]     = 1e-4
        prob.driver.options["maxiter"] = 500

    # ---- Per-station physical thickness (chord × t/c) ----------------------
    model.add_subsystem(
        "thickness_all",
        om.ExecComp(
            "phys_thick = chord_m * tc_ratio",
            chord_m=np.zeros(N_STATIONS),
            tc_ratio=np.zeros(N_STATIONS),
            phys_thick=np.zeros(N_STATIONS),
        ),
        promotes_inputs=["chord_m", "tc_ratio"],
        promotes_outputs=["phys_thick"],
    )

    # ---- Design variables (21 active: 1 rpm + 4×5 spline CPs) -------------
    prob.model.add_design_var("rpm",            lower=RPM_LOWER, upper=RPM_UPPER, units="rpm")
    prob.model.add_design_var("delta_twist_cp", lower=-6.0,  upper=6.0)
    prob.model.add_design_var("delta_chord_cp", lower=-0.05, upper=0.03)
    prob.model.add_design_var("sweep_cp",       lower=0.0,   upper=0.12)
    prob.model.add_design_var("delta_tc_cp",    lower=-0.03, upper=0.04)

    # ---- Objective -------------------------------------------------------
    prob.model.add_objective("SPL_weighted")

    # ---- Constraints -----------------------------------------------------
    prob.model.add_constraint("thrust_hover",  lower=thrust_min_hover)
    prob.model.add_constraint("thrust_cruise", lower=thrust_min_cruise)
    prob.model.add_constraint("max_stress",    upper=ALLOWABLE_STRESS,
                                               ref=ALLOWABLE_STRESS)
    inner_stations = list(range(N_STATIONS - 1))
    prob.model.add_constraint("phys_thick",    indices=inner_stations,
                                               lower=MIN_PRINT_THICKNESS,
                                               ref=MIN_PRINT_THICKNESS)
    prob.model.add_constraint("power_hover",   lower=0.0, ref=10.0)
    # Sweep control points must be non-decreasing (monotone aft-sweep)
    prob.model.add_constraint("sweep_cp_diff", lower=0.0)
    # Total twist at control points must be monotonically decreasing (wash-out)
    prob.model.add_constraint("twist_cp_diff", lower=0.0)

    return prob


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_baseline(rpm=RPM_HOVER_INIT, rho=1.225, verbose=True):
    """Run baseline analysis (no optimisation)."""
    prob = build_problem(rpm_init=rpm, rho=rho)
    prob.setup()
    prob.run_model()
    if verbose:
        print("\n=== Baseline Analysis ===")
        _print_results(prob)
    return prob


def run_optimization(thrust_min_hover=THRUST_HOVER_MIN,
                     thrust_min_cruise=THRUST_CRUISE_MIN,
                     rpm_init=RPM_HOVER_INIT, rho=1.225, verbose=True,
                     use_hybrid=True, fd_step=3e-4, optimizer="SLSQP"):
    """
    Run the full MDAO optimisation.

    Hybrid strategy (use_hybrid=True):
      Phase 1: SimpleGADriver (pop=200, gen=50) for global search
      Phase 2: ScipyOptimizeDriver SLSQP starting from GA best
    """
    if use_hybrid:
        return _run_hybrid(thrust_min_hover, thrust_min_cruise,
                           rpm_init, rho, verbose, fd_step, optimizer)
    else:
        return _run_slsqp(thrust_min_hover, thrust_min_cruise,
                          rpm_init, rho, verbose, fd_step, optimizer)


def run_multistart(n_starts=8, thrust_min_hover=THRUST_HOVER_MIN,
                   thrust_min_cruise=THRUST_CRUISE_MIN,
                   rho=1.225, seed=42, verbose=True,
                   fd_step=3e-4, optimizer="SLSQP",
                   return_all=False, start_from=0,
                   plot_feasible=False, plot_dir=None):
    """
    Run SLSQP from n_starts random starting points; return the best result.

    Start 0 is always the baseline (RPM_HOVER_INIT, zero perturbations).
    Starts 1..n-1 are random perturbations biased toward noise-reducing directions:
      - chord biased low (LETI ∝ chord²)
      - sweep biased high (cos⁴ correction)
      - RPM near hover range (thrust constraint limits RPM travel)

    Parameters
    ----------
    return_all : bool
        If True, return (best_prob, all_results) where all_results is a list
        of dicts — one per start — containing scalars + a "dvs" sub-dict.
        Infeasible starts are included so their DVs can be used with
        run_from_point.  If False (default), return best_prob only.
    """
    blade  = baseline_apc7x5e()
    n_def  = N_BLADE_STAT
    rng    = np.random.default_rng(seed)
    best_spl       = np.inf
    best_infeas_spl = np.inf
    best_prob      = None
    best_infeas_prob = None
    all_results    = []

    # Advance RNG to match the state it would be at after start_from starts,
    # so results are reproducible regardless of where we resume from.
    for _ in range(min(start_from, n_starts)):
        if _ > 0:  # start 0 uses no RNG draws
            rng.uniform(6000.0, 8500.0)
            rng.uniform(-2.0, 2.0, N_CP)
            rng.uniform(-0.05, 0.01, N_CP)
            rng.uniform(0.0, 0.10, N_CP)
            rng.uniform(-0.02, 0.02, N_CP)

    _plot_dir = plot_dir or os.path.join(os.path.dirname(__file__),
                                          "..", "results", "plots")

    for i in range(start_from, n_starts):
        if i == 0:
            rpm_i  = RPM_HOVER_INIT
            dt_cp  = np.zeros(N_CP)
            dc_cp  = np.zeros(N_CP)
            sw_cp  = np.zeros(N_CP)
            dtc_cp = np.zeros(N_CP)
        else:
            rpm_i  = float(rng.uniform(6000.0, 8500.0))
            dt_cp  = rng.uniform(-2.0, 2.0, N_CP)
            dc_cp  = rng.uniform(-0.05, 0.01, N_CP)
            sw_cp  = np.sort(rng.uniform(0.0, 0.10, N_CP))
            dtc_cp = rng.uniform(-0.02, 0.02, N_CP)

        prob_i = build_problem(thrust_min_hover=thrust_min_hover,
                               thrust_min_cruise=thrust_min_cruise,
                               rpm_init=rpm_i, rho=rho,
                               fd_step=fd_step, optimizer=optimizer)
        prob_i.setup()
        prob_i.set_val("rpm",            rpm_i)
        prob_i.set_val("delta_twist_cp", dt_cp)
        prob_i.set_val("delta_chord_cp", dc_cp)
        prob_i.set_val("sweep_cp",       sw_cp)
        prob_i.set_val("delta_tc_cp",    dtc_cp)
        prob_i.set_val("theta2",         120.0)
        prob_i.set_val("theta3",         240.0)
        prob_i.run_driver()

        spl_i      = float(prob_i.get_val("SPL_weighted")[0])
        t_hover_i  = float(prob_i.get_val("thrust_hover")[0])
        t_cruise_i = float(prob_i.get_val("thrust_cruise")[0])
        stress_i   = float(prob_i.get_val("max_stress")[0])
        thrust_ok  = (t_hover_i  >= thrust_min_hover  * 0.99 and
                      t_cruise_i >= thrust_min_cruise * 0.99)

        result_i = {
            "start":        i,
            "spl":          spl_i,
            "thrust_hover": t_hover_i,
            "thrust_cruise":t_cruise_i,
            "max_stress":   stress_i,
            "feasible":     thrust_ok,
            "dvs": {
                "rpm":             float(prob_i.get_val("rpm")[0]),
                "delta_twist_cp":  prob_i.get_val("delta_twist_cp").copy(),
                "delta_chord_cp":  prob_i.get_val("delta_chord_cp").copy(),
                "sweep_cp":        prob_i.get_val("sweep_cp").copy(),
                "delta_tc_cp":     prob_i.get_val("delta_tc_cp").copy(),
                "theta2":          float(prob_i.get_val("theta2")[0]),
                "theta3":          float(prob_i.get_val("theta3")[0]),
            },
        }
        all_results.append(result_i)

        if verbose:
            tag = "  *BEST*" if (thrust_ok and spl_i < best_spl) else \
                  " (infeas)" if not thrust_ok else ""
            print(f"  [start {i}] SPL={spl_i:.2f} dBA  "
                  f"T_hov={t_hover_i:.3f} N  T_cru={t_cruise_i:.3f} N"
                  f"  stress={stress_i/1e6:.1f} MPa{tag}")

        if plot_feasible and thrust_ok:
            try:
                from results.plots.geometry_viz import plot_geometry
                _b_opt = prob_to_blade(prob_i)
                _rpm   = float(prob_i.get_val("rpm")[0])
                _path  = os.path.join(_plot_dir, f"start_{i}_{spl_i:.2f}dBA.png")
                plot_geometry(baseline_apc7x5e(), _b_opt,
                              opt_label=f"Start {i}: {spl_i:.2f} dBA @ {_rpm:.0f} RPM",
                              save_path=_path, show=False)
                print(f"  [start {i}] Geometry -> {_path}")
            except Exception as _e:
                print(f"  [start {i}] Plot failed: {_e}")

        if thrust_ok and spl_i < best_spl:
            best_spl  = spl_i
            best_prob = prob_i
        if not thrust_ok and spl_i < best_infeas_spl:
            best_infeas_spl  = spl_i
            best_infeas_prob = prob_i

    if best_prob is None:
        best_prob = best_infeas_prob or prob_i   # fallback if all infeasible

    if verbose:
        print(f"\n=== Multistart Best (feasible): {best_spl:.2f} dBA ===")
        _print_results(best_prob)
        _print_design_vars(best_prob)
        if best_infeas_prob is not None:
            print(f"\n=== Best Infeasible: {best_infeas_spl:.2f} dBA "
                  f"(use run_from_point to probe this region) ===")

    if return_all:
        return best_prob, all_results
    return best_prob


def prob_to_blade(prob):
    """
    Extract optimised B-spline control points from a solved prob and return a
    perturbed BladeGeometry suitable for passing to plot_geometry.

    Mirrors GeometryFullComponent.compute: evaluate CubicSpline at blade.r_R.
    """
    blade  = baseline_apc7x5e()
    r_cp   = np.linspace(blade.r_R[0], blade.r_R[-1], N_CP)
    r_def  = blade.r_R

    def _spline(cp_vals):
        return CubicSpline(r_cp, cp_vals)(r_def)

    dc_cp  = np.ravel(prob.get_val("delta_chord_cp"))
    dt_cp  = np.ravel(prob.get_val("delta_twist_cp"))
    dtc_cp = np.ravel(prob.get_val("delta_tc_cp"))
    sw_cp  = np.ravel(prob.get_val("sweep_cp"))
    th2    = float(prob.get_val("theta2")[0])
    th3    = float(prob.get_val("theta3")[0])

    dc  = _spline(dc_cp)
    dt  = _spline(dt_cp)
    dtc = _spline(dtc_cp)
    sw  = np.clip(_spline(sw_cp), 0.0, 0.12)

    return (blade
            .perturb_twist(dt)
            .perturb_chord(dc)
            .perturb_tc(dtc)
            .set_sweep(sw)
            .set_blade_angles([0.0, th2, th3]))


def run_from_point(dvs, thrust_min_hover=THRUST_HOVER_MIN,
                   thrust_min_cruise=THRUST_CRUISE_MIN,
                   rho=1.225, verbose=True, fd_step=3e-4, optimizer="SLSQP"):
    """
    Warm-start SLSQP (or trust-constr) from a given DV dict.

    Parameters
    ----------
    dvs : dict with keys matching DV names:
        "rpm", "delta_twist_cp", "delta_chord_cp", "sweep_cp", "delta_tc_cp"
        Missing keys fall back to their default initial values (zeros for CPs).

    Returns the optimised prob.

    Typical use: probe the neighbourhood of an infeasible multistart result
    (e.g. start 7 at 66.78 dBA but T_cruise infeasible) with a more robust
    solver to verify whether 66–67 dBA is achievable and feasible.
    """
    rpm_init = float(dvs.get("rpm", RPM_HOVER_INIT))
    prob = build_problem(thrust_min_hover=thrust_min_hover,
                         thrust_min_cruise=thrust_min_cruise,
                         rpm_init=rpm_init, rho=rho,
                         fd_step=fd_step, optimizer=optimizer)
    prob.setup()

    prob.set_val("rpm",            dvs.get("rpm",            RPM_HOVER_INIT))
    prob.set_val("delta_twist_cp", dvs.get("delta_twist_cp", np.zeros(N_CP)))
    prob.set_val("delta_chord_cp", dvs.get("delta_chord_cp", np.zeros(N_CP)))
    prob.set_val("sweep_cp",       dvs.get("sweep_cp",       np.zeros(N_CP)))
    prob.set_val("delta_tc_cp",    dvs.get("delta_tc_cp",    np.zeros(N_CP)))
    prob.set_val("theta2",         120.0)
    prob.set_val("theta3",         240.0)

    if verbose:
        print(f"\n=== run_from_point: {optimizer}, fd_step={fd_step:.0e} ===")
        print("  Starting point:")
        prob.run_model()
        _print_results(prob)
        print("\n  Optimising…")

    prob.run_driver()

    if verbose:
        print("\n=== run_from_point result ===")
        _print_results(prob)
        _print_design_vars(prob)

    return prob


def _repair_tc(blade, dc_R, dtc, n_stations=N_STATIONS):
    """
    Repair GA geometry so that chord × tc >= MIN_PRINT_THICKNESS everywhere.

    Pass 1: raise delta_tc at violated stations (preferred — preserves chord).
    Pass 2: if t/c is already at its upper bound (+0.04) and still too thin,
            raise delta_chord_R just enough to close the gap.  This handles
            the case where chord is near its lower bound and t/c alone cannot
            satisfy the wall-thickness constraint.

    Returns (dc_R_repaired, dtc_repaired) — both arrays at the 18 definition
    stations, clipped to their respective DV bounds.
    """
    r_m, chord0, _, tc0, _, _ = blade.get_full_stations(n_stations)
    R      = blade.radius_m
    r_def  = blade.r_R * R

    dc_interp  = np.interp(r_m, r_def, dc_R).copy()
    dtc_interp = np.interp(r_m, r_def, dtc).copy()

    chord_m = np.maximum(chord0 + dc_interp * R, 1e-4)

    for i in range(len(r_m)):
        phys = chord_m[i] * (tc0[i] + dtc_interp[i])
        if phys >= MIN_PRINT_THICKNESS:
            continue

        # Pass 1: raise t/c
        dtc_needed = MIN_PRINT_THICKNESS / chord_m[i] - tc0[i]
        if dtc_needed <= 0.04:
            dtc_interp[i] = dtc_needed
            continue

        # Pass 2: t/c maxed — raise chord enough to cover the remaining gap
        dtc_interp[i] = 0.04
        tc_at_max     = tc0[i] + 0.04
        chord_needed  = MIN_PRINT_THICKNESS / tc_at_max
        dc_interp[i] += (chord_needed - chord_m[i]) / R
        chord_m[i]    = chord_needed

    dc_repaired  = np.clip(np.interp(r_def, r_m, dc_interp),  -0.05, 0.03)
    dtc_repaired = np.clip(np.interp(r_def, r_m, dtc_interp), -0.03, 0.04)
    return dc_repaired, dtc_repaired


def _run_slsqp(thrust_min_hover, thrust_min_cruise, rpm_init, rho, verbose,
               fd_step=3e-4, optimizer="SLSQP"):
    prob = build_problem(thrust_min_hover=thrust_min_hover,
                         thrust_min_cruise=thrust_min_cruise,
                         rpm_init=rpm_init, rho=rho,
                         fd_step=fd_step, optimizer=optimizer)
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


def _feasible_start(thrust_min, rho, rpm_lo=RPM_LOWER, rpm_hi=RPM_UPPER):
    """
    Find a thrust- and structure-feasible starting point for SLSQP when the
    GA result is infeasible.

    Strategy: use the structurally-strongest blade within DV bounds
    (max chord +0.03, max t/c +0.04). This geometry:
      - Achieves thrust_min at lower RPM (wider blade = more lift per rev)
      - Has larger cross-sectional area -> lower centrifugal stress
      - Has greater physical thickness -> satisfies min-wall constraint

    Returns (rpm, dc_R, dtc) where dc_R and dtc are the DV arrays that
    define the starting geometry.
    """
    blade = baseline_apc7x5e()
    n = len(blade.r_R)
    dc   = np.full(n, 0.025)   # +0.025 R chord -> stays within +0.03 bound
    dtc  = np.full(n, 0.035)   # +0.035 tc   -> stays within +0.04 bound
    blade_strong = blade.perturb_chord(dc).perturb_tc(dtc)

    lo, hi = rpm_lo, rpm_hi
    for _ in range(20):          # ~20 bisection steps => 1 RPM accuracy
        mid = 0.5 * (lo + hi)
        res = bem_solve(blade_strong, mid, v_inf=0.0, rho=rho,
                        n_stations=N_STATIONS)
        if res["thrust"] >= thrust_min:
            hi = mid
        else:
            lo = mid
    return float(hi), dc, dtc


def _run_hybrid(thrust_min_hover, thrust_min_cruise, rpm_init, rho, verbose,
                fd_step=3e-4, optimizer="SLSQP"):
    """GA global search -> SLSQP local refinement."""

    # ---- Phase 1: GA global search ----------------------------------------
    if verbose:
        print("\n=== Phase 1: GA global search (pop=200, gen=50) ===")

    prob_ga = build_problem(thrust_min_hover=thrust_min_hover,
                            thrust_min_cruise=thrust_min_cruise,
                            rpm_init=rpm_init, rho=rho,
                            fd_step=fd_step, optimizer=optimizer)
    prob_ga.driver = om.SimpleGADriver()
    prob_ga.driver.options["pop_size"]         = 200
    prob_ga.driver.options["max_gen"]          = 50
    prob_ga.driver.options["Pc"]               = 0.5    # crossover probability
    prob_ga.driver.options["Pm"]               = 0.01   # mutation probability
    prob_ga.driver.options["penalty_parameter"] = 100.0  # strong constraint enforcement
    prob_ga.setup()
    prob_ga.run_driver()

    # Capture GA result
    ga_rpm        = float(prob_ga.get_val("rpm")[0])
    ga_dt         = prob_ga.get_val("delta_twist_cp").copy()
    ga_dc         = prob_ga.get_val("delta_chord_cp").copy()
    ga_sw         = prob_ga.get_val("sweep_cp").copy()
    ga_dtc        = prob_ga.get_val("delta_tc_cp").copy()
    ga_th2        = float(prob_ga.get_val("theta2")[0])
    ga_th3        = float(prob_ga.get_val("theta3")[0])
    ga_thrust     = float(prob_ga.get_val("thrust_hover")[0])
    ga_min_thick  = float(prob_ga.get_val("min_thickness")[0])
    ga_max_stress = float(prob_ga.get_val("max_stress")[0])

    if verbose:
        print("GA result:")
        _print_results(prob_ga)
        print("\n=== Phase 2: SLSQP refinement ===")

    # ---- Feasibility check: SLSQP must start inside the feasible region ----
    ga_thrust_bad = ga_thrust    < thrust_min_hover  * 0.95
    ga_stress_bad = ga_max_stress > ALLOWABLE_STRESS  * 1.05
    ga_thick_bad  = ga_min_thick < MIN_PRINT_THICKNESS * 0.95

    if ga_thrust_bad or ga_stress_bad:
        if verbose:
            print(f"[HYBRID] GA hard infeasible: thrust={ga_thrust:.2f} N, "
                  f"stress={ga_max_stress/1e6:.1f} MPa — falling back to baseline CPs")
        start_rpm = RPM_HOVER_INIT
        start_dt  = np.zeros(N_CP)
        start_dc  = np.zeros(N_CP)
        start_sw  = np.zeros(N_CP)
        start_dtc = np.zeros(N_CP)
        start_th2 = 120.0
        start_th3 = 240.0
    else:
        start_rpm = ga_rpm
        start_dt  = ga_dt
        start_dc  = ga_dc
        start_sw  = ga_sw
        start_dtc = ga_dtc
        start_th2 = ga_th2
        start_th3 = ga_th3

    # ---- Phase 2: SLSQP local refinement ----------------------------------
    prob = build_problem(thrust_min_hover=thrust_min_hover,
                         thrust_min_cruise=thrust_min_cruise,
                         rpm_init=start_rpm, rho=rho,
                         fd_step=fd_step, optimizer=optimizer)
    prob.setup()

    prob.set_val("rpm",            start_rpm)
    prob.set_val("delta_twist_cp", start_dt)
    prob.set_val("delta_chord_cp", start_dc)
    prob.set_val("sweep_cp",       start_sw)
    prob.set_val("delta_tc_cp",    start_dtc)
    prob.set_val("theta2",         start_th2)
    prob.set_val("theta3",         start_th3)

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
    # phys_thick is the per-station constraint array; tip (last station, r/R=1)
    # is excluded from the 0.5 mm constraint because the APC baseline is 0.42 mm
    # there.  Report the inner-span minimum as the binding structural value.
    try:
        phys = np.ravel(prob.get_val('phys_thick'))
        inner_min_mm = float(np.min(phys[:-1])) * 1e3
        print(f"  Min thickness: {inner_min_mm:.2f} mm  (>=0.5 mm, inner span; tip excluded)")
    except Exception:
        print(f"  Min thickness: {float(prob.get_val('min_thickness')):.3f} m  (StressComponent scalar)")
    print(f"  Imbalance    : {_g('imbalance_factor'):.4f}  (<=0.05)")
    th2 = _g('theta2')
    th3 = _g('theta3')
    print(f"  Blade angles : 0 / {th2:.1f} / {th3:.1f} deg")


def _print_design_vars(prob):
    def _g(name):
        try:
            return prob.get_val(name)
        except Exception:
            return np.array([float("nan")])

    print(f"\n  dtwist_cp (deg) : {np.round(_g('delta_twist_cp'), 3)}")
    print(f"  dchord_cp (R)   : {np.round(_g('delta_chord_cp'), 4)}")
    print(f"  sweep_cp (R)    : {np.round(_g('sweep_cp'), 4)}")
    print(f"  delta_tc_cp     : {np.round(_g('delta_tc_cp'), 4)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run quiet-prop MDAO optimiser")
    parser.add_argument("--starts",      type=int,   default=8,       help="Number of multistart runs")
    parser.add_argument("--start-from",  type=int,   default=0,       help="Resume from this start index")
    parser.add_argument("--optimizer",   type=str,   default="SLSQP", help="SLSQP or trust-constr")
    parser.add_argument("--fd-step",     type=float, default=3e-4,    help="FD step size")
    parser.add_argument("--no-plot",     action="store_true",          help="Skip final geometry visualisation")
    parser.add_argument("--plot-starts", action="store_true",          help="Plot geometry after each feasible start")
    args = parser.parse_args()

    prob, all_results = run_multistart(
        n_starts=args.starts,
        optimizer=args.optimizer,
        fd_step=args.fd_step,
        return_all=True,
        start_from=args.start_from,
        plot_feasible=args.plot_starts,
    )

    # Report best infeasible start so the user can probe manually with run_from_point
    infeas = [r for r in all_results if not r["feasible"]]
    if infeas:
        best_infeas = min(infeas, key=lambda r: r["spl"])
        best_feasible_spl = float(prob.get_val("SPL_weighted")[0])
        if best_infeas["spl"] < best_feasible_spl:
            print(f"\n[INFO] Best infeasible start {best_infeas['start']} "
                  f"({best_infeas['spl']:.2f} dBA) beat best feasible "
                  f"({best_feasible_spl:.2f} dBA) — DVs saved in all_results.")

    if not args.no_plot:
        try:
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from results.plots.geometry_viz import plot_geometry

            blade_base = baseline_apc7x5e()
            blade_opt  = prob_to_blade(prob)
            spl_val    = float(prob.get_val("SPL_weighted")[0])
            rpm_val    = float(prob.get_val("rpm")[0])

            out_path = os.path.join(os.path.dirname(__file__),
                                    "..", "results", "plots", "optimum_geometry.png")
            plot_geometry(
                blade_base, blade_opt,
                opt_label=f"Optimum {spl_val:.2f} dBA @ {rpm_val:.0f} RPM",
                save_path=out_path,
            )
        except Exception as e:
            print(f"[WARN] Geometry plot failed: {e}")
