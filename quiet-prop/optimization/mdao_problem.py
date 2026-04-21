"""
OpenMDAO MDAO problem – Phase 2.

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

2. Structural constraint: centrifugal + bending root stress <= 22 MPa;
   minimum wall thickness >= 0.5 mm.

3. Hybrid optimizer: GA global search (pop=50, gen=30) -> SLSQP local
   refinement to escape local minima in the 57-variable space.

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

Total design variables: 1 + 18 + 18 + 18 + 18 + 10 + 1 + 1 = 85
"""

import sys
import os
import numpy as np
import openmdao.api as om

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.blade_generator import baseline_apc7x5e, BladeGeometry
from aerodynamics.ccblade_component import CCBladeComponent, bem_solve
from acoustics.bpm_component import BPMComponent, bpm_noise
from structures.structural_component import (StressComponent, ALLOWABLE_STRESS,
                                              MIN_PRINT_THICKNESS)

N_STATIONS   = 20
N_BLADE_STAT = 18          # blade definition stations (matches APC 7x5E baseline)
N_TIP_ZONES  = 10          # dihedral design var stations (outer 55–100% span)
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
        blade = self.options["blade"] or baseline_apc7x5e()
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

    def setup(self):
        n = len(self.options["baseline_twist"])
        self.add_input("delta_twist_deg", val=np.zeros(n))
        self.add_output("twist_def_diff",  val=np.zeros(n - 1))

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=1e-4)

    def compute(self, inputs, outputs):
        total = self.options["baseline_twist"] + inputs["delta_twist_deg"]
        outputs["twist_def_diff"] = total[:-1] - total[1:]


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
                  rpm_init=RPM_HOVER_INIT, rho=1.225, max_imbalance=0.05):
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

    # ---- Twist monotonicity at definition level (wash-out enforcement) ------
    model.add_subsystem(
        "twist_mono",
        TwistMonotonicityComponent(baseline_twist=blade.twist_deg.copy()),
        promotes_inputs=["delta_twist_deg"],
        promotes_outputs=["twist_def_diff"],
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

    # ---- Per-station physical thickness (chord × t/c) ----------------------
    # One constraint per station gives SLSQP gradient information on exactly
    # which stations are thin — much more tractable than a single min() scalar.
    # sweep_R and z_offset_R_tip are NOT design variables: BEM and BPM both
    # ignore 3D blade geometry, so they have zero gradient everywhere and
    # degrade the SLSQP Jacobian rank.  Keep them in the IVC at their
    # zero defaults but do not optimise over them.
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

    # ---- Design variables (57 active; sweep/dihedral excluded) -----------
    prob.model.add_design_var("rpm",             lower=RPM_LOWER, upper=RPM_UPPER, units="rpm")
    prob.model.add_design_var("delta_twist_deg", lower=-5.0,    upper=5.0)
    prob.model.add_design_var("delta_chord_R",   lower=-0.03,   upper=0.03)
    prob.model.add_design_var("delta_tc",        lower=-0.03,   upper=0.04)
    prob.model.add_design_var("theta2",          lower=105.0,   upper=135.0)
    prob.model.add_design_var("theta3",          lower=225.0,   upper=255.0)

    # ---- Objective -------------------------------------------------------
    prob.model.add_objective("SPL_weighted")

    # ---- Constraints -----------------------------------------------------
    prob.model.add_constraint("thrust_hover",     lower=thrust_min_hover)
    prob.model.add_constraint("thrust_cruise",    lower=thrust_min_cruise)
    prob.model.add_constraint("max_stress",       upper=ALLOWABLE_STRESS,
                                                  ref=ALLOWABLE_STRESS)
    # Per-station physical thickness: replaces scalar min_thickness to give
    # SLSQP station-level gradient information.
    # Station N_STATIONS-1 (r/R=1.0 exact tip) is excluded: the APC 7x5E
    # baseline tip chord is 5.3 mm with tc=0.078 → 0.41 mm < 0.5 mm even at
    # zero delta.  Enforcing 0.5 mm there is infeasible and traps SLSQP.
    inner_stations = list(range(N_STATIONS - 1))
    prob.model.add_constraint("phys_thick",       indices=inner_stations,
                                                  lower=MIN_PRINT_THICKNESS,
                                                  ref=MIN_PRINT_THICKNESS)
    prob.model.add_constraint("imbalance_factor", upper=max_imbalance)
    # Hover power must be positive — prevents BEM stalled-root artefacts.
    prob.model.add_constraint("power_hover",      lower=0.0,  ref=10.0)
    # Enforce monotonically decreasing twist (aerodynamic wash-out).
    prob.model.add_constraint("twist_def_diff",    lower=0.0)

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


def _repair_tc(blade, dc_R, dtc, n_stations=N_STATIONS):
    """
    Clip delta_tc upward so that chord × tc >= MIN_PRINT_THICKNESS at every
    evaluation station.  Preserves the GA's chord/twist/spacing solution and
    only adjusts t/c as needed — avoids the heavy-handed _feasible_start reset.
    """
    r_m, chord0, _, tc0, _, _ = blade.get_full_stations(n_stations)
    R = blade.radius_m

    # Interpolate delta_chord from 18 definition to n_stations evaluation
    r_def = blade.r_R * R
    dc_interp = np.interp(r_m, r_def, dc_R)
    dtc_interp = np.interp(r_m, r_def, dtc)

    chord_m = chord0 + dc_interp * R
    chord_m = np.maximum(chord_m, 1e-4)
    tc_min  = MIN_PRINT_THICKNESS / chord_m - tc0  # minimum delta_tc per station
    dtc_interp = np.maximum(dtc_interp, tc_min)
    dtc_interp = np.clip(dtc_interp, -0.03, 0.04)

    # Map repaired evaluation values back to 18 definition stations
    r_def = blade.r_R * R
    dtc_repaired = np.interp(r_def, r_m, dtc_interp)
    return np.clip(dtc_repaired, -0.03, 0.04)


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


def _run_hybrid(thrust_min_hover, thrust_min_cruise, rpm_init, rho, verbose):
    """GA global search -> SLSQP local refinement."""

    # ---- Phase 1: GA global search ----------------------------------------
    if verbose:
        print("\n=== Phase 1: GA global search (pop=50, gen=30) ===")

    prob_ga = build_problem(thrust_min_hover=thrust_min_hover,
                            thrust_min_cruise=thrust_min_cruise,
                            rpm_init=rpm_init, rho=rho)
    prob_ga.driver = om.SimpleGADriver()
    prob_ga.driver.options["pop_size"]         = 50
    prob_ga.driver.options["max_gen"]          = 30
    prob_ga.driver.options["Pc"]               = 0.5    # crossover probability
    prob_ga.driver.options["Pm"]               = 0.01   # mutation probability
    prob_ga.driver.options["penalty_parameter"] = 100.0  # strong constraint enforcement
    prob_ga.setup()
    prob_ga.run_driver()

    # Capture GA result
    ga_rpm       = float(prob_ga.get_val("rpm")[0])
    ga_dt        = prob_ga.get_val("delta_twist_deg").copy()
    ga_dc        = prob_ga.get_val("delta_chord_R").copy()
    ga_sw        = prob_ga.get_val("sweep_R").copy()
    ga_dtc       = prob_ga.get_val("delta_tc").copy()
    ga_z         = prob_ga.get_val("z_offset_R_tip").copy()
    ga_th2       = float(prob_ga.get_val("theta2")[0])
    ga_th3       = float(prob_ga.get_val("theta3")[0])
    ga_thrust    = float(prob_ga.get_val("thrust_hover")[0])
    ga_min_thick = float(prob_ga.get_val("min_thickness")[0])
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
        # Hard structural violation: fall back to the known-feasible geometry.
        rpm_feasible, dc_feas, dtc_feas = _feasible_start(thrust_min_hover, rho)
        if verbose:
            print(f"[HYBRID] GA hard infeasible: thrust={ga_thrust:.2f} N, "
                  f"stress={ga_max_stress/1e6:.1f} MPa")
            print(f"[HYBRID] Full fallback: RPM {ga_rpm:.0f} -> {rpm_feasible:.0f}")
        start_rpm = rpm_feasible
        start_dt  = np.zeros_like(ga_dt)
        start_dc  = dc_feas
        start_sw  = np.zeros_like(ga_sw)
        start_dtc = dtc_feas
        start_z   = np.zeros_like(ga_z)
        start_th2 = ga_th2
        start_th3 = ga_th3
    elif ga_thick_bad:
        # Only thickness violated: repair delta_tc, keep the rest of the GA solution.
        blade_tmp = baseline_apc7x5e()
        dtc_repaired = _repair_tc(blade_tmp, ga_dc, ga_dtc)
        if verbose:
            print(f"[HYBRID] GA thin blade (min={ga_min_thick*1e3:.2f} mm); "
                  f"repairing delta_tc — keeping RPM/twist/chord/spacing from GA.")
        start_rpm = ga_rpm
        start_dt  = ga_dt
        start_dc  = ga_dc
        start_sw  = ga_sw
        start_dtc = dtc_repaired
        start_z   = ga_z
        start_th2 = ga_th2
        start_th3 = ga_th3
    else:
        start_rpm = ga_rpm
        start_dt  = ga_dt
        start_dc  = ga_dc
        start_sw  = ga_sw
        start_dtc = ga_dtc
        start_z   = ga_z
        start_th2 = ga_th2
        start_th3 = ga_th3

    # ---- Phase 2: SLSQP local refinement ----------------------------------
    prob = build_problem(thrust_min_hover=thrust_min_hover,
                         thrust_min_cruise=thrust_min_cruise,
                         rpm_init=start_rpm, rho=rho)
    prob.setup()

    prob.set_val("rpm",             start_rpm)
    prob.set_val("delta_twist_deg", start_dt)
    prob.set_val("delta_chord_R",   start_dc)
    prob.set_val("sweep_R",         start_sw)
    prob.set_val("delta_tc",        start_dtc)
    prob.set_val("z_offset_R_tip",  start_z)
    prob.set_val("theta2",          start_th2)
    prob.set_val("theta3",          start_th3)

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

    print(f"\n  dtwist (deg)    : {np.round(_g('delta_twist_deg'), 3)}")
    print(f"  dchord (R)      : {np.round(_g('delta_chord_R'), 4)}")
    print(f"  sweep (R)       : {np.round(_g('sweep_R'), 4)}")
    print(f"  delta_tc        : {np.round(_g('delta_tc'), 4)}")
    print(f"  z_offset_R_tip  : {np.round(_g('z_offset_R_tip'), 4)}")


if __name__ == "__main__":
    # 928 g AUW, TWR 2.5 (5.69 N/rotor), RPM_HOVER_INIT = 7000, hybrid GA+SLSQP
    prob = run_optimization(use_hybrid=True)
