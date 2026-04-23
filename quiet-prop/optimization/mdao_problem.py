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

        r_R_n = np.linspace(blade.r_R[0], blade.r_R[-1], N)
        camber0 = np.interp(r_R_n, blade.r_R, blade.camber_dist)

        self.add_input("delta_twist_cp",  val=np.zeros(N_CP))
        self.add_input("delta_chord_cp",  val=np.zeros(N_CP))
        self.add_input("sweep_cp",        val=np.zeros(N_CP))
        self.add_input("delta_tc_cp",     val=np.zeros(N_CP))
        self.add_input("delta_camber_cp", val=np.zeros(N_CP))
        self.add_input("theta2",          val=120.0)
        self.add_input("theta3",          val=240.0)

        self.add_output("chord_m",          val=chord0,  units="m")
        self.add_output("twist_deg",        val=twist0)
        self.add_output("tc_ratio",         val=tc0)
        self.add_output("sweep_m",          val=np.zeros(N), units="m")
        self.add_output("z_offset_m",       val=np.zeros(N), units="m")
        self.add_output("blade_angles_deg", val=np.array([0.0, 120.0, 240.0]))
        self.add_output("phys_thick_diff",  val=np.zeros(N - 1))
        self.add_output("camber",           val=camber0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=self.options["fd_step"])

    def _spline(self, cp_vals, r_def):
        """Evaluate cubic spline defined by N_CP control points at r_def stations."""
        return CubicSpline(self._r_cp, cp_vals)(r_def)

    def compute(self, inputs, outputs):
        blade  = self._blade
        N      = self.options["n_stations"]
        r_def  = blade.r_R

        dc_def  = self._spline(inputs["delta_chord_cp"],  r_def)
        dt_def  = self._spline(inputs["delta_twist_cp"],  r_def)
        dtc_def = self._spline(inputs["delta_tc_cp"],     r_def)
        sw_def  = np.clip(self._spline(inputs["sweep_cp"], r_def), 0.0, 0.12)
        dcam_def = self._spline(inputs["delta_camber_cp"], r_def)

        perturbed = (blade
                     .perturb_twist(dt_def)
                     .perturb_chord(dc_def)
                     .perturb_tc(dtc_def)
                     .set_sweep(sw_def)
                     .set_camber(blade.camber_dist + dcam_def)
                     .set_blade_angles([0.0,
                                        float(inputs["theta2"][0]),
                                        float(inputs["theta3"][0])]))

        r_m, chord_m, twist_deg, tc, sw, zof = perturbed.get_full_stations(N)

        # Interpolate camber to N_STATIONS
        r_R_n = np.linspace(blade.r_R[0], blade.r_R[-1], N)
        camber_n = np.interp(r_R_n, blade.r_R, perturbed.camber_dist)

        outputs["chord_m"]          = chord_m
        outputs["twist_deg"]        = twist_deg
        outputs["tc_ratio"]         = tc
        outputs["sweep_m"]          = sw
        outputs["z_offset_m"]       = zof
        outputs["blade_angles_deg"] = perturbed.blade_angles_deg
        phys_thick = chord_m * tc
        outputs["phys_thick_diff"]  = phys_thick[1:] - phys_thick[:-1]
        outputs["camber"]           = camber_n


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
                  twist_smooth_max=2.0, chord_smooth_max=0.02,
                  geometry_dvs=True, le_type="sawtooth"):
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
    ivc.add_output("rpm",              val=rpm_init,        units="rpm")
    ivc.add_output("rho",              val=rho,             units="kg/m**3")
    ivc.add_output("v_hover",          val=0.0,             units="m/s")
    ivc.add_output("v_cruise",         val=CRUISE_VINF,     units="m/s")
    ivc.add_output("delta_twist_cp",   val=np.zeros(N_CP))
    ivc.add_output("delta_chord_cp",   val=np.zeros(N_CP))
    ivc.add_output("sweep_cp",         val=np.zeros(N_CP))
    ivc.add_output("delta_tc_cp",      val=np.zeros(N_CP))
    ivc.add_output("delta_camber_cp",  val=np.zeros(N_CP))
    ivc.add_output("h_s_cp",          val=np.zeros(N_CP), units="m")
    ivc.add_output("h_LE_cp",         val=np.zeros(N_CP), units="m")
    ivc.add_output("theta2",           val=120.0)
    ivc.add_output("theta3",           val=240.0)

    # ---- Geometry perturbation (B-spline) --------------------------------
    model.add_subsystem(
        "geom",
        GeometryFullComponent(blade=blade, n_stations=N_STATIONS, fd_step=fd_step),
        promotes_inputs=["delta_twist_cp", "delta_chord_cp",
                         "sweep_cp", "delta_tc_cp", "delta_camber_cp",
                         "theta2", "theta3"],
        promotes_outputs=["chord_m", "twist_deg", "tc_ratio",
                          "sweep_m", "z_offset_m", "blade_angles_deg",
                          "phys_thick_diff", "camber"],
    )

    # ---- Hover aerodynamics ----------------------------------------------
    model.add_subsystem(
        "hover_aero",
        CCBladeComponent(blade=blade, n_stations=N_STATIONS, fd_step=fd_step),
        promotes_inputs=[("rpm", "rpm"), ("v_inf", "v_hover"),
                         ("rho", "rho"), "chord_m", "twist_deg", "camber"],
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
            ("dT_dr",      "dT_dr_hover"),
            ("dQ_dr",      "dQ_dr_hover"),
        ],
    )

    # ---- Hover acoustics -------------------------------------------------
    model.add_subsystem(
        "hover_acoustics",
        BPMComponent(blade=blade, n_stations=N_STATIONS, n_cp=N_CP, r_obs=1.0,
                     fd_step=fd_step, le_type=le_type),
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
            ("dT_dr",            "dT_dr_hover"),
            ("dQ_dr",            "dQ_dr_hover"),
            "h_s_cp",
            "h_LE_cp",
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
                         ("rho", "rho"), "chord_m", "twist_deg", "camber"],
        promotes_outputs=[
            ("thrust",     "thrust_cruise"),
            ("torque",     "torque_cruise"),
            ("r_m",        "r_m_cruise"),
            ("v_rel",      "v_rel_cruise"),
            ("aoa_deg",    "aoa_deg_cruise"),
            ("cl",         "cl_cruise"),
            ("cd",         "cd_cruise"),
            ("x_tr_c",     "x_tr_c_cruise"),
            ("dT_dr",      "dT_dr_cruise"),
            ("dQ_dr",      "dQ_dr_cruise"),
        ],
    )

    # ---- Cruise acoustics ------------------------------------------------
    model.add_subsystem(
        "cruise_acoustics",
        BPMComponent(blade=blade, n_stations=N_STATIONS, n_cp=N_CP, r_obs=1.0,
                     fd_step=fd_step, le_type=le_type),
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
            ("dT_dr",            "dT_dr_cruise"),
            ("dQ_dr",            "dQ_dr_cruise"),
            "h_s_cp",
            "h_LE_cp",
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

    # ---- Structural smoothness constraints (CP-level) ----------------------
    # Sweep: monotone non-decreasing only (no rate cap — relaxed for convergence)
    model.add_subsystem(
        "sweep_cp_mono",
        SmoothnessComponent(input_name="sweep_cp", output_name="sweep_cp_diff",
                            n_def=N_CP, fd_step=fd_step),
        promotes_inputs=["sweep_cp"],
        promotes_outputs=["sweep_cp_diff"],
    )
    # t/c smoothness — prevents oscillating thickness distribution
    model.add_subsystem(
        "tc_cp_smooth",
        SmoothnessComponent(input_name="delta_tc_cp", output_name="delta_tc_diff",
                            n_def=N_CP, fd_step=fd_step),
        promotes_inputs=["delta_tc_cp"],
        promotes_outputs=["delta_tc_diff"],
    )
    # Camber smoothness — prevents oscillating camber distribution
    model.add_subsystem(
        "camber_cp_smooth",
        SmoothnessComponent(input_name="delta_camber_cp", output_name="delta_camber_diff",
                            n_def=N_CP, fd_step=fd_step),
        promotes_inputs=["delta_camber_cp"],
        promotes_outputs=["delta_camber_diff"],
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

    # ---- Design variables -----------------------------------------------
    prob.model.add_design_var("rpm", lower=RPM_LOWER, upper=RPM_UPPER, units="rpm")

    if geometry_dvs:
        prob.model.add_design_var("delta_twist_cp",  lower=-5.0,   upper=5.0)
        prob.model.add_design_var("delta_chord_cp",  lower=-0.015, upper=0.03)
        prob.model.add_design_var("sweep_cp",        lower=0.0,    upper=0.12)
        prob.model.add_design_var("delta_tc_cp",     lower=-0.03,  upper=0.04)
        prob.model.add_design_var("delta_camber_cp", lower=-0.02,  upper=0.03)
        prob.model.add_design_var("h_s_cp", lower=0.0, upper=0.008, units="m")

    # LE serration amplitude: per-CP chord-proportional upper bound.
    # Elegoo Saturn 4 Ultra 16K: 19 um XY pixel -> min reliable feature ~10 px = 0.19 mm.
    # With h/lambda = 0.5: min acoustic h_LE ~ 0.1 mm (lower = 0 allows no-serration).
    # Upper bound: 15% of local chord — keeps teeth structurally intact.
    _, chord_cp_le, _ = blade.get_stations(N_CP)
    h_LE_max_cp = 0.15 * chord_cp_le   # metres, per CP
    prob.model.add_design_var("h_LE_cp", lower=0.0, upper=h_LE_max_cp, units="m")
    # Blade angles locked at 120°/240° — unequal spacing tested but the imbalance
    # force at 0.05 factor corresponds to ~12 N rotating vibration (G250 quality),
    # destroying motor bearings before any acoustic benefit is realised.
    # For 3-blade rotors the balance constraint limits deviation to <1° → <0.001 dBA.

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
    # Sweep: non-decreasing (monotone only, no rate cap)
    prob.model.add_constraint("sweep_cp_diff", lower=0.0)
    # t/c smoothness: adjacent CP changes capped at ±0.02
    prob.model.add_constraint("delta_tc_diff",     lower=-0.02, upper=0.02)
    # Camber smoothness: adjacent CP changes capped at ±0.02
    prob.model.add_constraint("delta_camber_diff", lower=-0.02, upper=0.02)
    # Physical thickness non-increasing root→tip (inner span, exclude tip)
    inner_mono = list(range(N_STATIONS - 2))
    prob.model.add_constraint("phys_thick_diff", indices=inner_mono, upper=0.0,
                              ref=MIN_PRINT_THICKNESS)

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


def _start_worker(args):
    """
    Module-level worker for multiprocessing.Pool — required for Windows spawn.
    Suppresses all stdout/stderr to avoid interleaved output.
    Returns a result dict with all fields needed for reporting and visualisation.
    """
    import os, sys
    (i, rpm_i, dt_cp, dc_cp, sw_cp, dtc_cp, dcam_cp, hs_cp,
     thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer,
     plot_feasible, plot_dir) = args

    # Disable OpenMDAO HTML reports (all workers share the same dir → collision)
    os.environ["OPENMDAO_REPORTS"] = "0"

    # Suppress OpenMDAO / SLSQP output in worker processes
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = devnull

    try:
        prob_i = build_problem(thrust_min_hover=thrust_min_hover,
                               thrust_min_cruise=thrust_min_cruise,
                               rpm_init=rpm_i, rho=rho,
                               fd_step=fd_step, optimizer=optimizer)
        prob_i.setup()
        prob_i.set_val("rpm",              rpm_i)
        prob_i.set_val("delta_twist_cp",   dt_cp)
        prob_i.set_val("delta_chord_cp",   dc_cp)
        prob_i.set_val("sweep_cp",         sw_cp)
        prob_i.set_val("delta_tc_cp",      dtc_cp)
        prob_i.set_val("delta_camber_cp",  dcam_cp)
        prob_i.set_val("h_s_cp",           hs_cp)
        prob_i.set_val("theta2",           120.0)
        prob_i.set_val("theta3",           240.0)
        prob_i.run_driver()

        spl_i      = float(prob_i.get_val("SPL_weighted")[0])
        t_hover_i  = float(prob_i.get_val("thrust_hover")[0])
        t_cruise_i = float(prob_i.get_val("thrust_cruise")[0])
        stress_i   = float(prob_i.get_val("max_stress")[0])
        thrust_ok  = (t_hover_i  >= thrust_min_hover  * 0.99 and
                      t_cruise_i >= thrust_min_cruise * 0.99)

        result_i = {
            "start":         i,
            "spl":           spl_i,
            "thrust_hover":  t_hover_i,
            "thrust_cruise": t_cruise_i,
            "max_stress":    stress_i,
            "feasible":      thrust_ok,
            "dvs": {
                "rpm":             float(prob_i.get_val("rpm")[0]),
                "delta_twist_cp":  prob_i.get_val("delta_twist_cp").copy(),
                "delta_chord_cp":  prob_i.get_val("delta_chord_cp").copy(),
                "sweep_cp":        prob_i.get_val("sweep_cp").copy(),
                "delta_tc_cp":     prob_i.get_val("delta_tc_cp").copy(),
                "delta_camber_cp": prob_i.get_val("delta_camber_cp").copy(),
                "h_s_cp":          prob_i.get_val("h_s_cp").copy(),
                "h_LE_cp":         prob_i.get_val("h_LE_cp").copy(),
                "theta2":          float(prob_i.get_val("theta2")[0]),
                "theta3":          float(prob_i.get_val("theta3")[0]),
            },
        }

        if plot_feasible and thrust_ok:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                from results.plots.geometry_viz import plot_geometry
                _b_opt = dvs_to_blade(result_i["dvs"])
                _path  = os.path.join(plot_dir, f"start_{i}_{spl_i:.2f}dBA.png")
                plot_geometry(baseline_apc7x5e(), _b_opt,
                              opt_label=f"Start {i}: {spl_i:.2f} dBA @ {rpm_i:.0f} RPM",
                              save_path=_path, show=False)
                result_i["plot_path"] = _path
            except Exception as _e:
                result_i["plot_error"] = str(_e)
            sys.stdout = devnull
            sys.stderr = devnull

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    return result_i


def run_multistart(n_starts=8, thrust_min_hover=THRUST_HOVER_MIN,
                   thrust_min_cruise=THRUST_CRUISE_MIN,
                   rho=1.225, seed=42, verbose=True,
                   fd_step=3e-4, optimizer="SLSQP",
                   return_all=False, start_from=0,
                   plot_feasible=False, plot_dir=None,
                   n_jobs=1, seed_dvs=None):
    """
    Run SLSQP from n_starts starting points; return the best result.

    Parameters
    ----------
    seed_dvs : list of dicts, optional
        Known-good DV dicts to use as the first len(seed_dvs) starts.
        Each dict may contain any subset of DV keys; missing keys fall back
        to zeros/defaults. Remaining starts (up to n_starts) are random.
        If None, start 0 is always the baseline (zero perturbations).
    n_jobs : int
        Number of parallel worker processes.  1 = sequential (default).
        -1 = use all available CPU cores.  N = use N cores.
    return_all : bool
        If True, return (best_prob, all_results).
    """
    rng      = np.random.default_rng(seed)
    _plot_dir = plot_dir or os.path.join(os.path.dirname(__file__),
                                         "..", "results", "plots")

    # ---- Generate ALL start points in main process (reproducible RNG) --------
    start_points = []

    # Seeded starts (explicit, for diversity / warm-starting from known optima)
    if seed_dvs:
        for dvs in seed_dvs[:n_starts]:
            start_points.append((
                float(dvs.get("rpm", RPM_HOVER_INIT)),
                np.ravel(dvs.get("delta_twist_cp",  np.zeros(N_CP))),
                np.ravel(dvs.get("delta_chord_cp",  np.zeros(N_CP))),
                np.ravel(dvs.get("sweep_cp",        np.zeros(N_CP))),
                np.ravel(dvs.get("delta_tc_cp",     np.zeros(N_CP))),
                np.ravel(dvs.get("delta_camber_cp", np.zeros(N_CP))),
                np.ravel(dvs.get("h_s_cp",          np.zeros(N_CP))),
            ))

    # Random starts fill the remainder (first random = baseline if no seeds at all)
    n_seeded = len(start_points)
    for i in range(n_seeded, n_starts):
        if i == 0:
            start_points.append((RPM_HOVER_INIT,
                                  np.zeros(N_CP), np.zeros(N_CP),
                                  np.zeros(N_CP), np.zeros(N_CP), np.zeros(N_CP),
                                  np.zeros(N_CP)))
        else:
            rpm_i   = float(rng.uniform(6000.0, 8500.0))
            dt_cp   = rng.uniform(-2.0, 2.0, N_CP)
            dc_cp   = rng.uniform(-0.03, 0.01, N_CP)
            sw_cp   = np.sort(rng.uniform(0.0, 0.10, N_CP))
            dtc_cp  = rng.uniform(-0.02, 0.02, N_CP)
            dcam_cp = rng.uniform(-0.01, 0.02, N_CP)
            hs_cp_i = np.zeros(N_CP)  # start from no serrations
            start_points.append((rpm_i, dt_cp, dc_cp, sw_cp, dtc_cp, dcam_cp, hs_cp_i))

    active_points = start_points[start_from:]
    active_indices = list(range(start_from, n_starts))

    # Build args tuples for workers
    worker_args = [
        (i, rpm_i, dt_cp, dc_cp, sw_cp, dtc_cp, dcam_cp, hs_cp,
         thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer,
         plot_feasible, _plot_dir)
        for (i, (rpm_i, dt_cp, dc_cp, sw_cp, dtc_cp, dcam_cp, hs_cp))
        in zip(active_indices, active_points)
    ]

    # ---- Run starts (parallel or sequential) ---------------------------------
    if n_jobs == 1:
        # Sequential: stream results as they arrive
        all_results = []
        best_spl        = np.inf
        best_infeas_spl = np.inf
        for args in worker_args:
            result_i = _start_worker(args)
            all_results.append(result_i)
            spl_i     = result_i["spl"]
            thrust_ok = result_i["feasible"]
            if verbose:
                tag = ("  *BEST*" if (thrust_ok and spl_i < best_spl) else
                       " (infeas)" if not thrust_ok else "")
                print(f"  [start {result_i['start']}] SPL={spl_i:.2f} dBA  "
                      f"T_hov={result_i['thrust_hover']:.3f} N  "
                      f"T_cru={result_i['thrust_cruise']:.3f} N  "
                      f"stress={result_i['max_stress']/1e6:.1f} MPa{tag}")
            if "plot_path" in result_i:
                print(f"  [start {result_i['start']}] Geometry -> {result_i['plot_path']}")
            if thrust_ok and spl_i < best_spl:
                best_spl = spl_i
            if not thrust_ok and spl_i < best_infeas_spl:
                best_infeas_spl = spl_i
    else:
        from multiprocessing import Pool, cpu_count
        n_workers = cpu_count() if n_jobs == -1 else n_jobs
        if verbose:
            print(f"  [parallel] launching {len(worker_args)} starts across "
                  f"{n_workers} workers…")
        with Pool(n_workers) as pool:
            all_results = pool.map(_start_worker, worker_args)
        all_results.sort(key=lambda r: r["start"])

        best_spl        = np.inf
        best_infeas_spl = np.inf
        for result_i in all_results:
            spl_i     = result_i["spl"]
            thrust_ok = result_i["feasible"]
            if verbose:
                tag = ("  *BEST*" if (thrust_ok and spl_i < best_spl) else
                       " (infeas)" if not thrust_ok else "")
                print(f"  [start {result_i['start']}] SPL={spl_i:.2f} dBA  "
                      f"T_hov={result_i['thrust_hover']:.3f} N  "
                      f"T_cru={result_i['thrust_cruise']:.3f} N  "
                      f"stress={result_i['max_stress']/1e6:.1f} MPa{tag}")
            if "plot_path" in result_i:
                print(f"  [start {result_i['start']}] Geometry -> {result_i['plot_path']}")
            if thrust_ok and spl_i < best_spl:
                best_spl = spl_i
            if not thrust_ok and spl_i < best_infeas_spl:
                best_infeas_spl = spl_i

    # ---- Pick best and re-run to get a live prob object for printing/viz -----
    feasible = [r for r in all_results if r["feasible"]]
    infeas   = [r for r in all_results if not r["feasible"]]

    if feasible:
        best_result = min(feasible, key=lambda r: r["spl"])
    else:
        best_result = min(infeas,   key=lambda r: r["spl"]) if infeas else all_results[-1]

    best_prob = run_from_point(best_result["dvs"],
                               thrust_min_hover=thrust_min_hover,
                               thrust_min_cruise=thrust_min_cruise,
                               rho=rho, verbose=False,
                               fd_step=fd_step, optimizer=optimizer)

    if verbose:
        feasible_spl = best_result["spl"] if feasible else np.inf
        print(f"\n=== Multistart Best (feasible): {feasible_spl:.2f} dBA ===")
        _print_results(best_prob)
        _print_design_vars(best_prob)
        if infeas:
            best_infeas_spl = min(infeas, key=lambda r: r["spl"])["spl"]
            print(f"\n=== Best Infeasible: {best_infeas_spl:.2f} dBA "
                  f"(use run_from_point to probe this region) ===")

    if return_all:
        return best_prob, all_results
    return best_prob


def dvs_to_blade(dvs):
    """Reconstruct a BladeGeometry from a DVs dict (no prob object needed)."""
    blade = baseline_apc7x5e()
    r_cp  = np.linspace(blade.r_R[0], blade.r_R[-1], N_CP)
    r_def = blade.r_R

    def _sp(cp): return CubicSpline(r_cp, cp)(r_def)

    dc   = _sp(np.ravel(dvs["delta_chord_cp"]))
    dt   = _sp(np.ravel(dvs["delta_twist_cp"]))
    dtc  = _sp(np.ravel(dvs["delta_tc_cp"]))
    sw   = np.clip(_sp(np.ravel(dvs["sweep_cp"])), 0.0, 0.12)
    dcam = _sp(np.ravel(dvs.get("delta_camber_cp", np.zeros(N_CP))))
    th2  = float(dvs.get("theta2", 120.0))
    th3  = float(dvs.get("theta3", 240.0))

    return (blade
            .perturb_twist(dt)
            .perturb_chord(dc)
            .perturb_tc(dtc)
            .set_sweep(sw)
            .set_camber(blade.camber_dist + dcam)
            .set_blade_angles([0.0, th2, th3]))


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

    dc_cp   = np.ravel(prob.get_val("delta_chord_cp"))
    dt_cp   = np.ravel(prob.get_val("delta_twist_cp"))
    dtc_cp  = np.ravel(prob.get_val("delta_tc_cp"))
    sw_cp   = np.ravel(prob.get_val("sweep_cp"))
    dcam_cp = np.ravel(prob.get_val("delta_camber_cp"))
    th2     = float(prob.get_val("theta2")[0])
    th3     = float(prob.get_val("theta3")[0])

    dc   = _spline(dc_cp)
    dt   = _spline(dt_cp)
    dtc  = _spline(dtc_cp)
    sw   = np.clip(_spline(sw_cp), 0.0, 0.12)
    dcam = _spline(dcam_cp)

    return (blade
            .perturb_twist(dt)
            .perturb_chord(dc)
            .perturb_tc(dtc)
            .set_sweep(sw)
            .set_camber(blade.camber_dist + dcam)
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

    prob.set_val("rpm",              dvs.get("rpm",              RPM_HOVER_INIT))
    prob.set_val("delta_twist_cp",   dvs.get("delta_twist_cp",   np.zeros(N_CP)))
    prob.set_val("delta_chord_cp",   dvs.get("delta_chord_cp",   np.zeros(N_CP)))
    prob.set_val("sweep_cp",         dvs.get("sweep_cp",         np.zeros(N_CP)))
    prob.set_val("delta_tc_cp",      dvs.get("delta_tc_cp",      np.zeros(N_CP)))
    prob.set_val("delta_camber_cp",  dvs.get("delta_camber_cp",  np.zeros(N_CP)))
    prob.set_val("h_s_cp",           dvs.get("h_s_cp",           np.zeros(N_CP)))
    prob.set_val("theta2",           120.0)
    prob.set_val("theta3",           240.0)

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


# ---------------------------------------------------------------------------
# LE serration optimisation — baseline geometry, RPM + h_LE_cp only
# ---------------------------------------------------------------------------

def _le_start_worker(args):
    """
    Worker for LE-serration-only multistart.
    DVs: rpm, h_LE_cp (5 CPs). Geometry frozen at APC 7x5E baseline.
    """
    import os, sys
    (i, rpm_i, h_LE_cp_i,
     thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer,
     le_type) = args

    os.environ["OPENMDAO_REPORTS"] = "0"
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = devnull

    try:
        prob_i = build_problem(
            thrust_min_hover=thrust_min_hover,
            thrust_min_cruise=thrust_min_cruise,
            rpm_init=rpm_i, rho=rho,
            fd_step=fd_step, optimizer=optimizer,
            geometry_dvs=False, le_type=le_type,
        )
        prob_i.setup()
        prob_i.set_val("rpm",      rpm_i)
        prob_i.set_val("h_LE_cp",  h_LE_cp_i)
        prob_i.run_driver()

        spl_i      = float(prob_i.get_val("SPL_weighted")[0])
        t_hover_i  = float(prob_i.get_val("thrust_hover")[0])
        t_cruise_i = float(prob_i.get_val("thrust_cruise")[0])
        stress_i   = float(prob_i.get_val("max_stress")[0])
        thrust_ok  = (t_hover_i  >= thrust_min_hover  * 0.99 and
                      t_cruise_i >= thrust_min_cruise * 0.99)

        result_i = {
            "start":         i,
            "spl":           spl_i,
            "thrust_hover":  t_hover_i,
            "thrust_cruise": t_cruise_i,
            "max_stress":    stress_i,
            "feasible":      thrust_ok,
            "dvs": {
                "rpm":      float(prob_i.get_val("rpm")[0]),
                "h_LE_cp":  prob_i.get_val("h_LE_cp").copy(),
            },
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    return result_i


def run_le_multistart(n_starts=8,
                      thrust_min_hover=THRUST_HOVER_MIN,
                      thrust_min_cruise=THRUST_CRUISE_MIN,
                      rho=1.225, seed=42, verbose=True,
                      fd_step=3e-4, optimizer="SLSQP",
                      n_jobs=1, le_type="sawtooth"):
    """
    Multistart SLSQP optimising RPM + h_LE_cp on the fixed baseline geometry.

    Starting points span the full RPM range and sample h_LE uniformly.
    First start is always the no-serration baseline (h_LE = 0, RPM = 7000).
    """
    rng = np.random.default_rng(seed)

    # Per-CP chord-proportional upper bounds (Elegoo Saturn 4 Ultra 16K, 15% chord)
    blade_le = baseline_apc7x5e()
    _, chord_cp_le, _ = blade_le.get_stations(N_CP)
    h_LE_max_cp = 0.15 * chord_cp_le    # metres

    start_points = []
    # Start 0: no serrations, hover RPM
    start_points.append((RPM_HOVER_INIT, np.zeros(N_CP)))
    # Start 1: 50% of max serrations uniform
    start_points.append((RPM_HOVER_INIT, 0.5 * h_LE_max_cp))
    # Start 2: maximum serrations, baseline RPM
    start_points.append((RPM_HOVER_INIT, h_LE_max_cp.copy()))
    # Remaining: random RPM + random spanwise h_LE within per-CP bounds
    for _ in range(3, n_starts):
        rpm_i   = float(rng.uniform(5500.0, 8500.0))
        h_LE_i  = rng.uniform(0.0, 1.0, N_CP) * h_LE_max_cp
        start_points.append((rpm_i, h_LE_i))

    worker_args = [
        (i, rpm_i, h_LE_cp_i,
         thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer, le_type)
        for i, (rpm_i, h_LE_cp_i) in enumerate(start_points)
    ]

    if verbose:
        print(f"\n=== LE Serration Multistart ({n_starts} starts, {le_type} model) ===")

    if n_jobs == 1:
        all_results = []
        best_spl = np.inf
        for wa in worker_args:
            r = _le_start_worker(wa)
            all_results.append(r)
            tag = "  *BEST*" if (r["feasible"] and r["spl"] < best_spl) else \
                  " (infeas)" if not r["feasible"] else ""
            if verbose:
                h_mean = r["dvs"]["h_LE_cp"].mean() * 1000
                print(f"  [start {r['start']}] SPL={r['spl']:.2f} dBA  "
                      f"RPM={r['dvs']['rpm']:.0f}  "
                      f"h_LE_mean={h_mean:.2f} mm  "
                      f"T_hov={r['thrust_hover']:.3f} N{tag}")
            if r["feasible"] and r["spl"] < best_spl:
                best_spl = r["spl"]
    else:
        from multiprocessing import Pool, cpu_count
        n_workers = cpu_count() if n_jobs == -1 else n_jobs
        if verbose:
            print(f"  [parallel] {len(worker_args)} starts across {n_workers} workers")
        with Pool(n_workers) as pool:
            all_results = pool.map(_le_start_worker, worker_args)
        all_results.sort(key=lambda r: r["start"])

    feasible = [r for r in all_results if r["feasible"]]
    if not feasible:
        if verbose:
            print("  WARNING: no feasible starts found")
        best = min(all_results, key=lambda r: r["spl"])
    else:
        best = min(feasible, key=lambda r: r["spl"])

    if verbose:
        print(f"\n=== LE Multistart Best: {best['spl']:.2f} dBA ===")
        print(f"  RPM    : {best['dvs']['rpm']:.0f}")
        print(f"  h_LE_cp: {np.round(best['dvs']['h_LE_cp']*1000, 2)} mm")
        print(f"  Feasible: {best['feasible']}")

    return best, all_results


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

    prob.set_val("rpm",             start_rpm)
    prob.set_val("delta_twist_cp",  start_dt)
    prob.set_val("delta_chord_cp",  start_dc)
    prob.set_val("sweep_cp",        start_sw)
    prob.set_val("delta_tc_cp",     start_dtc)
    prob.set_val("delta_camber_cp", np.zeros(N_CP))
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
    try:
        itu = _g('hover_acoustics.SPL_itu468')
        mf  = prob.get_val('hover_acoustics.merit_factor')[0]
        print(f"  ITU-R 468    : {itu:.2f} dB(ITU)  |  Merit T/p: {mf:.1f} N/Pa")
    except Exception:
        pass
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

    print(f"\n  dtwist_cp (deg)   : {np.round(_g('delta_twist_cp'), 3)}")
    print(f"  dchord_cp (R)     : {np.round(_g('delta_chord_cp'), 4)}")
    print(f"  sweep_cp (R)      : {np.round(_g('sweep_cp'), 4)}")
    print(f"  delta_tc_cp       : {np.round(_g('delta_tc_cp'), 4)}")
    print(f"  delta_camber_cp   : {np.round(_g('delta_camber_cp'), 4)}")
    h_s_mm  = _g('h_s_cp')  * 1000
    h_LE_mm = _g('h_LE_cp') * 1000
    print(f"  h_s_cp  (mm)      : {np.round(h_s_mm,  3)}")
    print(f"  h_LE_cp (mm)      : {np.round(h_LE_mm, 3)}")
    try:
        th2 = prob.get_val('theta2')[0]; th3 = prob.get_val('theta3')[0]
        print(f"  theta2/theta3     : {th2:.2f} / {th3:.2f} deg")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run quiet-prop MDAO optimiser")
    parser.add_argument("--starts",      type=int,   default=8,       help="Number of multistart runs")
    parser.add_argument("--start-from",  type=int,   default=0,       help="Resume from this start index")
    parser.add_argument("--optimizer",   type=str,   default="SLSQP", help="SLSQP or trust-constr")
    parser.add_argument("--fd-step",     type=float, default=3e-4,    help="FD step size")
    parser.add_argument("--jobs",        type=int,   default=1,       help="Parallel workers (-1=all cores)")
    parser.add_argument("--no-plot",     action="store_true",          help="Skip final geometry visualisation")
    parser.add_argument("--plot-starts", action="store_true",          help="Plot geometry after each feasible start")
    parser.add_argument("--seed-best",   action="store_true",
                        help="Seed first 4 starts from known best geometries for diversity")
    parser.add_argument("--le-opt",      action="store_true",
                        help="LE-serration-only optimisation: fix baseline geometry, optimise RPM + h_LE_cp")
    parser.add_argument("--le-type",     type=str, default="sawtooth",
                        choices=["sawtooth", "tubercle"],
                        help="LE serration model (sawtooth=Lyu 2016, tubercle=Chaitanya 2017)")
    args = parser.parse_args()

    # ---- Known-best seed points (used when --seed-best is passed) ----------------
    # Four geometries covering different regions of the 28-DV design space.
    # Remaining starts (4..n-1) are random.
    _SEED_DVS = [
        # 1. Best overall: warm-start from 23-DV optimum, camber optimized (70.01 dBA)
        {
            "rpm": 6053.0,
            "delta_twist_cp":  np.array([-0.358,  2.946,  0.434,  5.0,    5.0   ]),
            "delta_chord_cp":  np.array([ 0.0074,-0.03,  -0.03,  -0.03,  -0.03  ]),
            "sweep_cp":        np.array([ 0.0,    0.0,    0.1136, 0.1136, 0.1183 ]),
            "delta_tc_cp":     np.array([-0.01,  -0.03,  -0.0202,-0.0055, 0.0145 ]),
            "delta_camber_cp": np.array([ 0.0,   -0.02,  -0.0096, 0.0104, 0.03  ]),
        },
        # 2. 8-start best feasible (70.62 dBA) — different twist profile
        {
            "rpm": 6403.0,
            "delta_twist_cp":  np.array([ 1.664, -0.214, -1.371,  4.374,  1.205 ]),
            "delta_chord_cp":  np.array([ 0.0074,-0.03,  -0.03,  -0.03,  -0.03  ]),
            "sweep_cp":        np.array([ 0.0,    0.0,    0.1117, 0.1117, 0.1197 ]),
            "delta_tc_cp":     np.array([-0.01,  -0.03,  -0.0202,-0.0055, 0.0145 ]),
            "delta_camber_cp": np.array([ 0.0053,-0.0047,-0.0102, 0.007,  0.0215 ]),
        },
        # 3. 23-DV optimum + zero camber (70.16 dBA) — different tc/twist basin
        {
            "rpm": 6053.0,
            "delta_twist_cp":  np.array([-0.349,  3.239,  0.438,  5.0,    5.0   ]),
            "delta_chord_cp":  np.array([ 0.0061,-0.03,  -0.03,  -0.03,  -0.0125]),
            "sweep_cp":        np.array([ 0.0,    0.0,    0.1136, 0.1136, 0.1198 ]),
            "delta_tc_cp":     np.array([-0.01,  -0.03,  -0.0234,-0.0186, 0.0014 ]),
            "delta_camber_cp": np.zeros(N_CP),
        },
        # 4. Baseline (clean descent — anchor for global coverage)
        {
            "rpm": RPM_HOVER_INIT,
            "delta_twist_cp":  np.zeros(N_CP),
            "delta_chord_cp":  np.zeros(N_CP),
            "sweep_cp":        np.zeros(N_CP),
            "delta_tc_cp":     np.zeros(N_CP),
            "delta_camber_cp": np.zeros(N_CP),
        },
    ]

    # ---- LE-serration-only optimisation (baseline geometry fixed) -----------
    if args.le_opt:
        best, all_results = run_le_multistart(
            n_starts=args.starts,
            optimizer=args.optimizer,
            fd_step=args.fd_step,
            n_jobs=args.jobs,
            le_type=args.le_type,
        )
        feasible = [r for r in all_results if r["feasible"]]
        print(f"\n{len(feasible)}/{len(all_results)} starts feasible")
        import sys; sys.exit(0)

    prob, all_results = run_multistart(
        n_starts=args.starts,
        optimizer=args.optimizer,
        fd_step=args.fd_step,
        return_all=True,
        start_from=args.start_from,
        plot_feasible=args.plot_starts,
        n_jobs=args.jobs,
        seed_dvs=_SEED_DVS if args.seed_best else None,
    )

    # Report best infeasible start
    infeas = [r for r in all_results if not r["feasible"]]
    if infeas:
        best_infeas      = min(infeas, key=lambda r: r["spl"])
        best_feasible_spl = float(prob.get_val("SPL_weighted")[0])
        if best_infeas["spl"] < best_feasible_spl:
            print(f"\n[INFO] Best infeasible start {best_infeas['start']} "
                  f"({best_infeas['spl']:.2f} dBA) beat best feasible "
                  f"({best_feasible_spl:.2f} dBA) — DVs saved in all_results.")

    if not args.no_plot:
        try:
            from results.plots.geometry_viz import plot_geometry
            feasible = [r for r in all_results if r["feasible"]]
            best_result = (min(feasible, key=lambda r: r["spl"]) if feasible
                           else min(all_results, key=lambda r: r["spl"]))
            blade_opt = dvs_to_blade(best_result["dvs"])
            spl_val   = best_result["spl"]
            rpm_val   = best_result["dvs"]["rpm"]
            out_path  = os.path.join(os.path.dirname(__file__),
                                     "..", "results", "plots", "optimum_geometry.png")
            plot_geometry(
                baseline_apc7x5e(), blade_opt,
                opt_label=f"Optimum {spl_val:.2f} dBA @ {rpm_val:.0f} RPM",
                save_path=out_path,
            )
            print(f"[VIZ] Saved -> {out_path}")
        except Exception as e:
            print(f"[WARN] Geometry plot failed: {e}")
