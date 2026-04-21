"""
Aeroacoustic noise model: BPM broadband + Gutin tonal.

Additions over Phase 1:
  - LBL-VS (Laminar Boundary Layer Vortex Shedding) noise model (BPM 1989 §3.2)
  - Transition-aware boundary layer: blends turbulent TBL-TE and laminar LBL-VS
    based on x_tr_c from Michel's criterion
  - Unequal blade spacing: Fourier interference factor on tonal harmonics
"""

import numpy as np
from scipy.special import jv as bessel_jv
import openmdao.api as om
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry.blade_generator import baseline_apc7x5e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_SOUND = 340.3    # m/s
NU_AIR  = 1.48e-5  # m2/s
P_REF   = 20e-6    # Pa

THIRD_OCT_FREQS = np.array([
     50,  63,  80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
], dtype=float)

A_WEIGHT = np.array([
    -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9,  -8.6,  -6.6,  -4.8,
     -3.2,  -1.9,  -0.8,   0.0,   0.6,   1.0,   1.2,   1.3,   1.2,
      1.0,   0.5,  -0.1,  -1.1,  -2.5,  -4.3,  -6.6,  -9.3,
], dtype=float)


# ---------------------------------------------------------------------------
# BPM helpers
# ---------------------------------------------------------------------------

def _bpm_K1(Re_c):
    if Re_c < 2.47e5:
        return -4.31 * np.log10(Re_c) + 156.3
    elif Re_c < 8.0e5:
        return -9.0  * np.log10(Re_c) + 181.6
    else:
        return 128.5


def _bpm_A(St_ratio):
    x = np.abs(np.log10(np.clip(St_ratio, 1e-10, 1e10)))
    return np.where(
        x <= 0.204,
        np.sqrt(np.maximum(67.552 - 886.788 * x ** 2, 0.0)) - 8.219,
        np.where(
            x <= 0.604,
            -32.665 * x + 3.981,
            -142.795 * x ** 3 + 103.656 * x ** 2 - 57.757 * x + 6.006,
        ),
    )


# ---------------------------------------------------------------------------
# Boundary layer thickness models
# ---------------------------------------------------------------------------

def _delta_star_turbulent(chord, v_rel, aoa_deg, turb_length=None):
    """
    Turbulent BL displacement thickness (Schlichting) for a section.
    turb_length: physical length over which turbulent BL grows (default = chord).
    """
    L = turb_length if turb_length is not None else chord
    Re_L = np.maximum(v_rel * L / NU_AIR, 1e3)
    ds0  = L * 0.0144 / Re_L ** 0.2

    alpha = np.abs(aoa_deg)
    ds_s  = ds0 * 10 ** (0.3 * alpha ** 0.5)
    ds_p  = ds0 * 10 ** (-0.1 * alpha)

    ds_s = np.clip(ds_s, 1e-7, chord * 0.5)
    ds_p = np.clip(ds_p, 1e-7, chord * 0.5)
    return ds_s, ds_p


def _delta_star_laminar(chord, v_rel):
    """Blasius laminar BL displacement thickness at x = chord (full chord laminar)."""
    Re_c = np.maximum(v_rel * chord / NU_AIR, 1e3)
    ds   = 1.72 * chord / np.sqrt(Re_c)
    return np.clip(ds, 1e-7, chord * 0.5)


def _delta_star(chord, v_rel, aoa_deg, x_tr_c=0.0):
    """
    Transition-aware BL displacement thickness.

    With transition at x_tr_c, turbulent BL grows only over the aft
    (1 - x_tr_c) fraction of chord. Returns (ds_suction, ds_pressure).
    """
    # Turbulent portion grows from x_tr to TE
    c_turb = chord * max(1.0 - x_tr_c, 1e-6)
    ds_s, ds_p = _delta_star_turbulent(chord, v_rel, aoa_deg, turb_length=c_turb)
    return ds_s, ds_p


# ---------------------------------------------------------------------------
# TBL-TE broadband (turbulent trailing edge)
# ---------------------------------------------------------------------------

def _tbl_te_spl(chord, v_rel, aoa_deg, dr, r_obs=1.0, x_tr_c=0.0):
    if v_rel < 1.0:
        return np.full(len(THIRD_OCT_FREQS), -200.0)

    M    = v_rel / C_SOUND
    Re_c = v_rel * chord / NU_AIR

    ds_s, ds_p = _delta_star(chord, v_rel, aoa_deg, x_tr_c)
    K1  = _bpm_K1(Re_c)

    St1  = 0.02 * M ** (-0.6)
    f    = THIRD_OCT_FREQS
    St_s = f * ds_s / (v_rel + 1e-12)
    St_p = f * ds_p / (v_rel + 1e-12)

    base_s = 10 * np.log10(np.maximum(ds_s * M ** 5 * dr / r_obs ** 2, 1e-300))
    base_p = 10 * np.log10(np.maximum(ds_p * M ** 5 * dr / r_obs ** 2, 1e-300))

    spl_s = base_s + K1 + _bpm_A(St_s / St1)
    spl_p = base_p + K1 + _bpm_A(St_p / St1)

    return 10 * np.log10(10 ** (spl_s / 10) + 10 ** (spl_p / 10) + 1e-300)


# ---------------------------------------------------------------------------
# LBL-VS: Laminar Boundary Layer Vortex Shedding (BPM 1989, §3.2)
# ---------------------------------------------------------------------------

def _lbl_vs_spl(chord, v_rel, aoa_deg, dr, r_obs=1.0):
    """
    LBL-VS SPL spectrum (BPM 1989).

    Occurs when suction-side BL is laminar at the TE. Typically 10–20 dB
    above TBL-TE at the same conditions in the Re ~ 1e5 UAV regime.
    """
    if v_rel < 1.0:
        return np.full(len(THIRD_OCT_FREQS), -200.0)

    M    = v_rel / C_SOUND
    Re_c = v_rel * chord / NU_AIR
    alpha = np.abs(aoa_deg)

    # Laminar suction-side displacement thickness (Blasius)
    ds = _delta_star_laminar(chord, v_rel)

    # BPM peak Strouhal (based on delta*, decreases with Re_c per BPM Fig.19)
    St_peak = np.clip(0.25 * (Re_c / 1.35e6) ** (-0.10), 0.08, 2.0)

    # Level calibration (BPM G2 + G3 merged, calibrated for Re ~ 1e4–5e5)
    # At Re_c = 1e5: K_lbl ≈ 135; at 5e5: ≈ 125 (turbulent BPF K1 is ~135 at 1e5)
    K_lbl = np.clip(160.0 - 5.0 * np.log10(Re_c), 110.0, 185.0)

    # AoA: LBL-VS strengthens with AoA up to ~12deg (BPM G3 trend)
    aoa_boost = np.clip(2.0 * alpha, 0.0, 20.0)

    f   = THIRD_OCT_FREQS
    St  = f * ds / (v_rel + 1e-12)
    dim = 10 * np.log10(np.maximum(ds * M ** 5 * dr / r_obs ** 2, 1e-300))

    return dim + K_lbl + aoa_boost + _bpm_A(St / St_peak)


# ---------------------------------------------------------------------------
# Tonal noise with unequal blade spacing interference
# ---------------------------------------------------------------------------

def _blade_spacing_factor(harmonic_m, blade_angles_deg):
    """
    Fourier interference factor for unequal blade spacing.

    For m-th BPF harmonic (f = m * B * n_rps) with blades at angles theta_k:
      F = |Σ_k exp(j·m·B·theta_k)| / B

    Equal spacing -> F = 1.0 (coherent addition, maximum tonal SPL).
    Unequal spacing -> F < 1.0 (partial cancellation).
    """
    B   = len(blade_angles_deg)
    th  = np.deg2rad(blade_angles_deg)
    vec = np.sum(np.exp(1j * harmonic_m * B * th))
    return float(np.abs(vec)) / B


def _gutin_tonal_spl(thrust, torque, rpm, num_blades, radius_m,
                     blade_angles_deg=None, r_obs=1.0, harmonic=1):
    """
    Far-field tonal SPL at the m-th BPF harmonic, broadside observer.
    Applies unequal-spacing interference factor when blade_angles_deg is given.

    Returns (spl_dB, freq_Hz).
    """
    omega  = rpm * 2 * np.pi / 60.0
    n_rps  = rpm / 60.0
    B      = num_blades
    m      = harmonic
    BPF    = B * n_rps
    f_tone = m * BPF

    R_eff  = 0.8 * radius_m
    order  = m * B
    x_T    = m * B * omega * R_eff / C_SOUND

    J_T = float(np.abs(bessel_jv(order, x_T)))
    J_Q = float(np.abs(bessel_jv(order + 1, x_T)))

    p_thrust = (m * n_rps * B) / (2.0 * r_obs * C_SOUND) * (thrust / B) * J_T
    p_torque = (m * n_rps * B) / (2.0 * r_obs * C_SOUND) * \
               (torque / (B * radius_m)) * J_Q * 0.5

    p_single = np.sqrt(p_thrust ** 2 + p_torque ** 2)

    # Unequal-spacing interference factor
    if blade_angles_deg is not None:
        F_int = _blade_spacing_factor(m, blade_angles_deg)
    else:
        F_int = 1.0

    p_total = p_single * F_int
    spl     = 20.0 * np.log10(np.maximum(p_total, 1e-30) / P_REF)
    return spl, f_tone


# ---------------------------------------------------------------------------
# Full BPM + Gutin model
# ---------------------------------------------------------------------------

def bpm_noise(r_m, chord_m, v_rel, aoa_deg, cl,
              thrust, torque, rpm, num_blades, radius_m,
              rho=1.225, r_obs=1.0,
              x_tr_c=None, blade_angles_deg=None):
    """
    Compute SPL spectrum (BPM broadband + Gutin tonal).

    Parameters
    ----------
    x_tr_c        : array (N,) or None
        Per-station transition location from Michel's criterion.
        If None, assumes fully turbulent (x_tr_c = 0).
    blade_angles_deg : array (B,) or None
        Azimuthal blade positions for interference calculation.
        If None, assumes equal spacing (interference factor = 1).

    Returns
    -------
    dict: SPL_total (dBA), SPL_broadband (dB), SPL_tonal (dB),
          freq (Hz), SPL_spectrum (dB unweighted)
    """
    N = len(r_m)
    if x_tr_c is None:
        x_tr_c = np.zeros(N)
    x_tr_c = np.asarray(x_tr_c, dtype=float)

    n_freqs      = len(THIRD_OCT_FREQS)
    SPL_spectrum = np.full(n_freqs, -200.0)

    # --- Broadband: sum over blade stations ---
    dr_arr = np.gradient(r_m)
    for i in range(N):
        c     = float(chord_m[i])
        vr    = float(v_rel[i])
        aoa_i = float(aoa_deg[i])
        dr_i  = float(dr_arr[i])
        xtr_i = float(x_tr_c[i])

        # Turbulent TBL-TE (weighted by turbulent fraction)
        w_turb = 1.0 - xtr_i
        if w_turb > 0.01:
            spl_tbl = _tbl_te_spl(c, vr, aoa_i, dr_i, r_obs, xtr_i)
            SPL_spectrum = 10 * np.log10(
                10 ** (SPL_spectrum / 10) +
                w_turb * 10 ** (spl_tbl / 10) + 1e-300)

        # LBL-VS (laminar fraction; activated when x_tr_c > 0.3)
        if xtr_i > 0.30:
            spl_lbl = _lbl_vs_spl(c, vr, aoa_i, dr_i, r_obs)
            SPL_spectrum = 10 * np.log10(
                10 ** (SPL_spectrum / 10) +
                xtr_i * 10 ** (spl_lbl / 10) + 1e-300)

    SPL_broadband = 10 * np.log10(np.sum(10 ** (SPL_spectrum / 10)) + 1e-300)

    # --- Tonal: Gutin first 3 harmonics with blade-spacing interference ---
    SPL_tonal = -200.0
    for m in [1, 2, 3]:
        spl_tone, f_tone = _gutin_tonal_spl(
            float(thrust), float(torque), float(rpm),
            num_blades, radius_m, blade_angles_deg=blade_angles_deg,
            r_obs=r_obs, harmonic=m)
        idx = np.argmin(np.abs(THIRD_OCT_FREQS - f_tone))
        SPL_spectrum[idx] = 10 * np.log10(
            10 ** (SPL_spectrum[idx] / 10) + 10 ** (spl_tone / 10) + 1e-300)
        SPL_tonal = 10 * np.log10(
            10 ** (SPL_tonal / 10) + 10 ** (spl_tone / 10) + 1e-300)

    SPL_A     = SPL_spectrum + A_WEIGHT
    SPL_total = 10 * np.log10(np.sum(10 ** (SPL_A / 10)) + 1e-300)

    return {
        "SPL_total":     SPL_total,
        "SPL_broadband": SPL_broadband,
        "SPL_tonal":     SPL_tonal,
        "freq":          THIRD_OCT_FREQS.copy(),
        "SPL_spectrum":  SPL_spectrum,
    }


# ---------------------------------------------------------------------------
# OpenMDAO component
# ---------------------------------------------------------------------------

class BPMComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("blade",      default=None)
        self.options.declare("n_stations", default=20)
        self.options.declare("r_obs",      default=1.0)

    def setup(self):
        self._blade = self.options["blade"] or baseline_apc7x5e()
        N      = self.options["n_stations"]
        n_freq = len(THIRD_OCT_FREQS)
        B      = self._blade.num_blades
        _, chord0, _ = self._blade.get_stations(N)

        self.add_input("r_m",             val=np.zeros(N), units="m")
        self.add_input("chord_m",         val=chord0,       units="m")
        self.add_input("v_rel",           val=np.zeros(N), units="m/s")
        self.add_input("aoa_deg",         val=np.zeros(N))
        self.add_input("cl",              val=np.zeros(N))
        self.add_input("x_tr_c",         val=np.zeros(N))   # from CCBladeComponent
        self.add_input("thrust",          val=0.0,    units="N")
        self.add_input("torque",          val=0.0,    units="N*m")
        self.add_input("rpm",             val=5000.0, units="rpm")
        self.add_input("rho",             val=1.225,  units="kg/m**3")
        # Azimuthal blade positions (design variables for unequal spacing)
        self.add_input("blade_angles_deg", val=self._blade.blade_angles_deg.copy())

        self.add_output("SPL_total",     val=0.0)
        self.add_output("SPL_broadband", val=0.0)
        self.add_output("SPL_tonal",     val=0.0)
        self.add_output("SPL_spectrum",  val=np.zeros(n_freq))
        self.add_output("freq",          val=THIRD_OCT_FREQS.copy(), units="Hz")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=1e-4)

    def compute(self, inputs, outputs):
        res = bpm_noise(
            r_m=inputs["r_m"],
            chord_m=inputs["chord_m"],
            v_rel=inputs["v_rel"],
            aoa_deg=inputs["aoa_deg"],
            cl=inputs["cl"],
            thrust=float(inputs["thrust"][0]),
            torque=float(inputs["torque"][0]),
            rpm=float(inputs["rpm"][0]),
            num_blades=self._blade.num_blades,
            radius_m=self._blade.radius_m,
            rho=float(inputs["rho"][0]),
            r_obs=self.options["r_obs"],
            x_tr_c=inputs["x_tr_c"],
            blade_angles_deg=inputs["blade_angles_deg"],
        )
        outputs["SPL_total"]     = res["SPL_total"]
        outputs["SPL_broadband"] = res["SPL_broadband"]
        outputs["SPL_tonal"]     = res["SPL_tonal"]
        outputs["SPL_spectrum"]  = res["SPL_spectrum"]
        outputs["freq"]          = res["freq"]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from aerodynamics.ccblade_component import bem_solve

    blade = baseline_apc7x5e()
    aero  = bem_solve(blade, rpm=5000, v_inf=0.0)
    _, chord_m, _ = blade.get_stations(20)

    # Equal spacing
    res_equal = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"], cl=aero["cl"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=5000, num_blades=blade.num_blades, radius_m=blade.radius_m,
        x_tr_c=aero["x_tr_c"],
        blade_angles_deg=np.array([0.0, 120.0, 240.0]),
    )

    # Unequal spacing
    res_unequal = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"], cl=aero["cl"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=5000, num_blades=blade.num_blades, radius_m=blade.radius_m,
        x_tr_c=aero["x_tr_c"],
        blade_angles_deg=np.array([0.0, 115.0, 235.0]),
    )

    print(f"Equal spacing:   SPL={res_equal['SPL_total']:.2f} dBA  "
          f"(broad={res_equal['SPL_broadband']:.1f} dB, tonal={res_equal['SPL_tonal']:.1f} dB)")
    print(f"Unequal spacing: SPL={res_unequal['SPL_total']:.2f} dBA  "
          f"(broad={res_unequal['SPL_broadband']:.1f} dB, tonal={res_unequal['SPL_tonal']:.1f} dB)")
    print(f"Mean x_tr/c: {np.mean(aero['x_tr_c']):.3f}  "
          f"(1.0=laminar, 0.0=turbulent)")
