"""
Aeroacoustic noise model: BPM broadband + Gutin tonal.

Broadband: Brooks-Pope-Marcolini (1989) TBL-TE self-noise model.
Tonal:     Garrick-Watkins Gutin formula with Bessel function correction
           (far-field, broadside observer, first 3 BPF harmonics).

Notes
-----
For small UAV propellers (Re ~ 15 000–50 000, M_tip ~ 0.1–0.15):
  - TBL-TE broadband peaks above 15 kHz due to thin BL → low audible broadband
  - Tonal noise (BPF harmonics) is the dominant audible source
  - Absolute SPL predictions carry ±5–10 dB uncertainty vs measurements
"""

import numpy as np
from scipy.special import jv as bessel_jv
import openmdao.api as om
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry.blade_generator import baseline_hqprop

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_SOUND = 340.3    # m/s
NU_AIR  = 1.48e-5  # m²/s  (sea level, 15°C)
P_REF   = 20e-6    # Pa  (acoustic reference pressure)

THIRD_OCT_FREQS = np.array([
     50,  63,  80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
], dtype=float)

# A-weighting (dB) for each 1/3-octave band above
A_WEIGHT = np.array([
    -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9,  -8.6,  -6.6,  -4.8,
     -3.2,  -1.9,  -0.8,   0.0,   0.6,   1.0,   1.2,   1.3,   1.2,
      1.0,   0.5,  -0.1,  -1.1,  -2.5,  -4.3,  -6.6,  -9.3,
], dtype=float)


# ---------------------------------------------------------------------------
# BPM helper: K1 calibration constant (function of chord Reynolds number)
# ---------------------------------------------------------------------------

def _bpm_K1(Re_c):
    """BPM Eq. 9 – Reynolds-number dependent level correction."""
    if Re_c < 2.47e5:
        return -4.31 * np.log10(Re_c) + 156.3
    elif Re_c < 8.0e5:
        return -9.0  * np.log10(Re_c) + 181.6
    else:
        return 128.5


# ---------------------------------------------------------------------------
# BPM spectral shape function A (Eq. 1, BPM 1989)
# ---------------------------------------------------------------------------

def _bpm_A(St_ratio):
    """
    BPM spectral shape function A(St/St_peak).
    Returns the level correction in dB.
    """
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
# Schlichting turbulent BL displacement thickness
# ---------------------------------------------------------------------------

def _delta_star(chord, v_rel, aoa_deg):
    """
    Trailing-edge displacement thickness for suction and pressure sides.
    Uses Schlichting flat-plate turbulent BL modified for angle of attack.

    Returns (delta_star_s, delta_star_p) in metres.
    """
    Re_c = np.maximum(v_rel * chord / NU_AIR, 1e3)
    # Turbulent flat-plate δ*/c (Schlichting, Ch.21)
    ds0 = chord * 0.0144 / Re_c ** 0.2

    alpha = np.abs(aoa_deg)
    # Suction side thickens with AoA; pressure side thins
    ds_s = ds0 * 10 ** (0.3 * alpha ** 0.5)
    ds_p = ds0 * 10 ** (-0.1 * alpha)

    ds_s = np.clip(ds_s, 1e-7, chord * 0.5)
    ds_p = np.clip(ds_p, 1e-7, chord * 0.5)
    return ds_s, ds_p


# ---------------------------------------------------------------------------
# TBL-TE broadband SPL for one blade element
# ---------------------------------------------------------------------------

def _tbl_te_spl(chord, v_rel, aoa_deg, dr, r_obs=1.0):
    """
    BPM TBL-TE SPL spectrum (dB re 20 µPa) for one blade element.

    Parameters
    ----------
    chord   : float  chord length (m)
    v_rel   : float  relative velocity (m/s)
    aoa_deg : float  angle of attack (deg)
    dr      : float  radial span of this element (m)
    r_obs   : float  observer distance (m)

    Returns array of length len(THIRD_OCT_FREQS).
    """
    if v_rel < 1.0:
        return np.full(len(THIRD_OCT_FREQS), -200.0)

    M    = v_rel / C_SOUND
    Re_c = v_rel * chord / NU_AIR

    ds_s, ds_p = _delta_star(chord, v_rel, aoa_deg)
    K1  = _bpm_K1(Re_c)

    # Peak Strouhal number (BPM Eq. 4)
    St1 = 0.02 * M ** (-0.6)

    # Strouhal numbers per frequency band
    f    = THIRD_OCT_FREQS
    St_s = f * ds_s / (v_rel + 1e-12)
    St_p = f * ds_p / (v_rel + 1e-12)

    # BPM Eqs. 2–4: include span dr in the dimensional term
    base_s = 10 * np.log10(np.maximum(ds_s * M ** 5 * dr / r_obs ** 2, 1e-300))
    base_p = 10 * np.log10(np.maximum(ds_p * M ** 5 * dr / r_obs ** 2, 1e-300))

    spl_s = base_s + K1 + _bpm_A(St_s / St1)
    spl_p = base_p + K1 + _bpm_A(St_p / St1)

    return 10 * np.log10(10 ** (spl_s / 10) + 10 ** (spl_p / 10) + 1e-300)


# ---------------------------------------------------------------------------
# Gutin tonal noise with Bessel function correction
# Garrick & Watkins (1954), simplified for broadside observer (θ = 90°)
# ---------------------------------------------------------------------------

def _gutin_tonal_spl(thrust, torque, rpm, num_blades, radius_m,
                     r_obs=1.0, harmonic=1):
    """
    Far-field tonal SPL at the m-th BPF harmonic, broadside observer.

    Formula (Garrick-Watkins, broadside):
        p_m = (m*B*n) / (2*r_obs*c) * (T/B) * J_{mB}(m*B*Ω*R_eff/c)

    Returns (spl_dB, freq_Hz).
    """
    omega  = rpm * 2 * np.pi / 60.0
    n_rps  = rpm / 60.0
    B      = num_blades
    m      = harmonic
    BPF    = B * n_rps
    f_tone = m * BPF

    # Bessel function argument at effective radius (0.8 R)
    R_eff  = 0.8 * radius_m
    order  = m * B
    x_T    = m * B * omega * R_eff / C_SOUND

    J_T    = float(np.abs(bessel_jv(order, x_T)))

    # Thrust-dominated far-field pressure (Pa)
    p_thrust = (m * n_rps * B) / (2.0 * r_obs * C_SOUND) * (thrust / B) * J_T

    # Torque contribution (minor but included for completeness)
    J_Q    = float(np.abs(bessel_jv(order + 1, x_T)))
    p_torque = (m * n_rps * B) / (2.0 * r_obs * C_SOUND) * \
               (torque / (B * radius_m)) * J_Q * 0.5

    p_total = np.sqrt(p_thrust ** 2 + p_torque ** 2)
    spl     = 20.0 * np.log10(np.maximum(p_total, 1e-30) / P_REF)
    return spl, f_tone


# ---------------------------------------------------------------------------
# Full BPM + Gutin model
# ---------------------------------------------------------------------------

def bpm_noise(r_m, chord_m, v_rel, aoa_deg, cl,
              thrust, torque, rpm, num_blades, radius_m,
              rho=1.225, r_obs=1.0):
    """
    Compute SPL spectrum (BPM broadband + Gutin tonal), return results dict.

    Returns
    -------
    dict: SPL_total (dBA), SPL_broadband (dB), SPL_tonal (dB),
          freq (Hz), SPL_spectrum (dB unweighted)
    """
    n_freqs      = len(THIRD_OCT_FREQS)
    SPL_spectrum = np.full(n_freqs, -200.0)

    # --- Broadband: sum TBL-TE over all blade stations ---
    dr_arr = np.gradient(r_m)   # radial step at each station (m)
    for i in range(len(r_m)):
        spl_elem    = _tbl_te_spl(chord_m[i], float(v_rel[i]),
                                   float(aoa_deg[i]),
                                   float(dr_arr[i]), r_obs)
        SPL_spectrum = 10 * np.log10(
            10 ** (SPL_spectrum / 10) + 10 ** (spl_elem / 10) + 1e-300)

    SPL_broadband = 10 * np.log10(np.sum(10 ** (SPL_spectrum / 10)) + 1e-300)

    # --- Tonal: Gutin first 3 harmonics ---
    SPL_tonal = -200.0
    for m in [1, 2, 3]:
        spl_tone, f_tone = _gutin_tonal_spl(
            float(thrust), float(torque), float(rpm),
            num_blades, radius_m, r_obs=r_obs, harmonic=m)
        # Inject into nearest 1/3-octave bin
        idx = np.argmin(np.abs(THIRD_OCT_FREQS - f_tone))
        SPL_spectrum[idx] = 10 * np.log10(
            10 ** (SPL_spectrum[idx] / 10) + 10 ** (spl_tone / 10) + 1e-300)
        SPL_tonal = 10 * np.log10(
            10 ** (SPL_tonal / 10) + 10 ** (spl_tone / 10) + 1e-300)

    # --- A-weighted overall ---
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
        self._blade = self.options["blade"] or baseline_hqprop()
        N      = self.options["n_stations"]
        n_freq = len(THIRD_OCT_FREQS)

        self.add_input("r_m",     val=np.zeros(N), units="m")
        self.add_input("v_rel",   val=np.zeros(N), units="m/s")
        self.add_input("aoa_deg", val=np.zeros(N))
        self.add_input("cl",      val=np.zeros(N))
        self.add_input("thrust",  val=0.0,    units="N")
        self.add_input("torque",  val=0.0,    units="N*m")
        self.add_input("rpm",     val=5000.0, units="rpm")
        self.add_input("rho",     val=1.225,  units="kg/m**3")

        self.add_output("SPL_total",     val=0.0)
        self.add_output("SPL_broadband", val=0.0)
        self.add_output("SPL_tonal",     val=0.0)
        self.add_output("SPL_spectrum",  val=np.zeros(n_freq))
        self.add_output("freq",          val=THIRD_OCT_FREQS.copy(), units="Hz")

    def compute(self, inputs, outputs):
        blade = self._blade
        _, chord_m, _ = blade.get_stations(self.options["n_stations"])

        res = bpm_noise(
            r_m=inputs["r_m"],
            chord_m=chord_m,
            v_rel=inputs["v_rel"],
            aoa_deg=inputs["aoa_deg"],
            cl=inputs["cl"],
            thrust=float(inputs["thrust"][0]),
            torque=float(inputs["torque"][0]),
            rpm=float(inputs["rpm"][0]),
            num_blades=blade.num_blades,
            radius_m=blade.radius_m,
            rho=float(inputs["rho"][0]),
            r_obs=self.options["r_obs"],
        )
        outputs["SPL_total"]     = res["SPL_total"]
        outputs["SPL_broadband"] = res["SPL_broadband"]
        outputs["SPL_tonal"]     = res["SPL_tonal"]
        outputs["SPL_spectrum"]  = res["SPL_spectrum"]
        outputs["freq"]          = res["freq"]


if __name__ == "__main__":
    from aerodynamics.ccblade_component import bem_solve

    blade = baseline_hqprop()
    aero  = bem_solve(blade, rpm=5000, v_inf=0.0)
    _, chord_m, _ = blade.get_stations(20)

    res = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"], cl=aero["cl"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=5000, num_blades=blade.num_blades, radius_m=blade.radius_m,
    )
    print(f"SPL total  (dBA) : {res['SPL_total']:.1f}")
    print(f"SPL broadband    : {res['SPL_broadband']:.1f} dB")
    print(f"SPL tonal  (BPF) : {res['SPL_tonal']:.1f} dB")
