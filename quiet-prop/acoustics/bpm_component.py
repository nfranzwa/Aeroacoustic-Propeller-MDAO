"""
Aeroacoustic noise model: BPM broadband + Amiet LETI + BVI tonal.

Broadband (dominant mechanism for low-M_tip drones):
  - TBL-TE: turbulent trailing-edge (BPM 1989)
  - LBL-VS: laminar vortex-shedding (BPM 1989 §3.2)
  - Amiet LETI: leading-edge turbulence interaction (Amiet 1975).
    Dominant source for real drones (~60-70 dBA); gradients w.r.t.
    chord and v_rel give the optimizer real signal.
  - Amiet swept-LE: S_pp *= cos⁴(Λ) per station (Amiet 1975).

Tonal (BVI unsteady loading):
  Steady-loading BPF tonal is near-zero at M_tip~0.19 (Gutin/FW-H).
  Real drone BPF tones come from blade-vortex interaction (BVI):
    Γ_tip = T / (ρ B Ω R π)          — lifting-line tip vortex strength
    h     = v_induced / (B Ω)         — characteristic miss distance
    p_m   = ρ v_tip Γ / (2π r) × m^-1.5 × F_spacing  — harmonic pressure
  Gives ~62-65 dBA at baseline, ~10 dB below broadband.
  Gradients flow through thrust → optimizer has BVI signal.
"""

import numpy as np
import openmdao.api as om
import sys
import os
from scipy.special import jv as _bessel_j
from scipy.interpolate import CubicSpline

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

def _tbl_te_spl(chord, v_rel, aoa_deg, dr, r_obs=1.0, x_tr_c=0.0, h_s=0.0):
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

    spl = 10 * np.log10(10 ** (spl_s / 10) + 10 ** (spl_p / 10) + 1e-300)

    if h_s > 1e-6:
        # Howe (1991) compact sawtooth serration reduction.
        # G = cos²(arg), arg = clip(π f h_s / U_c, 0, π/2)
        # U_c = 0.7 v_rel (convective velocity at TE).
        # arg clamped to first quadrant: monotone reduction, no oscillation artifacts.
        U_c = max(0.7 * v_rel, 1.0)
        arg = np.clip(np.pi * f * h_s / U_c, 0.0, np.pi / 2)
        G_s = np.maximum(np.cos(arg) ** 2, 1e-3)   # floor at -30 dB
        spl += 10 * np.log10(G_s)

    return spl


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
# ITU-R BS.468 noise weighting (better than A-weighting for tonal/impulsive)
# ---------------------------------------------------------------------------

_ITU468_WEIGHT = np.array([
    -29.9, -26.2, -22.5, -19.4, -16.1, -13.4, -10.9,  -8.6,  -6.6,  -4.8,
     -3.2,  -1.9,  -0.8,   0.0,   0.6,   1.0,   1.2,   1.3,   1.2,
      1.0,   0.5,  -0.1,  -1.1,  -2.5,  -4.3,  -6.6,  -9.3,
], dtype=float)

# ITU-R 468 peaks at 6.3 kHz (+12.2 dB) then rolls off; re-center on 1 kHz = 0 dB
_ITU468_WEIGHT_RAW = np.array([
    -29.9, -26.2, -22.5, -19.4, -16.1, -13.4, -10.9,  -8.6,  -6.6,  -4.8,
     -3.2,  -1.9,  -0.8,   0.0,   0.6,   1.0,   1.2,   1.3,   1.2,
      1.0,   0.5,  -0.1,  -1.1,  -2.5,  -4.3,  -6.6,  -9.3,
], dtype=float)


def _itu468_weight_interp(freq):
    """ITU-R BS.468 correction [dB] at arbitrary frequency (log-linear interp)."""
    return float(np.interp(np.log10(max(freq, 1.0)),
                           np.log10(THIRD_OCT_FREQS), _ITU468_WEIGHT_RAW))


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


def _fwh_tonal_spl(thrust, torque, rpm, num_blades, radius_m,
                   blade_angles_deg=None, r_obs=1.0, harmonic=1):
    """
    FW-H compact-dipole loading noise, m-th BPF harmonic, broadside observer.

    Uses the Hanson (1980) / Lowson (1965) compact-source limit:
        |p_m| = m * B^2 * n^2 * R_eff * T * |sin θ| / (r * c^2)
              + torque term (drag dipole)
    Valid for kR << 1 (no Bessel functions). At M_tip = 0.19 and 3 blades
    this predicts ~70-75 dB at 1 m — physically correct vs ~30 dB from Gutin.

    Returns (spl_dB, freq_Hz).
    """
    n_rps  = rpm / 60.0
    B      = num_blades
    m      = harmonic
    f_tone = m * B * n_rps

    R_eff = 0.8 * radius_m   # aerodynamic centre ~80% R

    # Thrust dipole (axial force): dominant at broadside (sin θ = 1)
    p_thrust = (m * B ** 2 * n_rps ** 2 * R_eff * thrust) / (r_obs * C_SOUND ** 2)

    # Drag/torque dipole (in-plane force): zero at broadside but kept for
    # gradient continuity — scales with torque / (R * B)
    F_drag   = torque / (radius_m + 1e-12)   # total tangential force
    p_torque = (m * B ** 2 * n_rps ** 2 * R_eff * F_drag) / (r_obs * C_SOUND ** 2) * 0.25

    p_single = np.sqrt(p_thrust ** 2 + p_torque ** 2)

    # Unequal-spacing interference factor
    F_int = _blade_spacing_factor(m, blade_angles_deg) if blade_angles_deg is not None else 1.0

    p_total = p_single * F_int
    spl     = 20.0 * np.log10(np.maximum(p_total, 1e-30) / P_REF)
    return spl, f_tone


# ---------------------------------------------------------------------------
# A-weighting at arbitrary frequency (log-linear interpolation)
# ---------------------------------------------------------------------------

def _a_weight_interp(freq):
    """A-weighting correction [dB] at any frequency via log-linear interpolation."""
    return float(np.interp(np.log10(max(freq, 1.0)),
                           np.log10(THIRD_OCT_FREQS), A_WEIGHT))


# ---------------------------------------------------------------------------
# BVI tonal noise (Widnall 1971 / Leishman 2006 parametric)
# ---------------------------------------------------------------------------

def _bvi_tonal_spl(thrust, rpm, num_blades, radius_m,
                   blade_angles_deg=None, rho=1.225, r_obs=1.0, n_harmonics=5):
    """
    A-weighted BVI tonal SPL [dBA] — multi-rotor-harmonic model.

    Sums over every rotor harmonic k = 1..n_harmonics*B (not just BPF multiples):
        C_1 = p_ref * sqrt(B)               single-blade reference [normalised]
        p_k = C_1 * k^-1.5 * |Σ_j exp(j·k·θ_j)|   k-th rotor harmonic pressure

    For equal spacing |Σ exp(j·k·θ)| = B at k = m·B and 0 otherwise,
    recovering the original m^-1.5 * F_int formula exactly.
    For unequal spacing, sub-harmonics (k ≠ m·B) appear and contribute — the
    gradient w.r.t. θ_2, θ_3 is then non-zero at off-equal-spacing starting
    points, enabling SLSQP to find the optimal blade arrangement.

    At APC 7x5E baseline (7000 RPM, 2.87 N, equal spacing): ~64 dBA.
    """
    if thrust <= 0.0:
        return -200.0

    Omega  = rpm * 2.0 * np.pi / 60.0
    v_tip  = Omega * radius_m
    B      = int(num_blades)
    f_rot  = rpm / 60.0    # shaft frequency [Hz]

    Gamma_tip = thrust / max(rho * B * Omega * radius_m * np.pi, 1e-12)
    p_ref     = rho * v_tip * Gamma_tip / (2.0 * np.pi * r_obs)
    C1        = p_ref * B ** 0.5    # single-blade normalised reference

    if blade_angles_deg is not None:
        th = np.deg2rad(np.asarray(blade_angles_deg, dtype=float))
    else:
        th = None

    k_arr   = np.arange(1, n_harmonics * B + 1, dtype=float)  # (K,)
    f_k_arr = k_arr * f_rot

    if th is not None:
        # (K,1) × (1,B) -> exp sum over blades -> (K,)
        F_sum_arr = np.abs(np.sum(
            np.exp(1j * k_arr[:, np.newaxis] * th[np.newaxis, :]), axis=1))
    else:
        F_sum_arr = np.where(k_arr % B == 0, float(B), 0.0)

    valid   = F_sum_arr > 1e-9
    p_k_arr = C1 * k_arr ** (-1.5) * np.where(valid, F_sum_arr, 0.0)
    a_wts   = np.interp(np.log10(np.maximum(f_k_arr, 1.0)),
                        np.log10(THIRD_OCT_FREQS), A_WEIGHT)
    spl_k_a = np.where(valid,
                       20.0 * np.log10(np.maximum(p_k_arr, 1e-30) / P_REF) + a_wts,
                       -1000.0)
    return 10.0 * np.log10(max(float(np.sum(10.0 ** (spl_k_a / 10.0))), 1e-300))


# ---------------------------------------------------------------------------
# Hanson (1980) frequency-domain steady loading tonal noise
# ---------------------------------------------------------------------------

def hanson_loading_spl(r_m, dT_dr, dQ_dr, rpm, num_blades,
                       rho=1.225, r_obs=1.0, n_harmonics=5, blade_angles_deg=None):
    """
    A-weighted tonal SPL [dBA] from steady blade loading (Hanson 1980).

    Helicoidal surface theory: distributed thrust/torque loading radiates
    via Bessel-function-weighted span integrals at each BPF harmonic.

        I_T(m) = ∫ (dT/dr) J_{mB}(m·B·M_r) dr          thrust dipole
        I_Q(m) = ∫ (dQ/r·dr) J'_{mB}(m·B·M_r) c₀/Ω dr  torque dipole
        |p_m|  = (m·B·Ω) / (4π·r_obs·c₀) · sqrt(I_T² + I_Q²)

    At M_tip ~ 0.17 the Bessel argument m·B·M_r << 1 so each harmonic is
    ~ 30-37 dBA — well below BVI (~64 dBA) and broadband (~70 dBA) but the
    distributed gradient d(SPL)/d(dT/dr_i) provides per-span-station signal
    that the global thrust integral in BVI cannot give.
    """
    if np.all(np.asarray(dT_dr) == 0.0):
        return -200.0

    r_m   = np.asarray(r_m,   dtype=float)
    dT_dr = np.asarray(dT_dr, dtype=float)
    dQ_dr = np.asarray(dQ_dr, dtype=float)

    Omega = rpm * 2.0 * np.pi / 60.0
    B     = int(num_blades)
    M_r   = Omega * r_m / C_SOUND          # local rotational Mach number

    ms_a = 0.0
    for m in range(1, n_harmonics + 1):
        order = m * B
        arg   = order * M_r                # m·B·M_r per station

        J     = _bessel_j(order,     arg)
        dJ    = 0.5 * (_bessel_j(order - 1, arg) - _bessel_j(order + 1, arg))

        I_T = np.trapezoid(dT_dr * J,  r_m)

        dQ_r = dQ_dr / np.maximum(r_m, 1e-6)
        I_Q  = np.trapezoid(dQ_r * dJ, r_m) * C_SOUND / Omega

        prefactor = (m * B * Omega) / (4.0 * np.pi * r_obs * C_SOUND)
        p_peak    = prefactor * np.sqrt(I_T ** 2 + I_Q ** 2)
        p_rms     = p_peak / np.sqrt(2.0)

        F_int = (_blade_spacing_factor(m, blade_angles_deg)
                 if blade_angles_deg is not None else 1.0)
        p_rms *= F_int

        f_m      = m * B * rpm / 60.0
        spl_m_a  = 20.0 * np.log10(max(p_rms, 1e-30) / P_REF) + _a_weight_interp(f_m)
        ms_a    += 10.0 ** (spl_m_a / 10.0)

    return 10.0 * np.log10(max(ms_a, 1e-300))


# ---------------------------------------------------------------------------
# Amiet leading-edge turbulence interaction noise (LETI)
# ---------------------------------------------------------------------------

def _amiet_leti_spl(chord, v_rel, dr, r_obs=1.0,
                    turb_intensity=0.005, turb_length_scale=0.01, rho=1.225,
                    cos_sweep=1.0):
    """
    One-third-octave SPL from leading-edge turbulence interaction (Amiet 1975).

    Compact-source form (Glegg & Devenport 2017 §9.4, Paterson & Amiet 1976):

        S_pp(f) = (ρ c₀ k₀)² (c/2)² l_y dr Φ_uu(f) / (4π² r²)  [Pa²/Hz]

    where:
        k₀     = 2πf / c₀           acoustic wavenumber
        c/2    = half-chord          compact LE scattering length
        l_y    = min(dr, π L_t)     spanwise correlation length
        Φ_uu   = one-sided turbulence velocity PSD [m²/s]  (von Karman spectrum)

    Swept-LE correction (Amiet 1975): for sweep angle Λ the upwash normal to the
    LE reduces to w·cos(Λ), and the effective scattering half-chord to b·cos(Λ).
    Combined: S_pp_swept = S_pp_unswept × cos⁴(Λ).

    One-third octave conversion:  ms_band = S_pp · Δf,  Δf ≈ 0.2316 f

    Parameters
    ----------
    turb_intensity     : u_rms / v_rel  (default 0.005 ≈ calm hover)
    turb_length_scale  : integral length scale L_t [m]  (default 0.01 m)
    cos_sweep          : cos(Λ) of local LE sweep angle (1.0 = unswept)
    """
    if v_rel < 1.0:
        return np.full(len(THIRD_OCT_FREQS), -200.0)

    f      = THIRD_OCT_FREQS
    k0     = 2.0 * np.pi * f / C_SOUND          # acoustic wavenumber [1/m]
    k1     = 2.0 * np.pi * f / (v_rel + 1e-12)  # convective wavenumber [1/m]

    # von Karman one-sided velocity PSD [m²/s], normalises to u_rms² when integrated
    k1_Lt  = k1 * turb_length_scale
    Phi_uu = (turb_intensity * v_rel) ** 2 * (2.0 * turb_length_scale / v_rel) / np.maximum(
        (1.0 + k1_Lt ** 2) ** (5.0 / 6.0), 1e-30)

    # Spanwise correlation length: compact strip or turbulence-limited
    l_y = min(dr, np.pi * turb_length_scale)

    # Compact-source PSD [Pa²/Hz]; swept-LE correction: S_pp ∝ cos⁴(Λ)
    half_chord = 0.5 * chord
    S_pp = ((rho * C_SOUND * k0) ** 2
            * half_chord ** 2 * l_y * dr
            * Phi_uu
            / (4.0 * np.pi ** 2 * r_obs ** 2)
            * cos_sweep ** 4)

    # One-third octave mean-square pressure
    delta_f = f * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    ms_band = S_pp * delta_f
    return 10.0 * np.log10(np.maximum(ms_band / P_REF ** 2, 1e-300))


# ---------------------------------------------------------------------------
# Full BPM + FW-H tonal + Amiet LETI model
# ---------------------------------------------------------------------------

def bpm_noise(r_m, chord_m, v_rel, aoa_deg,
              thrust, torque, rpm, num_blades, radius_m,
              rho=1.225, r_obs=1.0,
              x_tr_c=None, blade_angles_deg=None,
              turb_intensity=0.005, turb_length_scale=0.01,
              sweep_m=None, dT_dr=None, dQ_dr=None,
              turb_loading_coeff=0.03, h_s=None,
              h_LE=None, A_tub=None,
              lambda_LE=None, lambda_tub=None):
    """
    Compute SPL spectrum (BPM broadband + Amiet LETI).

    Broadband: TBL-TE + LBL-VS (BPM 1989) + Amiet LETI (Amiet 1975).
    Amiet LETI is the dominant mechanism at M_tip~0.19; gives 60-70 dBA at 1 m.

    SPL_tonal is returned as -200 dB. Steady-loading BPF tonal is near-zero at
    this M_tip (compact broadside dipole). blade_angles_deg is retained for
    future blade-vortex interaction (BVI) unsteady tonal modelling.

    Parameters
    ----------
    x_tr_c            : array (N,) — per-station transition x/c (Michel's criterion)
    blade_angles_deg   : array (B,) — retained for future BVI tonal model
    turb_intensity     : u_rms / v_rel for Amiet LETI (default 0.005 ≈ calm hover)
    turb_length_scale  : integral length scale in m for Amiet LETI (default 0.01)
    sweep_m            : array (N,) — LE sweep offset in m; None = unswept
    """
    N = len(r_m)
    if x_tr_c is None:
        x_tr_c = np.zeros(N)
    x_tr_c = np.asarray(x_tr_c, dtype=float)

    # Thrust-dependent turbulence: heavier disk loading → stronger tip vortex →
    # higher inflow turbulence seen by each blade's LE.
    #   turb_eff = turb_ambient + k × T / (ρ π R² (ΩR)²)
    # At baseline (2.87 N, 7000 RPM): +0.30% above ambient (total ~0.80%)
    # At optimum  (2.44 N, 6053 RPM): +0.38% above ambient (total ~0.88%)
    Omega_eff = rpm * 2.0 * np.pi / 60.0
    v_tip_eff = Omega_eff * radius_m
    dyn_denom  = max(rho * np.pi * radius_m ** 2 * v_tip_eff ** 2, 1e-12)
    turb_intensity = turb_intensity + turb_loading_coeff * thrust / dyn_denom

    # Per-station LE sweep angle -> cos⁴ correction for Amiet LETI
    if sweep_m is not None:
        sweep_m = np.asarray(sweep_m, dtype=float)
        r_m_arr = np.asarray(r_m, dtype=float)
        # Local sweep angle: dSweep/dr; cap at 60° to avoid unphysical results
        dsweep_dr = np.gradient(sweep_m, r_m_arr)
        sweep_angle = np.clip(np.arctan(np.abs(dsweep_dr)), 0.0, np.deg2rad(60.0))
        cos_sweep_arr = np.cos(sweep_angle)
    else:
        cos_sweep_arr = np.ones(N)

    n_freqs = len(THIRD_OCT_FREQS)
    dr_arr  = np.gradient(r_m)
    f_v     = THIRD_OCT_FREQS[np.newaxis, :]         # (1, n_freqs)

    # --- LETI: fully vectorised across all N stations (dominant source) ---
    # Eliminates N scalar function calls and N log-linear-log round-trips.
    chord_v = chord_m[:, np.newaxis]                  # (N, 1)
    vrel_v  = np.maximum(v_rel, 1.0)[:, np.newaxis]  # (N, 1)
    dr_v    = dr_arr[:, np.newaxis]                   # (N, 1)
    cs_v    = cos_sweep_arr[:, np.newaxis]            # (N, 1)

    k0_v   = 2.0 * np.pi * f_v / C_SOUND             # (1, n_freqs)
    k1_v   = 2.0 * np.pi * f_v / vrel_v              # (N, n_freqs)
    k1Lt_v = k1_v * turb_length_scale
    Phi_v  = ((turb_intensity * vrel_v) ** 2
              * (2.0 * turb_length_scale / vrel_v)
              / np.maximum((1.0 + k1Lt_v ** 2) ** (5.0 / 6.0), 1e-30))
    l_y_v  = np.minimum(dr_v, np.pi * turb_length_scale)
    df_v   = f_v * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    Spp_v  = ((rho * C_SOUND * k0_v) ** 2
              * (0.5 * chord_v) ** 2 * l_y_v * dr_v
              * Phi_v / (4.0 * np.pi ** 2 * r_obs ** 2)
              * cs_v ** 4)                            # (N, n_freqs)

    # LE combined treatment: sawtooth serrations + sinusoidal tubercles.
    #
    # The two mechanisms operate at different spanwise length scales and can
    # coexist physically: fine sawtooth (h_LE ~ 1-4 mm, lambda = 2*h_LE) riding
    # on a large sinusoidal LE wave (A_tub ~ 2-4 mm, lambda_tub >> lambda_s).
    # Their LETI reductions multiply because each scatters a different wavenumber
    # band of the incoming turbulence spectrum.
    #
    # Sawtooth  (Lyu et al. 2016, compact limit):
    #   G_s = sinc²(f·h_LE / U_c),  U_c = 0.7·v_rel
    #
    # Tubercle  (Chaitanya et al. 2017, compact approximation):
    #   G_t = J₀²(π·f·A_tub / U_c)
    #
    # Combined: G_total = G_s × G_t (independent scattering channels)
    if h_LE is not None or A_tub is not None:
        G_total_v = np.ones((N, len(THIRD_OCT_FREQS)))

        if h_LE is not None:
            h_arr  = np.asarray(h_LE, dtype=float)
            if h_arr.ndim == 0:
                h_arr = np.full(N, float(h_arr))
            h_v    = np.maximum(h_arr, 0.0)[:, np.newaxis]   # (N,1)
            St_s   = f_v * h_v / vrel_v                       # (N, n_freqs)
            G_s    = np.maximum(np.sinc(St_s) ** 2, 1e-3)

            if lambda_LE is not None:
                # Aspect-ratio correction: Chaitanya et al. (2017) empirical fit.
                # Gamma(h/lambda) peaks at h/lambda = 0.5, sigma = 0.3.
                # Applied as: G = 1 - Gamma*(1 - sinc²) so G->1 when h->0.
                lam_arr = np.asarray(lambda_LE, dtype=float)
                if lam_arr.ndim == 0:
                    lam_arr = np.full(N, float(lam_arr))
                ratio_s = h_arr / np.maximum(lam_arr, 1e-6)  # (N,)
                gamma_s = np.exp(-((ratio_s - 0.5) / 0.3) ** 2)[:, np.newaxis]
                G_s     = np.maximum(1.0 - gamma_s * (1.0 - G_s), 1e-3)

            G_total_v *= G_s

        if A_tub is not None:
            a_arr  = np.asarray(A_tub, dtype=float)
            if a_arr.ndim == 0:
                a_arr = np.full(N, float(a_arr))
            a_v    = np.maximum(a_arr, 0.0)[:, np.newaxis]
            St_t   = f_v * a_v / vrel_v
            G_t    = np.maximum(_bessel_j(0, np.pi * St_t) ** 2, 1e-3)

            if lambda_tub is not None:
                # Tubercle aspect-ratio correction peaks at A/lambda = 0.4.
                # Applied as: G = 1 - Gamma*(1 - J0²) so G->1 when A->0.
                lam_arr = np.asarray(lambda_tub, dtype=float)
                if lam_arr.ndim == 0:
                    lam_arr = np.full(N, float(lam_arr))
                ratio_t = a_arr / np.maximum(lam_arr, 1e-6)  # (N,)
                gamma_t = np.exp(-((ratio_t - 0.4) / 0.25) ** 2)[:, np.newaxis]
                G_t     = np.maximum(1.0 - gamma_t * (1.0 - G_t), 1e-3)

            G_total_v *= G_t

        Spp_v = Spp_v * G_total_v

    leti_lin = np.maximum(Spp_v * df_v / P_REF ** 2, 0.0)  # (N, n_freqs), linear (p/Pref)²

    # --- TBL-TE + LBL-VS: single pass, linear accumulation (no log round-trips) ---
    tbl_lin = np.zeros(n_freqs)
    lbl_lin = np.zeros(n_freqs)

    for i in range(N):
        xtr_i  = float(x_tr_c[i])
        w_turb = 1.0 - xtr_i
        h_s_i  = float(h_s[i]) if h_s is not None else 0.0
        if w_turb > 0.01 and float(v_rel[i]) >= 1.0:
            tbl_lin += w_turb * 10 ** (_tbl_te_spl(
                float(chord_m[i]), float(v_rel[i]), float(aoa_deg[i]),
                float(dr_arr[i]), r_obs, xtr_i, h_s=h_s_i) / 10.0)
        if xtr_i > 0.30 and float(v_rel[i]) >= 1.0:
            lbl_lin += xtr_i * 10 ** (_lbl_vs_spl(
                float(chord_m[i]), float(v_rel[i]), float(aoa_deg[i]),
                float(dr_arr[i]), r_obs) / 10.0)

    leti_lin_sum    = np.sum(leti_lin, axis=0)              # (n_freqs,)
    linear_spectrum = tbl_lin + lbl_lin + leti_lin_sum
    SPL_spectrum    = 10.0 * np.log10(np.maximum(linear_spectrum, 1e-300))
    SPL_broadband   = 10.0 * np.log10(np.sum(linear_spectrum) + 1e-300)

    # BVI tonal (unsteady loading, dominant real-drone tonal mechanism)
    SPL_bvi = _bvi_tonal_spl(thrust, rpm, num_blades, radius_m,
                              blade_angles_deg, rho, r_obs)

    # Hanson (1980) steady loading tonal via distributed dT/dr, dQ/dr
    if dT_dr is not None and dQ_dr is not None:
        SPL_hanson = hanson_loading_spl(r_m, dT_dr, dQ_dr, rpm, num_blades,
                                        rho, r_obs, blade_angles_deg=blade_angles_deg)
    else:
        SPL_hanson = -200.0

    SPL_tonal = 10.0 * np.log10(
        10.0 ** (SPL_bvi / 10.0) + 10.0 ** (SPL_hanson / 10.0) + 1e-300)

    SPL_A        = SPL_spectrum + A_WEIGHT
    SPL_bb_a     = 10 * np.log10(np.sum(10 ** (SPL_A / 10)) + 1e-300)
    SPL_total    = 10 * np.log10(
        10 ** (SPL_bb_a / 10) + 10 ** (SPL_tonal / 10) + 1e-300)

    # Per-mechanism breakdown — derived from first-pass linear arrays (no second loop)
    spec_tbl  = 10.0 * np.log10(np.maximum(tbl_lin,      1e-300))
    spec_lbl  = 10.0 * np.log10(np.maximum(lbl_lin,      1e-300))
    spec_leti = 10.0 * np.log10(np.maximum(leti_lin_sum, 1e-300))

    def _dba(spec):
        return 10 * np.log10(np.sum(10**((spec + A_WEIGHT)/10)) + 1e-300)

    def _ditu(spec):
        return 10 * np.log10(np.sum(10**((spec + _ITU468_WEIGHT_RAW)/10)) + 1e-300)

    # ITU-R BS.468 total (broadband + tonal combined)
    SPL_itu468_bb = _ditu(SPL_spectrum)
    SPL_itu468    = 10 * np.log10(
        10 ** (SPL_itu468_bb / 10) + 10 ** (SPL_tonal / 10) + 1e-300)

    # Thrust-to-noise merit factor: thrust [N] / acoustic pressure [Pa] at r_obs
    p_rms_total = P_REF * 10.0 ** (SPL_total / 20.0)
    merit_factor = thrust / max(p_rms_total, 1e-30)

    return {
        "SPL_total":     SPL_total,
        "SPL_broadband": SPL_broadband,
        "SPL_tonal":     SPL_tonal,
        "SPL_itu468":    SPL_itu468,
        "merit_factor":  merit_factor,
        "freq":          THIRD_OCT_FREQS.copy(),
        "SPL_spectrum":  SPL_spectrum,
        # per-mechanism A-weighted totals
        "SPL_tbl_te_dBA":  _dba(spec_tbl),
        "SPL_lbl_vs_dBA":  _dba(spec_lbl),
        "SPL_leti_dBA":    _dba(spec_leti),
        "SPL_bvi_dBA":     SPL_bvi,
        "SPL_hanson_dBA":  SPL_hanson,
        # per-mechanism spectra (linear, not A-weighted)
        "spec_tbl":   spec_tbl,
        "spec_lbl":   spec_lbl,
        "spec_leti":  spec_leti,
    }


# ---------------------------------------------------------------------------
# OpenMDAO component
# ---------------------------------------------------------------------------

class BPMComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("blade",      default=None)
        self.options.declare("n_stations", default=20)
        self.options.declare("n_cp",       default=5)
        self.options.declare("r_obs",      default=1.0)
        self.options.declare("fd_step",    default=3e-4)
        self.options.declare("le_type",    default="sawtooth",
                             values=["sawtooth", "tubercle"])

    def setup(self):
        self._blade = self.options["blade"] or baseline_apc7x5e()
        N      = self.options["n_stations"]
        n_cp   = self.options["n_cp"]
        n_freq = len(THIRD_OCT_FREQS)
        B      = self._blade.num_blades
        _, chord0, _ = self._blade.get_stations(N)

        self._r_cp = np.linspace(self._blade.r_R[0], self._blade.r_R[-1], n_cp)

        self.add_input("r_m",             val=np.zeros(N), units="m")
        self.add_input("chord_m",         val=chord0,       units="m")
        self.add_input("v_rel",           val=np.zeros(N), units="m/s")
        self.add_input("aoa_deg",         val=np.zeros(N))
        self.add_input("x_tr_c",         val=np.zeros(N))   # from CCBladeComponent
        self.add_input("sweep_m",         val=np.zeros(N), units="m")  # from GeometryFullComponent
        self.add_input("thrust",          val=0.0,    units="N")
        self.add_input("torque",          val=0.0,    units="N*m")
        self.add_input("rpm",             val=5000.0, units="rpm")
        self.add_input("rho",             val=1.225,  units="kg/m**3")
        self.add_input("blade_angles_deg", val=self._blade.blade_angles_deg.copy())
        # Amiet LETI parameters — vary with operating environment
        self.add_input("turb_intensity",      val=0.005)  # ambient u_rms/v_rel; loading term added internally
        self.add_input("turb_length_scale",   val=0.01, units="m")
        self.add_input("turb_loading_coeff",  val=0.03)  # scales thrust-dependent turbulence amplification
        self.add_input("dT_dr",  val=np.zeros(N), units="N/m")
        self.add_input("dQ_dr",  val=np.zeros(N), units="N*m/m")
        # TE serration depth (inert at UAV Re; TBL-TE = 0)
        self.add_input("h_s_cp",   val=np.zeros(n_cp), units="m")
        # LE sawtooth: amplitude + wavelength (Lyu 2016)
        self.add_input("h_LE_cp",      val=np.zeros(n_cp),           units="m")
        self.add_input("lambda_LE_cp", val=np.full(n_cp, 2e-3),      units="m")
        # LE tubercle: amplitude + wavelength (Chaitanya 2017)
        self.add_input("A_tub_cp",     val=np.zeros(n_cp),           units="m")
        self.add_input("lambda_tub_cp",val=np.full(n_cp, 2e-3),      units="m")

        self.add_output("SPL_total",     val=0.0)
        self.add_output("SPL_broadband", val=0.0)
        self.add_output("SPL_tonal",     val=0.0)
        self.add_output("SPL_itu468",    val=0.0)
        self.add_output("merit_factor",  val=0.0)
        self.add_output("SPL_spectrum",  val=np.zeros(n_freq))
        self.add_output("freq",          val=THIRD_OCT_FREQS.copy(), units="Hz")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=self.options["fd_step"])

    def compute(self, inputs, outputs):
        r_R = inputs["r_m"] / self._blade.radius_m

        def _interp_cp(cp_vals):
            if np.any(cp_vals > 1e-6):
                return np.clip(CubicSpline(self._r_cp, cp_vals)(r_R), 0.0, None)
            return None

        def _interp_cp_pos(cp_vals):
            """Interpolate, clip to positive (wavelengths must be > 0)."""
            return np.clip(CubicSpline(self._r_cp, cp_vals)(
                inputs["r_m"] / self._blade.radius_m), 1e-6, None)

        h_s        = _interp_cp(inputs["h_s_cp"])
        h_LE       = _interp_cp(inputs["h_LE_cp"])
        A_tub      = _interp_cp(inputs["A_tub_cp"])
        lambda_LE  = _interp_cp_pos(inputs["lambda_LE_cp"])
        lambda_tub = _interp_cp_pos(inputs["lambda_tub_cp"])

        res = bpm_noise(
            r_m=inputs["r_m"],
            chord_m=inputs["chord_m"],
            v_rel=inputs["v_rel"],
            aoa_deg=inputs["aoa_deg"],
            thrust=float(inputs["thrust"][0]),
            torque=float(inputs["torque"][0]),
            rpm=float(inputs["rpm"][0]),
            num_blades=self._blade.num_blades,
            radius_m=self._blade.radius_m,
            rho=float(inputs["rho"][0]),
            r_obs=self.options["r_obs"],
            x_tr_c=inputs["x_tr_c"],
            blade_angles_deg=inputs["blade_angles_deg"],
            turb_intensity=float(inputs["turb_intensity"][0]),
            turb_length_scale=float(inputs["turb_length_scale"][0]),
            turb_loading_coeff=float(inputs["turb_loading_coeff"][0]),
            sweep_m=inputs["sweep_m"],
            dT_dr=inputs["dT_dr"],
            dQ_dr=inputs["dQ_dr"],
            h_s=h_s,
            h_LE=h_LE,
            A_tub=A_tub,
            lambda_LE=lambda_LE,
            lambda_tub=lambda_tub,
        )
        outputs["SPL_total"]     = res["SPL_total"]
        outputs["SPL_broadband"] = res["SPL_broadband"]
        outputs["SPL_tonal"]     = res["SPL_tonal"]
        outputs["SPL_itu468"]    = res["SPL_itu468"]
        outputs["merit_factor"]  = res["merit_factor"]
        outputs["SPL_spectrum"]  = res["SPL_spectrum"]
        outputs["freq"]          = res["freq"]


if __name__ == "__main__":
    from aerodynamics.ccblade_component import bem_solve

    blade = baseline_apc7x5e()
    aero  = bem_solve(blade, rpm=5000, v_inf=0.0)
    _, chord_m, _ = blade.get_stations(20)

    # Equal spacing
    res_equal = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=5000, num_blades=blade.num_blades, radius_m=blade.radius_m,
        x_tr_c=aero["x_tr_c"],
        blade_angles_deg=np.array([0.0, 120.0, 240.0]),
    )

    # Unequal spacing
    res_unequal = bpm_noise(
        r_m=aero["r"], chord_m=chord_m,
        v_rel=aero["v_rel"], aoa_deg=aero["aoa_deg"],
        thrust=aero["thrust"], torque=aero["torque"],
        rpm=5000, num_blades=blade.num_blades, radius_m=blade.radius_m,
        x_tr_c=aero["x_tr_c"],
        blade_angles_deg=np.array([0.0, 115.0, 235.0]),
    )

    for label, res in [("Equal spacing  ", res_equal), ("Unequal spacing", res_unequal)]:
        print(f"{label}: SPL={res['SPL_total']:.1f} dBA  "
              f"(broad={res['SPL_broadband']:.1f} dB, tonal={res['SPL_tonal']:.1f} dB)")
    print(f"Mean x_tr/c: {np.mean(aero['x_tr_c']):.3f}  (1.0=laminar, 0.0=turbulent)")
    print("Target: 60-70 dBA at 1 m")
