"""
Blade Element Momentum (BEM) solver wrapped as an OpenMDAO ExplicitComponent.

Two-regime implementation:
  - Forward flight (V_inf >= 0.5 m/s): standard BEM, V_a = V_inf*(1-a)
  - Static/hover  (V_inf < 0.5 m/s):  direct phi iteration via momentum balance
    phi = arcsin(sqrt(sigma*Cn / (4*F)))

Sign convention (standard propeller BEM):
  phi   = arctan(V_axial / V_tangential)    [flow angle from rotation plane]
  alpha = twist - phi                        [angle of attack, positive = nose-up]
  dT    = 0.5*rho*V_rel^2 * c*B * (Cl*cos(phi) + Cd*sin(phi))
  dQ    = 0.5*rho*V_rel^2 * c*B * (Cl*sin(phi) - Cd*cos(phi)) * r
          (positive dQ => shaft inputs torque => propeller mode)
"""

import numpy as np
import openmdao.api as om
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry.blade_generator import baseline_hqprop


# ---------------------------------------------------------------------------
# NACA 4412 polar  (Re ~ 50 000–200 000, UAV propeller regime)
# ---------------------------------------------------------------------------

def _naca4412_polar(alpha_deg):
    alpha = np.asarray(alpha_deg, dtype=float)
    cl = np.zeros_like(alpha)
    cd = np.zeros_like(alpha)

    alpha0          = -4.0
    cl_slope        = 0.1097      # per degree
    alpha_stall_pos =  14.0
    alpha_stall_neg = -10.0

    att = (alpha > alpha_stall_neg) & (alpha < alpha_stall_pos)
    cl[att] = cl_slope * (alpha[att] - alpha0)

    sp = alpha >= alpha_stall_pos
    sn = alpha <= alpha_stall_neg
    cl[sp] = 0.95 * np.sin(np.deg2rad(2 * alpha[sp]))
    cl[sn] = 0.95 * np.sin(np.deg2rad(2 * alpha[sn]))

    cd0        = 0.010
    cl_mindrag = cl_slope * (2.0 - alpha0)
    cd[att]    = cd0 + (cl[att] - cl_mindrag) ** 2 / (np.pi * 0.85 * 6.0)
    cd[sp]     = 0.15 + 1.5 * np.sin(np.deg2rad(alpha[sp])) ** 2
    cd[sn]     = 0.15 + 1.5 * np.sin(np.deg2rad(alpha[sn])) ** 2

    return cl, cd


def _prandtl_loss(r, R, R_hub, B, phi):
    phi_safe = np.maximum(np.abs(phi), 0.005)
    f_tip = np.clip((B / 2.0) * (R - r)     / (r * np.sin(phi_safe)), 1e-4, 30)
    f_hub = np.clip((B / 2.0) * (r - R_hub) / (r * np.sin(phi_safe)), 1e-4, 30)
    F = ((2 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), 0, 1)) *
         (2 / np.pi) * np.arccos(np.clip(np.exp(-f_hub), 0, 1)))
    return np.maximum(F, 0.01)


# ---------------------------------------------------------------------------
# Static / hover BEM  (V_inf = 0)
# ---------------------------------------------------------------------------

def _bem_static(r, chord, twist_deg, sigma, omega, R, R_hub, B, rho,
                tol=1e-8, max_iter=300):
    """
    Solve for phi at each station via momentum balance for V_inf = 0.

    From equating blade-element thrust and actuator-disk momentum:
        4*F*sin^2(phi) = sigma*Cn(phi)
    Iterate:  phi_{n+1} = arcsin(sqrt(sigma*Cn(phi_n) / (4*F)))
    """
    phi = np.deg2rad(twist_deg) * 0.4    # initial guess: ~40% of geometric angle
    phi = np.clip(phi, 0.02, np.pi / 4)
    ap  = np.zeros_like(r)

    for _ in range(max_iter):
        phi_old = phi.copy()

        alpha_d = twist_deg - np.degrees(phi)
        cl, cd  = _naca4412_polar(alpha_d)
        cn      = cl * np.cos(phi) + cd * np.sin(phi)
        ct      = cl * np.sin(phi) - cd * np.cos(phi)

        F = _prandtl_loss(r, R, R_hub, B, phi)

        # Momentum-BEM match for phi
        sin2_new = np.clip(sigma * cn / (4.0 * F + 1e-12), 0.0, 0.98)
        phi_new  = np.arcsin(np.sqrt(sin2_new))
        phi_new  = np.clip(phi_new, 0.005, np.pi / 3)

        # Tangential induction (standard BEM)
        kappa_t = sigma * ct / (4.0 * F * np.sin(phi) * np.cos(phi) + 1e-12)
        ap      = np.clip(kappa_t / (1.0 - kappa_t + 1e-12), -0.3, 1.0)

        phi = 0.4 * phi + 0.6 * phi_new

        if np.max(np.abs(phi - phi_old)) < tol:
            break

    alpha_d = twist_deg - np.degrees(phi)
    cl, cd  = _naca4412_polar(alpha_d)
    cn      = cl * np.cos(phi) + cd * np.sin(phi)
    ct      = cl * np.sin(phi) - cd * np.cos(phi)

    v_i   = omega * r * np.tan(phi)          # induced axial velocity
    V_t   = omega * r * (1.0 + ap)
    v_rel = np.sqrt(v_i ** 2 + V_t ** 2)

    dT = 0.5 * rho * v_rel ** 2 * chord * B * cn
    dQ = 0.5 * rho * v_rel ** 2 * chord * B * ct * r

    return phi, alpha_d, cl, cd, v_rel, dT, dQ


# ---------------------------------------------------------------------------
# Forward-flight BEM  (V_inf >= threshold)
# ---------------------------------------------------------------------------

def _bem_forward(r, chord, twist_deg, sigma, omega, R, R_hub, B, rho, V,
                 tol=1e-8, max_iter=300):
    """
    Standard BEM for V_inf > 0 using PROPELLER momentum convention:
      V_a = V*(1+a)   [propeller accelerates flow, not decelerates]
      a   = κ/(1-κ)   where κ = σ*Cn/(4F·sin²φ)
    """
    a  = np.full(len(r), 0.05)
    ap = np.zeros(len(r))

    for _ in range(max_iter):
        a_old  = a.copy()
        ap_old = ap.copy()

        V_a = V * (1.0 + a)          # propeller accelerates axial flow
        V_t = omega * r * (1.0 + ap)
        phi = np.arctan2(V_a, V_t)

        alpha_d = twist_deg - np.degrees(phi)
        cl, cd  = _naca4412_polar(alpha_d)
        cn      = cl * np.cos(phi) + cd * np.sin(phi)
        ct      = cl * np.sin(phi) - cd * np.cos(phi)

        F = _prandtl_loss(r, R, R_hub, B, phi)

        # Propeller axial induction: a/(1+a) = κ → a = κ/(1-κ)
        kappa = np.clip(
            sigma * cn / (4.0 * F * np.sin(phi) ** 2 + 1e-12),
            0.0, 0.9)
        a_new = np.clip(kappa / (1.0 - kappa + 1e-12), 0.0, 1.5)

        # Tangential induction (same propeller form)
        kappa_t = np.clip(
            sigma * ct / (4.0 * F * np.sin(phi) * np.cos(phi) + 1e-12),
            -0.5, 0.9)
        ap_new = np.clip(kappa_t / (1.0 - kappa_t + 1e-12), -0.3, 1.0)

        a  = 0.5 * a  + 0.5 * a_new
        ap = 0.5 * ap + 0.5 * ap_new

        if (np.max(np.abs(a - a_old)) < tol and
                np.max(np.abs(ap - ap_old)) < tol):
            break

    V_a   = V * (1.0 + a)
    V_t   = omega * r * (1.0 + ap)
    phi   = np.arctan2(V_a, V_t)
    alpha_d = twist_deg - np.degrees(phi)
    cl, cd  = _naca4412_polar(alpha_d)
    cn      = cl * np.cos(phi) + cd * np.sin(phi)
    ct      = cl * np.sin(phi) - cd * np.cos(phi)
    v_rel   = np.sqrt(V_a ** 2 + V_t ** 2)

    dT = 0.5 * rho * v_rel ** 2 * chord * B * cn
    dQ = 0.5 * rho * v_rel ** 2 * chord * B * ct * r

    return phi, alpha_d, cl, cd, v_rel, dT, dQ


# ---------------------------------------------------------------------------
# Public BEM solver
# ---------------------------------------------------------------------------

STATIC_THRESHOLD = 0.5    # m/s — below this, use the static solver

def bem_solve(blade, rpm, v_inf, rho=1.225, n_stations=20,
              chord_override=None, twist_override=None,
              tol=1e-8, max_iter=300):
    """
    Blade Element Momentum solver for a propeller.

    Parameters
    ----------
    chord_override : array (n_stations,) or None
        If provided, replaces blade chord at the n_stations resolution.
    twist_override : array (n_stations,) or None
        If provided, replaces blade twist (deg) at the n_stations resolution.

    Returns dict: thrust, torque, power, efficiency, CT, CP,
                  r, v_rel, aoa_deg, cl, cd
    """
    omega = rpm * 2 * np.pi / 60.0
    R     = blade.radius_m
    R_hub = blade.r_R[0] * R
    B     = blade.num_blades
    V     = float(v_inf)

    r, chord, twist_deg = blade.get_stations(n_stations)
    if chord_override is not None:
        chord = np.asarray(chord_override, dtype=float)
    if twist_override is not None:
        twist_deg = np.asarray(twist_override, dtype=float)
    r      = np.clip(r, R_hub + 1e-6, R - 1e-6)
    sigma  = B * chord / (2.0 * np.pi * r)

    args = (r, chord, twist_deg, sigma, omega, R, R_hub, B, rho)

    if V < STATIC_THRESHOLD:
        phi, alpha_d, cl, cd, v_rel, dT, dQ = _bem_static(*args, tol=tol, max_iter=max_iter)
    else:
        phi, alpha_d, cl, cd, v_rel, dT, dQ = _bem_forward(*args, V, tol=tol, max_iter=max_iter)

    thrust = float(np.trapezoid(dT, r))
    torque = float(np.trapezoid(dQ, r))
    power  = torque * omega
    efficiency = (thrust * V / power) if (V >= STATIC_THRESHOLD and power > 1e-6) else 0.0

    n_rps = omega / (2 * np.pi)
    D     = 2 * R
    CT    = thrust / (rho * n_rps ** 2 * D ** 4)
    CP    = power  / (rho * n_rps ** 3 * D ** 5)

    return {
        "thrust": thrust, "torque": torque, "power": power,
        "efficiency": efficiency, "CT": CT, "CP": CP,
        "r": r, "v_rel": v_rel, "aoa_deg": alpha_d,
        "cl": cl, "cd": cd,
    }


# ---------------------------------------------------------------------------
# OpenMDAO ExplicitComponent
# ---------------------------------------------------------------------------

class CCBladeComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("blade",      default=None)
        self.options.declare("n_stations", default=20)

    def setup(self):
        self._blade = self.options["blade"] or baseline_hqprop()
        N = self.options["n_stations"]
        _, chord0, twist0 = self._blade.get_stations(N)

        self.add_input("rpm",      val=5000.0,    units="rpm")
        self.add_input("v_inf",    val=0.0,        units="m/s")
        self.add_input("rho",      val=1.225,      units="kg/m**3")
        self.add_input("chord_m",  val=chord0,     units="m")
        self.add_input("twist_deg",val=twist0)

        self.add_output("thrust",     val=0.0, units="N")
        self.add_output("torque",     val=0.0, units="N*m")
        self.add_output("power",      val=0.0, units="W")
        self.add_output("efficiency", val=0.0)
        self.add_output("CT",         val=0.0)
        self.add_output("CP",         val=0.0)
        self.add_output("r_m",    val=np.zeros(N), units="m")
        self.add_output("v_rel",  val=np.zeros(N), units="m/s")
        self.add_output("aoa_deg",val=np.zeros(N))
        self.add_output("cl",     val=np.zeros(N))
        self.add_output("cd",     val=np.zeros(N))

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step=1e-4)

    def compute(self, inputs, outputs):
        res = bem_solve(self._blade,
                        rpm=float(inputs["rpm"][0]),
                        v_inf=float(inputs["v_inf"][0]),
                        rho=float(inputs["rho"][0]),
                        n_stations=self.options["n_stations"],
                        chord_override=inputs["chord_m"],
                        twist_override=inputs["twist_deg"])
        for k in ("thrust", "torque", "power", "efficiency", "CT", "CP"):
            outputs[k] = res[k]
        outputs["r_m"]     = res["r"]
        outputs["v_rel"]   = res["v_rel"]
        outputs["aoa_deg"] = res["aoa_deg"]
        outputs["cl"]      = res["cl"]
        outputs["cd"]      = res["cd"]


if __name__ == "__main__":
    blade = baseline_hqprop()
    for rpm, v in [(5000, 0.0), (5000, 5.0), (7000, 0.0)]:
        res = bem_solve(blade, rpm=rpm, v_inf=v)
        print(f"RPM={rpm}, V={v}: T={res['thrust']:.3f}N  "
              f"CT={res['CT']:.4f}  CP={res['CP']:.4f}  "
              f"eta={res['efficiency']:.3f}  P={res['power']:.1f}W")
