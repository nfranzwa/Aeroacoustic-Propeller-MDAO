"""One-shot script to wire lambda_LE_cp and lambda_tub_cp into mdao_problem.py."""
import numpy as np

with open('optimization/mdao_problem.py', encoding='utf-8') as f:
    c = f.read()

def rep(old, new, n=1):
    count = c.count(old)
    assert count == n, f"Expected {n} found {count}: {old[:50]!r}"
    return c.replace(old, new)

# 1. IVC
c = rep(
    '    ivc.add_output("A_tub_cp",        val=np.zeros(N_CP), units="m")',
    '    ivc.add_output("A_tub_cp",        val=np.zeros(N_CP), units="m")\n'
    '    ivc.add_output("lambda_LE_cp",    val=np.full(N_CP, 2e-3), units="m")\n'
    '    ivc.add_output("lambda_tub_cp",   val=np.full(N_CP, 2e-3), units="m")',
)

# 2. acoustics promotes (hover + cruise share same string -> n=2)
c = rep(
    '            "A_tub_cp",\n        ],',
    '            "A_tub_cp",\n            "lambda_LE_cp",\n            "lambda_tub_cp",\n        ],',
    n=2,
)

# 3. DVs
c = rep(
    '    # LE amplitudes (15% chord printer bound)\n'
    '    prob.model.add_design_var("h_LE_cp",       lower=0.0,    upper=h_LE_max_cp, units="m")\n'
    '    prob.model.add_design_var("A_tub_cp",      lower=0.0,    upper=h_LE_max_cp, units="m")',
    '    # LE amplitudes (15% chord printer bound)\n'
    '    prob.model.add_design_var("h_LE_cp",       lower=0.0,    upper=h_LE_max_cp, units="m")\n'
    '    prob.model.add_design_var("A_tub_cp",      lower=0.0,    upper=h_LE_max_cp, units="m")\n'
    '    # LE wavelengths: min = 10 pixels = 0.19 mm (Saturn 4 Ultra 16K), max = 15 mm\n'
    '    prob.model.add_design_var("lambda_LE_cp",  lower=0.0004, upper=0.015, units="m")\n'
    '    prob.model.add_design_var("lambda_tub_cp", lower=0.0004, upper=0.015, units="m")',
)

# 4. le worker unpack
c = rep(
    '    (i, rpm_i, h_LE_cp_i, a_tub_cp_i,\n'
    '     thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer,\n'
    '     le_type, num_blades) = args',
    '    (i, rpm_i, h_LE_cp_i, a_tub_cp_i, lam_LE_i, lam_tub_i,\n'
    '     thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer,\n'
    '     le_type, num_blades) = args',
)

# 5. set_val
c = rep(
    '        prob_i.set_val("rpm",      rpm_i)\n'
    '        prob_i.set_val("h_LE_cp",  h_LE_cp_i)\n'
    '        prob_i.set_val("A_tub_cp", a_tub_cp_i)\n'
    '        prob_i.run_driver()',
    '        prob_i.set_val("rpm",           rpm_i)\n'
    '        prob_i.set_val("h_LE_cp",       h_LE_cp_i)\n'
    '        prob_i.set_val("A_tub_cp",      a_tub_cp_i)\n'
    '        prob_i.set_val("lambda_LE_cp",  lam_LE_i)\n'
    '        prob_i.set_val("lambda_tub_cp", lam_tub_i)\n'
    '        prob_i.run_driver()',
)

# 6. dvs dict
c = rep(
    '            "dvs": {\n'
    '                "rpm":            float(prob_i.get_val("rpm")[0]),\n'
    '                "h_LE_cp":        prob_i.get_val("h_LE_cp").copy(),\n'
    '                "A_tub_cp":       prob_i.get_val("A_tub_cp").copy(),\n'
    '            },',
    '            "dvs": {\n'
    '                "rpm":            float(prob_i.get_val("rpm")[0]),\n'
    '                "h_LE_cp":        prob_i.get_val("h_LE_cp").copy(),\n'
    '                "A_tub_cp":       prob_i.get_val("A_tub_cp").copy(),\n'
    '                "lambda_LE_cp":   prob_i.get_val("lambda_LE_cp").copy(),\n'
    '                "lambda_tub_cp":  prob_i.get_val("lambda_tub_cp").copy(),\n'
    '            },',
)

# 7. start_points
lam0 = 'np.full(N_CP, 2e-3)'
c = rep(
    '    # Start 0: no LE treatment\n'
    '    start_points.append((RPM_HOVER_INIT, np.zeros(N_CP), np.zeros(N_CP)))\n'
    '    # Start 1: half-amplitude serrations only\n'
    '    start_points.append((RPM_HOVER_INIT, 0.5*h_LE_max_cp, np.zeros(N_CP)))\n'
    '    # Start 2: half-amplitude tubercles only\n'
    '    start_points.append((RPM_HOVER_INIT, np.zeros(N_CP), 0.5*h_LE_max_cp))\n'
    '    # Start 3: both at max\n'
    '    start_points.append((RPM_HOVER_INIT, h_LE_max_cp.copy(), h_LE_max_cp.copy()))\n'
    '    # Remaining: random RPM + random split between both treatments\n'
    '    for _ in range(4, n_starts):\n'
    '        rpm_i   = float(rng.uniform(5500.0, 8500.0))\n'
    '        h_LE_i  = rng.uniform(0.0, 1.0, N_CP) * h_LE_max_cp\n'
    '        a_tub_i = rng.uniform(0.0, 1.0, N_CP) * h_LE_max_cp\n'
    '        start_points.append((rpm_i, h_LE_i, a_tub_i))',
    f'    # Start 0: no LE treatment, default wavelengths\n'
    f'    start_points.append((RPM_HOVER_INIT, np.zeros(N_CP), np.zeros(N_CP), {lam0}, {lam0}))\n'
    f'    # Start 1: max sawtooth, optimal wavelength = 2h\n'
    f'    start_points.append((RPM_HOVER_INIT, h_LE_max_cp, np.zeros(N_CP), 2*h_LE_max_cp, {lam0}))\n'
    f'    # Start 2: max tubercle, optimal wavelength\n'
    f'    start_points.append((RPM_HOVER_INIT, np.zeros(N_CP), h_LE_max_cp, {lam0}, 2*h_LE_max_cp))\n'
    f'    # Start 3: both at max\n'
    f'    start_points.append((RPM_HOVER_INIT, h_LE_max_cp.copy(), h_LE_max_cp.copy(), 2*h_LE_max_cp, 2*h_LE_max_cp))\n'
    '    # Remaining: random RPM, amplitudes, wavelengths\n'
    '    for _ in range(4, n_starts):\n'
    '        rpm_i    = float(rng.uniform(5500.0, 8500.0))\n'
    '        h_LE_i   = rng.uniform(0.0, 1.0, N_CP) * h_LE_max_cp\n'
    '        a_tub_i  = rng.uniform(0.0, 1.0, N_CP) * h_LE_max_cp\n'
    '        lam_LE_i = rng.uniform(0.0004, 0.010, N_CP)\n'
    '        lam_ti   = rng.uniform(0.0004, 0.010, N_CP)\n'
    '        start_points.append((rpm_i, h_LE_i, a_tub_i, lam_LE_i, lam_ti))',
)

# 8. worker_args unpack
c = rep(
    '    worker_args = [\n'
    '        (i, rpm_i, h_LE_cp_i, a_tub_cp_i,\n'
    '         thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer, le_type, num_blades)\n'
    '        for i, (rpm_i, h_LE_cp_i, a_tub_cp_i) in enumerate(start_points)\n'
    '    ]',
    '    worker_args = [\n'
    '        (i, rpm_i, h_LE_cp_i, a_tub_cp_i, lam_LE_i, lam_tub_i,\n'
    '         thrust_min_hover, thrust_min_cruise, rho, fd_step, optimizer, le_type, num_blades)\n'
    '        for i, (rpm_i, h_LE_cp_i, a_tub_cp_i, lam_LE_i, lam_tub_i) in enumerate(start_points)\n'
    '    ]',
)

# 9. print_design_vars
c = rep(
    '    a_tub_mm  = _g("A_tub_cp")     * 1000\n'
    '    print(f"  h_s_cp      (mm)  : {np.round(h_s_mm,   3)}")\n'
    '    print(f"  h_LE_cp     (mm)  : {np.round(h_LE_mm,  3)}")\n'
    '    print(f"  lambda_LE   (mm)  : {np.round(lam_LE_mm,3)}")\n'
    '    print(f"  A_tub_cp    (mm)  : {np.round(a_tub_mm, 3)}")\n'
    '    print(f"  lambda_tub  (mm)  : {np.round(lam_tb_mm,3)}")',
    '    a_tub_mm  = _g("A_tub_cp")     * 1000\n'
    '    lam_LE_mm = _g("lambda_LE_cp") * 1000\n'
    '    lam_tb_mm = _g("lambda_tub_cp")* 1000\n'
    '    print(f"  h_s_cp      (mm)  : {np.round(h_s_mm,   3)}")\n'
    '    print(f"  h_LE_cp     (mm)  : {np.round(h_LE_mm,  3)}")\n'
    '    print(f"  lambda_LE   (mm)  : {np.round(lam_LE_mm,3)}")\n'
    '    print(f"  A_tub_cp    (mm)  : {np.round(a_tub_mm, 3)}")\n'
    '    print(f"  lambda_tub  (mm)  : {np.round(lam_tb_mm,3)}")',
) if (
    '    a_tub_mm  = _g("A_tub_cp")     * 1000\n'
    '    print(f"  h_s_cp      (mm)  : {np.round(h_s_mm,   3)}")\n' in c
) else rep(
    '    a_tub_mm  = _g("A_tub_cp") * 1000\n'
    '    print(f"  h_s_cp  (mm)      : {np.round(h_s_mm,  3)}")\n'
    '    print(f"  h_LE_cp (mm)      : {np.round(h_LE_mm, 3)}")\n'
    '    print(f"  A_tub_cp(mm)      : {np.round(a_tub_mm,3)}")',
    '    a_tub_mm  = _g("A_tub_cp")     * 1000\n'
    '    lam_LE_mm = _g("lambda_LE_cp") * 1000\n'
    '    lam_tb_mm = _g("lambda_tub_cp")* 1000\n'
    '    print(f"  h_s_cp      (mm)  : {np.round(h_s_mm,   3)}")\n'
    '    print(f"  h_LE_cp     (mm)  : {np.round(h_LE_mm,  3)}")\n'
    '    print(f"  lambda_LE   (mm)  : {np.round(lam_LE_mm,3)}")\n'
    '    print(f"  A_tub_cp    (mm)  : {np.round(a_tub_mm, 3)}")\n'
    '    print(f"  lambda_tub  (mm)  : {np.round(lam_tb_mm,3)}")',
)

with open('optimization/mdao_problem.py', 'w', encoding='utf-8') as f:
    f.write(c)

print('All changes applied.')
