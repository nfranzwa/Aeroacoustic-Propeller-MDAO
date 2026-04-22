# quiet-prop — Aeroacoustic Propeller MDAO

Multidisciplinary design optimisation (MDAO) of a 3-blade UAV propeller for minimum A-weighted noise, subject to thrust, structural, and geometric constraints. Baseline is the APC 7×5E 3-blade propeller on a 7-inch 4-rotor UAV (928 g AUW).

## Result

| | Baseline | Optimum |
|---|---|---|
| SPL weighted | 73.45 dBA | **69.62 dBA** |
| SPL hover | 73.45 dBA | 69.54 dBA |
| SPL cruise | 73.57 dBA | 69.81 dBA |
| RPM | 7000 | 6056 |
| Thrust hover | — | 2.42 N (≥ 2.28 N ✓) |
| Thrust cruise | — | 2.33 N (≥ 2.33 N ✓) |
| Max stress | — | 9.25 MPa (≤ 22 MPa ✓) |
| Min wall thickness | — | 0.94 mm (≥ 0.5 mm ✓) |

**−3.83 dBA reduction.** Dominant mechanism: Amiet leading-edge turbulence interaction (LETI), reduced by narrowing chord (−0.05R inner span) and lowering RPM. BVI tonal contributes an additional −2.4 dBA.

![Optimum geometry](results/plots/optimum_geometry.png)
![Noise breakdown](results/plots/noise_breakdown.png)

## Noise model

| Mechanism | Implementation | Dominant? |
|---|---|---|
| TBL-TE | BPM 1989 turbulent trailing edge | No (blade is mostly laminar at UAV Re) |
| LBL-VS | BPM 1989 laminar vortex shedding | Minor |
| Amiet LETI | Amiet 1975 leading-edge turbulence + swept-LE cos⁴(Λ) | **Yes — ~73 dBA** |
| BVI tonal | Widnall/Leishman parametric (Γ_tip, miss distance) | Secondary — ~64 dBA |

## Design variables (21 total)

| Variable | Count | Bounds | Description |
|---|---|---|---|
| `rpm` | 1 | [3500, 10000] RPM | Rotor speed |
| `delta_twist_cp` | 5 | [−5, +5] deg | Twist perturbation B-spline CPs |
| `delta_chord_cp` | 5 | [−0.05, +0.03] R | Chord perturbation B-spline CPs |
| `sweep_cp` | 5 | [0, 0.12] R | Aft-sweep B-spline CPs |
| `delta_tc_cp` | 5 | [−0.03, +0.04] | t/c perturbation B-spline CPs |

Each array of 5 control points is evaluated at the 18 blade definition stations via `CubicSpline`, guaranteeing C2-continuous (kink-free) chord, twist, sweep, and t/c distributions.

## Constraints

| Constraint | Bound | Rationale |
|---|---|---|
| `thrust_hover` | ≥ 2.28 N | W/4 at hover (928 g AUW) |
| `thrust_cruise` | ≥ 2.33 N | Forward flight at 15 m/s (12.8° pitch) |
| `max_stress` | ≤ 22 MPa | Siraya Blu Tough (UTS 50 MPa, FoS 3.5 fatigue) |
| `phys_thick` (inner span) | ≥ 0.5 mm | Minimum 3D-print wall thickness |
| `sweep_cp_diff` | ≥ 0 | Monotone non-decreasing sweep (no sawtooth) |
| `twist_cp_diff` | ≥ 0 | Monotone wash-out at control points |

## Repository structure

```
quiet-prop/
├── acoustics/
│   └── bpm_component.py       BPM + Amiet LETI + BVI tonal OpenMDAO component
├── aerodynamics/
│   └── ccblade_component.py   CCBlade BEM OpenMDAO component + Michel transition
├── geometry/
│   ├── blade_generator.py     BladeGeometry class, APC 7×5E baseline definition
│   ├── blade_importer.py      Import blade geometry from CSV/APC data files
│   └── blade_stl_exporter.py  CadQuery STL export
├── structures/
│   └── structural_component.py Centrifugal + bending stress, wall thickness
├── optimization/
│   └── mdao_problem.py        OpenMDAO problem, multistart SLSQP driver
├── results/
│   ├── noise_breakdown.py     Post-process: per-mechanism SPL breakdown + plots
│   ├── plots/
│   │   ├── geometry_viz.py    Blade geometry visualisation (6-panel)
│   │   ├── optimum_geometry.png   Current best blade vs baseline
│   │   ├── noise_breakdown.png    Mechanism breakdown bar chart + 1/3-octave spectrum
│   │   └── start_*_*.png      Per-start geometry from last multistart run
│   └── stl/
│       ├── blade_baseline_single.stl
│       ├── rotor_APC_7x5E_3blade.stl
│       └── rotor_baseline_equal_sections.csv
├── tests/
│   └── test_baseline.py       Smoke test: baseline analysis sanity checks
├── test_install.py            Dependency verification script
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
pip install -r requirements.txt

# Verify installation
python test_install.py

# Run 8-start multistart optimisation (plots geometry after each feasible start)
python optimization/mdao_problem.py --starts 8 --plot-starts

# Resume from a specific start (e.g. after a crash)
python optimization/mdao_problem.py --starts 8 --start-from 4 --plot-starts

# Noise mechanism breakdown for a given optimum
python results/noise_breakdown.py \
  --rpm 6056 \
  --dtwist -1.455 2.454 2.486 5.0 2.557 \
  --dchord -0.05  -0.05  -0.05  -0.05  0.0142 \
  --sweep   0.0    0.0    0.0    0.0    0.1196 \
  --dtc     0.014  0.0269 -0.0176 0.0271 0.0327
```

## Physical basis

**Drone sizing (7-inch 4-rotor, 928 g AUW)**
- Motor: iFlight XING-E 2814 900KV on 4S (14.8V nominal)
- Hover RPM estimate: ~7000 RPM at 53% throttle
- Cruise: 15 m/s forward flight, pitched at 12.8° (arctan of drag/weight)
- Cruise axial inflow: V_axial = 15 × sin(12.8°) = 3.32 m/s

**Acoustic weighting**: 0.7 × SPL_hover + 0.3 × SPL_cruise (hover-dominant mission)

**Structural material**: Siraya Tech Blu Tough resin (UTS = 50 MPa, FoS = 3.5 fatigue → allowable 14.3 MPa; note 22 MPa used in constraint from UTS/2.3 with additional margin)

**Optimizer**: SLSQP via `scipy.optimize` through OpenMDAO `ScipyOptimizeDriver`. Finite-difference Jacobians at step 3×10⁻⁴. 8-start multistart with seed-reproducible random restarts.

## Known constraints and limitations

- TBL-TE is near-zero (~−188 dBA) because Michel's criterion predicts mostly laminar flow at UAV Reynolds numbers (Re_c ~ 10⁴–10⁵). This is physically consistent but means TBL-TE is not a useful optimisation lever at this scale.
- The chord lower bound (−0.05R) is active across most of the span in the current optimum — the 0.5 mm print floor is the binding constraint. A finer-resolution printer would allow thinner blades and potentially another 1–2 dBA reduction.
- The twist upper bound (+5 deg) is active at control point 3 in the current optimum — relaxing to +8 deg may unlock additional wash-out benefit.
