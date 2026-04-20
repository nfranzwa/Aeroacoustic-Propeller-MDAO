# Aeroacoustic Propeller MDAO

Multidisciplinary design optimization of a UAV propeller for minimum noise subject to a thrust constraint.

**Baseline:** HQProp 7x4x3 (7-inch, 3-blade, NACA 4412) at 5000 RPM, hover.

---

## Blade Geometry — Baseline vs Phase 1 Optimized

![Blade Geometry](quiet-prop/results/plots/blade_geometry.png)

---

## Phase 1 Results (Geometry Optimization)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| SPL total (dBA) | 28.73 | 21.22 | **−7.5 dBA** |
| SPL broadband (dB) | 33.64 | 27.15 | −6.5 dB |
| SPL tonal (dB) | 14.03 | 13.07 | −1.0 dB |
| Thrust (N) | 1.122 | 1.004 | −0.12 N |
| Power (W) | 4.49 | 3.74 | −0.75 W |
| CT | 0.1319 | 0.1182 | −10% |
| CP | 0.0356 | 0.0297 | −17% |

**Optimizer:** SLSQP — 37 design variables (RPM + 18× twist + 18× chord), thrust ≥ 1.0 N constraint.

Key design changes:
- **Twist:** reduced by up to −5° at mid-span (reduces angle of attack → lower TBL-TE broadband)
- **Chord:** increased by +0.03R across most of the span (maintains thrust at lower AoA)
- **RPM:** unchanged at 5000 (BPF tonal is Bessel-function suppressed for this small, low-Mach prop)

---

## Stack

```
quiet-prop/
  geometry/        blade_generator.py     — HQProp 7x4x3 parametric geometry
  aerodynamics/    ccblade_component.py   — BEM solver (static + forward flight)
  acoustics/       bpm_component.py       — BPM broadband + Gutin tonal
  optimization/    mdao_problem.py        — OpenMDAO MDAO problem (SLSQP)
  tests/           test_baseline.py       — 4/4 validation tests pass
  results/plots/   blade_geometry.png     — geometry figure
```

**Physics models:**
- BEM: two-regime (hover and forward flight), Prandtl tip/hub loss, NACA 4412 polar
- Broadband: Brooks-Pope-Marcolini (1989) TBL-TE self-noise, Schlichting BL thickness
- Tonal: Garrick-Watkins Gutin with Bessel function J_{mB}(x) correction

---

## Quick Start

```bash
# Install dependencies
pip install openmdao scipy numpy matplotlib

# Run validation tests
cd quiet-prop
python tests/test_baseline.py

# Run Phase 1 optimizer
python optimization/mdao_problem.py

# Regenerate geometry plot
python results/plots/geometry_viz.py
```

---

## Optimization Roadmap

| Phase | Variables | Models | Status |
|-------|-----------|--------|--------|
| 1 — Geometry | RPM, twist, chord | BEM + BPM | **Done** |
| 2 — LE/TE shaping | LE radius, TE angle, sweep | BEM + BPM + LBL-VS | Planned |
| 3 — Structural | Spar caps, skin thickness | FEM + BEM + BPM | Planned |
| 4 — CFD-informed | All + CFD surrogates | RANS + BPM | Planned |
