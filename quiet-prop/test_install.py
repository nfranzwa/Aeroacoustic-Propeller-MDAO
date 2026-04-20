import importlib, sys

packages = [
    ("openmdao", "openmdao"),
    ("ccblade", "ccblade"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("pymoo", "pymoo"),
    ("botorch", "botorch"),
]

all_ok = True
for module, pip_name in packages:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"  OK  {pip_name:<20} {version}")
    except ImportError:
        print(f"  MISSING  {pip_name}")
        all_ok = False

# CadQuery check (heavier import)
try:
    import cadquery as cq
    print(f"  OK  {'cadquery':<20} {cq.__version__}")
except ImportError:
    print(f"  MISSING  cadquery")
    all_ok = False

print()
if all_ok:
    import openmdao.api as om
    prob = om.Problem()
    prob.setup()
    print("OpenMDAO problem setup: OK")
    print("\nAll packages installed and core stack verified.")
else:
    print("Some packages are missing. Run: pip install -r requirements.txt")
    sys.exit(1)
