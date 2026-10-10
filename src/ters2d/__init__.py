"""ters2d: 2D FDTD simulation of gap-mode TERS near fields with Meep (or gpmeep).

Typical use::

    from ters2d import Model, Tip, Sample, Layer, Illumination, Numerics, materials, simulate

    model = Model(
        tip=Tip(apex_radius_nm=20, half_angle_deg=15),
        sample=Sample(materials.AU),
        gap_nm=2.0,
        illumination=Illumination(incidence_deg=60, divergence_deg=14),
        numerics=Numerics.small(resolution=500),
    )
    result = simulate(model, 633.0, "runs/demo")
    print(result.ef_probe)

Only the modules ``run`` and ``meep_model`` import Meep; everything else (model, layout,
analytic fields, materials, fitting, results, plotting) is pure Python.
"""

from . import materials
from ._version import __version__
from .materials import Material, Pole, UniaxialMaterial
from .model import Illumination, Layer, Model, Numerics, Outputs, Sample, Tip
from .results import FieldMap, Result, field_map, load_result, read_csv, write_csv
from .run import simulate, spectrum, sweep

__all__ = [
    "__version__",
    "FieldMap",
    "Illumination",
    "Layer",
    "Material",
    "Model",
    "Numerics",
    "Outputs",
    "Pole",
    "Result",
    "Sample",
    "Tip",
    "UniaxialMaterial",
    "field_map",
    "load_result",
    "materials",
    "read_csv",
    "simulate",
    "spectrum",
    "sweep",
    "write_csv",
]
