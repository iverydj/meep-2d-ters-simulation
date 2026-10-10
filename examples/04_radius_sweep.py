"""Apex-radius dependence of the enhancement (any nested field can be swept the same way,
e.g. "tip.half_angle_deg" or "illumination.incidence_deg").

Larger radii widen the shaft: the code checks that it stays clear of the side PML and of
the side plasmon absorbers and raises an error otherwise. Default: the production domain on a
coarse 2 nm grid (the 2 nm gap is one cell), ~11 min per value with mpirun -np 8 on a CPU
(~2-4 min on a GPU).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, sweep, write_csv
from ters2d.plotting import plot_sweep


@dataclass
class AppConfig:
    model: Model = field(
        default_factory=lambda: Model(sample=Sample(materials.AU), gap_nm=2.0, numerics=Numerics.production(resolution=500))
    )
    parameter: str = "tip.apex_radius_nm"
    values: list = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0])
    wavelength_nm: float = 633.0
    run_root: Path = Path("runs") / platform.node() / "04_radius_sweep"
    out_dir: Path = Path("results") / platform.node() / "04_radius_sweep"


def main(cfg: AppConfig) -> None:
    key = cfg.parameter.split(".")[-1]
    rows = []
    for v, (r,) in sweep(cfg.model, cfg.parameter, cfg.values, [cfg.wavelength_nm], cfg.run_root):
        rows.append({key: v, **r.row()})
        print(f"{cfg.parameter} = {v:g}: EF {r.ef_probe:.1f}")
    csv = write_csv(rows, cfg.out_dir / "sweep.csv")
    print(f"wrote {plot_sweep(csv, key, cfg.parameter, cfg.out_dir / 'sweep.png', logy=False)}")


if __name__ == "__main__":
    main(AppConfig())
