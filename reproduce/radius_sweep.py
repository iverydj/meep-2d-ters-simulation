"""Reference apex-radius sweep: Au tip over Au, 2 nm gap, 633 nm, R = 10-40 nm.

Writes radius_sweep_fdtd.csv. GPU: ~1 h on one RTX A6000 with gpmeep (4 geometries).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, sweep, write_csv


@dataclass
class AppConfig:
    radii_nm: list = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0])
    wavelength_nm: float = 633.0
    numerics: Numerics = field(default_factory=Numerics.production)
    run_root: Path = Path("runs") / platform.node() / "reference" / "radius_sweep"
    out_dir: Path = Path("results") / platform.node() / "reference"


def main(cfg: AppConfig) -> None:
    model = Model(sample=Sample(materials.AU), gap_nm=2.0, numerics=cfg.numerics)
    rows = []
    for radius, (r,) in sweep(model, "tip.apex_radius_nm", cfg.radii_nm, [cfg.wavelength_nm], cfg.run_root):
        rows.append({"radius_nm": radius, **r.row()})
    print(f"wrote {write_csv(rows, cfg.out_dir / 'radius_sweep_fdtd.csv')}")


if __name__ == "__main__":
    main(AppConfig())
