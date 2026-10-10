"""Reference gap sweep: Au tip over Au at 633 nm, gaps 1-10 nm (1 nm cells; the 1 nm gap is one cell).

Writes gap_sweep_fdtd.csv. GPU: ~1.6 h on one RTX A6000 with gpmeep (6 geometries).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, sweep, write_csv


@dataclass
class AppConfig:
    gaps_nm: list = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    wavelength_nm: float = 633.0
    numerics: Numerics = field(default_factory=Numerics.production)
    run_root: Path = Path("runs") / platform.node() / "reference" / "gap_sweep"
    out_dir: Path = Path("results") / platform.node() / "reference"


def main(cfg: AppConfig) -> None:
    model = Model(sample=Sample(materials.AU), numerics=cfg.numerics)
    rows = []
    for gap, (r,) in sweep(model, "gap_nm", cfg.gaps_nm, [cfg.wavelength_nm], cfg.run_root):
        rows.append({"sample": "Au", "gap_nm": gap, "air_gap_nm": gap, **r.row()})
    print(f"wrote {write_csv(rows, cfg.out_dir / 'gap_sweep_fdtd.csv')}")


if __name__ == "__main__":
    main(AppConfig())
