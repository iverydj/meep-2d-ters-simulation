"""Gap dependence of the enhancement at one wavelength.

The gap (apex to sample surface) must be a whole number of grid cells: multiples of 2 nm
on the default 2 nm grid, of 1 nm at resolution 1000. Each gap is its own geometry and run
directory. Use at least two cells across the gap for quantitative values (see
docs/validation.md: one cell underestimates EF by ~10-16 %).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, sweep, write_csv
from ters2d.plotting import plot_sweep


@dataclass
class AppConfig:
    model: Model = field(default_factory=lambda: Model(sample=Sample(materials.AU), numerics=Numerics.small(resolution=500)))
    gaps_nm: list = field(default_factory=lambda: [2.0, 4.0, 6.0, 10.0])
    wavelength_nm: float = 633.0
    run_root: Path = Path("runs") / platform.node() / "03_gap_sweep"
    out_dir: Path = Path("results") / platform.node() / "03_gap_sweep"


def main(cfg: AppConfig) -> None:
    rows = []
    for gap, (r,) in sweep(cfg.model, "gap_nm", cfg.gaps_nm, [cfg.wavelength_nm], cfg.run_root):
        rows.append({"gap_nm": gap, **r.row()})
        print(f"gap {gap:5.1f} nm  EF {r.ef_probe:8.1f}")
    csv = write_csv(rows, cfg.out_dir / "gap_sweep.csv")
    cells = cfg.model.numerics.dx_nm
    print(f"wrote {plot_sweep(csv, 'gap_nm', 'gap (nm)', cfg.out_dir / 'gap_sweep.png', flag_below=1.5 * cells, logx=True)}")


if __name__ == "__main__":
    main(AppConfig())
