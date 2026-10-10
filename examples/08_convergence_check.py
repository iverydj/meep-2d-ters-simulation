"""Convergence check for your own case: grid resolution and domain/boundary variants.

Two questions, answered separately (docs/setting_up_your_simulation.md, "Checklist"):

1. Grid: is EF converged with the cell size? Run the same model at several resolutions
   (gap and layers must be whole numbers of cells at every one). The discretization error
   is local to the gap, so this is done in the small domain, where fine grids are cheap,
   and judged by the change between resolutions.
2. Domain: is EF independent of where the boundaries are? Move the absorbers, the drive
   ramp and the free region, strengthen the absorbers and thicken the PML, one at a time,
   at the resolution you use. A good set-up changes EF by <~1 % (reference case: <= 1.0 %).

The defaults run on a CPU in ~20 min (small domain); for the production domain set
``domain_numerics`` to Numerics.production(1000) and use a GPU.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field, replace
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, simulate, write_csv


@dataclass
class AppConfig:
    model: Model = field(default_factory=lambda: Model(sample=Sample(materials.AU), gap_nm=2.0))
    wavelength_nm: float = 633.0
    resolutions: list = field(default_factory=lambda: [500, 1000])  # e.g. [1000, 1500, 2000, 3000] on a GPU
    domain_numerics: Numerics = field(default_factory=lambda: Numerics.small(resolution=500))
    # name -> Numerics overrides (applied to domain_numerics)
    variants: dict = field(
        default_factory=lambda: {
            "baseline": {},
            "moved 0.1 um": {
                "x_half_width_um": 0.5,
                "y_max_um": 0.7,
                "shaft_absorber_start_um": 0.45,
                "side_absorber_start_um": 0.3,
                "side_absorber_top_um": 0.45,
                "drive_ramp_um": (0.45, 0.7),
            },
            "absorber x1.5": {"absorber_conductivity": 60.0},
            "PML x2": {"pml_um": 0.8},
        }
    )
    run_root: Path = Path("runs") / platform.node() / "08_convergence"
    out_dir: Path = Path("results") / platform.node() / "08_convergence"


def main(cfg: AppConfig) -> None:
    rows = []
    for res in cfg.resolutions:
        m = replace(cfg.model, numerics=replace(cfg.domain_numerics, resolution=res))
        r = simulate(m, cfg.wavelength_nm, cfg.run_root / f"resolution_{res}")
        rows.append({"check": "resolution", "setting": f"{res} px/um ({1e3 / res:g} nm cells)", "EF_probe": r.ef_probe})
    ref = rows[-1]["EF_probe"]
    for row in rows:
        row["rel_to_reference"] = row["EF_probe"] / ref - 1
    base = None
    for name, over in cfg.variants.items():
        m = replace(cfg.model, numerics=replace(cfg.domain_numerics, **over))
        r = simulate(m, cfg.wavelength_nm, cfg.run_root / "domain" / name.replace(" ", "_"))
        base = r.ef_probe if base is None else base
        rows.append({"check": "domain", "setting": name, "EF_probe": r.ef_probe, "rel_to_reference": r.ef_probe / base - 1})
    for row in rows:
        print(f"{row['check']:10s} {row['setting']:24s} EF {row['EF_probe']:9.2f}  {row['rel_to_reference']:+.2%}")
    print(f"wrote {write_csv(rows, cfg.out_dir / 'convergence.csv')}")


if __name__ == "__main__":
    main(AppConfig())
