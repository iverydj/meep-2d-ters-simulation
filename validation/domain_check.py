"""Sensitivity of the production results to the domain, boundaries, absorbers and illuminated shaft.

Variants of the production domain, one factor at a time, at the production resolution:
  baseline         Numerics.production()
  taller           free region, absorbers and drive ramp moved up/out by 0.5 um / 0.2 um
                   (illuminated shaft length and absorber position)
  absorber_strong  absorber conductivity 1.5x (the absorbers' own reflection)
  thick_pml        PML twice as thick
GPU recommended (~1.5 h on an RTX A6000 with gpmeep). The FEM reference checks the same
design with more variants (fem_tip_reference.py, check_* groups).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, spectrum, write_csv

VARIANTS = {
    "baseline": {},
    "taller": dict(
        x_half_width_um=2.0,
        y_max_um=6.0,
        shaft_absorber_start_um=4.0,
        side_absorber_start_um=1.0,
        side_absorber_top_um=2.6,
        drive_ramp_um=(4.0, 5.0),
    ),
    "absorber_strong": {"absorber_conductivity": 15.0},
    "thick_pml": {"pml_um": 0.8},
}


@dataclass
class AppConfig:
    # (sample label, wavelengths, variants)
    plan: list = field(
        default_factory=lambda: [
            ("Au", [633.0, 700.0, 900.0], ["baseline"]),
            ("Au", [633.0, 900.0], ["taller"]),
            ("Au", [900.0], ["absorber_strong", "thick_pml"]),
            ("SiO2", [900.0], ["baseline", "taller"]),
        ]
    )
    resolution: int = 1000
    run_root: Path = Path("runs") / platform.node() / "domain_check"
    out_csv: Path = Path("results") / platform.node() / "domain_check" / "domain_check.csv"


def main(cfg: AppConfig) -> None:
    samples = {"Au": Sample(materials.AU), "SiO2": Sample(materials.SIO2)}
    rows = []
    for label, wls, names in cfg.plan:
        for name in names:
            model = Model(sample=samples[label], numerics=Numerics.production(cfg.resolution, **VARIANTS[name]))
            for r in spectrum(model, wls, cfg.run_root / f"{label}_{name}"):
                rows.append({"sample": label, "variant": name, **r.row()})
                print(f"{label} {name} {r.wavelength_nm:.0f} nm: EF_probe={r.ef_probe:.2f}", flush=True)
    print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
