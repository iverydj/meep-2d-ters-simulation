"""Reference spectra: Au tip over Au (gap mode) and over SiO2 (no gap mode), 2 nm gap.

Production set-up (Numerics.production(1000): 1 nm cells, ~27-29 million cells). One
simulation per wavelength, structure built once per sample. Writes
spectrum_Au_fdtd.csv and spectrum_SiO2_fdtd.csv (the tables in docs/figures/data).
GPU: ~1.7 h (Au, 13 wavelengths) + ~1 h (SiO2, 7 wavelengths) on one RTX A6000 with gpmeep.
The run directories are reused by field_maps.py.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d import Model, Numerics, Sample, materials, spectrum, write_csv

AU_SET = sorted({float(w) for w in np.arange(500.0, 901.0, 50.0)} | {525.0, 575.0, 633.0})
SIO2_SET = sorted({float(w) for w in np.arange(500.0, 901.0, 100.0)} | {550.0, 633.0})


@dataclass
class AppConfig:
    cases: list = field(
        default_factory=lambda: [("Au", Sample(materials.AU), AU_SET), ("SiO2", Sample(materials.SIO2), SIO2_SET)]
    )
    gap_nm: float = 2.0
    numerics: Numerics = field(default_factory=Numerics.production)
    run_root: Path = Path("runs") / platform.node() / "reference"
    out_dir: Path = Path("results") / platform.node() / "reference"


def main(cfg: AppConfig) -> None:
    for label, sample, wls in cfg.cases:
        model = Model(sample=sample, gap_nm=cfg.gap_nm, numerics=cfg.numerics)
        rows = [r.row() for r in spectrum(model, wls, cfg.run_root / f"{label}_g2nm_R20nm")]
        print(f"wrote {write_csv(rows, cfg.out_dir / f'spectrum_{label}_fdtd.csv')}")


if __name__ == "__main__":
    main(AppConfig())
