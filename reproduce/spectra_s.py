"""Reference spectra for s-polarized illumination (E along z): Au tip over Au and over SiO2,
2 nm gap, and the 633 nm field map over Au.

Same production set-up as spectra.py (1 nm cells) at a reduced wavelength set. Writes
spectrum_Au_s_fdtd.csv, spectrum_SiO2_s_fdtd.csv and fieldmap_Au_633nm_s.npz (normalized by
the incident beam at its focus, see docs/outputs.md). GPU: ~1.8 h on one RTX A6000 with gpmeep.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Illumination, Model, Numerics, Sample, field_map, materials, spectrum, write_csv

S_SET = [500.0, 550.0, 633.0, 700.0, 800.0, 900.0]


@dataclass
class AppConfig:
    cases: list = field(default_factory=lambda: [("Au", Sample(materials.AU)), ("SiO2", Sample(materials.SIO2))])
    wavelengths_nm: list = field(default_factory=lambda: list(S_SET))
    gap_nm: float = 2.0
    numerics: Numerics = field(default_factory=Numerics.production)
    map_case: str = "Au"
    map_wavelength_nm: float = 633.0
    run_root: Path = Path("runs") / platform.node() / "reference_s"
    out_dir: Path = Path("results") / platform.node() / "reference_s"


def main(cfg: AppConfig) -> None:
    for label, sample in cfg.cases:
        model = Model(sample=sample, gap_nm=cfg.gap_nm, illumination=Illumination(polarization="s"), numerics=cfg.numerics)
        results = spectrum(model, cfg.wavelengths_nm, cfg.run_root / f"{label}_g2nm_R20nm")
        print(f"wrote {write_csv([r.row() for r in results], cfg.out_dir / f'spectrum_{label}_s_fdtd.csv')}")
        if label == cfg.map_case:
            r = next(r for r in results if r.wavelength_nm == cfg.map_wavelength_nm)
            out = cfg.out_dir / f"fieldmap_{label}_{cfg.map_wavelength_nm:.0f}nm_s.npz"
            field_map(r, normalization="incident").save_npz(out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main(AppConfig())
