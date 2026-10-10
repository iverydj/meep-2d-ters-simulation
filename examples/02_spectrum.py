"""Enhancement spectrum of one geometry: one simulation per wavelength, structure built once.

The beam is synthesized at every wavelength, so the incidence angle is exact at every
point of the spectrum. Writes a CSV with all outputs and a plot.

Default: the production domain on a coarse 2 nm grid (6.7 million cells; the 2 nm gap is one
cell). This is a template, not a result: EF is 23 % (633 nm) and 18 % (900 nm) below the FEM
and EF(900)/EF(633) is 1.57 instead of 1.47 (docs/figures/data/coarse_checks.csv), but the
spectrum rises to the red as in the full model, which the small domain does not reproduce.
Cost per wavelength: ~11 min with mpirun -np 8 on a CPU, ~2-4 min on a GPU with gpmeep.
Numerics.production(1000) is the reference set-up (GPU: ~15 min for the first wavelength,
~7 min for each further one on an RTX A6000).
An interrupted run resumes from the files already written.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, spectrum, write_csv
from ters2d.plotting import plot_spectra


@dataclass
class AppConfig:
    model: Model = field(
        default_factory=lambda: Model(sample=Sample(materials.AU), gap_nm=2.0, numerics=Numerics.production(resolution=500))
    )
    wavelengths_nm: list = field(default_factory=lambda: [550.0, 633.0, 750.0, 900.0])
    run_dir: Path = Path("runs") / platform.node() / "02_spectrum"
    out_dir: Path = Path("results") / platform.node() / "02_spectrum"


def main(cfg: AppConfig) -> None:
    results = spectrum(cfg.model, cfg.wavelengths_nm, cfg.run_dir)
    csv = write_csv([r.row() for r in results], cfg.out_dir / "spectrum.csv")
    for r in results:
        print(f"{r.wavelength_nm:6.1f} nm  EF {r.ef_probe:8.1f}  EF_gap {r.ef_gap:8.1f}")
    print(f"wrote {csv}")
    print(f"wrote {plot_spectra([(cfg.model.sample.label, csv, '')], cfg.out_dir / 'spectrum.png')}")


if __name__ == "__main__":
    main(AppConfig())
