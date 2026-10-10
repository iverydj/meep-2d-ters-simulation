"""Enhancement spectrum of one geometry: one simulation per wavelength, structure built once.

The beam is synthesized at every wavelength, so the incidence angle is exact at every
point of the spectrum. Writes a CSV with all outputs and a plot. With the default coarse
small-domain grid this takes ~2-3 min per wavelength on one CPU core; for the reference
set-up use Numerics.production(1000) (GPU: ~15 min for the first wavelength, ~7 min for
each further one on an RTX A6000 with gpmeep). An interrupted run resumes from the
files already written.
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
        default_factory=lambda: Model(sample=Sample(materials.AU), gap_nm=2.0, numerics=Numerics.small(resolution=500))
    )
    wavelengths_nm: list = field(default_factory=lambda: [550.0, 600.0, 633.0, 700.0, 800.0])
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
