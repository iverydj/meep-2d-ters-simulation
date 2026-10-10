"""Reference near-field maps at 633 nm (Au and SiO2 substrates) from the runs of spectra.py
(no new simulation). Writes fieldmap_Au_633nm.npz and fieldmap_SiO2_633nm.npz."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import field_map, load_result
from ters2d.run import output_file


@dataclass
class AppConfig:
    run_root: Path = Path("runs") / platform.node() / "reference"
    cases: list = field(default_factory=lambda: ["Au", "SiO2"])
    wavelength_nm: float = 633.0
    out_dir: Path = Path("results") / platform.node() / "reference"


def main(cfg: AppConfig) -> None:
    for label in cfg.cases:
        r = load_result(output_file(cfg.run_root / f"{label}_g2nm_R20nm", cfg.wavelength_nm))
        out = cfg.out_dir / f"fieldmap_{label}_{cfg.wavelength_nm:.0f}nm.npz"
        field_map(r).save_npz(out)
        print(f"wrote {out} (EF at the gap centre {r.ef_probe:.1f})")


if __name__ == "__main__":
    main(AppConfig())
