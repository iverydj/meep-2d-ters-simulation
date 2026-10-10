"""Reference WS2 result: effect of a monolayer WS2 in the 2 nm gap (Au tip / 1L WS2 / Au vs Au tip / Au).

The 0.65 nm layer needs a grid on which it is representable (1.5 pixels/nm: 0.667 nm, one
cell). Both samples keep a 2 nm apex-to-Au distance (WS2: 0.667 nm layer + 1.333 nm air).
The effect is the ratio EF(WS2/Au) / EF(Au/Au) at equal resolution in the small domain;
the FEM (validation/fem_tip_reference.py) shows the ratio is the same in the production
domain. Writes ws2_on_au_fdtd.csv. GPU: ~45 min on one RTX A6000 with gpmeep.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Layer, Model, Numerics, Sample, materials, spectrum, write_csv


@dataclass
class AppConfig:
    resolution: int = 1500
    metal_metal_nm: float = 2.0
    wavelengths_nm: list = field(default_factory=lambda: [520.0, 600.0, 615.0, 633.0, 700.0, 900.0])  # B exciton ~520, A ~615 nm
    run_root: Path = Path("runs") / platform.node() / "reference" / "ws2_on_au"
    out_dir: Path = Path("results") / platform.node() / "reference"


def main(cfg: AppConfig) -> None:
    num = Numerics.small(resolution=cfg.resolution)
    au = spectrum(
        Model(sample=Sample(materials.AU), gap_nm=cfg.metal_metal_nm, numerics=num), cfg.wavelengths_nm, cfg.run_root / "Au"
    )
    t_layer = round(0.65 * cfg.resolution / 1e3) * 1e3 / cfg.resolution  # grid-snapped thickness [nm]
    ws2 = Model(
        sample=Sample(materials.AU, (Layer(materials.WS2_MONOLAYER, 0.65),)), gap_nm=cfg.metal_metal_nm - t_layer, numerics=num
    )
    wr = spectrum(ws2, cfg.wavelengths_nm, cfg.run_root / "WS2-Au")
    rows = []
    for a, w in zip(au, wr):
        rows.append(
            {
                "domain": "small",
                "resolution": cfg.resolution,
                "wavelength_nm": a.wavelength_nm,
                "EF_probe_Au": a.ef_probe,
                "EF_probe_WS2": w.ef_probe,
                "ratio_EF_probe": w.ef_probe / a.ef_probe,
                "EF_gap_Au": a.ef_gap,
                "EF_gap_WS2": w.ef_gap,
                "EF_layer_WS2": w.metrics["EF_layer"],
                "I_probe_tip_over_inc_Au": a.metrics["I_probe_tip_over_inc"],
                "I_probe_tip_over_inc_WS2": w.metrics["I_probe_tip_over_inc"],
            }
        )
        print(
            f"{a.wavelength_nm:.0f} nm: EF Au {a.ef_probe:.1f}, WS2/Au {w.ef_probe:.1f}, ratio {rows[-1]['ratio_EF_probe']:.3f}",
            flush=True,
        )
    print(f"wrote {write_csv(rows, cfg.out_dir / 'ws2_on_au_fdtd.csv')}")


if __name__ == "__main__":
    main(AppConfig())
