"""Grid convergence of the tip model in the small domain (where fine grids are affordable).

The discretization error is local to the gap, so it is measured in the small domain
(Numerics.small: free region |x| <= 0.4 um, y <= 0.6 um). Absolute values differ from the
production domain (shorter illuminated shaft); the ratios between resolutions are what
this check measures:
* Au tip / Au, 2 nm gap, 633 nm: 1, 1.5, 2 and 3 pixels/nm (2, 3, 4 and 6 cells in the gap);
* the same with a 1 nm gap, at 525 nm, and over SiO2, at 1 and 2 pixels/nm;
* WS2/Au (2 nm apex-Au distance) at 633 nm at 1.5 and 3 pixels/nm (the 0.65 nm monolayer
  is 1 and 2 cells thick).
GPU recommended (~1.2 h on an RTX A6000 with gpmeep for the whole set).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Layer, Model, Numerics, Sample, materials, simulate, write_csv


def _runs():
    au, sio2 = Sample(materials.AU), Sample(materials.SIO2)
    ws2 = Sample(materials.AU, (Layer(materials.WS2_MONOLAYER, 0.65),))
    return [
        *[("tip_Au_g2nm", au, 2.0, r, 633.0) for r in (1000, 1500, 2000, 3000)],
        *[("tip_Au_g1nm", au, 1.0, r, 633.0) for r in (1000, 2000)],
        *[("tip_Au_g2nm", au, 2.0, r, 525.0) for r in (1000, 2000)],
        *[("tip_SiO2_g2nm", sio2, 2.0, r, 633.0) for r in (1000, 2000)],
        *[("tip_WS2-Au_g2nm", ws2, 2.0 - 2.0 / 3.0, r, 633.0) for r in (1500, 3000)],
    ]


@dataclass
class AppConfig:
    runs: list = field(default_factory=_runs)  # (case, sample, air gap nm, resolution, wavelength nm)
    run_root: Path = Path("runs") / platform.node() / "resolution_series"
    out_csv: Path = Path("results") / platform.node() / "resolution_series" / "resolution_series.csv"


def main(cfg: AppConfig) -> None:
    rows = []
    for case, sample, gap, res, wl in cfg.runs:
        model = Model(sample=sample, gap_nm=gap, numerics=Numerics.small(resolution=res))
        r = simulate(model, wl, cfg.run_root / f"res{res}" / case)
        metal_gap = gap + sum(1e3 * (t - b) for b, t in r.meta["layout"]["layers"])
        rows.append(
            {
                "case": f"{case}_R20nm",
                "gap_nm": round(metal_gap, 6),
                "resolution": res,
                "dx_nm": 1e3 / res,
                "gap_cells": round(metal_gap * res / 1e3, 6),  # cells between apex and substrate
                **r.row(),
            }
        )
        print(f"{case} res={res} {wl:.0f} nm: EF_probe={r.ef_probe:.2f}", flush=True)
    print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
