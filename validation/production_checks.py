"""Two checks of the production set-up itself (Au/Au, 2 nm gap, 633 nm):

* resolution in the production domain: 1 and 1.5 pixels/nm (60 million cells), compared with
  the FEM value of the same model; the small-domain series only bounds this error from below;
* the stop rule: a run stops when the DFT fields have settled, but not before the light has
  crossed the free region once (~6.7 um/c). A run with a minimum of 14 um/c after the source
  also contains the echoes of the shaft plasmons from the top boundary region (round trip
  ~13 um/c), so the difference measures what the stop rule leaves out.

GPU: ~1 h on one RTX A6000 with gpmeep. The 1 pixel/nm baseline is shared with
regression_check.py (same run directory), so it is not recomputed if that script ran first.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, read_csv, simulate, write_csv

DATA = Path(__file__).resolve().parents[1] / "docs" / "figures" / "data"


@dataclass
class AppConfig:
    wavelength_nm: float = 633.0
    resolutions: list = field(default_factory=lambda: [1000, 1500])
    long_min_time: float = 14.0  # um / c after the end of the source
    baseline_dir: Path = Path("runs") / platform.node() / "regression_check" / "production_Au"
    run_root: Path = Path("runs") / platform.node() / "production_checks"
    out_csv: Path = Path("results") / platform.node() / "production_checks" / "production_checks.csv"


def main(cfg: AppConfig) -> None:
    fem = read_csv(DATA / "fem_tip_reference.csv")
    keys = zip(fem["EF_probe"], fem["group"], fem["sample"], fem["wavelength_nm"])
    ref = [e for e, g, s, w in keys if g == "spectrum" and s == "Au" and w == cfg.wavelength_nm]
    if len(ref) != 1:
        raise ValueError("no unique FEM reference value")
    rows = []
    for res in cfg.resolutions:
        out = cfg.baseline_dir if res == 1000 else cfg.run_root / f"res{res}"
        r = simulate(Model(numerics=Numerics.production(res)), cfg.wavelength_nm, out)
        rows.append(
            {
                "check": "resolution",
                "setting": f"{res} px/um",
                "EF_probe": r.ef_probe,
                "EF_FEM": ref[0],
                "rel_to_FEM": r.ef_probe / ref[0] - 1,
                "timesteps": r.meta["timesteps"],
                "time_stepping_s": r.meta["wall_s"],
                "setup_s": r.meta["init_s"],
                "n_cells": r.meta["layout"]["n_cells"],
            }
        )
    r = simulate(
        Model(numerics=Numerics.production(1000, min_time_after_sources=cfg.long_min_time)),
        cfg.wavelength_nm,
        cfg.run_root / "long_min_time",
    )
    rows.append(
        {
            "check": "stop rule",
            "setting": f"min. {cfg.long_min_time:g} um/c after the source",
            "EF_probe": r.ef_probe,
            "EF_FEM": ref[0],
            "rel_to_FEM": r.ef_probe / ref[0] - 1,
            "timesteps": r.meta["timesteps"],
            "time_stepping_s": r.meta["wall_s"],
            "setup_s": r.meta["init_s"],
            "n_cells": r.meta["layout"]["n_cells"],
        }
    )
    for row in rows:
        print(f"{row['check']:10s} {row['setting']:32s} EF {row['EF_probe']:.4f}  vs FEM {row['rel_to_FEM']:+.2%}", flush=True)
    print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
