"""FP32 (gpmeep, GPU) vs FP64 (upstream Meep, CPU) on the production Au/Au case at 633 nm.

Run the same script once with gpmeep on the GPU and once with upstream Meep:
    MEEP_GPU_BACKEND=cuda python validation/fp32_vs_fp64.py
    mpirun -np 32 python validation/fp32_vs_fp64.py
Each run writes a CSV row tagged with its backend (and the wall times), in the columns of
docs/figures/data/fp32_vs_fp64.csv (where the two rows were merged and given descriptive labels).
With MPI the default chunk layout is Meep's (Numerics.chunk_layout = "auto").
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Sample, materials, simulate, write_csv


@dataclass
class AppConfig:
    model: Model = field(default_factory=lambda: Model(sample=Sample(materials.AU)))
    wavelength_nm: float = 633.0
    run_root: Path = Path("runs") / platform.node() / "fp32_vs_fp64"
    out_dir: Path = Path("results") / platform.node() / "fp32_vs_fp64"


def main(cfg: AppConfig) -> None:
    import meep as mp

    gpu = getattr(mp, "gpu", None)
    tag = f"fp32_{gpu.requested_backend}" if gpu is not None else f"fp64_cpu{mp.count_processors()}"
    r = simulate(cfg.model, cfg.wavelength_nm, cfg.run_root / tag)
    row = {
        "run": tag,
        "meep_version": mp.__version__,
        "processes": mp.count_processors(),
        **{k: r.row()[k] for k in ("wavelength_nm", "EF_probe", "EF_gap", "I_probe_tip_over_inc")},
        "n_cells": r.meta["layout"]["n_cells"],
        "timesteps": r.meta["timesteps"],
        "time_stepping_s": r.meta["wall_s"],
        "setup_s": r.meta["init_s"],
    }
    if mp.am_master():
        print(row)
        print(f"wrote {write_csv([row], cfg.out_dir / f'{tag}.csv')}")


if __name__ == "__main__":
    main(AppConfig())
