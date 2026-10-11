"""Grid convergence of the FDTD for s polarization (E along z), against the FEM.

For s the field in the gap is what is left after the field scattered by the tip nearly
cancels the tip-free field (over gold at the gap centre E_scat ~ -1.06 E_bg), so a small
relative error of the scattered field becomes a several times larger one in |E|^2. This
check measures the resulting FDTD error:
* small domain (Numerics.small), Au tip over Au and over SiO2, 2 nm gap, 633 nm, at 1 to
  3 pixels/nm, against the FEM of the same domain (fem_tip_reference.csv, group small_s);
* production domain, Au, 633 nm, at 1 and 1.5 pixels/nm, against the FEM (group
  spectrum_s). The 1 pixel/nm run is shared with reproduce/spectra_s.py (same run
  directory), so it is not recomputed if that script ran first.
GPU: ~1 h on one RTX A6000 with gpmeep.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, read_csv, simulate, write_csv

DATA = Path(__file__).resolve().parents[1] / "docs" / "figures" / "data"
SAMPLES = {"Au": materials.AU, "SiO2": materials.SIO2}


@dataclass
class AppConfig:
    wavelength_nm: float = 633.0
    small_runs: list = field(
        default_factory=lambda: [("Au", 1000), ("Au", 1500), ("Au", 2000), ("Au", 3000), ("SiO2", 1000), ("SiO2", 2000)]
    )
    production_resolutions: list = field(default_factory=lambda: [1000, 1500])
    baseline_dir: Path = Path("runs") / platform.node() / "reference_s" / "Au_g2nm_R20nm"
    run_root: Path = Path("runs") / platform.node() / "s_convergence"
    out_csv: Path = Path("results") / platform.node() / "s_convergence" / "s_convergence.csv"


def _fem(group: str, sample: str, wavelength_nm: float) -> float:
    fem = read_csv(DATA / "fem_tip_reference.csv")
    keys = zip(fem["EF_probe"], fem["group"], fem["sample"], fem["wavelength_nm"], fem["order"])
    ref = [e for e, g, s, w, o in keys if g == group and s == sample and w == wavelength_nm and o == 4]
    if len(ref) != 1:
        raise ValueError(f"no unique FEM value for {group} {sample} {wavelength_nm} nm")
    return ref[0]


def main(cfg: AppConfig) -> None:
    runs = [("small", s, res) for s, res in cfg.small_runs] + [("production", "Au", res) for res in cfg.production_resolutions]
    rows = []
    for domain, sample, res in runs:
        num = Numerics.small(resolution=res) if domain == "small" else Numerics.production(res)
        model = Model(sample=Sample(SAMPLES[sample]), gap_nm=2.0, numerics=num).with_value("illumination.polarization", "s")
        if domain == "production" and res == 1000:
            out = cfg.baseline_dir
        else:
            out = cfg.run_root / f"{domain}_{sample}_res{res}"
        r = simulate(model, cfg.wavelength_nm, out)
        ef_fem = _fem("small_s" if domain == "small" else "spectrum_s", sample, cfg.wavelength_nm)
        rows.append(
            {
                "domain": domain,
                "sample": sample,
                "resolution": res,
                "dx_nm": 1e3 / res,
                "wavelength_nm": cfg.wavelength_nm,
                "EF_probe": r.ef_probe,
                "I_probe_tip_over_inc": r.metrics["I_probe_tip_over_inc"],
                "EF_FEM": ef_fem,
                "rel_to_FEM": r.ef_probe / ef_fem - 1,
                "timesteps": r.meta["timesteps"],
                "time_stepping_s": r.meta["wall_s"],
                "setup_s": r.meta["init_s"],
                "n_cells": r.meta["layout"]["n_cells"],
            }
        )
        rel = rows[-1]["rel_to_FEM"]
        print(f"{domain:10s} {sample:4s} {res:5d} px/um: EF {r.ef_probe:.5f} vs FEM {ef_fem:.5f} ({rel:+.2%})", flush=True)
    print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
