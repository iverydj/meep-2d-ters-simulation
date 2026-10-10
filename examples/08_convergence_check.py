"""Convergence check for your own case: resolution, domain/boundaries, and the converged reference.

Three questions, answered separately (docs/setting_up_your_simulation.md, "Checklist"):

1. Resolution: how does EF change with the cell size? The same model at several resolutions
   (the gap and the layers must be whole numbers of cells at every one). Run in the small
   domain, where fine grids are cheap, this gives the trend and a LOWER bound of the
   discretization error of a production run (validation.md: the small-domain series suggests
   ~-5 % at 1 nm cells, the production FDTD is -7.7 % below the FEM).
2. Domain: is EF independent of where the boundaries are? One variant at a time, relative to
   the domain you use for your numbers (here the production domain): free region, absorbers
   and drive ramp moved up/out; drive ramp alone moved up; absorbers 1.5x stronger; PML 2x
   thicker. A good set-up changes EF by <~1 % (reference case: <= 1.0 %). Do not run this in
   the small domain: it fails by design (its drive ramp cuts an illuminated shaft).
3. Converged value (optional, fem=True, needs NGSolve): the FEM solves the same model without
   staircasing, in about a minute on a CPU; the FDTD deviation from it is the discretization
   error of your run.

Defaults: resolution part in the small domain at 2 and 1 nm cells (~15 min on one CPU core);
domain part in the production domain at 2 nm cells (5 runs, ~11 min each with mpirun -np 8 on
a CPU, ~2-4 min each on a GPU with gpmeep).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field, replace
from pathlib import Path

from ters2d import Model, Numerics, Sample, materials, simulate, write_csv


def domain_variants(num: Numerics, shift_um: float, absorber_factor: float, pml_factor: float) -> dict:
    """One variant at a time, relative to ``num``."""
    r0, r1 = num.drive_ramp_um
    return {
        "baseline": num,
        f"free region, absorbers, ramp +{shift_um:g} um": replace(
            num,
            y_max_um=num.y_max_um + shift_um,
            shaft_absorber_start_um=num.shaft_absorber_start_um + shift_um,
            drive_ramp_um=(r0 + shift_um, r1 + shift_um),
            x_half_width_um=num.x_half_width_um + 0.4 * shift_um,
            side_absorber_start_um=num.side_absorber_start_um + 0.4 * shift_um,
            side_absorber_top_um=num.side_absorber_top_um + 0.6 * shift_um,
        ),
        f"drive ramp +{shift_um:g} um": replace(num, drive_ramp_um=(r0 + shift_um, min(r1 + shift_um, num.y_max_um))),
        f"absorber conductivity x{absorber_factor:g}": replace(
            num, absorber_conductivity=num.absorber_conductivity * absorber_factor
        ),
        f"PML x{pml_factor:g}": replace(num, pml_um=num.pml_um * pml_factor),
    }


@dataclass
class AppConfig:
    model: Model = field(default_factory=lambda: Model(sample=Sample(materials.AU), gap_nm=2.0))
    wavelength_nm: float = 633.0
    resolutions: list = field(default_factory=lambda: [500, 1000])  # e.g. [1000, 1500, 2000, 3000] on a GPU
    resolution_numerics: Numerics = field(default_factory=lambda: Numerics.small(resolution=500))
    domain_numerics: Numerics = field(default_factory=lambda: Numerics.production(resolution=500))
    shift_um: float = 0.5
    absorber_factor: float = 1.5
    pml_factor: float = 2.0
    fem: bool = False
    run_root: Path = Path("runs") / platform.node() / "08_convergence"
    out_dir: Path = Path("results") / platform.node() / "08_convergence"


def main(cfg: AppConfig) -> None:
    rows = []
    res_rows = []
    for res in cfg.resolutions:
        m = replace(cfg.model, numerics=replace(cfg.resolution_numerics, resolution=res))
        r = simulate(m, cfg.wavelength_nm, cfg.run_root / "resolution" / f"res{res}")
        res_rows.append(
            {"check": "resolution (small domain)", "setting": f"{res} px/um ({1e3 / res:g} nm cells)", "EF_probe": r.ef_probe}
        )
    for row in res_rows:
        row["rel_to_reference"] = row["EF_probe"] / res_rows[-1]["EF_probe"] - 1
    rows += res_rows
    base = None
    for name, num in domain_variants(cfg.domain_numerics, cfg.shift_um, cfg.absorber_factor, cfg.pml_factor).items():
        r = simulate(
            replace(cfg.model, numerics=num), cfg.wavelength_nm, cfg.run_root / "domain" / name.replace(" ", "_").replace(",", "")
        )
        base = r.ef_probe if base is None else base
        rows.append({"check": "domain", "setting": name, "EF_probe": r.ef_probe, "rel_to_reference": r.ef_probe / base - 1})
    if cfg.fem:
        from ters2d.fem import solve

        f = solve(replace(cfg.model, numerics=cfg.domain_numerics), cfg.wavelength_nm)
        rows.append(
            {
                "check": "FEM (same model)",
                "setting": "no staircasing",
                "EF_probe": f["EF_probe"],
                "rel_to_reference": base / f["EF_probe"] - 1,
            }
        )
    for row in rows:
        print(f"{row['check']:26s} {row['setting']:40s} EF {row['EF_probe']:9.2f}  {row['rel_to_reference']:+.2%}")
    print("(resolution: relative to the finest grid; domain: to the baseline; FEM line: FDTD baseline relative to the FEM)")
    print(f"wrote {write_csv(rows, cfg.out_dir / 'convergence.csv')}")


if __name__ == "__main__":
    main(AppConfig())
