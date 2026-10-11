"""p vs s polarization for the same tip and sample.

``Illumination(polarization="s")`` puts E along the invariant z axis (Ez, Hx, Hy): parallel
to every surface of the 2D tip and sample, so there is no surface charge and no gap mode. In
2D the two polarizations are decoupled, so a beam polarized at an angle psi to the plane of
incidence gives |E|^2 = cos^2(psi) |E_p|^2 + sin^2(psi) |E_s|^2 from these two runs (with
E2_probe_tip, which has the same units for both).

Two normalizations are printed. EF = |E|^2 / |E_ref|^2 compares with the tip-free field at
the same point, which for s above a metal is itself suppressed (Ez is tangential: ~0.09 of the
incident intensity on gold at 633 nm). |E|^2 / |E_inc|^2 compares with the incident beam
alone at its focus and is the number to compare between p and s.

Default: the production domain on a coarse 2 nm grid (a template: absolute values are not
converged), ~11 min per run with mpirun -np 8 on a CPU, ~2-4 min on a GPU with gpmeep.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Sample, field_map, materials, simulate, write_csv
from ters2d.plotting import plot_field_map


@dataclass
class AppConfig:
    model: Model = field(default_factory=lambda: Model(sample=Sample(materials.AU), numerics=Numerics.production(resolution=500)))
    polarizations: list = field(default_factory=lambda: ["p", "s"])
    wavelength_nm: float = 633.0
    run_root: Path = Path("runs") / platform.node() / "09_s_polarization"
    out_dir: Path = Path("results") / platform.node() / "09_s_polarization"


def main(cfg: AppConfig) -> None:
    rows = []
    for pol in cfg.polarizations:
        model = cfg.model.with_value("illumination.polarization", pol)
        r = simulate(model, cfg.wavelength_nm, cfg.run_root / pol)
        m = r.metrics
        rows.append({"polarization": pol, **r.row()})
        print(
            f"{pol}: EF = {m['EF_probe']:.3g}, |E|^2/|E_inc|^2 = {m['I_probe_tip_over_inc']:.3g} "
            f"(tip-free: {m['I_probe_ref_over_inc']:.3g})"
        )
        print(f"wrote {plot_field_map(field_map(r, normalization='incident'), cfg.out_dir / f'fieldmap_{pol}.png', vmin=1e-4)}")
    print(f"wrote {write_csv(rows, cfg.out_dir / 'p_vs_s.csv')}")


if __name__ == "__main__":
    main(AppConfig())
