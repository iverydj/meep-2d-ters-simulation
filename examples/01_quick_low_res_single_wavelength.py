"""Quick start: one wavelength of the gap-mode case on a coarse grid (plain CPU Meep, ~2-3 min on one core).

Au tip (apex radius 20 nm, half-angle 15 deg) 2 nm above Au, 60 deg p-polarized Gaussian
beam at 633 nm, in the small domain at 2 nm cells (the gap is a single cell). This is a
smoke test and a template, not a converged number: the reference value of this geometry
(production domain, 1 nm cells) is EF = 140 (FDTD) / 152 (FEM); the small domain gives
larger absolute values (see docs/setting_up_your_simulation.md).

    python examples/01_quick_low_res_single_wavelength.py
    MEEP_GPU_BACKEND=cuda python examples/01_quick_low_res_single_wavelength.py   # gpmeep on a GPU
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Illumination, Model, Numerics, Sample, Tip, field_map, materials, simulate
from ters2d.plotting import plot_field_map


@dataclass
class AppConfig:
    model: Model = field(
        default_factory=lambda: Model(
            tip=Tip(apex_radius_nm=20.0, half_angle_deg=15.0),
            sample=Sample(materials.AU),
            gap_nm=2.0,
            illumination=Illumination(incidence_deg=60.0, divergence_deg=14.0),
            numerics=Numerics.small(resolution=500),  # 2 nm cells; Numerics.production(1000) for the reference set-up
        )
    )
    wavelength_nm: float = 633.0
    run_dir: Path = Path("runs") / platform.node() / "01_quick"
    figure: Path = Path("results") / platform.node() / "01_quick" / "fieldmap.png"


def main(cfg: AppConfig) -> None:
    r = simulate(cfg.model, cfg.wavelength_nm, cfg.run_dir)
    m = r.metrics
    print(f"EF at the gap centre      {m['EF_probe']:.1f}")
    print(f"EF averaged over the gap  {m['EF_gap']:.1f}")
    print(f"|E|^4 estimate (EF4_gap)  {m['EF4_gap']:.3g}")
    print(f"|E|^2 / incident beam     {m['I_probe_tip_over_inc']:.1f}")
    print(f"time stepping {r.meta['wall_s']:.0f} s, {r.meta['timesteps']} steps, {r.meta['layout']['n_cells']} cells")
    print(f"wrote {plot_field_map(field_map(r), cfg.figure)}")


if __name__ == "__main__":
    main(AppConfig())
