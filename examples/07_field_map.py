"""Near-field maps: |E|^2 around the apex and over the whole domain.

Every run stores the DFT field of a box around the apex at full resolution
(Outputs.near_box_nm, default 300 x 130 nm); Outputs(full_map=True) also stores the whole
PML-free region (down-sampled by full_map_stride); inside the plasmon absorbers that map is not
physical (they damp only the scattered field). Maps are normalized by the tip-free
intensity at the gap centre, so their value at the gap centre is EF. The maps of an
existing run can be made without re-running (field_map(load_result(npz))). Default: the
production domain on a coarse 2 nm grid (a template: EF ~20 % low), ~11 min with mpirun -np 8 on
a CPU (~2-4 min on a GPU).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Model, Numerics, Outputs, Sample, field_map, materials, simulate
from ters2d.plotting import plot_field_map


@dataclass
class AppConfig:
    model: Model = field(
        default_factory=lambda: Model(
            sample=Sample(materials.AU),
            numerics=Numerics.production(resolution=500),
            outputs=Outputs(near_box_nm=(-150.0, 150.0, -10.0, 120.0), full_map=True, full_map_stride=4),
        )
    )
    wavelength_nm: float = 633.0
    run_dir: Path = Path("runs") / platform.node() / "07_field_map"
    out_dir: Path = Path("results") / platform.node() / "07_field_map"


def main(cfg: AppConfig) -> None:
    r = simulate(cfg.model, cfg.wavelength_nm, cfg.run_dir)
    near = field_map(r, "near")
    near.save_npz(cfg.out_dir / "fieldmap_near.npz")
    print(f"wrote {plot_field_map(near, cfg.out_dir / 'fieldmap_near.png')}")
    full = field_map(r, "full")
    print(f"wrote {plot_field_map(full, cfg.out_dir / 'fieldmap_full.png', vmin=1e-3)}")
    ratio = r.e2_at(10.0, 1.0) / r.e2_at(10.0, 1.0, kind="b")
    print(f"|E|^2 / |E_ref|^2 at (10 nm, 1 nm), both at that point: {ratio:.1f}")


if __name__ == "__main__":
    main(AppConfig())
