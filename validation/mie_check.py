"""Scattered-field formulation vs the exact 2D Mie solution (Au cylinder in the focused beam).

Validates the source construction, its normalization (no fitted constants) and the Au
model: a cylinder (radius 20 nm, free space) is driven exactly like the tip and its total
field is compared with the Mie series for the same plane-wave superposition, in three
shells: 1-3 nm from the surface (where the gap probes sit), 1.5-4 cells (staircase zone)
and from 4 cells out to 55 nm. Upstream Meep or gpmeep (CPU or GPU).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d import write_csv
from ters2d.analysis import mie_cylinder_hz, rel_l2
from ters2d.background import GaussianBeam, Stack, beam_fields_grid
from ters2d.materials import AU
from ters2d.sources import volume_sources


@dataclass
class AppConfig:
    resolutions: list = field(default_factory=lambda: [500, 1000, 1500])
    wavelengths_nm: list = field(default_factory=lambda: [500.0, 633.0, 850.0])
    radius_um: float = 0.02
    incidence_deg: float = 60.0
    divergence_deg: float = 14.0
    half_box_um: float = 0.4
    pml_um: float = 0.4
    out_csv: Path = Path("results") / platform.node() / "mie_check" / "mie_check.csv"


class _Cylinder:
    def __init__(self, r):
        self.r = r

    def contains(self, x, y):
        return x**2 + y**2 <= self.r**2


def run_one(cfg: AppConfig, res: int, wavelength_nm: float) -> dict:
    import meep as mp

    from ters2d.meep_model import meep_medium

    wl_um = wavelength_nm * 1e-3
    f0 = 1 / wl_um
    dx = 1 / res
    R = cfg.radius_um
    beam = GaussianBeam(wl_um, cfg.incidence_deg, cfg.divergence_deg)
    stack = Stack.isotropic(1.0)
    eps = complex(AU.epsilon(wavelength_nm))
    src = mp.GaussianSource(f0, fwidth=1.5 * f0, cutoff=4.0)
    box = (-R - 2 * dx, R + 2 * dx, -R - 2 * dx, R + 2 * dx)
    srcs, _ = volume_sources(_Cylinder(R), box, dx, beam, stack, eps, src)
    L = 2 * (cfg.half_box_um + cfg.pml_um)
    sim = mp.Simulation(
        cell_size=mp.Vector3(L, L),
        resolution=res,
        geometry=[mp.Cylinder(R, material=meep_medium(AU))],
        sources=srcs,
        boundary_layers=[mp.PML(cfg.pml_um)],
        eps_averaging=False,
    )
    dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Hz], [f0], center=mp.Vector3(), size=mp.Vector3(0.12, 0.12))
    sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, minimum_run_time=2))
    scale = complex(src.fourier_transform(f0)) * (0.5 if sim.fields.is_real else 1.0)
    x, y, _, _ = sim.get_array_metadata(dft_cell=dft)
    x, y = np.asarray(x), np.asarray(y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    bg = beam_fields_grid(beam, stack, x, y)
    sim_f = [
        bg[i] + np.asarray(sim.get_dft_array(dft, c, 0)).reshape(len(x), len(y)) / scale
        for i, c in enumerate((mp.Hz, mp.Ex, mp.Ey))
    ]
    th, w = beam.angles_weights()
    r = np.hypot(X, Y)
    row = {"resolution": res, "wavelength_nm": wavelength_nm}
    shells = {
        "shell_1_3nm": (R + 0.001, R + 0.003),
        "surface_shell": (R + 1.5 * dx, R + 4 * dx),
        "near_shell": (R + 4 * dx, 0.055),
    }
    for name, (r0, r1) in shells.items():
        m = (r > r0) & (r < r1)
        mie = mie_cylinder_hz(eps, R, wl_um, th, w, (0, 0), (0, 0), X[m], Y[m])
        e2s = np.abs(sim_f[1][m]) ** 2 + np.abs(sim_f[2][m]) ** 2
        e2m = np.abs(mie[1]) ** 2 + np.abs(mie[2]) ** 2
        row[f"{name}_Hz_relL2"] = rel_l2(sim_f[0][m], mie[0])
        row[f"{name}_E_relL2"] = rel_l2(np.r_[sim_f[1][m], sim_f[2][m]], np.r_[mie[1], mie[2]])
        row[f"{name}_mean_E2_ratio"] = float(e2s.mean() / e2m.mean())
    return row


def main(cfg: AppConfig) -> None:
    import meep as mp

    rows = [run_one(cfg, r, wl) for wl in cfg.wavelengths_nm for r in cfg.resolutions]
    if mp.am_master():
        for row in rows:
            print(row, flush=True)
        print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
