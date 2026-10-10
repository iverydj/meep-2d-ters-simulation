"""Nanogap benchmark, FDTD side: Au cylinder (R = 10-40 nm) 1-10 nm above Au.

Same scattered-field formulation, beam, materials, grid alignment, PML treatment and
substrate-plasmon absorber as the tip model, but with a compact scatterer whose
frequency-domain solution (nanogap_fem.py, curved elements) is cheap to converge. The
difference is the staircase error of the uniform Meep grid for a nm gap between curved
and flat gold.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d import write_csv
from ters2d.background import GaussianBeam, Stack, beam_fields_grid
from ters2d.materials import AU
from ters2d.results import interp2
from ters2d.sources import volume_sources


@dataclass
class AppConfig:
    # (gap_nm, radius_nm, resolution, wavelength_nm)
    runs: list = field(
        default_factory=lambda: [
            *[(2.0, 20.0, r, wl) for wl in (633.0, 785.0) for r in (1000, 1500, 2000)],
            *[(1.0, 20.0, r, 633.0) for r in (1000, 2000)],
            (4.0, 20.0, 1000, 633.0),
            (10.0, 20.0, 1000, 633.0),
            (2.0, 10.0, 1000, 633.0),
            (2.0, 40.0, 1000, 633.0),
        ]
    )
    incidence_deg: float = 60.0
    divergence_deg: float = 14.0
    half_width_um: float = 1.3
    height_um: float = 0.5
    substrate_um: float = 0.16  # with the other sizes: even pixel counts at all resolutions
    pml_um: float = 0.4
    dft_decay_tol: float = 1e-5
    side_absorber_start_um: float = 0.3  # matched lossy air above the substrate for |x| > start
    absorber_conductivity: float = 10.0
    absorber_power: int = 3
    absorber_segments: int = 30
    pml_metal_conductivity: float = 20.0
    out_csv: Path = Path("results") / platform.node() / "nanogap" / "nanogap_fdtd.csv"


class _Cylinder:
    def __init__(self, center, r):
        self.c, self.r = center, r

    def contains(self, x, y):
        return (x - self.c[0]) ** 2 + (y - self.c[1]) ** 2 <= self.r**2


def run_one(cfg: AppConfig, gap_nm: float, radius_nm: float, res: int, wavelength_nm: float) -> dict:
    import meep as mp

    from ters2d.meep_model import meep_medium

    au = meep_medium(AU)
    dx = 1.0 / res
    g, R = gap_nm * 1e-3, radius_nm * 1e-3
    if abs(g * res - round(g * res)) > 1e-6:
        raise ValueError("gap must be a whole number of cells")
    for size in (2 * (cfg.half_width_um + cfg.pml_um), cfg.substrate_um + cfg.height_um + cfg.pml_um):
        if round(size * res) % 2:
            raise ValueError("odd number of pixels: Meep would shift the grid by half a cell")
    wl_um = wavelength_nm * 1e-3
    f0 = 1.0 / wl_um
    eps_au = complex(AU.epsilon(wavelength_nm))
    x0, x1 = -cfg.half_width_um - cfg.pml_um, cfg.half_width_um + cfg.pml_um
    y0, y1 = -cfg.substrate_um, cfg.height_um + cfg.pml_um
    nudge = 1e-3 * dx
    yc, h = 0.5 * (y0 - 1 + nudge), nudge - y0 + 1
    geom = [mp.Block(center=mp.Vector3(0, yc), size=mp.Vector3(2 * cfg.half_width_um, h), material=au)]
    bm = mp.Medium(epsilon=1.0, D_conductivity=cfg.pml_metal_conductivity)  # gold inside the side PML
    edges = np.linspace(cfg.side_absorber_start_um, cfg.half_width_um, cfg.absorber_segments + 1)
    for sgn in (-1.0, 1.0):
        geom.append(
            mp.Block(
                center=mp.Vector3(sgn * 0.5 * (cfg.half_width_um + x1 + 1), yc),
                size=mp.Vector3(x1 + 1 - cfg.half_width_um, h),
                material=bm,
            )
        )
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            sig = cfg.absorber_conductivity * ((k + 0.5) / cfg.absorber_segments) ** cfg.absorber_power
            geom.append(
                mp.Block(
                    center=mp.Vector3(sgn * 0.5 * (a + b), 0.5 * (nudge + y1)),
                    size=mp.Vector3(b - a, y1 - nudge),
                    material=mp.Medium(epsilon=1.0, D_conductivity=sig, B_conductivity=sig),
                )
            )
    geom.append(mp.Cylinder(R, center=mp.Vector3(0, g + R), material=au))
    beam = GaussianBeam(wl_um, cfg.incidence_deg, cfg.divergence_deg)
    stack = Stack.isotropic(eps_au)
    src = mp.GaussianSource(f0, fwidth=1.5 * f0, cutoff=4.0)
    box = (-R - 2 * dx, R + 2 * dx, g - 0.5 * dx, g + 2 * R + 2 * dx)
    srcs, _ = volume_sources(_Cylinder((0.0, g + R), R), box, dx, beam, stack, eps_au, src)
    sim = mp.Simulation(
        cell_size=mp.Vector3(x1 - x0, y1 - y0),
        geometry_center=mp.Vector3(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
        resolution=res,
        geometry=geom,
        sources=srcs,
        boundary_layers=[mp.PML(cfg.pml_um, direction=mp.X), mp.PML(cfg.pml_um, direction=mp.Y, side=mp.High)],
        eps_averaging=False,
    )
    mon = (-0.01, 0.01, -0.002, 0.006)
    dft = sim.add_dft_fields(
        [mp.Ex, mp.Ey], [f0], center=mp.Vector3(0, 0.5 * (mon[2] + mon[3])), size=mp.Vector3(mon[1] - mon[0], mon[3] - mon[2])
    )
    sim.run(until_after_sources=mp.stop_when_dft_decayed(cfg.dft_decay_tol, minimum_run_time=2.0))
    scale = complex(src.fourier_transform(f0)) * (0.5 if sim.fields.is_real else 1.0)
    x, y, _, _ = sim.get_array_metadata(dft_cell=dft)
    x, y = np.asarray(x), np.asarray(y)
    _, bx, by = beam_fields_grid(beam, stack, x, y)
    ex = bx + np.asarray(sim.get_dft_array(dft, mp.Ex, 0)).reshape(len(x), len(y)) / scale
    ey = by + np.asarray(sim.get_dft_array(dft, mp.Ey, 0)).reshape(len(x), len(y)) / scale
    p = (0.0, 0.5 * g)
    e2 = abs(interp2(x, y, ex, *p)) ** 2 + abs(interp2(x, y, ey, *p)) ** 2
    e2b = abs(interp2(x, y, bx, *p)) ** 2 + abs(interp2(x, y, by, *p)) ** 2
    return {
        "gap_nm": gap_nm,
        "radius_nm": radius_nm,
        "resolution": res,
        "dx_nm": 1e3 * dx,
        "gap_cells": round(g * res),
        "wavelength_nm": wavelength_nm,
        "EF_probe": float(e2 / e2b),
    }


def main(cfg: AppConfig) -> None:
    import meep as mp

    rows = []
    for gap, radius, res, wl in cfg.runs:
        rows.append(run_one(cfg, gap, radius, res, wl))
        if mp.am_master():
            print(rows[-1], flush=True)
            write_csv(rows, cfg.out_csv)  # rewritten after every run
    print(f"wrote {cfg.out_csv}")


if __name__ == "__main__":
    main(AppConfig())
