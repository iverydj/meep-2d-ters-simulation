"""Tip-free reference vs Meep: specular reflection of the sample at the beam's angles.

For each sample and wavelength a Bloch-periodic Meep cell without tip is illuminated by a
p- or s-polarized plane wave (``AppConfig.polarization``) whose in-plane wavevector
k sin(theta) is set for that wavelength, so the angle is exact. The out-of-plane field (Hz
for p, Ez for s) between source and sample is split into down- and up-going waves;
r = up / down at the surface is compared with the transfer-matrix
coefficient that the analytic tip-free field uses (same grid-snapped layers, same media).
This validates the reference field against the discretized Meep materials at the
simulated resolution and precision. CPU, one process per (sample, wavelength, angle);
with gpmeep use MEEP_GPU_BACKEND=cpu (the same FP32 numerics as its GPU lane).
"""

from __future__ import annotations

import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d import Layer, Model, Numerics, Sample, materials, write_csv
from ters2d.analysis import fit_up_down
from ters2d.background import stack_coefficients
from ters2d.layout import NUDGE_CELLS, build_layout


def _cases():
    return [
        ("Au", Sample(materials.AU), 1500),
        ("SiO2", Sample(materials.SIO2), 1500),
        ("WS2/Au", Sample(materials.AU, (Layer(materials.WS2_MONOLAYER, 0.65),)), 1500),
    ]


@dataclass
class AppConfig:
    cases: list = field(default_factory=_cases)  # (label, sample, resolution)
    polarization: str = "p"
    incidence_deg: float = 60.0
    wavelengths_nm: list = field(default_factory=lambda: [500, 550, 600, 633, 650, 700, 750, 800, 850])
    angle_scan_deg: list = field(default_factory=lambda: [40.0, 50.0, 70.0])  # extra angles at one wavelength
    angle_scan_wavelength_nm: float = 633.0
    workers: int = 14
    cell_width_um: float = 0.06
    air_um: float = 0.8  # air between the surface and the top PML (source at 0.75 of it)
    substrate_um: float = 0.24  # sizes chosen for even pixel counts (Meep grid alignment)
    pml_um: float = 0.6
    fwidth_rel: float = 0.15
    out_csv: Path = Path("results") / platform.node() / "fresnel_check" / "fresnel_check.csv"  # _s.csv for s


def reflection(label, sample, resolution, wavelength_nm, incidence_deg, cfg: AppConfig) -> dict:
    import meep as mp

    from ters2d.meep_model import meep_medium

    f0 = 1e3 / wavelength_nm
    k = 2 * np.pi * f0
    th = np.deg2rad(incidence_deg)
    kx = k * np.sin(th)
    dx = 1.0 / resolution
    lay = build_layout(Model(sample=sample, numerics=Numerics(resolution=resolution)), [wavelength_nm])
    T = lay.thickness
    metal = lay.substrate_metallic
    y_lo = -T - cfg.substrate_um - (0.0 if metal else cfg.pml_um)
    y_hi = cfg.air_um + cfg.pml_um
    nudge = NUDGE_CELLS * dx
    big = 10.0
    geom = [
        mp.Block(
            center=mp.Vector3(0, 0.5 * (-T + nudge - big)), size=mp.Vector3(mp.inf, big), material=meep_medium(sample.substrate)
        )
    ]
    for layer, (yb, yt) in zip(sample.layers, lay.layers):
        geom.append(
            mp.Block(
                center=mp.Vector3(0, 0.5 * (yb + yt) + nudge),
                size=mp.Vector3(mp.inf, yt - yb),
                material=meep_medium(layer.material),
            )
        )
    layers = [mp.PML(cfg.pml_um, direction=mp.Y, side=mp.High)]
    if not metal:
        layers.append(mp.PML(cfg.pml_um, direction=mp.Y, side=mp.Low))
    ny = round((y_hi - y_lo) * resolution)
    if ny % 2:  # even pixel count (Meep grid alignment): extend the substrate by one cell
        y_lo -= dx
    if round(cfg.cell_width_um * resolution) % 2:
        raise ValueError("odd number of pixels across the cell")
    ys = 0.75 * cfg.air_um
    src = [
        mp.Source(
            # keep the pulse spectrum away from the grazing cutoff f0 sin(theta) of the Bloch cell
            mp.GaussianSource(f0, fwidth=min(cfg.fwidth_rel, 1 - np.sin(th)) * f0),
            component=mp.Ex if cfg.polarization == "p" else mp.Ez,
            center=mp.Vector3(0, ys),
            size=mp.Vector3(cfg.cell_width_um, 0),
            amp_func=lambda p: np.exp(1j * kx * p.x),
        )
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(cfg.cell_width_um, y_hi - y_lo),
        geometry_center=mp.Vector3(0, 0.5 * (y_hi + y_lo)),
        resolution=resolution,
        geometry=geom,
        sources=src,
        boundary_layers=layers,
        k_point=mp.Vector3(kx / (2 * np.pi)),
        ensure_periodicity=False,  # do not replicate the (taller-than-cell) substrate block along y
        eps_averaging=False,
    )
    comp = mp.Hz if cfg.polarization == "p" else mp.Ez
    line = sim.add_dft_fields([comp], [f0], center=mp.Vector3(0, 0.5 * ys), size=mp.Vector3(0, ys))
    sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-6, minimum_run_time=2.0))
    _, y, _, _ = sim.get_array_metadata(dft_cell=line)
    y = np.asarray(y)
    hz = np.asarray(sim.get_dft_array(line, comp, 0)).ravel()
    sel = (y > 0.05) & (y < ys - 0.05)
    a, b, resid = fit_up_down(y[sel], hz[sel], k * np.cos(th))
    r_an = complex(stack_coefficients(lay.stack(wavelength_nm), np.array([kx]), k, cfg.polarization)["r"][0])
    r_sim = b / a
    return {
        "sample": label,
        "polarization": cfg.polarization,
        "resolution": resolution,
        "wavelength_nm": float(wavelength_nm),
        "incidence_deg": incidence_deg,
        "R_sim": abs(r_sim) ** 2,
        "R_fresnel": abs(r_an) ** 2,
        "R_rel_err": abs(r_sim) ** 2 / abs(r_an) ** 2 - 1,
        "abs_r_rel_err": abs(r_sim) / abs(r_an) - 1,
        "abs_r_err": abs(r_sim - r_an),
        "phase_err_deg": float(np.degrees(np.angle(r_sim / r_an))),
        "complex_rel_err": abs(r_sim - r_an) / abs(r_an),
        "fit_residual": resid,
    }


def _job(args):
    return reflection(*args)


def main(cfg: AppConfig) -> None:
    jobs = [(lab, s, res, wl, cfg.incidence_deg, cfg) for lab, s, res in cfg.cases for wl in cfg.wavelengths_nm]
    jobs += [(lab, s, res, cfg.angle_scan_wavelength_nm, a, cfg) for lab, s, res in cfg.cases for a in cfg.angle_scan_deg]
    out = cfg.out_csv if cfg.polarization == "p" else cfg.out_csv.with_name(cfg.out_csv.stem + "_s.csv")
    rows = []
    with ProcessPoolExecutor(cfg.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for fut in as_completed([ex.submit(_job, j) for j in jobs]):
            row = fut.result()
            rows.append(row)
            print(
                f"{row['sample']:7s} {row['wavelength_nm']:5.0f} nm {row['incidence_deg']:4.0f} deg R_sim={row['R_sim']:.5f} "
                f"R_analytic={row['R_fresnel']:.5f} |dr|={row['abs_r_err']:.2e} dphase={row['phase_err_deg']:+.3f} deg",
                flush=True,
            )
            rows.sort(key=lambda r: (r["sample"], r["incidence_deg"], r["wavelength_nm"]))
            write_csv(rows, out)  # rewritten after every case
    print(f"wrote {out} ({len(rows)} of {len(jobs)} cases)")


if __name__ == "__main__":
    main(AppConfig())
