"""Re-create the packaged WS2 models (src/ters2d/data/ws2_*_lorentz.json) from the n, k tables.

In-plane:      monolayer data of Ermolaev et al., J. Vac. Sci. Technol. B 38, 014002 (2020)
Out-of-plane:  bulk extraordinary data of Munkhbat et al., ACS Photonics 9, 2398 (2022), the
               monolayer out-of-plane proxy (no broadband monolayer data exist).
Both tables are CC0 data of the refractiveindex.info database (materials/nk/).

Writes the two JSON files, the data-vs-fit table and figure to ``out_dir`` and prints how
far they are from the packaged models. Set out_dir to src/ters2d/data to replace them.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d.fitting import fit_lorentz, fit_quality, read_nk_csv
from ters2d.materials import WS2_MONOLAYER
from ters2d.plotting import plot_lorentz_fit

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    nk_dir: Path = ROOT / "materials" / "nk"
    out_dir: Path = Path("results") / platform.node() / "ws2_fit"
    fit_window_nm: tuple = (450.0, 950.0)
    valid_wavelength_nm: tuple = (480.0, 900.0)
    # in-plane starting poles (energy eV, width eV): A (two terms), B (two), C (two), UV background
    inplane_init_eV: list = field(
        default_factory=lambda: [(2.01, 0.04), (2.08, 0.2), (2.40, 0.12), (2.55, 0.3), (2.85, 0.3), (3.5, 1.0), (5.0, 2.0)]
    )
    max_rel_rms: float = 0.03
    min_gamma_um_inv: float = 0.02  # every pole gets a finite width: lossless poles ring indefinitely in a pulsed run


def main(cfg: AppConfig) -> None:
    wl_ip, n_ip, k_ip = read_nk_csv(cfg.nk_dir / "ws2_monolayer_inplane_ermolaev2020.csv")
    wl_op, n_op, k_op = read_nk_csv(cfg.nk_dir / "ws2_bulk_outofplane_munkhbat2022.csv")
    inplane = fit_lorentz(
        wl_ip,
        n_ip,
        k_ip,
        cfg.inplane_init_eV,
        name="WS2 monolayer, in-plane (Lorentz fit)",
        fit_window_nm=cfg.fit_window_nm,
        valid_wavelength_nm=cfg.valid_wavelength_nm,
        min_gamma_um_inv=cfg.min_gamma_um_inv,
        reference="data: G. A. Ermolaev et al., J. Vac. Sci. Technol. B 38, 014002 (2020), doi:10.1116/1.5122683",
    )
    # the bulk extraordinary data are lossless in the window: one UV pole with a small fixed width
    outplane = fit_lorentz(
        wl_op,
        n_op,
        k_op,
        [(5.0, 0.0)],
        name="WS2 out-of-plane (bulk extraordinary data, Lorentz fit)",
        fit_window_nm=cfg.fit_window_nm,
        valid_wavelength_nm=cfg.valid_wavelength_nm,
        fixed_gamma_um_inv=cfg.min_gamma_um_inv,
        reference="data: B. Munkhbat et al., ACS Photonics 9, 2398 (2022), doi:10.1021/acsphotonics.2c00433",
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, mat, (wl, n, k), packaged, fname in (
        ("inplane", inplane, (wl_ip, n_ip, k_ip), WS2_MONOLAYER.in_plane, "ws2_1L_inplane_lorentz.json"),
        ("outofplane", outplane, (wl_op, n_op, k_op), WS2_MONOLAYER.out_of_plane, "ws2_outofplane_lorentz.json"),
    ):
        q = fit_quality(mat, wl, n, k)
        if q["min_im_eps_200nm_to_20um"] < 0:
            raise RuntimeError(f"{tag} fit is not passive")
        if tag == "inplane" and q["rel_rms"] > cfg.max_rel_rms:
            raise RuntimeError(f"in-plane fit relative RMS {q['rel_rms']:.3f} > {cfg.max_rel_rms}")
        mat.save_json(cfg.out_dir / fname, extra={"fit_quality": q})
        w = np.linspace(*cfg.valid_wavelength_nm, 421)
        diff = np.max(np.abs(mat.epsilon(w) - packaged.epsilon(w)) / np.abs(packaged.epsilon(w)))
        print(f"{tag}: rel. RMS error {q['rel_rms']:.2%}, max {q['rel_max']:.2%}; max |eps - packaged| / |eps| = {diff:.1e}")
        print(json.dumps(q))
        band = (wl >= cfg.valid_wavelength_nm[0]) & (wl <= cfg.valid_wavelength_nm[1])
        nk = np.sqrt(mat.epsilon(wl[band]))
        rows += [(tag, x * 1e-3, a, b, c.real, c.imag) for x, a, b, c in zip(wl[band], n[band], k[band], nk)]
    table = cfg.out_dir / "ws2_lorentz_fit.csv"
    with table.open("w", encoding="utf-8") as fh:
        fh.write("component,wavelength_um,n_data,k_data,n_fit,k_fit\n")
        for r in rows:
            fh.write("{},{:.4f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(*r))
    print(f"wrote {table}, {plot_lorentz_fit(table, cfg.out_dir / 'ws2_lorentz_fit.png')}")


if __name__ == "__main__":
    main(AppConfig())
