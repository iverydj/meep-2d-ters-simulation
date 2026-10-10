"""Add a material from tabulated n, k: passive Lorentz fit -> check -> save -> use.

Meep needs eps_inf + Lorentz/Drude poles. ``fit_lorentz`` fits them with all strengths and
widths >= 0, so the model is passive (Im eps >= 0) at every frequency, not only inside the
data window. The demo input is the monolayer WS2 in-plane table shipped in materials/nk/
with the starting poles of the packaged fit; for your material:

1. write a CSV with the header ``wavelength_nm,n,k`` (# lines are comments);
2. put one pole at each absorption feature (exciton, interband edge) and one or two broad
   poles above the window for the background; start widths at the feature widths;
3. fit a window slightly wider than the wavelengths you will simulate, and set
   valid_wavelength_nm to the range you trust (simulations outside it raise an error);
4. check rel_rms (a few % is typical), min Im(eps) >= 0 and the pole warnings (poles at the
   minimum width or outside the window), and look at the plot;
5. use the saved JSON: Material.load_json(path); for an anisotropic layer combine two fits
   in UniaxialMaterial(name, in_plane=..., out_of_plane=...).

No Meep needed; runs in seconds.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d import Layer, Material, Sample, UniaxialMaterial, materials
from ters2d.fitting import fit_lorentz, fit_quality, pole_warnings, read_nk_csv
from ters2d.plotting import plot_lorentz_fit

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    nk_csv: Path = ROOT / "materials" / "nk" / "ws2_monolayer_inplane_ermolaev2020.csv"
    name: str = "my WS2 in-plane fit"
    # (resonance energy eV, width eV): A exciton (two terms), B exciton (two), C (two), UV background
    initial_poles_eV: list = field(
        default_factory=lambda: [(2.01, 0.04), (2.08, 0.2), (2.40, 0.12), (2.55, 0.3), (2.85, 0.3), (3.5, 1.0), (5.0, 2.0)]
    )
    fit_window_nm: tuple = (450.0, 950.0)
    valid_wavelength_nm: tuple = (480.0, 900.0)
    max_rel_rms: float = 0.03  # fail if the fit is worse
    out_dir: Path = Path("results") / platform.node() / "06_material_fit"


def main(cfg: AppConfig) -> None:
    wl, n, k = read_nk_csv(cfg.nk_csv)
    mat = fit_lorentz(
        wl,
        n,
        k,
        cfg.initial_poles_eV,
        name=cfg.name,
        fit_window_nm=cfg.fit_window_nm,
        valid_wavelength_nm=cfg.valid_wavelength_nm,
        reference=str(cfg.nk_csv.name),
    )
    q = fit_quality(mat, wl, n, k)
    print(
        f"relative error of eps: rms {q['rel_rms']:.2%}, max {q['rel_max']:.2%}; min Im eps {q['min_im_eps_200nm_to_20um']:.2e}"
    )
    if q["rel_rms"] > cfg.max_rel_rms or q["min_im_eps_200nm_to_20um"] < 0:
        raise RuntimeError("fit rejected: adjust initial_poles_eV or the window")
    for w in pole_warnings(mat, cfg.fit_window_nm):
        print("check:", w)  # e.g. the packaged WS2 fit has a narrow pole at 410 nm (C exciton side)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    js = cfg.out_dir / "material.json"
    mat.save_json(js, extra={"fit_quality": q})
    band = (wl >= cfg.valid_wavelength_nm[0]) & (wl <= cfg.valid_wavelength_nm[1])
    nk_fit = np.sqrt(mat.epsilon(wl[band]))
    table = cfg.out_dir / "fit.csv"
    with table.open("w", encoding="utf-8") as fh:
        fh.write("component,wavelength_um,n_data,k_data,n_fit,k_fit\n")
        for w, a, b, c in zip(wl[band], n[band], k[band], nk_fit):
            fh.write(f"fit,{w * 1e-3:.5f},{a:.5f},{b:.5f},{c.real:.5f},{c.imag:.5f}\n")
    print(f"wrote {js}, {plot_lorentz_fit(table, cfg.out_dir / 'fit.png')}")

    # use it: an anisotropic layer with the packaged out-of-plane model, 0.65 nm on Au
    loaded = Material.load_json(js)
    layer = Layer(UniaxialMaterial("WS2 (my fit)", loaded, materials.WS2_MONOLAYER.out_of_plane), 0.65)
    sample = Sample(materials.AU, (layer,))
    print(f"sample {sample.label}: eps at 633 nm (in plane, out of plane) = {layer.material.epsilon_xy(633.0)}")
    print("simulate it like any other sample, e.g. in examples/05_custom_substrate_and_2d_layer.py")


if __name__ == "__main__":
    main(AppConfig())
