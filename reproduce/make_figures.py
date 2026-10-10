"""Render every result and validation figure of docs/ from the tables in docs/figures/data (no Meep needed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ters2d import plotting

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    data_dir: Path = ROOT / "docs" / "figures" / "data"
    fig_dir: Path = ROOT / "docs" / "figures"


def main(cfg: AppConfig) -> None:
    d, f = cfg.data_dir, cfg.fig_dir
    fem = d / "fem_tip_reference.csv"
    done = [
        plotting.plot_spectra(
            [
                ("Au tip / Au (gap mode)", d / "spectrum_Au_fdtd.csv", "Au"),
                ("Au tip / SiO$_2$ (no gap mode)", d / "spectrum_SiO2_fdtd.csv", "SiO2"),
            ],
            f / "spectra.png",
            fem,
        ),
        plotting.plot_sweep(
            d / "gap_sweep_fdtd.csv", "gap_nm", "gap (nm)", f / "gap_sweep.png", fem, "gap", flag_below=1.5, logx=True
        ),
        plotting.plot_sweep(
            d / "radius_sweep_fdtd.csv", "radius_nm", "apex radius (nm)", f / "radius_sweep.png", fem, "radius", logy=False
        ),
        plotting.plot_ws2(d / "ws2_on_au_fdtd.csv", f / "ws2_on_au.png", fem),
        plotting.plot_field_map(d / "fieldmap_Au_633nm.npz", f / "fieldmap_Au_633nm.png"),
        plotting.plot_field_map(d / "fieldmap_SiO2_633nm.npz", f / "fieldmap_SiO2_633nm.png"),
        plotting.plot_fresnel(d / "fresnel_check.csv", f / "fresnel_check.png"),
        plotting.plot_nanogap(d / "nanogap_fdtd.csv", d / "nanogap_fem.csv", f / "nanogap_benchmark.png"),
        plotting.plot_resolution_series(d / "resolution_series.csv", f / "resolution_series.png"),
        plotting.plot_lorentz_fit(d / "ws2_lorentz_fit.csv", f / "ws2_lorentz_fit.png"),
    ]
    for p in done:
        print(f"wrote {p}")


if __name__ == "__main__":
    main(AppConfig())
