"""Render every result and validation figure of docs/ from the tables in docs/figures/data (no Meep needed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ters2d import FieldMap, plotting, read_csv

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
    # s polarization
    done.append(
        plotting.plot_spectra(
            [("Au tip / Au", d / "spectrum_Au_s_fdtd.csv", "Au"), ("Au tip / SiO$_2$", d / "spectrum_SiO2_s_fdtd.csv", "SiO2")],
            f / "spectra_s.png",
            fem,
            fem_group="spectrum_s",
            title_suffix=", s-pol.",
        )
    )
    done.append(
        plotting.plot_p_vs_s(
            [
                ("Au", 0, d / "spectrum_Au_fdtd.csv", d / "spectrum_Au_s_fdtd.csv", "Au"),
                ("SiO$_2$", 2, d / "spectrum_SiO2_fdtd.csv", d / "spectrum_SiO2_s_fdtd.csv", "SiO2"),
            ],
            f / "p_vs_s.png",
            fem,
        )
    )
    p_map = FieldMap.load_npz(d / "fieldmap_Au_633nm.npz")  # normalized by E_ref(gap centre): renormalize to the incident beam
    sp = read_csv(d / "spectrum_Au_fdtd.csv")
    i_ref = float(sp["I_probe_ref_over_inc"][sp["wavelength_nm"] == 633.0][0])
    p_map.e2_rel = p_map.e2_rel * i_ref
    s_map = FieldMap.load_npz(d / "fieldmap_Au_633nm_s.npz")
    done.append(
        plotting.plot_field_maps_shared(
            [p_map, s_map],
            ["(a) p-polarized, Au, 633 nm", "(b) s-polarized, Au, 633 nm"],
            f / "fieldmap_Au_633nm_p_vs_s.png",
        )
    )
    for p in done:
        print(f"wrote {p}")


if __name__ == "__main__":
    main(AppConfig())
