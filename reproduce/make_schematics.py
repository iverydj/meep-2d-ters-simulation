"""Schematic figures of docs/model.md and the README (pure Python, no Meep needed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Layer, Model, Sample, plotting
from ters2d.layout import build_layout
from ters2d.materials import AU, WS2_MONOLAYER

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    model: Model = field(default_factory=Model)  # the default (reference) gap-mode case
    wavelength_nm: float = 633.0
    fig_dir: Path = ROOT / "docs" / "figures"
    field_map: Path = ROOT / "docs" / "figures" / "data" / "fieldmap_Au_633nm.npz"


def main(cfg: AppConfig) -> None:
    lay = build_layout(cfg.model, [cfg.wavelength_nm])
    f = cfg.fig_dir
    done = [
        plotting.plot_geometry(lay, cfg.wavelength_nm, f / "geometry.png"),
        plotting.plot_scattered_field_idea(lay, cfg.wavelength_nm, cfg.field_map, f / "scattered_field.png"),
        plotting.plot_boundaries(lay, f / "boundaries.png"),
        plotting.plot_beam_synthesis(
            cfg.model.illumination, cfg.wavelength_nm, lay.stack(cfg.wavelength_nm), f / "beam_synthesis.png"
        ),
    ]
    # apex of a layered sample (WS2 monolayer on Au, 2 nm metal-metal distance at 1.5 px/nm)
    ws2 = Model(sample=Sample(AU, (Layer(WS2_MONOLAYER, 0.65),)), gap_nm=4.0 / 3.0)
    ws2 = ws2.with_value("numerics.resolution", 1500)
    plt = plotting._plt()
    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    plotting._draw_apex(ax, build_layout(ws2, [cfg.wavelength_nm]))
    ax.set_title("WS$_2$ monolayer in the gap")
    done.append(plotting.save(fig, f / "apex_ws2.png"))
    for p in done:
        print(f"wrote {p}")


if __name__ == "__main__":
    main(AppConfig())
