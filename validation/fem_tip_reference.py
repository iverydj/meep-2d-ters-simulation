"""Frequency-domain (FEM, NGSolve) reference of the full tip model for every result, plus
the FEM's own convergence and boundary checks. Writes the table behind the FEM lines of the
result figures (docs/figures/data/fem_tip_reference.csv).

Each case is a ters2d Model solved by ters2d.fem.solve: the same tip, sample, beam, drive
ramp, absorbers and conductor-in-PML as the FDTD, with curved elements instead of the
staircased grid. The table has the groups
  spectrum, gap, radius, ws2 (production domain), small, ws2_small (small domain),
  radius_div10 (divergence 10 deg), radius_local (shaft plasmons absorbed from 0.3 um), and
  check_* (order 5, finer mesh, gold continued into the PML, absorbers 0.7x / 1.5x,
  absorbers + ramp + free region moved by 0.5 um, ramp moved by +-0.5 um).
gap_nm is the apex-to-substrate (metal-metal) distance; air_gap_nm excludes the layer.
~1.5 h on 16 cores for all 162 cases. Requires NGSolve (pip install ngsolve).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ters2d import Illumination, Layer, Model, Numerics, Sample, Tip, materials, write_csv
from ters2d.fem import FemSettings, solve

AU_SET = sorted({float(w) for w in np.arange(500.0, 901.0, 25.0)} | {633.0})
SIO2_SET = sorted({float(w) for w in np.arange(500.0, 901.0, 50.0)} | {633.0})
WS2_SMALL_SET = [500.0, 520.0, 560.0, 600.0, 615.0, 633.0, 660.0, 700.0, 800.0, 900.0]
WS2_SET = sorted(
    set(WS2_SMALL_SET) | {float(w) for w in range(500, 700, 10)} | {605.0, 625.0, 725.0, 750.0, 775.0, 825.0, 850.0, 875.0}
)
MOVED = dict(
    x_half_width_um=2.0,
    y_max_um=6.0,
    shaft_absorber_start_um=4.0,
    side_absorber_start_um=1.0,
    side_absorber_top_um=2.6,
    drive_ramp_um=(4.0, 5.0),
)
FINE = FemSettings(h_metal_um=0.01, h_air_um=0.025, h_gap_rel=0.075)


def _model(sample: str, metal_metal_nm=2.0, radius=20.0, numerics: Numerics | None = None, divergence=14.0) -> Model:
    """sample: 'Au', 'SiO2' or 'WS2/Au' (0.65 nm monolayer, snapped to 2/3 nm at 1.5 px/nm)."""
    if sample == "WS2/Au":
        s = Sample(materials.AU, (Layer(materials.WS2_MONOLAYER, 0.65),))
        gap = metal_metal_nm - 2.0 / 3.0
    else:
        s = Sample(materials.AU if sample == "Au" else materials.SIO2)
        gap = metal_metal_nm
    num = numerics if numerics is not None else Numerics(resolution=1500)  # 1.5 px/nm: the layer grid of the FDTD
    return Model(
        tip=Tip(apex_radius_nm=radius), sample=s, gap_nm=gap, illumination=Illumination(divergence_deg=divergence), numerics=num
    )


def _cases():
    """(group, sample, Model, wavelength_nm, FemSettings)."""
    base, c = FemSettings(), []
    small = Numerics.small(resolution=1500)
    c += [("spectrum", "Au", _model("Au"), wl, base) for wl in AU_SET]
    c += [("spectrum", "SiO2", _model("SiO2"), wl, base) for wl in SIO2_SET]
    c += [("ws2", s, _model(s), wl, base) for wl in WS2_SET for s in ("Au", "WS2/Au")]
    c += [
        ("gap", "Au", _model("Au", metal_metal_nm=g, numerics=Numerics(resolution=1000)), 633.0, base)
        for g in (1.0, 2.0, 3.0, 4.0, 6.0, 10.0)
    ]
    c += [("radius", "Au", _model("Au", radius=r), 633.0, base) for r in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)]
    for s, wl in (("Au", 633.0), ("Au", 700.0), ("Au", 900.0), ("SiO2", 900.0)):
        c.append(("check_taller", s, _model(s, numerics=Numerics(resolution=1500, **MOVED)), wl, base))
        c.append(("check_order5", s, _model(s), wl, replace(base, order=5)))
        c.append(("check_pml_gold", s, _model(s), wl, replace(base, pml_metal="gold")))
    for wl in (633.0, 900.0):
        c.append(
            (
                "check_absorber_strong",
                "Au",
                _model("Au", numerics=Numerics(resolution=1500, absorber_conductivity=15.0)),
                wl,
                base,
            )
        )
        c.append(
            ("check_absorber_weak", "Au", _model("Au", numerics=Numerics(resolution=1500, absorber_conductivity=7.0)), wl, base)
        )
    for wl in (633.0, 900.0):
        c.append(("check_mesh_fine", "Au", _model("Au"), wl, FINE))
        c.append(("check_ramp_early", "Au", _model("Au", numerics=Numerics(resolution=1500, drive_ramp_um=(3.0, 4.0))), wl, base))
        c.append(("check_ramp_late", "Au", _model("Au", numerics=Numerics(resolution=1500, drive_ramp_um=(4.0, 5.0))), wl, base))
    local = Numerics(resolution=1500, shaft_absorber_start_um=0.3, side_absorber_top_um=0.3)
    for r in (10.0, 20.0, 30.0, 40.0):
        c.append(("radius_div10", "Au", _model("Au", radius=r, divergence=10.0), 633.0, base))
        c.append(("radius_local", "Au", _model("Au", radius=r, numerics=local), 633.0, base))
    c.append(("small", "Au", _model("Au", numerics=small), 633.0, base))
    c.append(("small", "Au", _model("Au", numerics=small), 900.0, base))
    c.append(("small", "Au", _model("Au", metal_metal_nm=1.0, numerics=Numerics.small(resolution=1000)), 633.0, base))
    c.append(("small", "Au", _model("Au", numerics=small), 525.0, base))
    c.append(("small", "SiO2", _model("SiO2", numerics=small), 633.0, base))
    c += [("ws2_small", s, _model(s, numerics=small), wl, base) for wl in WS2_SMALL_SET for s in ("Au", "WS2/Au")]
    return c


@dataclass
class AppConfig:
    cases: list = field(default_factory=_cases)
    threads: int = 16
    out_csv: Path = Path("results") / platform.node() / "fem_tip_reference" / "fem_tip_reference.csv"


def main(cfg: AppConfig) -> None:
    rows = []
    for group, sample, model, wl, settings in cfg.cases:
        r = solve(model, wl, replace(settings, threads=cfg.threads))
        layers = 2.0 / 3.0 if sample == "WS2/Au" else 0.0
        rows.append(
            {
                "group": group,
                "sample": sample,
                "gap_nm": round(model.gap_nm + layers, 6),
                "air_gap_nm": model.gap_nm,
                "radius_nm": model.tip.apex_radius_nm,
                "wavelength_nm": wl,
                "order": settings.order,
                "pml_metal": settings.pml_metal,
                "divergence_deg": model.illumination.divergence_deg,
                **{k: v for k, v in r.items() if k != "wavelength_nm"},
            }
        )
        print(rows[-1], flush=True)
        write_csv(rows, cfg.out_csv)  # rewritten after every case
    print(f"wrote {cfg.out_csv}")


if __name__ == "__main__":
    main(AppConfig())
