"""Your own sample: a 2D material (uniaxial) on different substrates, with a reference without it.

Samples are a semi-infinite substrate plus layers listed from the substrate upward:

    Sample(substrate, layers=(Layer(material, thickness_nm), ...))

* any Material or UniaxialMaterial (optic axis along the layer normal) can be a layer or
  the substrate; ``Material.from_meep("Ag")`` takes an entry of Meep's material library,
  ``Material.constant("hBN", index=2.1)`` a lossless dielectric, and
  examples/06_add_material_from_nk.py fits your own n, k data;
* every layer is snapped to whole grid cells (within 5 %): a 0.65 nm monolayer needs
  1.5 pixels/nm (0.667 nm, 1 cell) or 3 pixels/nm (2 cells);
* ``gap_nm`` is the AIR gap above the top layer, so a 2 nm metal-to-metal distance with the
  monolayer is gap_nm = 2 - 2/3 nm at 1.5 pixels/nm;
* the boundary treatment follows the materials: a metallic substrate gets no bottom PML
  and side plasmon absorbers, a dielectric one a bottom PML (see docs/model.md).

Cost: four runs at 1.5 pixels/nm in the small domain, ~3-10 min each on a GPU (gpmeep), far
longer on one CPU core (use MPI: mpirun -np 16 python ...). Compare samples by ratios in the
small domain (docs/setting_up_your_simulation.md explains why).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Layer, Material, Model, Numerics, Sample, materials, simulate, write_csv


def _samples() -> dict:
    ws2 = Layer(materials.WS2_MONOLAYER, 0.65)
    return {
        "Au": Sample(materials.AU),
        "WS2/Au": Sample(materials.AU, (ws2,)),
        "WS2/Ag": Sample(Material.from_meep("Ag"), (ws2,)),
        "WS2/SiO2": Sample(materials.SIO2, (ws2,)),
    }


@dataclass
class AppConfig:
    samples: dict = field(default_factory=_samples)
    metal_metal_distance_nm: float = 2.0  # apex to the substrate surface; the air gap is this minus the layers
    numerics: Numerics = field(default_factory=lambda: Numerics.small(resolution=1500))
    wavelength_nm: float = 633.0
    run_root: Path = Path("runs") / platform.node() / "05_custom_sample"
    out_csv: Path = Path("results") / platform.node() / "05_custom_sample" / "samples.csv"


def main(cfg: AppConfig) -> None:
    from ters2d.layout import snapped_thickness

    dx = 1e-3 * cfg.numerics.dx_nm
    rows = []
    for name, sample in cfg.samples.items():
        layers_nm = sum(1e3 * snapped_thickness(lay.thickness_nm * 1e-3, dx) for lay in sample.layers)
        model = Model(sample=sample, gap_nm=cfg.metal_metal_distance_nm - layers_nm, numerics=cfg.numerics)
        r = simulate(model, cfg.wavelength_nm, cfg.run_root / name.replace("/", "-"))
        rows.append({"sample": name, "air_gap_nm": model.gap_nm, "layers_nm": layers_nm, **r.row()})
        extra = f", EF in the layer {r.metrics['EF_layer']:.1f}" if "EF_layer" in r.metrics else ""
        print(f"{name:9s} air gap {model.gap_nm:.3f} nm: EF {r.ef_probe:.1f}{extra}")
    print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
