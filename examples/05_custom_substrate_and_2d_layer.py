"""Your own sample: the effect of a 2D material (uniaxial) on different substrates.

Samples are a semi-infinite substrate plus layers listed from the substrate upward:

    Sample(substrate, layers=(Layer(material, thickness_nm), ...))

* any Material or UniaxialMaterial (optic axis along the layer normal) can be a layer or
  the substrate; ``Material.from_meep("Ag")`` takes an entry of Meep's material library,
  ``Material.constant("hBN", index=2.1)`` a lossless dielectric, and
  examples/06_add_material_from_nk.py fits your own n, k data;
* every layer is snapped to whole grid cells (within 5 %): a 0.65 nm monolayer needs
  1.5 pixels/nm (0.667 nm, 1 cell) or 3 pixels/nm (2 cells);
* ``gap_nm`` is the AIR gap above the top layer; here the apex-to-substrate distance is kept at
  2 nm, so the air gap with the monolayer is 2 - 2/3 nm;
* the boundary treatment follows the materials: a metallic substrate gets no bottom PML and
  side plasmon absorbers, a dielectric one a bottom PML (docs/model.md).

For each substrate the script runs the bare substrate and the substrate with the monolayer and
reports EF(with layer) / EF(without): the effect of the layer. That ratio is the one quantity
the cheap small domain gives correctly (FEM, small vs production domain: within 1.3 % for gold
over 500-900 nm, and at 633 nm +0.4 % for silver and +0.6 % for SiO2;
validation/layer_transfer_fem.py). Absolute EF values and comparisons between substrates need
Numerics.production. For a new substrate or wavelength, repeat that FEM comparison first (a
few minutes on a CPU).

Cost: six runs at 1.5 pixels/nm in the small domain, ~2-8 min each on a GPU (gpmeep), far
longer on one CPU core (use MPI: mpirun -np 16 python ...).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Layer, Material, Model, Numerics, Sample, materials, simulate, write_csv
from ters2d.layout import snapped_thickness


@dataclass
class AppConfig:
    substrates: dict = field(default_factory=lambda: {"Au": materials.AU, "Ag": Material.from_meep("Ag"), "SiO2": materials.SIO2})
    layer: Layer = field(default_factory=lambda: Layer(materials.WS2_MONOLAYER, 0.65))
    apex_to_substrate_nm: float = 2.0  # air gap = this minus the (snapped) layer
    numerics: Numerics = field(default_factory=lambda: Numerics.small(resolution=1500))
    wavelength_nm: float = 633.0
    run_root: Path = Path("runs") / platform.node() / "05_custom_sample"
    out_csv: Path = Path("results") / platform.node() / "05_custom_sample" / "layer_effect.csv"


def main(cfg: AppConfig) -> None:
    t_layer = 1e3 * snapped_thickness(cfg.layer.thickness_nm * 1e-3, 1e-3 * cfg.numerics.dx_nm)
    name = cfg.layer.material.name
    rows = []
    for sub_name, substrate in cfg.substrates.items():
        bare = Model(sample=Sample(substrate), gap_nm=cfg.apex_to_substrate_nm, numerics=cfg.numerics)
        layered = Model(sample=Sample(substrate, (cfg.layer,)), gap_nm=cfg.apex_to_substrate_nm - t_layer, numerics=cfg.numerics)
        rb = simulate(bare, cfg.wavelength_nm, cfg.run_root / sub_name)
        rl = simulate(layered, cfg.wavelength_nm, cfg.run_root / f"{name}-{sub_name}")
        rows.append(
            {
                "substrate": sub_name,
                "layer_nm": t_layer,
                "air_gap_nm": layered.gap_nm,
                "wavelength_nm": cfg.wavelength_nm,
                "EF_bare": rb.ef_probe,
                "EF_layered": rl.ef_probe,
                "layer_effect": rl.ef_probe / rb.ef_probe,
                "EF_in_layer": rl.metrics["EF_layer"],
            }
        )
        print(
            f"{sub_name:5s}: EF {rb.ef_probe:7.1f} -> {rl.ef_probe:7.1f} with {name} (x{rows[-1]['layer_effect']:.3f}), "
            f"EF inside the layer {rl.metrics['EF_layer']:.1f}  [small-domain absolute values]"
        )
    print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
