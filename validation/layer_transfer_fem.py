"""Does the effect of a thin layer, computed in the small domain, transfer to the production domain?

For each substrate, EF(with the WS2 monolayer) / EF(without) at a fixed apex-to-substrate
distance (2 nm), in the small and in the production domain, with the FEM reference
(ters2d.fem). This is the check behind using the small domain for layer effects (reproduce/
ws2_on_au.py, examples/05); it was validated over 500-900 nm for Au (fem_tip_reference.csv,
groups ws2 and ws2_small) and is checked here for other substrates. CPU, ~10 min on 16 cores.
Requires NGSolve.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from ters2d import Layer, Model, Numerics, Sample, materials, write_csv
from ters2d.fem import FemSettings, solve


@dataclass
class AppConfig:
    substrates: dict = field(default_factory=lambda: {"Au": materials.AU, "Ag": materials.AG, "SiO2": materials.SIO2})
    wavelengths_nm: list = field(default_factory=lambda: [633.0])
    apex_to_substrate_nm: float = 2.0
    threads: int = 16
    out_csv: Path = Path("results") / platform.node() / "layer_transfer_fem" / "layer_transfer_fem.csv"


def main(cfg: AppConfig) -> None:
    layer = Layer(materials.WS2_MONOLAYER, 0.65)
    t = 2.0 / 3.0  # the layer snapped to one cell at 1.5 px/nm
    settings = FemSettings(threads=cfg.threads)
    rows = []
    for name, sub in cfg.substrates.items():
        for wl in cfg.wavelengths_nm:
            ratio = {}
            for domain, num in (("small", Numerics.small(resolution=1500)), ("production", Numerics.production(resolution=1500))):
                bare = solve(Model(sample=Sample(sub), gap_nm=cfg.apex_to_substrate_nm, numerics=num), wl, settings)
                lay = Model(sample=Sample(sub, (layer,)), gap_nm=cfg.apex_to_substrate_nm - t, numerics=num)
                with_layer = solve(lay, wl, settings)
                ratio[domain] = with_layer["EF_probe"] / bare["EF_probe"]
                rows.append(
                    {
                        "substrate": name,
                        "wavelength_nm": wl,
                        "domain": domain,
                        "EF_bare": bare["EF_probe"],
                        "EF_layer_sample": with_layer["EF_probe"],
                        "ratio": ratio[domain],
                    }
                )
            rows[-1]["small_vs_production"] = ratio["small"] / ratio["production"] - 1
            print(
                f"{name} {wl:.0f} nm: layer effect x{ratio['small']:.3f} (small) vs x{ratio['production']:.3f} (production), "
                f"{100 * rows[-1]['small_vs_production']:+.2f} %",
                flush=True,
            )
            write_csv(rows, cfg.out_csv)
    print(f"wrote {cfg.out_csv}")


if __name__ == "__main__":
    main(AppConfig())
