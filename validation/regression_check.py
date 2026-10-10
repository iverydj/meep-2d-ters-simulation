"""Re-run a few reference cases and compare them with the committed result tables.

Checks that the code in this repository reproduces the numbers behind docs/results.md:
the WS2/Au and Au/Au pair of the small domain at 1.5 pixels/nm, the production Au/Au case
at three wavelengths and the production SiO2 case at 633 nm. Prints and writes the
relative differences of EF_probe (and of the WS2/Au to Au/Au ratio).

GPU (gpmeep): MEEP_GPU_BACKEND=cuda python validation/regression_check.py  (~1 GPU-hour on
an RTX A6000). On CPU Meep the production cases take ~1 h each with 32 MPI processes.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d import Layer, Model, Numerics, Sample, read_csv, spectrum, write_csv
from ters2d.materials import AU, SIO2, WS2_MONOLAYER

DATA = Path(__file__).resolve().parents[1] / "docs" / "figures" / "data"


@dataclass
class AppConfig:
    run_root: Path = Path("runs") / platform.node() / "regression_check"
    out_csv: Path = Path("results") / platform.node() / "regression_check" / "regression_check.csv"
    au_wavelengths_nm: list = field(default_factory=lambda: [633.0, 525.0, 900.0])
    sio2_wavelengths_nm: list = field(default_factory=lambda: [633.0])
    ws2_wavelengths_nm: list = field(default_factory=lambda: [633.0])


def _reference(csv_name: str, column: str, wavelength_nm: float) -> float:
    d = read_csv(DATA / csv_name)
    m = np.isclose(d["wavelength_nm"], wavelength_nm)
    if m.sum() != 1:
        raise ValueError(f"{csv_name}: no unique row for {wavelength_nm} nm")
    return float(d[column][m][0])


def main(cfg: AppConfig) -> None:
    rows = []

    def compare(case, results, csv_name, column):
        for r in results:
            ref = _reference(csv_name, column, r.wavelength_nm)
            rows.append(
                {
                    "case": case,
                    "wavelength_nm": r.wavelength_nm,
                    "EF_probe_now": r.ef_probe,
                    "EF_probe_reference": ref,
                    "rel_diff": r.ef_probe / ref - 1,
                    "timesteps": r.meta["timesteps"],
                    "time_stepping_s": r.meta["wall_s"],
                    "setup_s": r.meta["init_s"],
                    "backend": r.meta["backend"]["active_backend"],
                }
            )
            print(
                f"{case} {r.wavelength_nm:.0f} nm: EF {r.ef_probe:.6f} vs reference {ref:.6f} ({rows[-1]['rel_diff']:+.2e})",
                flush=True,
            )

    # WS2: 0.65 nm snaps to 0.667 nm (1 cell) at 1.5 pixels/nm; metal-metal distance 2 nm = 0.667 nm + 1.333 nm air
    num = Numerics.small(resolution=1500)
    au_s = spectrum(Model(sample=Sample(AU), gap_nm=2.0, numerics=num), cfg.ws2_wavelengths_nm, cfg.run_root / "small_Au_res1500")
    ws2 = Model(sample=Sample(AU, (Layer(WS2_MONOLAYER, 0.65),)), gap_nm=2.0 - 2.0 / 3.0, numerics=num)
    ws2_s = spectrum(ws2, cfg.ws2_wavelengths_nm, cfg.run_root / "small_WS2-Au_res1500")
    compare("small Au, 1.5 px/nm", au_s, "ws2_on_au_fdtd.csv", "EF_probe_Au")
    compare("small WS2/Au, 1.5 px/nm", ws2_s, "ws2_on_au_fdtd.csv", "EF_probe_WS2")
    for a, w in zip(au_s, ws2_s):
        ref = _reference("ws2_on_au_fdtd.csv", "ratio_EF_probe", a.wavelength_nm)
        ratio = w.ef_probe / a.ef_probe
        rows.append(
            {
                "case": "ratio WS2/Au to Au/Au",
                "wavelength_nm": a.wavelength_nm,
                "EF_probe_now": ratio,
                "EF_probe_reference": ref,
                "rel_diff": ratio / ref - 1,
            }
        )
        print(f"ratio {a.wavelength_nm:.0f} nm: {ratio:.6f} vs reference {ref:.6f} ({ratio / ref - 1:+.2e})", flush=True)
    au = spectrum(Model(sample=Sample(AU)), cfg.au_wavelengths_nm, cfg.run_root / "production_Au")
    compare("production Au", au, "spectrum_Au_fdtd.csv", "EF_probe")
    sio2 = spectrum(Model(sample=Sample(SIO2)), cfg.sio2_wavelengths_nm, cfg.run_root / "production_SiO2")
    compare("production SiO2", sio2, "spectrum_SiO2_fdtd.csv", "EF_probe")

    write_csv(rows, cfg.out_csv)
    print(f"wrote {cfg.out_csv}")


if __name__ == "__main__":
    main(AppConfig())
