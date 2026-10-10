"""Reading run files and computing the outputs: enhancement factors and field maps (pure NumPy).

Each run writes ``wl<wavelength>nm.npz`` (complex DFT fields on the monitor grids) and a
``.json`` with the model, the layout and the run statistics. On every monitor grid
(prefix ``near_`` or ``full_``) the file stores the scattered field (``sEx``, ``sEy``,
``sHz``), the analytic tip-free field (``bEx`` ...) and the total field (``Ex`` ...), plus
the grid coordinates ``x``, ``y`` [um] (origin: the sample surface below the apex).

Definitions (all |E|^2 = |Ex|^2 + |Ey|^2 of the complex amplitude at the run wavelength):

* EF_probe = |E_tip(probe)|^2 / |E_ref(probe)|^2, E_ref the tip-free field at the same point;
  the probe is the gap centre unless ``Outputs.probe_nm`` says otherwise;
* EF_gap = mean |E_tip|^2 / mean |E_ref|^2 along the tip axis across the air gap;
* EF4_gap = mean |E_tip|^4 / mean |E_ref|^4 along the axis (the usual |E|^4 Raman
  estimate with equal enhancement at the laser and Stokes wavelengths), EF4_probe = EF_probe^2;
* EF_layer = mean |E|^2 / mean |E_ref|^2 along the axis inside the top layer (if any);
* I_probe_tip_over_inc, I_probe_ref_over_inc = |E|^2 at the probe relative to the incident
  beam alone (no sample) at its focus.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .background import GaussianBeam, Stack, beam_fields
from .layout import tip_shape


def interp2(x, y, field, x0, y0):
    """Bilinear interpolation of field[ix, iy] on the grid (x, y) at (x0, y0)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if not (x[0] <= x0 <= x[-1] and y[0] <= y0 <= y[-1]):
        raise ValueError(f"point ({x0}, {y0}) outside the monitor grid")
    i = int(np.clip(np.searchsorted(x, x0) - 1, 0, len(x) - 2))
    j = int(np.clip(np.searchsorted(y, y0) - 1, 0, len(y) - 2))
    tx = (x0 - x[i]) / (x[i + 1] - x[i])
    ty = (y0 - y[j]) / (y[j + 1] - y[j])
    return (
        (1 - tx) * (1 - ty) * field[i, j]
        + tx * (1 - ty) * field[i + 1, j]
        + (1 - tx) * ty * field[i, j + 1]
        + tx * ty * field[i + 1, j + 1]
    )


def intensity(data: dict, prefix: str = "near", kind: str = "") -> np.ndarray:
    """|E|^2 on a monitor grid; kind '' total, 'b' tip-free (reference), 's' scattered."""
    return np.abs(data[f"{prefix}_{kind}Ex"]) ** 2 + np.abs(data[f"{prefix}_{kind}Ey"]) ** 2


def axis_profile(data: dict, prefix: str = "near", kind: str = ""):
    """Complex (Ex, Ey) on the tip axis x = 0, interpolated between the adjacent pixel columns."""
    x = data[f"{prefix}_x"]
    i = int(np.searchsorted(x, 0.0)) - 1
    if not (x[i] < 0 < x[i + 1]):
        raise ValueError("monitor grid does not straddle x = 0")
    t = (0.0 - x[i]) / (x[i + 1] - x[i])
    ex = (1 - t) * data[f"{prefix}_{kind}Ex"][i] + t * data[f"{prefix}_{kind}Ex"][i + 1]
    ey = (1 - t) * data[f"{prefix}_{kind}Ey"][i] + t * data[f"{prefix}_{kind}Ey"][i + 1]
    return data[f"{prefix}_y"], ex, ey


def gap_metrics(data: dict, meta: dict) -> dict:
    """|E|^2 of the total and the reference field at the probe, along the gap and in the top layer."""
    lay = meta["layout"]
    x, y = data["near_x"], data["near_y"]
    px, py = lay["probe"]
    g, dx = lay["gap_um"], lay["dx_um"]
    out = {}
    for kind, tag in (("", "tip"), ("b", "ref")):
        ex = interp2(x, y, data[f"near_{kind}Ex"], px, py)
        ey = interp2(x, y, data[f"near_{kind}Ey"], px, py)
        out[f"E2_probe_{tag}"] = float(abs(ex) ** 2 + abs(ey) ** 2)
        ya, axa, aya = axis_profile(data, kind=kind)
        col = np.abs(axa) ** 2 + np.abs(aya) ** 2
        air = (ya > 0.25 * dx) & (ya < g - 0.25 * dx)
        if not air.any():
            raise ValueError("no monitor points inside the air gap")
        out[f"E2_gap_{tag}"] = float(col[air].mean())
        out[f"E4_gap_{tag}"] = float((col[air] ** 2).mean())
        if lay["layers"]:
            yb, yt = lay["layers"][-1]
            inlay = (ya > yb + 0.25 * dx) & (ya < yt - 0.25 * dx)
            if not inlay.any():
                raise ValueError("no monitor points inside the top layer")
            out[f"E2_layer_{tag}"] = float(col[inlay].mean())
    return out


def incident_focus_intensity(meta: dict) -> float:
    """|E|^2 of the free-space incident beam (no sample) at its focus."""
    ill = meta["model"]["illumination"]
    fx, fy = meta["layout"]["focus"]
    beam = GaussianBeam(meta["wavelength_nm"] * 1e-3, ill["incidence_deg"], ill["divergence_deg"], (fx, fy), ill["n_waves"])
    _, ex, ey = beam_fields(beam, Stack.isotropic(1.0), np.array([fx]), np.array([fy]), part="incident")
    return float(abs(ex[0]) ** 2 + abs(ey[0]) ** 2)


def enhancement_metrics(data: dict, meta: dict) -> dict:
    """All scalar outputs of one run (see the module docstring)."""
    m = gap_metrics(data, meta)
    out = {
        "EF_probe": m["E2_probe_tip"] / m["E2_probe_ref"],
        "EF_gap": m["E2_gap_tip"] / m["E2_gap_ref"],
        "EF4_gap": m["E4_gap_tip"] / m["E4_gap_ref"],
    }
    out["EF4_probe"] = out["EF_probe"] ** 2
    if "E2_layer_tip" in m:
        out["EF_layer"] = m["E2_layer_tip"] / m["E2_layer_ref"]
    out.update(m)
    i0 = incident_focus_intensity(meta)
    out["I_probe_tip_over_inc"] = m["E2_probe_tip"] / i0
    out["I_probe_ref_over_inc"] = m["E2_probe_ref"] / i0
    return out


@dataclass
class Result:
    """Outputs of one wavelength of one geometry.

    wavelength_nm; metrics: dict of the scalar outputs (EF_probe, EF_gap, EF4_gap,
    EF4_probe, EF_layer if a layer exists, the underlying |E|^2 values, I_probe_*_over_inc);
    file: the .npz run file; meta: its JSON metadata.
    """

    wavelength_nm: float
    metrics: dict
    file: Path
    meta: dict

    @property
    def ef_probe(self) -> float:
        return self.metrics["EF_probe"]

    @property
    def ef_gap(self) -> float:
        return self.metrics["EF_gap"]

    def row(self) -> dict:
        return {"wavelength_nm": self.wavelength_nm, **self.metrics}

    def fields(self) -> dict:
        return dict(np.load(self.file))

    def e2_at(self, x_nm: float, y_nm: float, kind: str = "") -> float:
        """|E|^2 at a point of the near monitor [nm]; kind '' total, 'b' reference, 's' scattered."""
        d = self.fields()
        ex = interp2(d["near_x"], d["near_y"], d[f"near_{kind}Ex"], x_nm * 1e-3, y_nm * 1e-3)
        ey = interp2(d["near_x"], d["near_y"], d[f"near_{kind}Ey"], x_nm * 1e-3, y_nm * 1e-3)
        return float(abs(ex) ** 2 + abs(ey) ** 2)


def load_result(npz_path: Path) -> Result:
    npz_path = Path(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not meta["dft_converged"]:
        raise ValueError(f"{npz_path}: DFT fields did not converge")
    data = dict(np.load(npz_path))
    return Result(meta["wavelength_nm"], enhancement_metrics(data, meta), npz_path, meta)


def write_csv(rows: list[dict], path: Path) -> Path:
    """Write dict rows (e.g. ``[r.row() for r in results]``) to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path: Path) -> dict:
    """CSV -> dict of columns (numeric columns as float arrays)."""
    with Path(path).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[k] = vals
    return out


@dataclass
class FieldMap:
    """|E|^2 / |E_ref(probe)|^2 on a grid (NaN inside the tip and the substrate).

    x_nm, y_nm: grid [nm]; e2_rel[ix, iy]: |E|^2 normalized by the reference intensity at
    the probe, so that e2_rel at the probe equals EF_probe; outline_nm: tip polygon;
    layers_nm: (y_bottom, y_top) of each layer; meta: wavelength_nm, label, EF_probe.
    """

    x_nm: np.ndarray
    y_nm: np.ndarray
    e2_rel: np.ndarray
    outline_nm: np.ndarray
    layers_nm: np.ndarray
    meta: dict

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x_nm=self.x_nm.astype(np.float32),
            y_nm=self.y_nm.astype(np.float32),
            e2_rel=self.e2_rel.astype(np.float32),
            outline_nm=self.outline_nm.astype(np.float32),
            layers_nm=np.asarray(self.layers_nm, dtype=np.float32).reshape(-1, 2),
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "FieldMap":
        d = np.load(path)
        return cls(d["x_nm"], d["y_nm"], d["e2_rel"], d["outline_nm"], d["layers_nm"], json.loads(str(d["meta"])))


def field_map(result: Result, region: str = "near", stride: int = 1) -> FieldMap:
    """Field map of a run: region 'near' (around the apex, full resolution) or 'full'
    (the whole PML-free region, stored only with ``Outputs(full_map=True)``)."""
    d = result.fields()
    if f"{region}_x" not in d:
        raise ValueError(f"run file has no '{region}' monitor (set Outputs(full_map=True) for 'full')")
    meta = result.meta
    norm = result.metrics["E2_probe_ref"]
    x, y = d[f"{region}_x"][::stride], d[f"{region}_y"][::stride]
    e2 = intensity(d, region)[::stride, ::stride] / norm
    tip = meta["model"]["tip"]
    lay = meta["layout"]
    t = tip_shape(tip["apex_radius_nm"] * 1e-3, tip["half_angle_deg"], lay["gap_um"])
    X, Y = np.meshgrid(x, y, indexing="ij")
    sub_top = lay["layers"][0][0] if lay["layers"] else 0.0
    e2 = np.where(t.contains(X, Y) | (Y < sub_top), np.nan, e2)
    return FieldMap(
        x * 1e3,
        y * 1e3,
        e2,
        t.outline(float(y.max())) * 1e3,
        np.asarray(lay["layers"], dtype=float).reshape(-1, 2) * 1e3,
        {"wavelength_nm": result.wavelength_nm, "label": lay["label"], "EF_probe": result.ef_probe},
    )
