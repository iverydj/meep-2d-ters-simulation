"""Meep driver: ``simulate`` (one wavelength), ``spectrum`` (one geometry, many wavelengths),
``sweep`` (one parameter varied).

Scattered-field formulation (``sources.py``): Meep computes the field scattered by the
tip, E_s; the analytic tip-free field E_bg is added afterwards. The structure is built
once per geometry; for each wavelength the fields are reset, the tip sources are rebuilt
for that wavelength (exact beam angles), fresh DFT monitors are attached, and the run
continues until the DFT fields at that wavelength have converged. Works with upstream
Meep (CPU, FP64, optionally MPI) and gpmeep (CUDA, FP32: ``MEEP_GPU_BACKEND=cuda``).

Each wavelength writes ``wl<wavelength>nm.npz`` + ``.json`` into ``out_dir``. Existing
files are reused if they were computed with the same model (so an interrupted spectrum
resumes); a file computed with different settings raises an error.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._version import __version__
from .background import beam_fields, beam_fields_grid
from .layout import Layout, build_layout
from .model import Model
from .results import Result, load_result
from .sources import background_beam, tip_sources


def _print(msg: str) -> None:
    print(msg, flush=True)


def output_file(out_dir: Path, wavelength_nm: float, suffix: str = ".npz") -> Path:
    return Path(out_dir) / f"wl{wavelength_nm:.1f}nm{suffix}"


def model_record(model: Model) -> dict:
    """JSON-compatible record of the model (used to decide whether a run file can be reused)."""
    return json.loads(json.dumps(asdict(model)))


def backend_info() -> dict:
    import meep as mp

    info = {"meep_version": mp.__version__, "mpi_processes": int(mp.count_processors())}
    gpu = getattr(mp, "gpu", None)
    info["gpmeep"] = gpu is not None
    info["active_backend"] = str(gpu.active_backend) if gpu is not None else "cpu"
    return info


def shaft_illumination(layout: Layout, wavelength_nm: float) -> float:
    """max |E_bg|^2 on the shaft faces at the start of the drive ramp, relative to |E_bg|^2 at the probe.

    Small values (<~1e-3) mean the beam no longer reaches the shaft where its drive is
    ramped down, so the result does not depend on where the shaft is truncated.
    """
    beam = background_beam(layout, wavelength_nm)
    stack = layout.stack(wavelength_nm)
    y0 = layout.drive_ramp[0]
    hw = float(layout.tip.half_width(y0)) + layout.dx
    _, ex, ey = beam_fields(beam, stack, np.array([-hw, hw, layout.probe[0]]), np.array([y0, y0, layout.probe[1]]))
    e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    return float(max(e2[0], e2[1]) / e2[2])


def _monitor_regions(layout: Layout) -> dict:
    """name -> (x0, x1, y0, y1) [um]."""
    x0, x1, y0, y1 = (v * 1e-3 for v in layout.model.outputs.near_box_nm)
    regions = {"near": (x0, x1, y0, y1)}
    if layout.model.outputs.full_map:
        (px0, py0), (px1, py1) = layout.phys_lo, layout.phys_hi
        regions["full"] = (px0, px1, py0, py1)
    return regions


def _add_monitors(sim, regions: dict, f0: float) -> dict:
    import meep as mp

    return {
        name: sim.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Hz], [f0], center=mp.Vector3(0.5 * (x0 + x1), 0.5 * (y0 + y1)), size=mp.Vector3(x1 - x0, y1 - y0)
        )
        for name, (x0, x1, y0, y1) in regions.items()
    }


def _collect(sim, mons: dict, regions: dict, layout: Layout) -> dict:
    """Scattered-field DFT arrays cropped to the requested region (Meep pads one pixel)."""
    import meep as mp

    out = {}
    for name, m in mons.items():
        x0, x1, y0, y1 = regions[name]
        x, y, _, _ = sim.get_array_metadata(dft_cell=m)
        x, y = np.asarray(x), np.asarray(y)
        tol = 0.01 * layout.dx
        ix = np.where((x >= x0 - tol) & (x <= x1 + tol))[0]
        iy = np.where((y >= y0 - tol) & (y <= y1 + tol))[0]
        stride = layout.model.outputs.full_map_stride if name == "full" else 1
        ix, iy = ix[::stride], iy[::stride]
        out[f"{name}_x"], out[f"{name}_y"] = x[ix], y[iy]
        for c, comp in (("Ex", mp.Ex), ("Ey", mp.Ey), ("Hz", mp.Hz)):
            arr = np.asarray(sim.get_dft_array(m, comp, 0)).reshape(len(x), len(y))[np.ix_(ix, iy)]
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"monitor '{name}' {c}: non-finite DFT fields (the simulation diverged)")
            if np.any((arr == 0).all(axis=0)) or np.any((arr == 0).all(axis=1)):
                raise RuntimeError(f"monitor '{name}' {c}: an all-zero row/column inside the region")
            out[f"{name}_s{c}"] = arr.astype(np.complex64)
    return out


def _add_background(data: dict, layout: Layout, wavelength_nm: float, scale: complex) -> None:
    """Store E_bg and the total field E_bg + E_s / scale on every monitor grid."""
    beam = background_beam(layout, wavelength_nm)
    stack = layout.stack(wavelength_nm)
    for name in [k[:-2] for k in list(data) if k.endswith("_x")]:
        bg = beam_fields_grid(beam, stack, data[f"{name}_x"], data[f"{name}_y"])
        for c, b in zip(("Hz", "Ex", "Ey"), bg):
            data[f"{name}_b{c}"] = b.astype(np.complex64)
            data[f"{name}_{c}"] = (b + data[f"{name}_s{c}"] / scale).astype(np.complex64)


def _layout_record(layout: Layout) -> dict:
    return {
        "label": layout.model.sample.label,
        "dx_um": layout.dx,
        "cell_lo": list(layout.cell_lo),
        "cell_hi": list(layout.cell_hi),
        "phys_lo": list(layout.phys_lo),
        "phys_hi": list(layout.phys_hi),
        "layers": [list(lay) for lay in layout.layers],
        "gap_um": layout.gap,
        "probe": list(layout.probe),
        "focus": list(layout.focus),
        "n_cells": layout.n_cells,
        "substrate_metallic": layout.substrate_metallic,
        "bottom_pml": layout.bottom_pml,
        "side_absorbers": layout.side_absorbers,
        "boundary_layers": [list(b) for b in layout.boundary_layers],
    }


def _check_existing(out_dir: Path, model: Model, wavelengths) -> list[float]:
    todo = []
    rec = model_record(model)
    for w in wavelengths:
        f = output_file(out_dir, w)
        if not f.exists():
            todo.append(w)
            continue
        old = json.loads(output_file(out_dir, w, ".json").read_text(encoding="utf-8"))
        if old["model"] != rec or old.get("ters2d_version") != __version__:
            raise ValueError(f"{f} exists but was computed with other settings or another ters2d version; use another out_dir")
    return todo


def check_shaft_illumination(layout: Layout, wavelengths) -> dict:
    """shaft_illumination per wavelength; raises if one exceeds Numerics.max_shaft_illumination."""
    vals = {w: shaft_illumination(layout, w) for w in wavelengths}
    limit = layout.model.numerics.max_shaft_illumination
    bad = {w: v for w, v in vals.items() if limit is not None and v > limit}
    if bad:
        raise ValueError(
            f"the beam still reaches the shaft where its drive is cut (|E_bg|^2 at the ramp / at the probe = "
            f"{', '.join(f'{v:.1e} at {w:g} nm' for w, v in bad.items())} > {limit:g}): move y_max_um, "
            "shaft_absorber_start_um and drive_ramp_um up together, or set Numerics.max_shaft_illumination=None to accept it"
        )
    return vals


def spectrum(model: Model, wavelengths_nm, out_dir: Path, log=_print) -> list[Result]:
    """Run one tip geometry at several wavelengths (structure built once); one Result per wavelength."""
    wls = [float(w) for w in wavelengths_nm]
    if len(set(wls)) != len(wls):
        raise ValueError("duplicate wavelengths")
    layout = build_layout(model, wls)
    check_shaft_illumination(layout, wls)
    todo = _check_existing(out_dir, model, wls)
    if todo:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        _run(layout, todo, Path(out_dir), log)
    return [load_result(output_file(out_dir, w)) for w in wls]


def simulate(model: Model, wavelength_nm: float, out_dir: Path, log=_print) -> Result:
    """Run one geometry at one wavelength."""
    return spectrum(model, [wavelength_nm], out_dir, log)[0]


def sweep(model: Model, parameter: str, values, wavelengths_nm, out_root: Path, log=_print) -> list[tuple[object, list[Result]]]:
    """Vary one (possibly nested) model field, e.g. ``"gap_nm"`` or ``"tip.apex_radius_nm"``.

    Each value gets its own run directory ``out_root/<parameter>=<value>``.
    Returns [(value, [Result per wavelength]), ...].
    """
    out = []
    for v in values:
        m = model.with_value(parameter, v)
        tag = f"{parameter}={v:g}" if isinstance(v, (int, float)) else f"{parameter}={v}"
        out.append((v, spectrum(m, wavelengths_nm, Path(out_root) / tag, log)))
    return out


def _run(layout: Layout, wavelengths, out_dir: Path, log) -> None:
    import meep as mp

    model = layout.model
    num = model.numerics
    regions = _monitor_regions(layout)
    (px0, py0), (px1, py1) = layout.phys_lo, layout.phys_hi
    transit = float(np.hypot(px1 - px0, py1 - py0))  # let scattered light cross the free region once
    from .meep_model import make_simulation

    t_init = time.time()
    sim = make_simulation(layout)
    first = True
    for wl in wavelengths:
        f0 = 1e3 / wl
        if not first:
            sim.restart_fields()
            sim.clear_dft_monitors()
        src = mp.GaussianSource(f0, fwidth=num.pulse_fwidth_rel * f0, cutoff=num.pulse_cutoff)
        srcs, n_src = tip_sources(layout, wl, src)
        sim.change_sources(srcs)
        mons = _add_monitors(sim, regions, f0)
        init_s = 0.0
        if first:
            sim.init_sim()
            init_s = time.time() - t_init
        # DFT of a real-field run driven by Re[A s(t)]: E_s(f0) = (response to A) * S(f0) / 2
        scale = complex(src.fourier_transform(f0)) * (0.5 if sim.fields.is_real else 1.0)
        t0 = time.time()
        state = {"timeout": False}
        decay = mp.stop_when_dft_decayed(tol=num.dft_decay_tol)
        t_src = sim.fields.last_source_time()
        t_min = t_src + max(num.min_time_after_sources, transit)

        def stop(s, decay=decay, t_src=t_src, t_min=t_min):
            if s.meep_time() > t_src + num.max_time_after_sources:
                state["timeout"] = True
                return True
            return decay(s) and s.meep_time() >= t_min

        sim.run(until=stop)
        wall = time.time() - t0
        data = _collect(sim, mons, regions, layout)
        _add_background(data, layout, wl, scale)
        stack = layout.stack(wl)
        if state["timeout"]:  # nothing is written, so a re-run with other settings starts cleanly
            raise RuntimeError(f"{wl} nm: DFT fields did not converge within max_time_after_sources (t = {sim.meep_time():.1f})")
        meta = {
            "ters2d_version": __version__,
            "wavelength_nm": wl,
            "frequency_um_inv": f0,
            "model": model_record(model),
            "layout": _layout_record(layout),
            "stack": {
                "substrate": [[z.real, z.imag] for z in stack.substrate],
                "layers_top_down": [[[ex.real, ex.imag], [ey.real, ey.imag], t] for ex, ey, t in stack.layers],
            },
            "eps_tip": [complex(model.tip.material.epsilon(wl)).real, complex(model.tip.material.epsilon(wl)).imag],
            "shaft_illumination_at_ramp": shaft_illumination(layout, wl),
            "source_scale": [scale.real, scale.imag],
            "n_source_points": n_src,
            "meep_time": float(sim.meep_time()),
            "source_end_time": float(t_src),
            "timesteps": int(round(sim.meep_time() / sim.fields.dt)),
            "dft_converged": not state["timeout"],
            "wall_s": wall,
            "init_s": init_s,
            "backend": backend_info(),
        }
        if mp.am_master():
            np.savez_compressed(output_file(out_dir, wl), **data)
            output_file(out_dir, wl, ".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        mp.all_wait()  # MPI runs: other ranks may read the file next
        log(
            f"[{model.sample.label} gap {model.gap_nm:g} nm R {model.tip.apex_radius_nm:g} nm] wl={wl:.1f} nm "
            f"steps={meta['timesteps']} wall={wall:.0f}s init={init_s:.0f}s source points={n_src} "
            f"shaft illumination at ramp={meta['shaft_illumination_at_ramp']:.1e} converged={meta['dft_converged']}"
        )
        first = False
