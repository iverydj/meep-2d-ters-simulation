# Outputs

## What a run returns

`simulate`, `spectrum` and `sweep` return `Result` objects (one per wavelength):

```python
r = simulate(model, 633.0, "runs/demo")
r.ef_probe            # EF at the probe point (default: the gap centre)
r.metrics["EF_gap"]   # every scalar output, see the table below
r.meta                # the run's JSON metadata (model, layout, timing, backend)
r.fields()            # the stored complex DFT fields (dict of arrays)
r.e2_at(10.0, 1.0)    # |E|^2 at (x, y) = (10 nm, 1 nm), anywhere in the near box
r.row()               # dict for a CSV row; write_csv([r.row() for r in results], path)
```

## Scalar outputs

All intensities are |E|² = |Ex|² + |Ey|² of the complex field amplitude at the run
wavelength. "Reference" is the tip-free field at the same point (the analytic E_bg), so a
value of 1 means "no enhancement".

| key | definition |
|---|---|
| `EF_probe` | \|E_tip\|² / \|E_ref\|² at the probe point (default: gap centre, x = 0, y = gap/2) |
| `EF_gap` | mean \|E_tip\|² / mean \|E_ref\|² along the tip axis across the air gap (points closer than a quarter cell to either surface excluded) |
| `EF4_gap` | mean \|E_tip\|⁴ / mean \|E_ref\|⁴ along the axis: the usual \|E\|⁴ Raman estimate, assuming equal enhancement at the laser and Stokes wavelengths |
| `EF4_probe` | `EF_probe`² |
| `EF_layer` | mean \|E\|² / mean \|E_ref\|² along the axis inside the top layer (only with layers): the field that drives the Raman scattering of a 2D material itself |
| `I_probe_tip_over_inc` | \|E_tip\|² at the probe divided by \|E\|² of the incident beam alone (no sample) at its focus |
| `I_probe_ref_over_inc` | the same for the tip-free field: what the sample alone does (≈ 1.96 above gold at 60°, 633 nm; 0.89 above SiO2, close to the Brewster angle) |
| `E2_probe_tip`, `E2_probe_ref`, `E2_gap_*`, `E4_gap_*`, `E2_layer_*` | the underlying intensities, in the beam's units |

`EF` (relative to the tip-free field) isolates what the tip does. `I_..._over_inc` relative
to the incident beam also contains what the sample does to the beam (for example the
constructive interference of the reflected beam above gold), which matters when comparing
substrates.

Raman with a Stokes shift: `ters2d.analysis.raman_enhancement(wavelength_nm, ef, laser_nm,
shift_cm1)` returns EF(laser) × EF(Stokes) interpolated from a computed EF spectrum.

## Field maps

`field_map(result)` returns a `FieldMap`: |E|² / |E_ref(probe)|² on the near-field box (by
default 300 × 130 nm around the apex at full resolution), NaN inside the tip and the
substrate, plus the tip outline and the layer positions; at the probe its value is EF.
`plot_field_map` draws it on a log scale. With `Outputs(full_map=True)` every run also stores
the whole PML-free region (down-sampled by `full_map_stride`), available as
`field_map(result, "full")`.

![field map](figures/fieldmap_Au_633nm.png)

## Files

Each wavelength writes two files into the run directory:

* `wl633.0nm.npz`: complex64 arrays on each monitor grid, with the prefix `near_` (and
  `full_` with `full_map=True`): `x`, `y` (grid coordinates in µm, origin at the sample
  surface below the apex), the scattered field `sEx`, `sEy`, `sHz` (Meep), the tip-free field
  `bEx`, `bEy`, `bHz` (analytic) and the total field `Ex`, `Ey`, `Hz`. The amplitudes are in
  the arbitrary units of the beam's plane-wave weights (`background.GaussianBeam`; |E|² of the
  incident beam at its focus is ≈ 0.18 for the default beam), so only ratios are physical.
* `wl633.0nm.json`: the complete model (`asdict(model)`), the layout (cell, free region,
  layers, gap, probe, boundary choices), the permittivities used, the run statistics
  (`timesteps`, `meep_time`, `wall_s` time stepping, `init_s` structure set-up, which is
  only paid by the first wavelength of a geometry), `dft_converged`,
  `shaft_illumination_at_ramp` and the backend (Meep version, MPI processes, gpmeep CUDA or
  CPU).

A run directory is reused when the stored model equals the requested one, so an interrupted
spectrum resumes; a file computed with other settings raises an error instead of being
mixed in. Load any run with `load_result(path_to_npz)`.

The examples and reproduction scripts put raw runs under `runs/<hostname>/<run-id>/` and
tables and figures under `results/<hostname>/<run-id>/` (both are git-ignored).

## CSV tables of the reference results

The tables in `docs/figures/data/` have one row per run with the keys above (columns of
`Result.row()`), plus sweep parameters. For the WS2 tables, `gap_nm` is the apex-to-gold
distance (2 nm = 0.667 nm WS2 + 1.333 nm air). `fem_tip_reference.csv` has the FEM values
(`EF_probe`, `I_probe_tip_over_inc`, `EF_layer`) of every case with its `group` (see
`validation/fem_tip_reference.py`).
