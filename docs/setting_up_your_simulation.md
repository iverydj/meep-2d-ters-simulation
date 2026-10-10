# Setting up your simulation

This guide walks through adapting the model to your own TERS case: tip, gap, sample, 2D
layer, materials, illumination, outputs and numerical settings, then how to check that the
answer is converged and what it costs. The physics behind each choice is in
[model.md](model.md).

## 1. The workflow

```python
from ters2d import Illumination, Layer, Model, Numerics, Outputs, Sample, Tip, materials, simulate, spectrum, sweep

model = Model(
    tip=Tip(apex_radius_nm=20, half_angle_deg=15, material=materials.AU),
    sample=Sample(materials.AU, layers=(Layer(materials.WS2_MONOLAYER, 0.65),)),
    gap_nm=4 / 3,                                   # air gap above the top layer
    illumination=Illumination(incidence_deg=60, divergence_deg=14),
    numerics=Numerics.small(resolution=1500),       # or Numerics.production(...)
    outputs=Outputs(),                              # probe at the gap centre
)
r = simulate(model, 633.0, "runs/my_case")          # one wavelength
rs = spectrum(model, [550, 600, 633, 700], "runs/my_case")      # one geometry, many wavelengths
sw = sweep(model, "tip.apex_radius_nm", [10, 20, 40], [633], "runs/my_radius")
```

A `Model` is immutable; derive variants with `model.with_value("gap_nm", 3.0)` or
`dataclasses.replace`. Everything except `simulate`/`spectrum`/`sweep` is pure Python, so a
model can be built and checked (`ters2d.layout.build_layout(model, wavelengths)`) without
Meep. The scripts in `examples/` are the same steps as runnable templates, each with an
`AppConfig` at the top.

## 2. Tip

`Tip(apex_radius_nm, half_angle_deg, material)`: a circular apex with a wedge tangent to it.

* Any isotropic `Material` works as the tip material (e.g. `Material.from_meep("Ag")` for a
  silver tip). A dispersive tip is replaced by the conductor inside the top PML.
* A larger radius or opening angle widens the shaft; the layout raises an error if the shaft
  would reach the side PML or the side plasmon absorbers at their top edge
  (`side_absorber_top_um`). Move `side_absorber_start_um` out (or widen the domain) if so.
* The 2D tip is a ridge, not a cone: compare trends, not absolute values, with 3D tips or
  experiments ([limitations.md](limitations.md)).

## 3. Gap

`gap_nm` is the air gap between the apex and the top of the sample.

* It must be a whole number of cells at the chosen resolution (2 nm at 1 nm cells: 2 cells;
  1.333 nm at 1.5 pixels/nm: 2 cells).
* With a layer on a metal, the metal-to-metal distance is `gap_nm` plus the snapped layer
  thickness. To keep the tip-to-metal distance fixed while adding a layer, subtract the
  snapped thickness (example 05 does this).
* Use at least two cells across the air gap. With one cell the FDTD underestimates EF by
  10–16 % (1 nm gap at 1 nm cells: −16 % against the FEM reference; −9.5 % for the cylinder
  nanogap); with two cells by 5–16 % depending on the wavelength (−8 % at 633 nm, −16 % at
  525 nm, where |ε_Au| is small and the staircase of the apex matters more).
* Below ~1 nm, local optics itself becomes questionable (no nonlocality or tunnelling in the
  model).

## 4. Sample: substrate and layers

`Sample(substrate, layers=(Layer(material, thickness_nm), ...))`, layers listed from the
substrate upward; the last one faces the tip.

* **Metal or dielectric substrate.** The boundary treatment follows from the material at
  the simulated wavelengths (metal: Re ε < 0): a metal substrate ends at the bottom of the
  cell without a PML (a check requires it to be opaque over `substrate_depth_um`), is
  replaced by the conductor in the side PML and gets side plasmon absorbers; a dielectric
  substrate continues into the side and bottom PML. A spectrum across a sign change of
  Re ε raises an error; split it.
* **2D material.** Use a `UniaxialMaterial` (in-plane and out-of-plane permittivity, optic
  axis along the layer normal). The gap field is mostly out of plane, so the out-of-plane
  permittivity controls the gap mode; the in-plane one controls the excitonic absorption
  seen by Ex and the field inside the layer.
* **Thickness snapping.** Each layer becomes a whole number of cells, within 5 % of the
  requested thickness (otherwise an error names the nearest representable thickness). A
  monolayer of 0.65 nm needs 1.5 pixels/nm (1 cell, 0.667 nm) or 3 pixels/nm (2 cells); a
  ≥ 2 nm film works at 1 pixel/nm. The snapped thickness is used everywhere, including
  the analytic reference.
* **Thick films** (e.g. SiO2 on Si): the free region extends `substrate_depth_um` below the
  lowest layer, so every layer is inside it. Dispersive layers stop at the side PML; guided
  modes of a thick dispersive film are then reflected there unless the side absorbers
  remove them first. Check with the domain variants of the checklist.
* **Absorbing dielectric substrates** (e.g. Si as a dispersive Lorentz model) cross the side
  and bottom PML; this combination is supported but was not part of the validation. If the
  fields diverge the run fails with "non-finite DFT fields" or "did not converge".

Example 05 runs WS2 on Au, on Ag (Meep's library) and on SiO2.

## 5. Materials

| you have | use |
|---|---|
| a built-in | `materials.AU`, `materials.SIO2`, `materials.WS2_MONOLAYER`, `materials.AIR` |
| an entry of Meep's library | `Material.from_meep("Ag")` (any isotropic entry: Ag, Al, Cu, Pt, Ti, cSi, fused_quartz, ...) |
| a constant index | `Material.constant("hBN", index=2.1)` (lossless) |
| tabulated n, k | `fitting.fit_lorentz(...)` → `Material`, then `save_json` / `Material.load_json` |
| a pole model from the literature | `Material(name, eps_inf, poles=(Pole("lorentz", f0, gamma, sigma), ...))` in Meep units (1/µm) |
| anisotropy | `UniaxialMaterial(name, in_plane, out_of_plane)` |

Fitting your own n, k data (example 06):

1. CSV with the header `wavelength_nm,n,k`.
2. Starting poles `(energy_eV, width_eV)`: one at each absorption feature (excitons,
   interband edges), plus one or two broad poles above the window for the background. For a
   metal, start from a Drude–Lorentz model from the literature instead.
3. Fit a window slightly wider than the wavelengths you will simulate, set
   `valid_wavelength_nm` to the range you trust; simulations outside it raise an error.
4. Accept the fit when the relative RMS error of ε is a few % (the WS2 fit: 2.3 %) and
   `min_im_eps` ≥ 0 (guaranteed by the bounds). Look at the plot near weak absorption:
   Lorentz fits overestimate absorption below a band edge.
5. Every pole keeps a width of at least 0.02 µm⁻¹ (`min_gamma_um_inv`). Narrow poles ring for
   a long time after the pulse and slow the DFT convergence: the WS2 runs need 1.5–2.5
   times more time steps than gold alone.

## 6. Illumination

`Illumination(incidence_deg, divergence_deg, n_waves)`, p polarization.

* `incidence_deg ± 2 divergence_deg` must lie inside (0°, 90°) (the angular spectrum is cut
  there).
* The divergence sets the beam waist, w0 ≈ λ/(π θd), and therefore how much of the
  semi-infinite shaft is illuminated. The drive of the shaft is ramped to zero inside the
  shaft absorber (`drive_ramp_um`), which is only harmless where the beam no longer reaches
  the shaft. Every run records `shaft_illumination_at_ramp` (|E_bg|² on the shaft faces at
  the start of the ramp relative to the gap centre) in its metadata and log; keep it below
  ~10⁻³. Production domain, default beam: ≤ 1.0 × 10⁻³ (500–900 nm, Au and SiO2). A narrower
  angular spectrum (larger waist) illuminates more of the shaft and needs a taller domain
  with the absorber and ramp moved up: with 10° the number reaches 1.3 × 10⁻² at 900 nm (at
  633 nm it stays at 1.6 × 10⁻⁴; there the narrower beam itself lowers EF by ~3 %, a physical
  change of the illumination, not an artefact).
* Wavelengths: one run per wavelength; the beam is synthesized at each one, so the angle is
  exact. All materials must be valid at every wavelength.
* Polarization: only p (TM). In 2D, s polarization is the decoupled TE problem with E along
  the invariant z axis; it has no field component along the tip axis and no gap mode.

## 7. Outputs

`Outputs(near_box_nm, probe_nm, full_map, full_map_stride)`:

* `probe_nm=(x, y)` moves the EF probe (relative to the sample surface below the apex); the
  default is the gap centre. Any other point of the near box can be evaluated afterwards
  with `result.e2_at(x_nm, y_nm)`.
* `near_box_nm` is the region stored at full resolution (default 300 × 130 nm).
* `full_map=True` stores the whole PML-free region (down-sampled), for large-scale maps.

Definitions of every output: [outputs.md](outputs.md).

## 8. Numerical settings

Two presets:

* `Numerics.production(resolution=1000)`: the reference set-up (free region |x| ≤ 1.8 µm,
  −0.2 ≤ y ≤ 5.5 µm, long absorbers, drive ramp 3.5–4.5 µm). Gives absolute values; ~27
  million cells at 1 nm cells.
* `Numerics.small(resolution)`: free region |x| ≤ 0.4 µm, y ≤ 0.6 µm, short steep absorbers,
  drive ramped down at 0.35–0.6 µm. The shaft is illuminated over a much shorter length
  (`shaft_illumination_at_ramp` ≈ 0.4–1.3), so absolute values differ (EF 352.6 instead of
  139.7 for the default case at 1 nm cells) and depend at first order on where the ramp sits
  relative to the tip; but ratios at equal settings (resolution series, with/without a layer)
  transfer to the production domain: the FEM gives the same WS2/Au-to-Au ratio in both
  domains within 1.3 % (0.5 % except on the steep flank of the B exciton at 560 nm). 15 times
  fewer cells, so fine grids are affordable.

What the knobs do:

| knob | default (production) | what it controls | change it when |
|---|---|---|---|
| `resolution` | 1000 px/µm (1 nm) | cells across the gap and layers; staircasing of the apex | the gap has < 2 cells, a layer cannot be snapped, or the resolution series is not converged |
| `x_half_width_um`, `y_max_um` | 1.8, 5.5 | size of the free region | the beam or the absorbers need more room (narrow angular spectrum, large tip) |
| `substrate_depth_um` | 0.2 | substrate inside the free region below the layers | an opaque substrate fails the skin-depth check (lossy dielectrics need a bottom PML instead) |
| `pml_um` | 0.4 | PML thickness | the PML check of the checklist moves EF |
| `bottom_pml`, `side_absorbers` | automatic | the boundary treatment of section 4 | to force a choice |
| `pml_metal_conductivity` | 20 | conductor replacing dispersive metal inside the PML | rarely; the result does not depend on it (FEM: same result with gold in the PML, ≤ 0.04 %) |
| `shaft_absorber_start_um` | 3.5 | where the shaft absorber starts (height on the faces) | together with the drive ramp, when the domain changes |
| `side_absorber_start_um`, `side_absorber_top_um` | 0.8, 2.3 | the side plasmon absorbers above a metal | wider tips (shaft must stay clear), other domain sizes |
| `absorber_conductivity`, `absorber_power`, `absorber_segments` | 10, 3, 30 | strength and grading of the absorbers | only with the checklist: long and gentle beats short and strong |
| `drive_ramp_um` | (3.5, 4.5) | where the scattered-field drive of the shaft is ramped to zero | keep it inside the shaft absorber where `shaft_illumination_at_ramp` ≲ 10⁻³ |
| `courant` | 0.5 | time step = courant × dx / c | instability with unusual materials (lower it) |
| `pulse_fwidth_rel`, `pulse_cutoff` | 1.5, 4 | pulse bandwidth (only the centre frequency is used) | rarely |
| `dft_decay_tol`, `min/max_time_after_sources` | 10⁻⁴, 2, 80 | when a run stops / fails | narrow material poles (raise `max_time_after_sources`) |
| `chunk_layout` | "auto" | banded chunks for single-process runs, Meep's for MPI | never needed |

Grid cost: the number of cells grows as resolution² and the number of steps as resolution,
so the work grows as resolution³ for a fixed domain.

## 9. Convergence checklist

Do this once for every new kind of case (new sample type, wavelength range, tip family),
not for every run.

1. **Representation.** `build_layout(model, wavelengths)` succeeds; ≥ 2 cells across the air
   gap; layers snapped within 5 %; `shaft_illumination_at_ramp` ≲ 10⁻³ in the production
   domain (read it from the first run's log).
2. **Resolution.** In the small domain, run 2–4 resolutions (example 08 or
   `validation/resolution_series.py`). Reference case: EF = 352.6 / 362.5 / 370.2 / 372.2 at
   1 / 1.5 / 2 / 3 pixels/nm (+5.6 % from 1 to 3 pixels/nm, +0.5 % between the two finest);
   1 nm gap +12.9 % and 525 nm +7.6 % from 1 to 2 pixels/nm; WS2/Au +6.2 % from 1.5 to 3
   pixels/nm. Use the change between the two finest grids as the discretization uncertainty,
   or quote the FEM value (step 4).
3. **Domain and boundaries.** At the resolution you use, change one thing at a time: move
   the free region, absorbers and drive ramp by 0.5 µm; absorber conductivity × 1.5; PML × 2
   (example 08, `validation/domain_check.py`). Reference case: ≤ 1.0 %.
4. **Converged reference (optional, CPU).** `ters2d.fem.solve(model, wavelength)` gives the
   staircase-free answer of the same model in under a minute (pip install ngsolve). The FDTD
   at 1 nm cells is 5–16 % below it for the reference cases (−8 % at 633 nm); the FEM itself
   is converged to < 0.1 % and moves < 1 % with its boundaries.
5. **Precision.** gpmeep runs in FP32. For the reference case FP32 and FP64 agree to
   2.7 × 10⁻⁶ (`validation/fp32_vs_fp64.py`); repeat once if your case is very different
   (much longer runs, very large cells).

## 10. Running it, and what it costs

* **CPU, upstream Meep** (conda-forge `pymeep`, FP64): `python script.py`, or
  `mpirun -np 16 python script.py`. With MPI the default chunk layout is Meep's.
* **GPU, gpmeep** (FP32, the same `import meep`): install gpmeep in its own environment and
  select the backend before Python starts, `MEEP_GPU_BACKEND=cuda CUDA_VISIBLE_DEVICES=0
  python script.py`. One process per GPU; no code changes.
* An interrupted `spectrum` or `sweep` resumes from the files already written.

Measured costs (one wavelength; the structure set-up is paid once per geometry):

| case | cells | time steps | gpmeep FP32, 1 RTX A6000 | upstream Meep FP64, CPU |
|---|---|---|---|---|
| quick start: small domain, 2 nm cells (example 01) | 0.44 M | 11 400 | — | 105 s on 1 core (+5 s set-up), 21 s with 8 MPI processes; 0.36 GB RAM |
| small domain, 1 nm cells, Au | 1.76 M | 13 700–19 000 | 52–72 s + 18 s set-up | — |
| small domain, 0.67 nm cells, Au / WS2/Au | 3.96 M | 22 000–30 000 / 32 000–67 000 | 87–120 s / 209–429 s, + 40–80 s set-up, ~1 GB GPU memory | — |
| small domain, 0.5 nm cells | 7.04 M | 29 000–41 000 | 165–227 s + 71 s set-up | — |
| small domain, 0.33 nm cells, Au / WS2/Au | 15.8 M | 46 000 / 99 000 | 482 s / 1370 s, + 160–320 s set-up | — |
| production, 1 nm cells, Au | 26.8 M | 19 000–25 000 | 374–496 s + 510 s set-up, 5.7 GB GPU memory | 3614 s with 32 MPI processes (+58 s set-up) |
| production, 1 nm cells, SiO2 | 28.6 M | 19 000–23 000 | 385–473 s + 630 s set-up | — |

(`docs/figures/data/run_times.csv` has every run of the reference results.) Rules of thumb:
gpmeep advanced ~1.4 × 10⁹ cell-updates/s on the production domain and 32 CPU cores
~1.5 × 10⁸; the GPU was 9 times faster in time stepping and 4 times faster end to end for a
single wavelength, because gpmeep builds the structure on one host thread (510 s vs 58 s on
32 processes). For a spectrum the set-up is shared, so the GPU advantage approaches 9×.
Layers with narrow absorption lines (excitons) need 1.5–2.5 times more time steps.

## 11. Errors you may meet

| message | meaning / fix |
|---|---|
| `gap ... is not a whole number of ... cells` | change `gap_nm` or `resolution` |
| `a ... nm layer cannot be represented on a ... grid` | use a resolution at which the layer is ≈ whole cells |
| `... wavelengths ... outside the model's range` | a material's `valid_wavelength_nm` excludes a wavelength |
| `Re(eps) changes sign within the wavelengths` | split the spectrum (metal vs dielectric boundary treatment) |
| `substrate depth ... is not opaque` | increase `substrate_depth_um` or set `bottom_pml=True` |
| `a non-metallic substrate needs a bottom PML` | do not set `bottom_pml=False` for a dielectric |
| `the tip shaft would enter the side ...` | widen the domain or move `side_absorber_start_um` out |
| `... exists but was computed with other settings` | a run directory is reused only for the identical model; use another directory |
| `DFT fields did not converge` / `non-finite DFT fields` | narrow poles (raise `max_time_after_sources`) or an unstable material/PML combination |
