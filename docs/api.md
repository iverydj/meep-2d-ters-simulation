# API reference

`import ters2d`. Lengths of the tip, sample and gap in nm, sizes of the domain in µm,
wavelengths in nm, angles in degrees. All dataclasses are frozen (immutable); change them with
`dataclasses.replace` or `Model.with_value`. The docstrings in `src/ters2d/` are the
authoritative description of every field.

## Set-up (`ters2d.model`)

| class | fields (defaults) |
|---|---|
| `Tip` | `apex_radius_nm=20.0`, `half_angle_deg=15.0`, `material=materials.AU` (isotropic `Material`) |
| `Layer` | `material` (`Material` or `UniaxialMaterial`), `thickness_nm` |
| `Sample` | `substrate=materials.AU`, `layers=()` (from the substrate upward); `.label` e.g. "WS2/Au" |
| `Illumination` | `incidence_deg=60.0`, `divergence_deg=14.0` (Gaussian angular spectrum, 1/e² intensity half-angle), `n_waves=721`; p polarization |
| `Numerics` | grid, domain, boundaries, absorbers, drive ramp, time stepping (see the table in [setting_up_your_simulation.md](setting_up_your_simulation.md)); presets `Numerics.production(resolution=1000, **overrides)`, `Numerics.small(resolution=1000, **overrides)`; `.dx_nm` |
| `Outputs` | `near_box_nm=(-150, 150, -10, 120)`, `probe_nm=None` (gap centre), `full_map=False`, `full_map_stride=8` |
| `Model` | `tip`, `sample`, `gap_nm=2.0` (air gap), `illumination`, `numerics`, `outputs`; `.with_value("tip.apex_radius_nm", 30)` |

## Running (`ters2d.run`, needs Meep)

| function | returns |
|---|---|
| `simulate(model, wavelength_nm, out_dir, log=print)` | `Result` |
| `spectrum(model, wavelengths_nm, out_dir, log=print)` | `[Result]` (structure built once; resumes from existing files) |
| `sweep(model, parameter, values, wavelengths_nm, out_root, log=print)` | `[(value, [Result])]`; one directory `out_root/<parameter>=<value>` per value |

## Results (`ters2d.results`, pure Python)

| name | what |
|---|---|
| `Result` | `.wavelength_nm`, `.metrics` (dict, see [outputs.md](outputs.md)), `.ef_probe`, `.ef_gap`, `.meta`, `.file`, `.fields()`, `.e2_at(x_nm, y_nm, kind="")`, `.row()` |
| `load_result(npz_path)` | `Result` of an existing run file |
| `field_map(result, region="near", stride=1)` | `FieldMap` (`x_nm`, `y_nm`, `e2_rel`, `outline_nm`, `layers_nm`, `meta`; `save_npz`, `load_npz`) |
| `write_csv(rows, path)`, `read_csv(path)` | tables |

## Materials (`ters2d.materials`, `ters2d.fitting`)

| name | what |
|---|---|
| `Pole(kind, frequency, gamma, sigma)` | one Lorentz (`"lorentz"`) or Drude (`"drude"`) term, Meep units (1/µm); passive only |
| `Material(name, eps_inf, poles, valid_wavelength_nm, reference)` | isotropic model; `.epsilon(wavelength_nm)`, `.epsilon_xy(...)`, `.dispersive`, `.check_wavelengths(...)`, `save_json` / `load_json` |
| `Material.constant(name, index=... or epsilon=...)` | lossless dielectric |
| `Material.from_meep(name)` | an isotropic entry of `meep.materials` (needs Meep) |
| `UniaxialMaterial(name, in_plane, out_of_plane)` | optic axis along the layer normal (y) |
| `AU`, `SIO2`, `AIR`, `WS2_MONOLAYER` | built-ins ([model.md](model.md#materials)) |
| `is_metallic(material, wavelengths_nm)` | Re ε < 0 at all wavelengths (error if mixed) |
| `fitting.read_nk_csv(path)` | `(wavelength_nm, n, k)` from a `wavelength_nm,n,k` CSV |
| `fitting.fit_lorentz(wl, n, k, initial_poles_eV, name=..., fit_window_nm=..., valid_wavelength_nm=..., min_gamma_um_inv=0.02, fixed_gamma_um_inv=None)` | passive Lorentz `Material` |
| `fitting.fit_quality(material, wl, n, k)` | `rel_rms`, `rel_max`, `min_im_eps_200nm_to_20um` |

## Lower-level modules

| module | contents |
|---|---|
| `ters2d.layout` | `build_layout(model, wavelengths_nm) -> Layout` (grid-snapped geometry, boundary choices, all pre-run checks), `TipShape`, `tip_shape`, `snapped_thickness`, `chunk_bands` |
| `ters2d.background` | `GaussianBeam`, `Stack`, `stack_coefficients` (N-layer uniaxial TM transfer), `beam_fields` (points), `beam_fields_grid` (tensor grid): the analytic tip-free field |
| `ters2d.sources` | `volume_sources` (scattered-field current on the Yee points of any shape with a `contains(x, y)` method), `tip_sources`, `drive_weight` |
| `ters2d.meep_model` | `meep_medium`, `meep_geometry`, `meep_boundary_layers`, `meep_chunk_layout`, `make_simulation` |
| `ters2d.fem` | `solve(model, wavelength_nm, FemSettings())`: frequency-domain reference of the same model (needs NGSolve) |
| `ters2d.analysis` | `fresnel_rp`, `fit_up_down`, `mie_cylinder_hz` (exact 2D Mie series), `rel_l2`, `stokes_wavelength_nm`, `raman_enhancement` |
| `ters2d.plotting` | `plot_spectra`, `plot_sweep`, `plot_field_map`, `plot_ws2`, `plot_fresnel`, `plot_nanogap`, `plot_resolution_series`, `plot_lorentz_fit`, schematics |
