# Results

These are the reference results computed with this code for the default gap-mode geometry and
its variations. Unless stated otherwise: Au tip with apex radius R = 20 nm and half-angle 15°,
2 nm gap, 60° p-polarized Gaussian beam (divergence 14°), production domain at 1 pixel/nm,
gpmeep on one GPU; s-polarized illumination is in its own section at the end. In the figures, markers are FDTD and lines are the frequency-domain (FEM)
solution of exactly the same model, which has no staircasing. The numbers in the text are FEM
values, the accurate numbers of this 2D model; the FDTD values are in the tables. For the 2 nm
gap and R = 20 nm the FDTD at 1 pixel/nm is 5.3–15.9 % below the FEM (−7.7 % at 633 nm; less
for larger gaps and radii, see [validation.md](validation.md)). The tables, scripts and
figures are listed at the end.

## Gap mode vs no gap mode

![spectra](figures/spectra.png)

*(a) EF = |E|²/|E_ref|² at the gap centre, E_ref the tip-free field at the same point. (b)
|E|² at the gap centre relative to the incident beam at its focus (without sample).*

With a gold substrate the intensity |E|² in the gap is 151 times that of the tip-free field at
633 nm (222 at 900 nm); above SiO2 it is 19 times (32). The image charges in the gold mirror
turn the apex field into a gap plasmon. Relative to the incident beam the contrast is larger,
296 vs 17 at 633 nm, because the sample alone already changes the intensity at the gap: on
gold the reflected beam nearly doubles it (1.96 times the incident intensity at the focus), on
SiO2 at 60°, close to the Brewster angle, the sample lowers it by 11 % (0.89 times).

For R = 20 nm and a 2 nm gap this 2D model has no gap-plasmon resonance between 500 and 900
nm: the enhancement rises towards the red, and below ~600 nm it is limited by the interband
absorption of gold (EF 8.6 at 500 nm). The |E|⁴ Raman estimate is EF² ≈ 2.3 × 10⁴ at 633 nm
(FEM; the FDTD columns `EF4_probe` and `EF4_gap` give 1.95 × 10⁴ and 1.96 × 10⁴).

## Near field around the apex

![field map Au](figures/fieldmap_Au_633nm.png)
![field map SiO2](figures/fieldmap_SiO2_633nm.png)

*|E|² at 633 nm normalized by the tip-free intensity at the gap centre (FDTD, 1 nm cells; top:
Au substrate, bottom: SiO2; the beam comes from the upper left).*

Above gold the field is concentrated in the gap below the apex. Above SiO2 the apex gives a
weaker (EF 18 in this FDTD map) and less confined lightning-rod enhancement. The dark region
to the right of the tip is the geometric shadow of the semi-infinite 2D wedge, which light
cannot pass around; a 3D cone does not cast such a shadow.

## Gap and apex radius at 633 nm

![gap sweep](figures/gap_sweep.png) ![radius sweep](figures/radius_sweep.png)

EF falls from 601 (1 nm gap) to 151 (2 nm), 49 (4 nm) and 13 (10 nm): it scales about as g⁻²
below 2 nm (exponent 1.99) and as g⁻¹·⁵ between 2 and 10 nm (exponent 1.53). The 1 nm point is
a single FDTD cell (open marker, −15.9 % against the FEM); local optics also becomes
questionable at that distance ([limitations.md](limitations.md)).

The radius dependence is not monotonic: EF = 187, 161, 151, 151, 156, 165, 178 for R = 10, 15,
20, 25, 30, 35 and 40 nm, with a minimum near 25 nm. It is a property of the apex and the gap,
not of the beam or the shaft: with the shaft plasmons absorbed 0.3 µm above the apex the
values change by only −2.4 %, and with a narrower beam (divergence 10° instead of 14°) by −2.6
% to −2.9 %. It is specific to the 2D wedge: an isolated gold cylinder of the same radius
above gold gives a monotonic trend (EF 24, 101 and 309 for R = 10, 20 and 40 nm,
`nanogap_fem.csv`).

## WS2 monolayer in the gap

![WS2 on Au](figures/ws2_on_au.png)

*Ratio of the gap-centre EF with and without a WS2 monolayer on the gold (same tip, same 2 nm
distance between the tip and the gold). Lower panel: absorption of the in-plane WS2 model.*

The monolayer (0.667 nm on the grid; out-of-plane permittivity ε⊥ ≈ 6.6 at 633 nm) fills a
third of the 2 nm gap. The normal field in the remaining 1.33 nm of air is then larger, and EF
at the centre of the air gap rises by a factor 1.6–2.0 on the red side of the excitons (1.96
at 633 nm). At the excitons the absorption of the layer damps the gap mode: the ratio has its
minimum, 0.67, at 530 nm on the red side of the B exciton (512 nm), and dips to 1.60 at the A
exciton (610 nm), between 1.81 at 590 nm and 1.96 at 633 nm. The field inside the layer, which
drives the Raman scattering of WS2 itself, is weaker:
|E|²/|E_ref|² averaged over the layer is 36 at 610 nm and 58 at 633 nm (column
`EF_layer` of `fem_tip_reference.csv`).

The FDTD ratios are computed in the small domain at 1.5 pixels/nm (the layer is one cell). The
FEM gives the same ratio in the small domain and in the production domain within 1.3 % (0.5 %
except at 560 nm, on the steep flank of the B exciton), and the FDTD agrees with the FEM
within −4.8 % to +1.4 % at the sampled wavelengths.

## s-polarized illumination

![s spectra](figures/spectra_s.png)

*As the first figure, for an s-polarized beam (E along z, the axis of the 2D tip): (a) EF =
|Ez|²/|Ez,ref|² at the gap centre, (b) relative to the incident beam at its focus. Markers:
FDTD at six wavelengths; lines: FEM.*

With E along z the field is parallel to every surface of the tip and the sample. It induces no
surface charge, so there is neither a lightning-rod field nor a gap mode, and the gold tip
screens the field instead: at 633 nm the intensity at the gap centre is 0.069 times the
tip-free value above gold and 0.079 times above SiO2. The screening grows as |ε_Au| grows
towards the red: EF falls from 0.19 at 500 nm to 0.015 at 900 nm above gold, and from 0.17 to
0.038 above SiO2. This screening is a property of the 2D ridge; what carries over to a 3D tip
is discussed in [limitations.md](limitations.md).

Here EF compares with a tip-free field that is itself small above gold, where the surface
suppresses the tangential Ez: at the gap centre |E_ref|² is 0.091 of the incident intensity at
633 nm (0.18 at 500 nm, 0.036 at 900 nm); above SiO2 it is 0.34. Relative to the incident beam
the intensity at the gap centre is then 0.0063 above gold and 0.027 above SiO2 at 633 nm. What
EF and this ratio mean for s is explained in
[outputs.md](outputs.md#s-polarization-which-number-to-read).

![p vs s](figures/p_vs_s.png)

*|E|² at the gap centre relative to the incident beam at its focus, p (filled markers, solid
lines) and s (open markers, dashed lines).*

The gap mode is a p effect. Relative to the incident beam, p gives 296 above gold at 633 nm
and s 0.0063, a ratio of 4.7 × 10⁴; the ratio grows from 3.1 × 10² at 500 nm to 1.0 × 10⁶ at
900 nm. Above SiO2 the ratio is 6.4 × 10² at 633 nm (92 at 500 nm, 2.2 × 10³ at 900 nm). The
substrate acts in opposite directions: replacing SiO2 by gold raises the gap intensity 17-fold
for p and lowers it to 0.24 for s at 633 nm (p: 2.1 at 500 nm and 20 at 900 nm; s: 0.61 and
0.042). These ratios use the incident intensity at the focus of each beam; at equal beam power
the p/s ratios are 2.9 % smaller ([outputs.md](outputs.md#s-polarization-which-number-to-read)). In this 2D model a beam polarized
at ψ to the plane of incidence therefore gives the p result scaled by cos²ψ; the s part
matters only within 0.3° of pure s above gold at 633 nm (2.3° above SiO2).

![field maps p vs s](figures/fieldmap_Au_633nm_p_vs_s.png)

*|E|² above gold at 633 nm, normalized by the incident intensity at the focus of each beam
(FDTD, 1 nm cells; the beam comes from the upper left): (a) p, (b) s, on one color scale.*

For s the field is lowest next to the metal surfaces, in the gap and along the tip, and grows
with the distance from the gold, as the tip-free standing wave of the incident and the
reflected beam does. Both polarizations show the shadow of the wedge on the right.

The FDTD at 1 pixel/nm is 0.1–0.7 % below the FEM above SiO2, but 1.6 % (500 nm) to 12.4 %
(900 nm) below it above gold (−6.2 % at 633 nm). Above gold the field scattered by the tip
nearly cancels the tip-free field in the gap, so a small error of the scattered field becomes
a larger relative error of the remainder ([validation.md](validation.md#s-polarization)).

## Uncertainty

For the results as a model of a 2D tip, the numerical uncertainty is small compared with the
modelling choices:

* FEM values (quoted in the text): converged to < 0.1 % in polynomial order and mesh; their
  boundary checks move them by ≤ 1.0 % in the cases checked (Au at 525, 633, 700 and 900 nm,
  SiO2 at 900 nm, WS2/Au at 633 nm; for s, Au and SiO2 at 633 nm: ≤ 0.07 %).
* FDTD values at 1 pixel/nm (markers, tables): biased low by the amounts in
  [validation.md](validation.md) (−7.7 % at 633 nm for the default case; for s over gold −1.6 %
  at 500 nm to −12.4 % at 900 nm); boundary effects ≤ 1 %; FP32 rounding negligible.
* Model parameters that are choices, not errors: the beam divergence (10° instead of 14°: −2.7
  % at 633 nm), the illuminated length of the shaft (shaft plasmons absorbed 0.3 µm above the
  apex: −2.4 %), the WS2 thickness and out-of-plane permittivity, the bulk gold permittivity.
* The 2D geometry itself is the dominant difference from a real tip and is not quantified here
  ([limitations.md](limitations.md)).

## Data and scripts

| figure | table(s) in `figures/data/` | produced by |
|---|---|---|
| `spectra.png` | `spectrum_Au_fdtd.csv`, `spectrum_SiO2_fdtd.csv`, `fem_tip_reference.csv` (group `spectrum`) | `reproduce/spectra.py`, `validation/fem_tip_reference.py` |
| `fieldmap_Au_633nm.png`, `fieldmap_SiO2_633nm.png` | `fieldmap_Au_633nm.npz`, `fieldmap_SiO2_633nm.npz` | `reproduce/field_maps.py` (from the spectrum runs) |
| `gap_sweep.png` | `gap_sweep_fdtd.csv`, FEM group `gap` | `reproduce/gap_sweep.py` |
| `radius_sweep.png` | `radius_sweep_fdtd.csv`, FEM groups `radius`, `radius_div10`, `radius_local` | `reproduce/radius_sweep.py` |
| `ws2_on_au.png` | `ws2_on_au_fdtd.csv`, FEM groups `ws2`, `ws2_small` | `reproduce/ws2_on_au.py` |
| `ws2_lorentz_fit.png` | `ws2_lorentz_fit.csv` | `materials/fit_ws2_lorentz.py` |
| `spectra_s.png`, `p_vs_s.png` | `spectrum_Au_s_fdtd.csv`, `spectrum_SiO2_s_fdtd.csv`, FEM groups `spectrum_s` (and `spectrum` for p) | `reproduce/spectra_s.py`, `validation/fem_tip_reference.py` |
| `fieldmap_Au_633nm_p_vs_s.png` | `fieldmap_Au_633nm.npz` (rescaled by `I_probe_ref_over_inc` of the p spectrum), `fieldmap_Au_633nm_s.npz` | `reproduce/field_maps.py`, `reproduce/spectra_s.py` |
| validation figures | see [validation.md](validation.md) | `validation/*.py` |
| — | `run_times.csv` (time steps, wall times, cells, backend and polarization of every run) | the run metadata |

`reproduce/make_figures.py` renders every figure from these tables (no Meep needed);
`reproduce/make_schematics.py` the schematics of [model.md](model.md). The FDTD tables were
computed with the code before the restructuring of this repository; re-running a subset with
the current code reproduces them to ≤ 2 × 10⁻⁶ ([validation.md](validation.md)). The FEM table
was computed with the current `ters2d.fem`.
