# Validation

Every check has a script in `validation/` and its data in `docs/figures/data/`. The checks go
from the parts (the reference field, the source, the materials) to the whole model, and end
with the numerical settings.

| check (script) | what is compared | result |
|---|---|---|
| Tip-free reference vs Meep (`fresnel_check.py`) | specular reflection of Au, SiO2 and WS2/Au computed by Meep (Bloch-periodic cell, no tip, 1.5 px/nm) vs the transfer-matrix coefficient used for E_bg; 500–850 nm at 60°, 633 nm at 40–70° | \|Δr\| ≤ 0.011, \|ΔR\|/R ≤ 0.6 %, phase ≤ 0.7° (Au, WS2/Au) / 1.5° (SiO2) |
| Scattered-field formulation vs Mie (`mie_check.py`) | Au cylinder (R = 20 nm) in free space, driven exactly like the tip, vs the exact 2D Mie series for the same beam; no fitted constants | ≥ 4 cells from the surface: \|E\| rel. L2 error 0.4–0.9 %, mean \|E\|² within 0.4 %; 1–3 nm from the surface (where gap probes sit): rel. L2 4–7 % (staircase), mean \|E\|² within 1.3 % (1–1.5 px/nm) |
| Nanogap FDTD vs FEM (`nanogap_fdtd.py`, `nanogap_fem.py`) | Au cylinder 1–10 nm above Au, R = 10–40 nm; FDTD at 1–2 px/nm vs curved-element FEM converged to 0.1 % | 2 nm gap, R = 20 nm: −4.6 / −2.8 / −1.0 % at 1 / 1.5 / 2 px/nm (633 nm), −4.7 / −2.8 / −1.6 % (785 nm); 1 nm gap: −9.5 % (1 px/nm), −1.0 % (2 px/nm); at 1 px/nm: 4 nm −1.1 %, 10 nm +1.8 %, R = 10 nm −7.1 %, R = 40 nm −4.9 % |
| Full model FDTD vs FEM (`fem_tip_reference.py`) | the same tip model at every wavelength, gap and radius of the results | Au, 2 nm: FDTD lower by 5.5 % (900 nm) to 15.8 % (525 nm), −8.0 % at 633 nm; SiO2: −6.5 to −11.8 %; gap 1 / 2 / 3 / 4 / 6 / 10 nm: −16.0 / −8.0 / −5.1 / −3.2 / −1.1 / +0.5 %; R = 10 / 20 / 30 / 40 nm: −12.7 / −8.0 / −7.5 / −4.9 % |
| Resolution (`resolution_series.py`, small domain) | FDTD at 1–3 px/nm | EF = 352.6 / 362.5 / 370.2 / 372.2 at 1 / 1.5 / 2 / 3 px/nm (633 nm, 2 nm gap); from 1 to 2 px/nm: +12.9 % (1 nm gap), +7.6 % (525 nm), +4.3 % (SiO2); WS2/Au +6.2 % from 1.5 to 3 px/nm |
| WS2 layer (`reproduce/ws2_on_au.py`) | FDTD ratio EF(WS2/Au)/EF(Au/Au) at 1.5 px/nm (layer = 1 cell), small domain, vs FEM | 520–900 nm: −4.4 % to +1.8 %; at 633 nm 1.869 (1.5 px/nm) and 1.934 (3 px/nm) vs FEM 1.955 |
| Domain and boundaries (`domain_check.py`) | free region, absorbers and drive ramp moved by 0.5 µm; absorbers 1.5× stronger; PML 2× thicker; production resolution | ≤ 1.0 % (Au at 633 and 900 nm, SiO2 at 900 nm) |
| FEM reference itself (`fem_tip_reference.py`, `check_*`) | polynomial order 4 → 5; finer mesh; absorber σ 7–15; absorbers moved by 0.5 µm; drive ramp moved by ±0.5 µm; gold continued into the PML instead of the conductor | ≤ 0.04 %, < 0.01 %, ≤ 0.4 %, ≤ 0.7 %, ≤ 1.1 %, ≤ 0.04 % |
| FP32 vs FP64 (`fp32_vs_fp64.py`) | gpmeep (FP32, one GPU) vs upstream Meep 1.34 (FP64, 32 MPI processes), Au/Au, 633 nm, production domain | EF 139.66402 vs 139.66364: 2.7 × 10⁻⁶ |
| This code vs the reference runs (`regression_check.py`) | see below | ≤ 10⁻⁶ for Au and SiO2; WS2 see below |

![Fresnel check](figures/fresnel_check.png)

*Tip-free reflectance at 60°: Meep (markers) and the analytic coefficient (lines), and the
error of the complex reflection coefficient.*

![nanogap benchmark](figures/nanogap_benchmark.png)

*Cylinder nanogap: FDTD EF relative to the converged FEM value. (a) Grid convergence. (b) At
the production grid of 1 nm.*

![resolution series](figures/resolution_series.png)

*Resolution series of the tip model in the small domain, relative to the finest grid of each
series (for WS2/Au the gap is the apex-to-gold distance).*

## What the numbers say

**The FDTD deviation from the FEM is discretization.** It grows as the gap has fewer cells
(from +0.5 % at 10 cells to −16 % at one), for smaller apex radii and at shorter wavelengths,
where |ε_Au| is smaller and the staircase of the apex and of the wedge faces matters more; it
does not change when the boundaries are moved. For the cylinder nanogap the FDTD converges to
the FEM value (−4.6, −2.8 and −1.0 % at 1, 1.5 and 2 px/nm). For the tip, the small-domain
series rises by 5.6 % from 1 to 3 px/nm at 633 nm and by 7.6 % from 1 to 2 px/nm at 525 nm.
These series bound the tip's discretization error only to a few per cent, because the small
domain has short, abrupt absorbers whose stepped (FDTD) and continuous (FEM) forms differ:
there the FDTD at 3 px/nm ends 2 % above the FEM, and the −8 % deviation of the production
FDTD at 633 nm is larger than the ~−5 % the series alone suggests.

**The FEM values are the accurate numbers of this 2D model.** The FEM solves the continuous
problem, is converged to < 0.1 % and moves by < 1 % with its boundaries. The FDTD at 1 pixel/nm
reproduces every trend and is 5–16 % low in absolute value (−8 % at 633 nm). The text of
[results.md](results.md) therefore quotes FEM values; the FDTD values are in the CSV files.

**FP32 is not a limitation here.** The production case agrees between gpmeep (FP32) and
upstream Meep (FP64) to 2.7 × 10⁻⁶, with identical step counts.

## Regression check of this code

The code of this repository was restructured from the one that produced the reference runs
(same physics, new API, N-layer transfer matrix, sample surface as the origin).
`validation/regression_check.py` re-runs a subset with gpmeep on one GPU and compares it with
the committed tables:

REGRESSION_TABLE

The Au and SiO2 cases contain no layer and reproduce the reference values to the last digit
of FP32. For WS2/Au the difference comes from one deliberate change of convention: the
heights of the domain, absorbers and drive ramp are now measured from the top of the sample
(the WS2 surface) instead of the gold surface, so for the layered sample they all sit one cell
(0.667 nm) higher relative to the tip. In the small domain, where the drive is ramped down on a
still strongly illuminated shaft, this shifts EF at first order: the FEM of the two conventions
differs by the same amount (+1.6 × 10⁻³ in the small domain, +1.3 × 10⁻⁶ in the production
domain). WS2_CONVENTION_SENTENCE
