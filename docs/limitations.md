# Limitations

What the model does not describe, and what that means for using its numbers.

**2D wedge, not a 3D cone.** The tip is an infinitely long ridge and the beam a 2D Gaussian
beam. Field concentration, radiation and the spreading of plasmons are different in 3D: the
absolute enhancement factors, the spectral shape (for example the absence of a gap resonance
between 500 and 900 nm for R = 20 nm) and the substrate contrast are those of the 2D model and
should not be compared one-to-one with 3D tips or experiments. The semi-infinite wedge casts a
full shadow (see the field maps), and its shaft plasmons propagate without geometric spreading;
here they add only ~3 % to the gap field, but they made the termination of the shaft the
hardest numerical problem of the model. What is expected to carry over to 3D, without being
tested here (no 3D or axisymmetric calculation is part of this repository), are mechanisms and
qualitative trends: the gap mode over a metal and its rapid rise as the gap closes, the
weaker, less confined field over a dielectric, and the effect of a high-permittivity layer in
the gap. The non-monotonic radius dependence is specific to the 2D wedge (an isolated
cylinder above gold gives a monotonic trend) and should not be expected in 3D.

**Local, bulk optics.** All materials are bulk permittivities. There is no nonlocal response,
electron spill-out or quantum tunnelling, which matter for gaps below ~1 nm and reduce the
enhancement there; the 1 nm point of the gap sweep is outside the range where local optics is
reliable.

**2D material model.** The WS2 out-of-plane permittivity is the bulk extraordinary index (no
broadband monolayer data exist), and the in-plane Lorentz fit overestimates the weak absorption
below the A exciton. The monolayer thickness (0.65 nm, 0.667 nm on the grid) is a convention;
the thickness assumed in the ellipsometric extraction of the in-plane data was not checked.
The layer is a homogeneous slab with sharp interfaces, with no excitonic nonlocality or strain,
and its Raman response is not modelled (`EF_layer` is the local field that would drive it).

**The FEM reference shares the model.** The frequency-domain reference has the same materials,
beam, drive ramp, absorbers and boundary layout. It validates the discretization and the
time-domain implementation (grid, boundaries, pulse/DFT), not the modelling choices; those are
probed by the domain, beam and absorber variations in [validation.md](validation.md).

**Staircased geometry.** Meep does not smooth dispersive interfaces, so the curved apex and the
wedge faces are staircased. At 1 nm cells this makes the FDTD 5.3–15.9 % low for the 2 nm gap
and R = 20 nm (−7.7 % at 633 nm; less for larger gaps and radii, more for a 1 nm gap); finer
grids or the FEM give the converged value. Layers are snapped to whole cells. For s
polarization above a metal the error has a second cause: in the gap the field scattered by the
tip nearly cancels the tip-free field (E_scat ≈ −(1.05 + 0.25i) E_bg at the gap centre over gold
at 633 nm), so a small error of the scattered field becomes a much larger one of what remains.
The FDTD at 1 nm cells is then 1.6 % (500 nm) to 12.4 % (900 nm) low over gold (−6.2 % at
633 nm, −4.2 % at 0.67 nm cells); over SiO2 it is within 0.7 %.

**Illumination.** p and s polarization, each a separate run (in 2D they do not couple; any
other polarization is their superposition, see [model.md](model.md#polarization); there is no
depolarization as by a 3D tip); one in-plane direction of incidence,
from the air side; a Gaussian angular spectrum cut at ±2 divergence
angles, so incidence ± 2 divergence must stay below 90° (no high-NA beams). The illuminated
length of the semi-infinite shaft is a modelling choice set by the beam width and the drive
ramp; the default domain covers incidence near 60° and 500–900 nm (the shaft-illumination
check refuses anything else until the domain is enlarged).

**Ambient and tip.** The space around the tip is air (ε = 1): the source current uses ε_tip − 1
and the tip-free field has air above the sample. Immersion media and coated or layered tips are
not modelled; the tip is one homogeneous isotropic material.

**Raman estimate.** EF² (or the axis average `EF4_gap`) assumes equal enhancement at the laser
and Stokes wavelengths and the field at the gap centre; `analysis.raman_enhancement` uses the
computed spectrum at both wavelengths, but neither is a Raman cross-section.

**Boundary treatments outside the validated set.** Validated: Au tip; Au and SiO2 substrates;
the WS2 monolayer on Au. Supported but not validated here: other metals (e.g. Meep's Ag),
metal or thick dispersive layers, and dispersive non-metallic substrates crossing the PML.
Run the checklist of [setting_up_your_simulation.md](setting_up_your_simulation.md) before
trusting such a case.

**Small domain.** The small domain is a different model (a shorter illuminated shaft): its
absolute values are 2.5 times those of the production domain at 633 nm (1.4 times
at 900 nm), they depend at first order on where the drive ramp sits relative to the tip, and the
domain does not preserve spectral shapes or substrate contrasts. Only the effect of a thin layer
on the same substrate at the same wavelength (checked with the FEM for gold over 500–900 nm, and
for silver and SiO2 at 633 nm) and changes with resolution are taken from it.

**Validation coverage.** The boundary design was chosen with the FEM on Au; its checks cover Au
at 525, 633, 700 and 900 nm, SiO2 at 900 nm and WS2/Au at 633 nm (FEM), and Au at 633 and
900 nm and SiO2 at 900 nm (FDTD). Silver appears only in the FEM layer-transfer check. The
Fresnel check of the tip-free field ran at 1.5 pixels/nm (production: 1 pixel/nm). For s
polarization the Fresnel and Mie checks, FDTD vs FEM for Au and SiO2 at 500–900 nm, the grid
convergence at 633 nm and two FEM checks (order 5, boundaries moved up) at 633 nm were run;
the other boundary variations were run for p only. Other wavelengths, materials and geometries rely on the
same design and on the checklist of [setting_up_your_simulation.md](setting_up_your_simulation.md).
