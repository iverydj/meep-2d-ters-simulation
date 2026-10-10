# Limitations

What the model does not describe, and what that means for using its numbers.

**2D wedge, not a 3D cone.** The tip is an infinitely long ridge and the beam a 2D Gaussian
beam. Field concentration, radiation and the spreading of plasmons are different in 3D: the
absolute enhancement factors, the spectral shape (for example the absence of a gap resonance
between 500 and 900 nm for R = 20 nm) and the substrate contrast are those of the 2D model and
should not be compared one-to-one with 3D tips or experiments. The semi-infinite wedge casts a
full shadow (see the field maps), and its shaft plasmons propagate without geometric spreading;
here they add only ~3 % to the gap field, but they made the termination of the shaft the
hardest numerical problem of the model. What transfers to 3D are mechanisms and trends: the
gap mode over a metal, the gap and radius dependence, the effect of a dielectric layer in the
gap.

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
wedge faces are staircased. At 1 nm cells this makes the FDTD 5–16 % low (−8 % at 633 nm);
finer grids or the FEM give the converged value. Layers are snapped to whole cells.

**Illumination.** p polarization only (the 2D TE problem has no gap mode); one in-plane
direction of incidence; a Gaussian angular spectrum cut at ±2 divergence angles. The
illuminated length of the semi-infinite shaft is a modelling choice set by the beam width and
the drive ramp.

**Raman estimate.** EF² (or the axis average `EF4_gap`) assumes equal enhancement at the laser
and Stokes wavelengths and the field at the gap centre; `analysis.raman_enhancement` uses the
computed spectrum at both wavelengths, but neither is a Raman cross-section.

**Boundary treatments outside the validated set.** Validated: Au tip; Au and SiO2 substrates;
the WS2 monolayer on Au. Supported but not validated here: other metals (e.g. Meep's Ag),
metal or thick dispersive layers, and dispersive non-metallic substrates crossing the PML.
Run the checklist of [setting_up_your_simulation.md](setting_up_your_simulation.md) before
trusting such a case.

**Small domain.** The small domain gives absolute values 2.5 times those of the production
domain and depends at first order on where the drive ramp sits relative to the tip; only
ratios at equal settings are meaningful there.
