# Documentation

`ters2d` simulates the near field of gap-mode tip-enhanced Raman spectroscopy (TERS) in 2D
with Meep: a metal tip a few nm above a metal or dielectric sample, optionally covered by a
2D material, illuminated by a focused p-polarized beam. It is not a general-purpose FDTD
library; it is one validated TERS model, organized so that the tip, gap, sample, materials,
illumination, outputs and numerical settings can be changed for another case.

| page | read it for |
|---|---|
| [model.md](model.md) | the physics and the numerical method: geometry, materials, beam synthesis, scattered-field formulation, discretization, boundaries and absorbers, and why the design is as it is |
| [setting_up_your_simulation.md](setting_up_your_simulation.md) | changing the tip, gap, sample, 2D layer, materials (from your own n, k data), illumination and numerics; the convergence checklist; run times on CPU Meep and gpmeep |
| [outputs.md](outputs.md) | what a run returns and writes: enhancement factors, field maps, files |
| [validation.md](validation.md) | every check against analytic and frequency-domain references, and the regression check of this code |
| [results.md](results.md) | the results: Au vs SiO2 spectra, near-field maps, gap and radius dependence, a WS2 monolayer in the gap |
| [limitations.md](limitations.md) | what the model does not describe |
| [api.md](api.md) | classes and functions |

Scripts: `examples/` (templates, simplest first), `reproduce/` (the reference results and all
figures), `validation/` (the checks), `materials/` (n, k data and the WS2 fit).
