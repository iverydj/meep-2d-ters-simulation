"""Nanogap benchmark, FEM side: frequency-domain FEM (NGSolve) of an Au cylinder
(R = 10-40 nm) 1-10 nm above Au in the same focused beam.

The FDTD counterpart is nanogap_fdtd.py. The FEM uses curved high-order
elements with local refinement in the gap, so it has no staircasing; comparing the two at
several FDTD resolutions measures the staircase error of the Meep grid for exactly this
kind of gap. Same scattered-field formulation for Hz:

    int eps^-1 grad(H_s).grad(v) - k^2 H_s v  =  - int_cyl (eps_Au^-1 - 1) grad(H_bg).grad(v),

with PML on all sides (frequency domain: no instability with metal in the PML).
Requires NGSolve (pip install ngsolve). Pure CPU.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ters2d import write_csv
from ters2d.background import GaussianBeam, Stack, beam_fields, stack_coefficients
from ters2d.materials import AU


@dataclass
class AppConfig:
    out_csv: Path = Path("results") / platform.node() / "nanogap" / "nanogap_fem.csv"
    # (gap_nm, radius_nm, wavelength_nm, polynomial order, mesh size in the gap zone / gap,
    #  half width of the PML-free region [um], PML thickness [um])
    runs: list = field(
        default_factory=lambda: [
            *[(2.0, 20.0, wl, o, h, 0.5, 0.3) for wl in (633.0, 785.0) for o, h in ((3, 0.2), (4, 0.125), (5, 0.075), (6, 0.05))],
            (2.0, 20.0, 633.0, 5, 0.075, 0.8, 0.5),  # larger domain and PML
            *[(g, 20.0, 633.0, o, h, 0.5, 0.3) for g in (1.0, 4.0, 10.0) for o, h in ((4, 0.125), (5, 0.075))],
            *[(2.0, r, 633.0, o, h, 0.5, 0.3) for r in (10.0, 40.0) for o, h in ((4, 0.125), (5, 0.075))],
        ]
    )
    incidence_deg: float = 60.0
    divergence_deg: float = 14.0
    n_waves: int = 241
    height_um: float = 0.5
    substrate_um: float = 0.15


def solve(cfg: AppConfig, gap_nm, radius_nm, wl_um, order, h_rel, half_width_um, pml_um, substrate: str = "Au") -> dict:
    """EF at the gap centre; substrate='air' gives the free-space cylinder (checked against Mie)."""
    import ngsolve as ngs
    from netgen.occ import Circle, Glue, MoveTo, OCCGeometry

    k = 2 * np.pi / wl_um
    eps_au = complex(AU.epsilon(wl_um * 1e3))
    eps_sub = eps_au if substrate == "Au" else 1.0
    R, g = radius_nm * 1e-3, gap_nm * 1e-3
    h_gap = h_rel * g
    Lx, Ly, Ds, T = half_width_um, cfg.height_um, cfg.substrate_um, pml_um

    outer = MoveTo(-Lx - T, -Ds - T).Rectangle(2 * (Lx + T), Ds + Ly + 2 * T).Face()
    inner = MoveTo(-Lx, -Ds).Rectangle(2 * Lx, Ds + Ly).Face()
    sub_all = MoveTo(-Lx - T, -Ds - T).Rectangle(2 * (Lx + T), Ds + T).Face()
    cyl = Circle((0, g + R), R).Face()
    zone = Circle((0, 0.5 * g), 8 * g).Face()
    sub_in = sub_all * inner
    sub_pml = sub_all - inner
    air_in = inner - sub_all - cyl
    gapz = air_in * zone
    air_rest = air_in - zone
    air_pml = outer - inner - sub_all
    cyl_zone = cyl * zone
    cyl_rest = cyl - zone
    sub_zone = sub_in * zone
    sub_rest = sub_in - zone
    for f, name, h in (
        (cyl_zone, "cyl", h_gap),
        (cyl_rest, "cyl", 0.002),
        (sub_zone, "sub", h_gap),
        (sub_rest, "sub", 0.01),
        (sub_pml, "sub_pml", 0.04),
        (gapz, "gap", h_gap),
        (air_rest, "air", 0.04),
        (air_pml, "air_pml", 0.06),
    ):
        f.faces.name = name
        f.faces.maxh = h
    geo = Glue([cyl_zone, cyl_rest, sub_zone, sub_rest, sub_pml, gapz, air_rest, air_pml])
    mesh = ngs.Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=0.06, grading=0.25))
    mesh.Curve(order)
    mesh.SetPML(ngs.pml.Cartesian(mins=(-Lx, -Ds), maxs=(Lx, Ly), alpha=2j), definedon="sub_pml|air_pml")

    eps = mesh.MaterialCF({"cyl": eps_au, "sub": eps_sub, "sub_pml": eps_sub}, default=1.0)
    fes = ngs.H1(mesh, order=order, complex=True)
    u, v = fes.TnT()
    a = ngs.BilinearForm(fes, symmetric=True)
    a += (1 / eps) * ngs.grad(u) * ngs.grad(v) * ngs.dx - k**2 * u * v * ngs.dx
    a.Assemble()

    # analytic background (air formula) and its gradient inside the cylinder
    beam = GaussianBeam(wl_um, cfg.incidence_deg, cfg.divergence_deg, (0.0, 0.0), n_waves=cfg.n_waves)
    th, w = beam.angles_weights()
    kx = k * np.sin(th)
    c = stack_coefficients(Stack.isotropic(eps_sub), kx, k)
    ky, r = c["ky"][0].real, c["r"]
    dhx = ngs.CoefficientFunction(0)
    dhy = ngs.CoefficientFunction(0)
    for wj, kxj, kyj, rj in zip(w, kx, ky, r):
        ex = ngs.exp(1j * kxj * ngs.x)
        dn, up = ngs.exp(-1j * kyj * ngs.y), ngs.exp(1j * kyj * ngs.y)
        dhx += wj * 1j * kxj * ex * (dn + rj * up)
        dhy += wj * ex * (-1j * kyj * dn + 1j * kyj * rj * up)
    grad_bg = ngs.CoefficientFunction((dhx, dhy)).Compile()
    f = ngs.LinearForm(fes)
    f += -(1 / eps_au - 1) * grad_bg * ngs.grad(v) * ngs.dx(definedon=mesh.Materials("cyl"))
    f.Assemble()
    gfu = ngs.GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    p = mesh(0.0, 0.5 * g)
    gx, gy = ngs.grad(gfu)(p)
    es = np.array([1j / k * gy, -1j / k * gx])  # E = (i/k) (dH/dy, -dH/dx) in air
    _, ebx, eby = beam_fields(
        GaussianBeam(wl_um, cfg.incidence_deg, cfg.divergence_deg, (0.0, 0.0)),
        Stack.isotropic(eps_sub),
        np.array([0.0]),
        np.array([0.5 * g]),
    )
    eb = np.array([ebx[0], eby[0]])
    ef = float(np.sum(np.abs(eb + es) ** 2) / np.sum(np.abs(eb) ** 2))
    return {
        "gap_nm": gap_nm,
        "radius_nm": radius_nm,
        "wavelength_nm": wl_um * 1e3,
        "half_width_um": half_width_um,
        "pml_um": pml_um,
        "substrate": substrate,
        "order": order,
        "h_gap_nm": h_gap * 1e3,
        "ndof": fes.ndof,
        "EF_probe": ef,
        "E_tot_x": complex(eb[0] + es[0]),
        "E_tot_y": complex(eb[1] + es[1]),
    }


def main(cfg: AppConfig) -> None:
    import ngsolve as ngs

    ngs.SetNumThreads(8)
    rows = []
    with ngs.TaskManager():
        for gap, radius, wl, order, h_rel, hw, pml in cfg.runs:
            row = solve(cfg, gap, radius, wl * 1e-3, order, h_rel, hw, pml)
            rows.append(row)
            print(row, flush=True)
    print(f"wrote {write_csv(rows, cfg.out_csv)}")


if __name__ == "__main__":
    main(AppConfig())
