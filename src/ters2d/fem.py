"""Frequency-domain FEM reference of the same model (NGSolve; pure CPU, requires ``pip install ngsolve``).

Solves the scattered-field problem for Hz of exactly the model the FDTD simulates (tip,
sample, Gaussian beam, drive ramp, plasmon absorbers, conductor replacement inside the
PML; see ``meep_model``), but with curved high-order elements (no staircasing, refined in
the gap and in the layers) and a complex coordinate-stretching PML:

    int (eps^-1 grad H_s) . grad v - k^2 mu H_s v  =  - int_tip w(y) (eps_tip^-1 - 1) grad H_bg . grad v

with the lossy-air absorbers as eps = mu = 1 + i sigma / omega (continuous profile instead
of the FDTD steps). A typical case solves in well under a minute, so it is the fastest way
to see the converged answer of a model and to separate the FDTD discretization error from
the model itself. It shares every modelling choice with the FDTD, so it does not validate
those choices (see docs/limitations.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .background import GaussianBeam, Stack, beam_fields, stack_coefficients
from .layout import tip_shape
from .materials import is_metallic
from .model import Model


@dataclass(frozen=True)
class FemSettings:
    """Mesh and solver settings (lengths in um).

    order: polynomial order; pml_um, pml_alpha: PML thickness and stretching
    (x -> x + i alpha (x - x_pml)); h_gap_rel: element size in the gap zone / (gap + top
    layer); h_metal_um: element size in the metal near its surface (skin depth ~25 nm);
    h_air_um, h_pml_um: elsewhere; n_waves: plane waves of the drive beam;
    pml_metal: "conductor" (as in the FDTD) or "gold" (the metal continues into the PML,
    possible in the frequency domain).
    """

    order: int = 4
    pml_um: float = 0.5
    pml_alpha: float = 2.0
    h_gap_rel: float = 0.125
    h_metal_um: float = 0.02
    h_air_um: float = 0.04
    h_pml_um: float = 0.04
    n_waves: int = 241
    pml_metal: str = "conductor"
    threads: int = 16


def _absorber_sigma(ngs, lay, side: bool):
    """Conductivity of the lossy-air absorbers: the layout and profile of meep_model, but continuous."""
    tip, num = lay.tip, lay.model.numerics
    a = tip.half_angle
    smax, pw = num.absorber_conductivity, num.absorber_power
    d_abs = (lay.shaft_absorber_start - tip.tangent_y) / np.cos(a)
    d_top = (lay.y_top - tip.tangent_y) / np.cos(a)
    ax = ngs.IfPos(ngs.x, ngs.x, -ngs.x)
    d = (ax - tip.tangent_x) * np.sin(a) + (ngs.y - tip.tangent_y) * np.cos(a)
    u = (d - d_abs) / (d_top - d_abs)
    top = ngs.IfPos(u, ngs.IfPos(u - 1.0, 0.0, smax * u**pw), 0.0)
    if not side:
        return top
    us = (ax - num.side_absorber_start_um) / (num.x_half_width_um - num.side_absorber_start_um)
    sd = smax * ngs.IfPos(us - 1.0, 1.0, us) ** pw
    return ngs.IfPos(us, ngs.IfPos(ngs.y - lay.side_absorber_top, top, sd), top)


def _inv_tensor(ngs, ex, ey):
    """Coefficient of grad H: d/dx (1/eps_y) d/dx + d/dy (1/eps_x) d/dy."""
    return ngs.CoefficientFunction((1 / ey, 0, 0, 1 / ex), dims=(2, 2))


def solve(model: Model, wavelength_nm: float, settings: FemSettings = FemSettings()) -> dict:
    """EF at the probe (and in the top layer) of ``model`` at one wavelength.

    Uses the grid-snapped layer thicknesses of the FDTD at ``model.numerics.resolution``,
    so that FDTD and FEM describe the same structure.
    """
    import ngsolve as ngs
    from netgen.occ import Circle, Glue, MoveTo, OCCGeometry

    from .layout import build_layout

    lay = build_layout(model, [wavelength_nm])
    num = model.numerics
    wl = wavelength_nm * 1e-3
    k = 2 * np.pi / wl
    T, P = lay.thickness, settings.pml_um
    lx, ly = num.x_half_width_um, lay.y_top
    ds = T + num.substrate_depth_um
    g = lay.gap
    t_top = (lay.layers[-1][1] - lay.layers[-1][0]) if lay.layers else 0.0
    g_ref = g + t_top
    eps_tip = complex(model.tip.material.epsilon(wavelength_nm))
    tip = tip_shape(model.tip.apex_radius_nm * 1e-3, model.tip.half_angle_deg, g)
    y_end = ly + P + 0.05  # the wedge is cut just outside the top PML, on its true face lines
    hw_end = float(tip.half_width(y_end))
    if hw_end >= lx:
        raise ValueError("tip shaft would enter the side PML")

    outer = MoveTo(-lx - P, -ds - P).Rectangle(2 * (lx + P), ds + ly + 2 * P).Face()
    inner = MoveTo(-lx, -ds).Rectangle(2 * lx, ds + ly).Face()
    sub_all = MoveTo(-lx - P, -ds - P).Rectangle(2 * (lx + P), ds - T + P).Face()
    tx, ty = tip.tangent_x, tip.tangent_y
    wedge = MoveTo(-tx, ty).LineTo(tx, ty).LineTo(hw_end, y_end).LineTo(-hw_end, y_end).Close().Face()
    tip_all = (wedge + Circle(tip.center, tip.radius).Face()) * outer
    zone = Circle((0, 0.5 * g), 8 * g_ref).Face()
    skin = MoveTo(-lx - P, -T - 0.06).Rectangle(2 * (lx + P), 0.06).Face()
    wl_list = [wavelength_nm]
    regions = []
    air_all = outer - sub_all - tip_all
    layer_names = []
    for i, (layer, (yb, yt)) in enumerate(zip(model.sample.layers, lay.layers)):
        t = yt - yb
        through = not layer.material.dispersive  # as in the FDTD: dispersive layers stop at the side PML
        band = MoveTo(-lx - P if through else -lx, yb).Rectangle(2 * (lx + P) if through else 2 * lx, t).Face()
        h_out = min(1.5 * t, settings.h_metal_um if is_metallic(layer.material, wl_list) else settings.h_air_um)
        name = f"lay{i}"
        layer_names.append(name)
        regions += [(band * inner * zone, name, settings.h_gap_rel * g_ref), (band * inner - zone, name, h_out)]
        if through:
            regions.append((band - inner, name + "_pml", settings.h_pml_um))
        air_all = air_all - band
    regions += [
        (tip_all * inner * zone, "tip", settings.h_gap_rel * g_ref),
        (tip_all * inner - zone, "tip", settings.h_metal_um),
        (tip_all - inner, "tip_pml", min(settings.h_pml_um, 0.04)),
        (sub_all * inner * zone, "sub", settings.h_gap_rel * g_ref),
        (sub_all * inner * skin - zone, "sub", settings.h_metal_um),
        (sub_all * inner - skin - zone, "sub", 0.05),
        (sub_all - inner, "sub_pml", settings.h_pml_um),
        (air_all * inner * zone, "air", settings.h_gap_rel * g_ref),
        (air_all * inner - zone, "air", settings.h_air_um),
        (air_all - inner, "air_pml", settings.h_pml_um),
    ]
    for face, name, h in regions:
        face.faces.name = name
        face.faces.maxh = h
    mesh = ngs.Mesh(OCCGeometry(Glue([r[0] for r in regions]), dim=2).GenerateMesh(maxh=0.08, grading=0.3))
    mesh.Curve(settings.order)
    pml_regions = "|".join(
        ["tip_pml", "sub_pml", "air_pml"] + [n + "_pml" for n in layer_names if n + "_pml" in mesh.GetMaterials()]
    )
    mesh.SetPML(ngs.pml.Cartesian(mins=(-lx, -ds), maxs=(lx, ly), alpha=1j * settings.pml_alpha), definedon=pml_regions)

    def iso(e):
        return _inv_tensor(ngs, e, e)

    lossy = 1.0 + 1j * _absorber_sigma(ngs, lay, side=lay.side_absorbers) / k  # matched: eps = mu = 1 + i sigma / omega
    lossy_pml = 1.0 + 1j * _absorber_sigma(ngs, lay, side=False) / k  # the shaft absorber continues into the PML
    cond = 1.0 + 1j * num.pml_metal_conductivity / k
    if settings.pml_metal not in ("conductor", "gold"):
        raise ValueError("pml_metal must be 'conductor' or 'gold'")
    tip_pml = eps_tip if (settings.pml_metal == "gold" or not model.tip.material.dispersive) else cond
    sx, sy = model.sample.substrate.epsilon_xy(wavelength_nm)
    sx, sy = complex(sx), complex(sy)
    if lay.substrate_metallic and settings.pml_metal == "conductor":
        sub_pml = _inv_tensor(ngs, cond, cond)
    else:
        sub_pml = _inv_tensor(ngs, sx, sy)
    inv = {
        "tip": iso(eps_tip),
        "tip_pml": iso(tip_pml),
        "sub": _inv_tensor(ngs, sx, sy),
        "sub_pml": sub_pml,
        "air": iso(lossy),
        "air_pml": iso(lossy_pml),
    }
    layer_eps = []
    for layer, name in zip(model.sample.layers, layer_names):
        ex, ey = layer.material.epsilon_xy(wavelength_nm)
        ex, ey = complex(ex), complex(ey)
        layer_eps.append((ex, ey))
        inv[name] = inv[name + "_pml"] = _inv_tensor(ngs, ex, ey)
    inv_eps = mesh.MaterialCF(inv, default=iso(1.0))
    mu = mesh.MaterialCF({"air": lossy, "air_pml": lossy_pml}, default=1.0)
    fes = ngs.H1(mesh, order=settings.order, complex=True)
    u, v = fes.TnT()
    a = ngs.BilinearForm(fes, symmetric=True)
    a += (inv_eps * ngs.grad(u)) * ngs.grad(v) * ngs.dx - k**2 * mu * u * v * ngs.dx

    ill = model.illumination
    stack = lay.stack(wavelength_nm)
    beam = GaussianBeam(wl, ill.incidence_deg, ill.divergence_deg, (0.0, 0.0), settings.n_waves)
    th, w = beam.angles_weights()
    kx = k * np.sin(th)
    c = stack_coefficients(stack, kx, k)
    ky, r = c["ky"][0].real, c["r"]
    dhx = ngs.CoefficientFunction(0)
    dhy = ngs.CoefficientFunction(0)
    for wj, kxj, kyj, rj in zip(w, kx, ky, r):  # gradient of the tip-free Hz in air (the tip is in air)
        ex_ = ngs.exp(1j * kxj * ngs.x)
        dn, up = ngs.exp(-1j * kyj * ngs.y), ngs.exp(1j * kyj * ngs.y)
        dhx += wj * 1j * kxj * ex_ * (dn + rj * up)
        dhy += wj * ex_ * (-1j * kyj * dn + 1j * kyj * rj * up)
    r0, r1 = lay.drive_ramp
    u_r = ngs.IfPos(ngs.y - r0, ngs.IfPos(ngs.y - r1, 1.0, (ngs.y - r0) / (r1 - r0)), 0.0)
    weight = ngs.cos(0.5 * np.pi * u_r) ** 2  # sources.drive_weight
    grad_bg = ngs.CoefficientFunction((dhx, dhy)).Compile()
    f = ngs.LinearForm(fes)
    f += -(1 / eps_tip - 1) * weight * grad_bg * ngs.grad(v) * ngs.dx(definedon=mesh.Materials("tip"))
    ngs.SetNumThreads(settings.threads)
    with ngs.TaskManager():
        a.Assemble()
        f.Assemble()
        gfu = ngs.GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    full = GaussianBeam(wl, ill.incidence_deg, ill.divergence_deg, (0.0, 0.0), ill.n_waves)
    px, py = lay.probe
    gx, gy = ngs.grad(gfu)(mesh(px, py))
    es = np.array([1j / k * gy, -1j / k * gx])  # E = (i/k) (dH/dy, -dH/dx) in air
    _, ebx, eby = beam_fields(full, stack, np.array([px]), np.array([py]))
    eb = np.array([ebx[0], eby[0]])
    _, eix, eiy = beam_fields(full, Stack.isotropic(1.0), np.array([0.0]), np.array([0.0]), part="incident")
    e2 = float(np.sum(np.abs(eb + es) ** 2))
    out = {
        "wavelength_nm": wavelength_nm,
        "ndof": fes.ndof,
        "EF_probe": e2 / float(np.sum(np.abs(eb) ** 2)),
        "I_probe_tip_over_inc": e2 / float(abs(eix[0]) ** 2 + abs(eiy[0]) ** 2),
        "EF_layer": float("nan"),
    }
    if lay.layers:  # mean |E|^2 on the axis inside the top layer, E = (i/k) (dH/dy / eps_x, -dH/dx / eps_y)
        yb, yt = lay.layers[-1]
        ex_l, ey_l = layer_eps[-1]
        ys = yb + np.linspace(0.1, 0.9, 5) * (yt - yb)
        _, lbx, lby = beam_fields(full, stack, np.zeros_like(ys), ys)
        tip_e2 = ref_e2 = 0.0
        for yy, bx, by in zip(ys, lbx, lby):
            gx, gy = ngs.grad(gfu)(mesh(0.0, yy))
            tip_e2 += abs(bx + 1j / (k * ex_l) * gy) ** 2 + abs(by - 1j / (k * ey_l) * gx) ** 2
            ref_e2 += abs(bx) ** 2 + abs(by) ** 2
        out["EF_layer"] = float(tip_e2 / ref_e2)
    return out
