"""Meep objects for a Layout: media, geometry, boundary layers, chunk layout (imports Meep).

Boundary rules (see docs/model.md, "Boundaries"):

* Meep's PML is unstable where a dispersive metal crosses it (Meep FAQ, "Why are the
  fields blowing up in my simulation?"). Inside the PML the tip shaft (if dispersive) and
  a metallic substrate are therefore replaced by a non-dispersive conductor
  (``Numerics.pml_metal_conductivity``). Non-dispersive media continue into the PML.
* Dispersive layers stop at the side PML (a run with the anisotropic Lorentz WS2 layer
  crossing Meep's PML diverged); non-dispersive layers continue into it.
* Surface plasmons of the shaft and of a metallic surface are absorbed by graded,
  impedance-matched lossy air (equal electric and magnetic conductivity) before they reach
  the PML: around the shaft above ``shaft_absorber_start_um`` and, with side absorbers,
  above the sample for |x| >= ``side_absorber_start_um``. The metal itself is unchanged.
"""

from __future__ import annotations

import numpy as np

from .layout import NUDGE_CELLS, Layout, chunk_bands
from .materials import AnyMaterial, Material, UniaxialMaterial


def _valid_range(materials):
    import meep as mp

    ranges = [m.valid_wavelength_nm for m in materials if m.valid_wavelength_nm is not None]
    if not ranges:
        return mp.FreqRange(min=-1e20, max=1e20)
    lo = max(r[0] for r in ranges)
    hi = min(r[1] for r in ranges)
    return mp.FreqRange(min=1e3 / hi, max=1e3 / lo)


def _susceptibilities(material: Material, sigma_diag=None):
    import meep as mp

    out = []
    for p in material.poles:
        kw = {"frequency": p.frequency, "gamma": p.gamma}
        if sigma_diag is None:
            kw["sigma"] = p.sigma
        else:
            kw["sigma_diag"] = mp.Vector3(*(p.sigma * s for s in sigma_diag))
        cls = mp.LorentzianSusceptibility if p.kind == "lorentz" else mp.DrudeSusceptibility
        out.append(cls(**kw))
    return out


def meep_medium(material: AnyMaterial):
    """Meep medium of a Material (isotropic) or UniaxialMaterial (optic axis along y)."""
    import meep as mp

    if isinstance(material, Material):
        if not material.dispersive:
            return mp.Medium(epsilon=material.eps_inf)
        return mp.Medium(
            epsilon=material.eps_inf,
            E_susceptibilities=_susceptibilities(material),
            valid_freq_range=_valid_range([material]),
        )
    if isinstance(material, UniaxialMaterial):
        ip, op = material.in_plane, material.out_of_plane
        susc = _susceptibilities(ip, (1.0, 0.0, 1.0)) + _susceptibilities(op, (0.0, 1.0, 0.0))
        return mp.Medium(
            epsilon_diag=mp.Vector3(ip.eps_inf, op.eps_inf, ip.eps_inf),
            E_susceptibilities=susc,
            valid_freq_range=_valid_range([ip, op]),
        )
    raise TypeError(f"not a material: {material!r}")


def pml_conductor(layout: Layout):
    """Non-dispersive lossy medium that replaces a dispersive metal inside a PML (stable there)."""
    import meep as mp

    return mp.Medium(epsilon=1.0, D_conductivity=layout.model.numerics.pml_metal_conductivity)


def absorber_media(layout: Layout):
    """Graded impedance-matched lossy air (equal D and B conductivity), one medium per segment."""
    import meep as mp

    num = layout.model.numerics
    n = num.absorber_segments
    return [
        mp.Medium(epsilon=1.0, D_conductivity=sig, B_conductivity=sig)
        for sig in (num.absorber_conductivity * ((k + 0.5) / n) ** num.absorber_power for k in range(n))
    ]


def meep_geometry(layout: Layout) -> list:
    """Meep geometric objects. Later objects take precedence over earlier ones."""
    import meep as mp

    model = layout.model
    num = model.numerics
    nudge = NUDGE_CELLS * layout.dx
    big = 100.0
    objs = []
    tip = layout.tip
    absorber = absorber_media(layout)

    # tip: slab above the tangent line (conductor inside the top PML if dispersive); the two
    # sides are carved by rotated blocks: air up to the shaft absorber, graded lossy air along
    # the faces from shaft_absorber_start_um to the top of the free region, air in the PML
    tip_med = meep_medium(model.tip.material)
    tip_pml = pml_conductor(layout) if model.tip.material.dispersive else tip_med
    a = tip.half_angle
    y_pml = layout.phys_hi[1]
    y_far = layout.cell_hi[1] + 1.0
    for y0, y1, mat in ((tip.tangent_y, y_pml, tip_med), (y_pml, y_far, tip_pml)):
        objs.append(mp.Block(center=mp.Vector3(0, 0.5 * (y0 + y1)), size=mp.Vector3(2 * big, y1 - y0, mp.inf), material=mat))
    d_abs = (num.shaft_absorber_start_um - tip.tangent_y) / np.cos(a)  # distance along a face
    d_top = (y_pml - tip.tangent_y) / np.cos(a)
    edges = np.linspace(d_abs, d_top, num.absorber_segments + 1)
    bands = [(-big, d_abs, mp.air)] + [(d0, d1, m) for d0, d1, m in zip(edges[:-1], edges[1:], absorber)] + [(d_top, big, mp.air)]
    for s in (-1.0, 1.0):
        n = np.array([s * np.cos(a), -np.sin(a)])  # outward normal of the face
        d = np.array([s * np.sin(a), np.cos(a)])  # up along the face
        p = np.array([s * tip.tangent_x, tip.tangent_y])
        for d0, d1, mat in bands:
            c = p + n * (0.5 * big) + d * (0.5 * (d0 + d1))
            objs.append(
                mp.Block(
                    center=mp.Vector3(*c),
                    size=mp.Vector3(big, d1 - d0, mp.inf),
                    e1=mp.Vector3(*n),
                    e2=mp.Vector3(*d),
                    e3=mp.Vector3(0, 0, 1),
                    material=mat,
                )
            )
    objs.append(mp.Cylinder(radius=tip.radius, center=mp.Vector3(*tip.center), material=tip_med))

    # substrate: below the lowest layer; a metallic substrate becomes the conductor inside the PML
    sub_top = -layout.thickness + nudge
    sub = meep_medium(model.sample.substrate)
    x_pml = layout.phys_hi[0]
    if layout.substrate_metallic:
        cond = pml_conductor(layout)
        segs = [(0.0, x_pml, sub), (x_pml, big, cond)]
    else:
        segs = [(0.0, big, sub)]
    for x0, x1, mat in segs:  # |x| ranges, mirrored to both sides
        for sgn in (1.0,) if x0 == 0.0 else (-1.0, 1.0):
            cx = 0.0 if x0 == 0.0 else sgn * 0.5 * (x0 + x1)
            w = 2 * x1 if x0 == 0.0 else x1 - x0
            objs.append(
                mp.Block(center=mp.Vector3(cx, 0.5 * (sub_top - big)), size=mp.Vector3(w, sub_top + big, mp.inf), material=mat)
            )
    if layout.substrate_metallic and layout.bottom_pml:
        y_b = layout.phys_lo[1]
        objs.append(
            mp.Block(
                center=mp.Vector3(0, 0.5 * (y_b - big)), size=mp.Vector3(2 * big, big, mp.inf), material=pml_conductor(layout)
            )
        )

    if layout.side_absorbers:
        # surface-plasmon absorber: graded lossy air above the sample near the sides
        edges = np.linspace(num.side_absorber_start_um, x_pml, num.absorber_segments + 1)
        y0, y1 = nudge, num.side_absorber_top_um
        for x0, x1, mat in zip(edges[:-1], edges[1:], absorber):
            for sgn in (-1.0, 1.0):
                objs.append(
                    mp.Block(
                        center=mp.Vector3(sgn * 0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                        size=mp.Vector3(x1 - x0, y1 - y0, mp.inf),
                        material=mat,
                    )
                )

    # layers (bottom up); dispersive layers end at the side PML
    for lay, (yb, yt) in zip(model.sample.layers, layout.layers):
        if lay.material.dispersive:
            x0, x1 = layout.phys_lo[0], layout.phys_hi[0]
        else:
            x0, x1 = -big, big
        y0, y1 = yb + nudge, yt + nudge
        objs.append(
            mp.Block(
                center=mp.Vector3(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                size=mp.Vector3(x1 - x0, y1 - y0, mp.inf),
                material=meep_medium(lay.material),
            )
        )
    return objs


def meep_boundary_layers(layout: Layout) -> list:
    import meep as mp

    out = []
    for d, axis, side in layout.boundary_layers:
        kw = {"direction": mp.X if axis == "x" else mp.Y}
        if side != "both":
            kw["side"] = mp.High if side == "high" else mp.Low
        out.append(mp.PML(d, **kw))
    return out


def meep_chunk_layout(layout: Layout):
    """BinaryPartition that keeps air out of the dispersive chunks, or None for Meep's default."""
    import meep as mp

    mode = layout.model.numerics.chunk_layout
    nproc = mp.count_processors()
    if mode == "meep" or (mode == "auto" and nproc > 1):
        return None
    if nproc > 1:
        raise ValueError("chunk_layout='banded' is a single-process layout; use 'auto' or 'meep' with MPI")
    y_sub, bands = chunk_bands(layout)
    tree = 0
    for y_top, hw in reversed(bands):
        band = [(mp.X, -hw), 0, [(mp.X, hw), 0, 0]]
        tree = band if tree == 0 else [(mp.Y, y_top), band, tree]
    return mp.BinaryPartition(data=[(mp.Y, y_sub), 0, tree])


def make_simulation(layout: Layout):
    import meep as mp

    kw = {}
    chunks = meep_chunk_layout(layout)
    if chunks is not None:
        kw["chunk_layout"] = chunks
    sx, sy = layout.cell_size
    return mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        geometry_center=mp.Vector3(*layout.cell_center),
        resolution=layout.model.numerics.resolution,
        geometry=meep_geometry(layout),
        sources=[],
        boundary_layers=meep_boundary_layers(layout),
        Courant=layout.model.numerics.courant,
        eps_averaging=False,
        **kw,
    )
