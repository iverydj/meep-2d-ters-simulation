"""Scattered-field excitation of the tip by the analytic tip-free beam.

The total field is split as E = E_bg + E_s, with E_bg the analytic tip-free field of
``background.py`` (focused beam plus the exact response of the layered sample).
Subtracting the tip-free equations from Maxwell's equations of the full structure gives

    curl H_s = -i omega eps E_s + J,     J = -i omega (eps_tip - 1) E_bg   inside the tip,

so the tip is driven by its own induced polarization current; no source sits in air next
to the sample. Meep propagates E_s in the full structure (tip, sample, boundary layers).
The tip-free reference is E_s = 0, i.e. E_ref = E_bg exactly, so it needs no simulation.

Why not a line or plane source in air: any localized source near a metal surface launches
surface plasmon polaritons through its near field, which a focused beam in free space
cannot do. A Huygens line source 0.2 um above Au (633 nm) produced an SPP with ~40 % of the
reflected-field amplitude at the gap; a total-field/scattered-field plane crossing the Au
surface produced a comparable artefact. The scattered-field formulation has neither.

The amplitudes are evaluated at the Yee positions of Ex = ((i + 1/2) dx, j dx) and
Ey = (i dx, (j + 1/2) dx) with the same inside test as the staircased Meep geometry, for
the single frequency of the run (exact beam angles at that wavelength).
"""

from __future__ import annotations

import numpy as np

from .background import GaussianBeam, Stack, beam_fields_grid
from .layout import Layout


def background_beam(layout: Layout, wavelength_nm: float) -> GaussianBeam:
    ill = layout.model.illumination
    return GaussianBeam(wavelength_nm * 1e-3, ill.incidence_deg, ill.divergence_deg, layout.focus, ill.n_waves)


def tip_box(layout: Layout):
    """Rectangle (x0, x1, y0, y1) containing the driven part of the tip (apex to the end of the drive ramp)."""
    tip = layout.tip
    dx = layout.dx
    y0, y1 = tip.apex_y - dx, layout.model.numerics.drive_ramp_um[1]
    hw = float(tip.half_width(np.array([y1]))[0]) + 2 * dx
    return (-hw, hw, y0, y1)


def _yee_axis(lo, hi, dx, offset):
    """Yee coordinates (n + offset) dx inside [lo, hi]."""
    n0 = int(np.ceil(lo / dx - offset - 1e-9))
    n1 = int(np.floor(hi / dx - offset + 1e-9))
    return (np.arange(n0, n1 + 1) + offset) * dx


def drive_weight(y, ramp):
    """cos^2 ramp from 1 at ramp[0] to 0 at ramp[1]."""
    y = np.asarray(y, dtype=float)
    u = np.clip((y - ramp[0]) / (ramp[1] - ramp[0]), 0.0, 1.0)
    return np.cos(0.5 * np.pi * u) ** 2


def volume_sources(shape, box, dx: float, beam: GaussianBeam, stack: Stack, eps_scatterer: complex, src_time, ramp=None):
    """Meep sources J = -i omega (eps - 1) E_bg on the Ex/Ey Yee points inside ``shape``.

    ``shape.contains(x, y)`` must reproduce the staircased Meep geometry (cells with an even
    number of pixels); ``box`` = (x0, x1, y0, y1) must contain the scatterer (only its top
    edge may cut it). The scatterer must sit in air. ``ramp`` = (y0, y1) optionally weights
    the drive by ``drive_weight`` (illuminated length of a semi-infinite shaft).

    Meep maps amp_data element i to the centre of the i-th of n equal slices of the source
    volume and gives the outermost grid rows fractional weights, so each array gets one
    row/column of zeros on every side and the volume extends half a cell beyond them:
    every Yee point then receives exactly its own amplitude.
    Returns (sources, number of driven points).
    """
    import meep as mp

    omega = 2 * np.pi / beam.wavelength
    pref = -1j * omega * (eps_scatterer - 1.0)
    x0, x1, y0, y1 = box
    sources = []
    n_points = 0
    for comp, (ox, oy), idx in ((mp.Ex, (0.5, 0.0), 1), (mp.Ey, (0.0, 0.5), 2)):
        xs = _yee_axis(x0, x1, dx, ox)
        ys = _yee_axis(y0, y1, dx, oy)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        inside = shape.contains(X, Y)
        if not inside.any():
            raise ValueError("source box does not contain any point of the scatterer")
        if inside[0].any() or inside[-1].any() or inside[:, 0].any():
            raise ValueError("source box cuts the scatterer (only the top edge may)")
        e_bg = beam_fields_grid(beam, stack, xs, ys)[idx]
        amp = np.zeros((xs.size + 2, ys.size + 2, 1), dtype=np.complex128)
        w = drive_weight(Y, ramp) if ramp is not None else 1.0
        amp[1:-1, 1:-1, 0] = np.where(inside, pref * e_bg * w, 0.0)
        n_points += int(inside.sum())
        center = mp.Vector3(0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1]))
        size = mp.Vector3((xs.size + 2) * dx, (ys.size + 2) * dx)
        sources.append(mp.Source(src_time, component=comp, center=center, size=size, amp_data=amp))
    return sources, n_points


def tip_sources(layout: Layout, wavelength_nm: float, src_time):
    """Scattered-field sources of the tip at one wavelength."""
    beam = background_beam(layout, wavelength_nm)
    stack = layout.stack(wavelength_nm)
    eps_tip = complex(layout.model.tip.material.epsilon(wavelength_nm))
    ramp = layout.model.numerics.drive_ramp_um
    return volume_sources(layout.tip, tip_box(layout), layout.dx, beam, stack, eps_tip, src_time, ramp=ramp)
