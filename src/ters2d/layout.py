"""Grid-aligned layout of the isolated tip / gap / sample problem (pure Python, no Meep).

The cell is closed by PML (no PML below an opaque metal substrate). The substrate and
the layers run through the side PML (dispersive layers stop at it), the tip shaft runs
through the top PML (semi-infinite tip). There are no periodic boundaries.

Grid alignment: Meep does not sub-pixel average dispersive materials, so all interfaces
are staircased. The sample surface, every layer interface and the lowest point of the
apex lie on grid lines: the gap and the layer thicknesses are whole numbers of cells.
The curved apex and the wedge faces are staircased differently at each resolution. The
cell must have an even number of pixels along x and y: Meep centres an odd-sized grid
half a cell off, which would misplace the scattered-field sources relative to the tip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .background import Stack
from .materials import is_metallic
from .model import Model

OPAQUE_ROUND_TRIP = 1e-2  # max amplitude of a wave crossing an opaque substrate without bottom PML and back
NUDGE_CELLS = (
    1e-3  # interfaces are shifted up by this fraction of a cell: a grid point on an interface belongs to the medium below
)


def _snap(v: float, dx: float, mode: str) -> float:
    n = v / dx
    n = np.floor(n + 1e-9) if mode == "floor" else np.ceil(n - 1e-9)
    return float(n * dx)


@dataclass(frozen=True)
class TipShape:
    """Circle of radius ``radius`` centred on the axis, plus the wedge tangent to it (um, rad)."""

    radius: float
    half_angle: float
    apex_y: float
    center: tuple[float, float]
    tangent_x: float  # the wedge faces touch the circle at (+-tangent_x, tangent_y)
    tangent_y: float

    def half_width(self, y):
        """Half width of the tip at height y (NaN below the apex)."""
        y = np.asarray(y, dtype=float)
        cy = self.center[1]
        out = np.where(
            y >= self.tangent_y,
            self.tangent_x + (y - self.tangent_y) * np.tan(self.half_angle),
            np.sqrt(np.clip(self.radius**2 - (y - cy) ** 2, 0, None)),
        )
        return np.where(y < self.apex_y, np.nan, out)

    def contains(self, x, y):
        """Points inside the tip (the same inside test as the staircased Meep geometry)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cx, cy = self.center
        in_circle = (x - cx) ** 2 + (y - cy) ** 2 <= self.radius**2
        in_wedge = (y >= self.tangent_y) & (np.abs(x) <= self.tangent_x + (y - self.tangent_y) * np.tan(self.half_angle))
        return in_circle | in_wedge

    def outline(self, y_top: float, n_arc: int = 121) -> np.ndarray:
        """Closed polygon (x, y) of the tip cross-section up to height y_top (for plots)."""
        a = self.half_angle
        cx, cy = self.center
        phi = np.linspace(-np.pi / 2 - (np.pi / 2 - a), -np.pi / 2 + (np.pi / 2 - a), n_arc)
        arc = np.c_[cx + self.radius * np.cos(phi), cy + self.radius * np.sin(phi)]
        xr = self.tangent_x + (y_top - self.tangent_y) * np.tan(a)
        return np.vstack([[-xr, y_top], arc, [xr, y_top], [-xr, y_top]])


def tip_shape(radius_um: float, half_angle_deg: float, apex_y_um: float) -> TipShape:
    a = np.deg2rad(half_angle_deg)
    cy = apex_y_um + radius_um
    return TipShape(radius_um, a, apex_y_um, (0.0, cy), radius_um * np.cos(a), cy - radius_um * np.sin(a))


def snapped_thickness(thickness_um: float, dx: float, rel_tol: float = 0.05) -> float:
    """Layer thickness rounded to whole cells; error if that is off by more than rel_tol."""
    n = int(round(thickness_um / dx))
    t = n * dx
    if n < 1 or abs(t - thickness_um) > rel_tol * thickness_um:
        raise ValueError(
            f"a {thickness_um * 1e3:.3f} nm layer cannot be represented on a {dx * 1e3:.4f} nm grid within "
            f"{rel_tol:.0%} (nearest: {t * 1e3:.3f} nm); choose another resolution"
        )
    return t


@dataclass(frozen=True)
class Layout:
    """Everything about one geometry that is fixed by the grid (lengths in um)."""

    model: Model
    dx: float
    cell_lo: tuple[float, float]
    cell_hi: tuple[float, float]
    phys_lo: tuple[float, float]  # the PML-free region
    phys_hi: tuple[float, float]
    layers: tuple[tuple[float, float], ...]  # (y_bottom, y_top) of each layer, in Sample.layers order (bottom up)
    tip: TipShape
    gap: float  # apex height above the sample surface
    probe: tuple[float, float]
    focus: tuple[float, float]  # beam focus: the sample surface below the apex
    substrate_metallic: bool
    metallic_layers: tuple[bool, ...]
    bottom_pml: bool
    side_absorbers: bool
    boundary_layers: tuple  # (thickness, axis, side)

    @property
    def thickness(self) -> float:
        """Total (snapped) thickness of the layers."""
        return -self.layers[0][0] if self.layers else 0.0

    @property
    def cell_size(self):
        return (self.cell_hi[0] - self.cell_lo[0], self.cell_hi[1] - self.cell_lo[1])

    @property
    def cell_center(self):
        return (0.5 * (self.cell_hi[0] + self.cell_lo[0]), 0.5 * (self.cell_hi[1] + self.cell_lo[1]))

    @property
    def n_cells(self) -> int:
        sx, sy = self.cell_size
        return int(round(sx / self.dx)) * int(round(sy / self.dx))

    def stack(self, wavelength_nm: float) -> Stack:
        """Analytic description of the sample at one wavelength (snapped thicknesses)."""
        s = self.model.sample
        layers = []
        for lay, (y0, y1) in zip(reversed(s.layers), reversed(self.layers)):
            ex, ey = lay.material.epsilon_xy(wavelength_nm)
            layers.append((complex(ex), complex(ey), y1 - y0))
        ex, ey = s.substrate.epsilon_xy(wavelength_nm)
        return Stack((complex(ex), complex(ey)), tuple(layers))


def skin_depth_round_trip(material, depth_um: float, wavelength_nm) -> np.ndarray:
    """Amplitude attenuation exp(-2 k0 Im(n) d) of a normally incident wave through depth d and back."""
    ex, _ = material.epsilon_xy(np.atleast_1d(np.asarray(wavelength_nm, dtype=float)))
    n_im = np.sqrt(ex + 0j).imag
    k0 = 2 * np.pi / (np.atleast_1d(np.asarray(wavelength_nm, dtype=float)) * 1e-3)
    return np.exp(-2 * k0 * np.abs(n_im) * depth_um)


def build_layout(model: Model, wavelengths_nm) -> Layout:
    """Grid-snapped layout; checks everything that can be checked before Meep runs.

    The wavelengths are needed to classify the materials (metal: Re eps < 0) and to check
    that an opaque substrate without a bottom PML is thick enough.
    """
    num = model.numerics
    res = int(num.resolution)
    dx = 1.0 / res
    wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    for mat in model.sample.materials() + [model.tip.material]:
        mat.check_wavelengths(wl)
    gap = model.gap_nm * 1e-3
    if abs(gap * res - round(gap * res)) > 1e-6:
        raise ValueError(f"gap {model.gap_nm} nm is not a whole number of {dx * 1e3:.4f} nm cells")

    # layers: bottom up, the top one ends at y = 0
    thick = [snapped_thickness(lay.thickness_nm * 1e-3, dx) for lay in model.sample.layers]
    total = float(sum(thick))
    layers, y = [], -total
    for t in thick:
        layers.append((y, y + t))
        y += t
    if layers:
        layers[-1] = (layers[-1][0], 0.0)

    sub_metal = is_metallic(model.sample.substrate, wl)
    lay_metal = tuple(is_metallic(lay.material, wl) for lay in model.sample.layers)
    bottom_pml = (not sub_metal) if num.bottom_pml is None else bool(num.bottom_pml)
    side_abs = (sub_metal or any(lay_metal)) if num.side_absorbers is None else bool(num.side_absorbers)

    phys_lo = [_snap(-num.x_half_width_um, dx, "floor"), _snap(-total - num.substrate_depth_um, dx, "floor")]
    phys_hi = [_snap(num.x_half_width_um, dx, "ceil"), _snap(num.y_max_um, dx, "ceil")]
    pml = _snap(num.pml_um, dx, "ceil")
    bottom = pml if bottom_pml else 0.0
    ny = round((phys_hi[1] - phys_lo[1] + pml + bottom) * res)
    if ny % 2:  # Meep would centre an odd grid half a cell off: extend the substrate by one cell
        phys_lo[1] -= dx
    cell_lo = (phys_lo[0] - pml, phys_lo[1] - bottom)
    cell_hi = (phys_hi[0] + pml, phys_hi[1] + pml)
    for lo, hi, axis in ((cell_lo[0], cell_hi[0], "x"), (cell_lo[1], cell_hi[1], "y")):
        if round((hi - lo) * res) % 2:
            raise ValueError(f"cell has an odd number of pixels along {axis}")
    boundary = [(pml, "x", "both"), (pml, "y", "high")] + ([(bottom, "y", "low")] if bottom_pml else [])

    if not bottom_pml:
        if not sub_metal:
            raise ValueError("a non-metallic substrate needs a bottom PML (Numerics.bottom_pml)")
        depth = -phys_lo[1] - total
        att = skin_depth_round_trip(model.sample.substrate, depth, wl)
        if np.any(att > OPAQUE_ROUND_TRIP):
            raise ValueError(
                f"substrate depth {depth:.3f} um is not opaque at {wl[att > OPAQUE_ROUND_TRIP]} nm (round-trip "
                f"attenuation {att.max():.1e} > {OPAQUE_ROUND_TRIP}): increase substrate_depth_um or use bottom_pml=True"
            )

    tip = tip_shape(model.tip.apex_radius_nm * 1e-3, model.tip.half_angle_deg, gap)
    if model.outputs.probe_nm is None:
        probe = (0.0, 0.5 * gap)
    else:
        probe = (model.outputs.probe_nm[0] * 1e-3, model.outputs.probe_nm[1] * 1e-3)
        if tip.contains(*probe) or probe[1] <= -total - num.substrate_depth_um:
            raise ValueError("the probe point lies inside the tip or below the free region")
    x0, x1, y0, y1 = (v * 1e-3 for v in model.outputs.near_box_nm)
    if not (phys_lo[0] < x0 and x1 < phys_hi[0] and phys_lo[1] < y0 and y1 < phys_hi[1]):
        raise ValueError("near_box_nm must lie inside the PML-free region")
    if not (x0 <= probe[0] <= x1 and y0 <= probe[1] <= y1):
        raise ValueError("the probe must lie inside near_box_nm")
    hw_top = float(tip.half_width(cell_hi[1]))
    if hw_top >= phys_hi[0]:
        raise ValueError("the tip shaft would enter the side PML; increase x_half_width_um")
    if side_abs and float(tip.half_width(num.side_absorber_top_um)) >= num.side_absorber_start_um:
        raise ValueError("the tip shaft would enter the side plasmon absorbers; increase side_absorber_start_um")
    return Layout(
        model=model,
        dx=dx,
        cell_lo=cell_lo,
        cell_hi=cell_hi,
        phys_lo=tuple(phys_lo),
        phys_hi=tuple(phys_hi),
        layers=tuple(layers),
        tip=tip,
        gap=gap,
        probe=probe,
        focus=(0.0, 0.0),
        substrate_metallic=sub_metal,
        metallic_layers=lay_metal,
        bottom_pml=bottom_pml,
        side_absorbers=side_abs,
        boundary_layers=tuple(boundary),
    )


def chunk_bands(layout: Layout) -> tuple[float, list]:
    """Chunk layout that keeps air out of the dispersive chunks (single-process runs).

    Returns the y of the split above the sample and a list of (y_top, half_width) bands
    above it; each band is split in x at +-half_width so that only its middle chunk
    contains the tip.
    """
    y_sub = 4 * layout.dx
    bands = []
    tops = [y for y in (0.3, 0.8, 1.5, 2.3, 3.0) if y_sub + 0.05 < y < layout.cell_hi[1] - 0.05] + [layout.cell_hi[1]]
    for y in tops:
        hw = float(layout.tip.half_width(np.array([y]))[0]) + 0.02
        bands.append((y, min(hw, 0.5 * layout.cell_size[0] - layout.dx)))
    return y_sub, bands
