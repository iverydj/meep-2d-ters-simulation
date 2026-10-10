"""The simulation set-up: tip, sample, illumination, numerical settings, outputs.

Coordinates (2D cross-section in the xy plane; z is the invariant direction):

* y = 0 is the top surface of the sample (the surface the tip faces); the sample fills
  y < 0: the layers (top layer directly below y = 0) and the semi-infinite substrate;
* the tip axis is x = 0 and the lowest point of the apex is at y = gap (the air gap);
* the beam arrives from the upper left (x < 0, y > 0), incidence angle measured from the
  surface normal, p-polarized: in-plane E = (Ex, Ey), Hz out of plane.

Units: geometry of the tip, sample and gap in nm; sizes of the computational domain in
um (Meep's length unit); wavelengths in nm; angles in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace

from .materials import AU, AnyMaterial, Material


@dataclass(frozen=True)
class Tip:
    """Semi-infinite 2D tip: a wedge whose faces are tangent to a circular apex.

    apex_radius_nm: radius of the apex circle [nm].
    half_angle_deg: half opening angle of the wedge [deg].
    material: isotropic material of the tip (default Au, Rakic Lorentz-Drude).

    The shaft leaves the domain through the top boundary (no tip end, no back-reflection).
    In 2D the wedge is an infinitely long ridge along z, not a cone (see docs/limitations.md).
    """

    apex_radius_nm: float = 20.0
    half_angle_deg: float = 15.0
    material: Material = AU

    def __post_init__(self):
        if not self.apex_radius_nm > 0:
            raise ValueError("apex_radius_nm must be > 0")
        if not 0 < self.half_angle_deg < 60:
            raise ValueError("half_angle_deg must be in (0, 60)")
        if not isinstance(self.material, Material):
            raise TypeError("the tip material must be an isotropic Material")


@dataclass(frozen=True)
class Layer:
    """A film on the substrate, infinite in x (e.g. a 2D material or a dielectric spacer).

    material: Material or UniaxialMaterial (optic axis along the layer normal).
    thickness_nm: thickness [nm]. It is snapped to a whole number of grid cells; the
        snapped value must lie within 5 % of the requested one (otherwise an error asks
        for another resolution) and is used consistently in the FDTD and the analytic field.
    """

    material: AnyMaterial
    thickness_nm: float

    def __post_init__(self):
        if not self.thickness_nm > 0:
            raise ValueError("thickness_nm must be > 0")


@dataclass(frozen=True)
class Sample:
    """Semi-infinite substrate plus optional layers.

    substrate: Material or UniaxialMaterial filling y below the lowest layer.
    layers: films listed from the substrate upward; the last one is the top surface
        (y = 0) that faces the tip. Example: WS2 on 90 nm SiO2 on Si is
        ``Sample(si, layers=(Layer(SIO2, 90), Layer(WS2_MONOLAYER, 0.65)))``.
    """

    substrate: AnyMaterial = AU
    layers: tuple[Layer, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))
        for lay in self.layers:
            if not isinstance(lay, Layer):
                raise TypeError("layers must be Layer objects")

    @property
    def label(self) -> str:
        """e.g. 'Au', 'WS2/Au' (top layer first)."""
        names = [lay.material.name for lay in reversed(self.layers)] + [self.substrate.name]
        return "/".join(n.split(" ")[0] for n in names)

    def materials(self):
        return [self.substrate] + [lay.material for lay in self.layers]


@dataclass(frozen=True)
class Illumination:
    """Focused p-polarized 2D Gaussian beam.

    incidence_deg: angle of the beam axis from the surface normal [deg]; the beam comes
        from the upper left (x < 0).
    divergence_deg: half-angle of the Gaussian angular spectrum [deg] (1/e^2 of the
        intensity); waist w0 ~ wavelength / (pi * divergence), 1.3 wavelengths for 14 deg.
        The spectrum is cut at +-2 divergence_deg, so incidence_deg +- 2 divergence_deg
        must lie in (0, 90) deg.
    n_waves: number of plane waves in the angular superposition.

    The beam is in phase at the focus, the point of the sample surface below the apex
    (x = 0, y = 0), and is synthesized separately at every wavelength (exact angles).
    Only p polarization (TM: Hz, Ex, Ey) is modelled: in 2D, s polarization is the
    decoupled TE problem with E along the invariant z axis, which has no field component
    along the tip axis and no gap mode.
    """

    incidence_deg: float = 60.0
    divergence_deg: float = 14.0
    n_waves: int = 721

    def __post_init__(self):
        if not self.divergence_deg > 0:
            raise ValueError("divergence_deg must be > 0")
        if not (0 < self.incidence_deg - 2 * self.divergence_deg and self.incidence_deg + 2 * self.divergence_deg < 90):
            raise ValueError("incidence_deg +- 2 divergence_deg must lie in (0, 90) deg")
        if self.n_waves < 3:
            raise ValueError("n_waves must be >= 3")


@dataclass(frozen=True)
class Numerics:
    """Grid, computational domain, boundary layers, absorbers and time stepping.

    The defaults are the production settings of the reference results (1 nm cells,
    free region 3.6 um x 5.7 um, ~27 million cells). See docs/setting_up_your_simulation.md
    for what each knob does and how to check it.

    Grid
        resolution: pixels per um (1000 -> 1 nm cells). The gap and every layer must be a
            whole number of cells (layers within 5 %).
    Domain (the 'free region' is the part without PML)
        x_half_width_um: free region spans |x| <= x_half_width_um.
        y_max_um: top of the free region above the sample surface.
        substrate_depth_um: substrate kept inside the free region below the lowest layer
            (one extra cell is added if needed to make the pixel count even).
    Boundaries
        pml_um: PML thickness (sides, top, and bottom if used).
        bottom_pml: PML below the substrate. None: no PML below a metallic substrate (it
            is opaque; a skin-depth check enforces this), PML below anything else.
        pml_metal_conductivity: Meep's PML is unstable where a dispersive metal crosses
            it; inside the PML the tip shaft and a metallic substrate are replaced by this
            non-dispersive conductor (Meep D_conductivity, epsilon = 1).
    Plasmon absorbers (graded, impedance-matched lossy air; act on the scattered field)
        shaft_absorber_start_um: height on the shaft faces where the shaft absorber starts;
            it fills the air beside the shaft up to the top of the free region.
        side_absorbers: absorbers above the sample for |x| >= side_absorber_start_um and
            y <= side_absorber_top_um, for the plasmons of a metal surface. None: on if the
            substrate or a layer is metallic.
        side_absorber_start_um, side_absorber_top_um: extent of the side absorbers.
        absorber_conductivity: electric and magnetic conductivity at the outer end, in Meep's
            angular-frequency units (c/um): eps = mu = 1 + i sigma / omega, omega = 2 pi / wavelength_um;
            the profile grows as (s / L) ** absorber_power in absorber_segments steps.
    Drive of the shaft
        drive_ramp_um: (y0, y1): the scattered-field drive on the tip has full weight up
            to y0 and falls to 0 at y1 (cos^2). Place it inside the shaft absorber, where
            the beam no longer reaches the shaft.
        max_shaft_illumination: a run is refused if |E_bg|^2 on the shaft faces at the start
            of the ramp exceeds this fraction of |E_bg|^2 at the probe (the cut of the drive
            would then change EF by more than ~1 %); None disables the check.
    Heights (y_max_um, shaft_absorber_start_um, side_absorber_top_um, drive_ramp_um) are
    measured from the substrate surface, i.e. from the bottom of the layers.
    Time stepping
        courant: Courant number (Meep default 0.5).
        pulse_fwidth_rel, pulse_cutoff: Gaussian pulse width / centre frequency and cutoff;
            only the centre frequency is analysed (one run per wavelength).
        dft_decay_tol: stop when the DFT fields change by less than this (relative) ...
        min_time_after_sources: ... but not before this time after the source, nor before
            the light has crossed the free region once ...
        max_time_after_sources: ... and fail if not converged by this time [um / c].
        chunk_layout: "auto" (banded layout for single-process runs, Meep's default for
            MPI), "banded" (keeps air out of the dispersive chunks; roughly halves the time
            per step on a GPU) or "meep".
    """

    resolution: int = 1000
    x_half_width_um: float = 1.8
    y_max_um: float = 5.5
    substrate_depth_um: float = 0.2
    pml_um: float = 0.4
    bottom_pml: bool | None = None
    pml_metal_conductivity: float = 20.0
    shaft_absorber_start_um: float = 3.5
    side_absorbers: bool | None = None
    side_absorber_start_um: float = 0.8
    side_absorber_top_um: float = 2.3
    absorber_conductivity: float = 10.0
    absorber_power: int = 3
    absorber_segments: int = 30
    drive_ramp_um: tuple[float, float] = (3.5, 4.5)
    max_shaft_illumination: float | None = 2e-3
    courant: float = 0.5
    pulse_fwidth_rel: float = 1.5
    pulse_cutoff: float = 4.0
    dft_decay_tol: float = 1e-4
    min_time_after_sources: float = 2.0
    max_time_after_sources: float = 80.0
    chunk_layout: str = "auto"

    def __post_init__(self):
        object.__setattr__(self, "drive_ramp_um", tuple(float(v) for v in self.drive_ramp_um))
        if int(self.resolution) != self.resolution or self.resolution <= 0:
            raise ValueError("resolution must be a positive integer (pixels per um)")
        if not (self.x_half_width_um > 0 and self.y_max_um > 0 and self.substrate_depth_um > 0):
            raise ValueError("x_half_width_um, y_max_um and substrate_depth_um must be > 0")
        if self.pml_um <= 0 or self.pml_metal_conductivity <= 0:
            raise ValueError("pml_um and pml_metal_conductivity must be > 0")
        if not 0 < self.shaft_absorber_start_um < self.y_max_um:
            raise ValueError("shaft_absorber_start_um must lie inside the free region")
        if not 0 < self.side_absorber_start_um < self.x_half_width_um:
            raise ValueError("side_absorber_start_um must lie inside the free region")
        if not 0 < self.side_absorber_top_um <= self.shaft_absorber_start_um:
            raise ValueError("side_absorber_top_um must lie between the sample and shaft_absorber_start_um")
        if self.absorber_conductivity <= 0 or self.absorber_segments < 1 or self.absorber_power < 1:
            raise ValueError("absorber_conductivity must be > 0, absorber_segments and absorber_power >= 1")
        r0, r1 = self.drive_ramp_um
        if not 0 < r0 < r1 <= self.y_max_um:
            raise ValueError("drive_ramp_um must satisfy 0 < start < end <= y_max_um")
        if self.max_shaft_illumination is not None and not self.max_shaft_illumination > 0:
            raise ValueError("max_shaft_illumination must be > 0 or None")
        if self.chunk_layout not in ("auto", "banded", "meep"):
            raise ValueError("chunk_layout must be 'auto', 'banded' or 'meep'")
        if not 0 < self.courant <= 0.5 * 2**0.5:
            raise ValueError("courant must be in (0, 1/sqrt(2)] for a 2D grid")

    @property
    def dx_nm(self) -> float:
        return 1e3 / self.resolution

    @classmethod
    def production(cls, resolution: int = 1000, **overrides) -> "Numerics":
        """Domain of the reference results (free region |x| <= 1.8 um, -0.2 <= y <= 5.5 um)."""
        return cls(resolution=resolution, **overrides)

    @classmethod
    def small(cls, resolution: int = 1000, **overrides) -> "Numerics":
        """Small domain for resolution and model studies (free region |x| <= 0.4 um, y <= 0.6 um).

        Same boundary treatment as the production domain, but short, steep absorbers and a
        shaft drive ramped down between 0.35 and 0.6 um, i.e. the shaft is illuminated
        over a much shorter length. Absolute enhancement factors differ from the production
        domain (2.5 times larger for the default case: EF 352.6 vs 139.7 at 1 nm cells), move
        at first order with the position of the drive ramp relative to the tip, and spectral
        shapes and substrate contrasts are not preserved. Two things transfer: the effect of a
        thin layer on the same substrate at the same wavelength, and changes with resolution.
        """
        kw = dict(
            x_half_width_um=0.4,
            y_max_um=0.6,
            substrate_depth_um=0.1,
            shaft_absorber_start_um=0.35,
            side_absorber_start_um=0.2,
            side_absorber_top_um=0.35,
            absorber_conductivity=40.0,
            absorber_power=2,
            absorber_segments=10,
            drive_ramp_um=(0.35, 0.6),
            max_shaft_illumination=None,  # by design: the drive is cut on an illuminated shaft
        )
        kw.update(overrides)
        return cls(resolution=resolution, **kw)


@dataclass(frozen=True)
class Outputs:
    """What is stored and evaluated (positions relative to the sample surface below the apex).

    near_box_nm: (x0, x1, y0, y1) region around the apex whose DFT fields are stored at
        full resolution [nm]. It must contain the probe and the whole gap.
    probe_nm: (x, y) point at which EF is evaluated [nm]; None = the gap centre (0, gap/2).
    full_map: also store the DFT fields of the whole PML-free region (down-sampled by
        full_map_stride) for large-scale field maps.
    """

    near_box_nm: tuple[float, float, float, float] = (-150.0, 150.0, -10.0, 120.0)
    probe_nm: tuple[float, float] | None = None
    full_map: bool = False
    full_map_stride: int = 8

    def __post_init__(self):
        object.__setattr__(self, "near_box_nm", tuple(float(v) for v in self.near_box_nm))
        if self.probe_nm is not None:
            object.__setattr__(self, "probe_nm", tuple(float(v) for v in self.probe_nm))
        x0, x1, y0, y1 = self.near_box_nm
        if not (x0 < 0 < x1 and y0 < 0 < y1):
            raise ValueError("near_box_nm must contain the point below the apex (0, 0)")
        if self.full_map_stride < 1:
            raise ValueError("full_map_stride must be >= 1")


@dataclass(frozen=True)
class Model:
    """Everything that defines one tip geometry (all wavelengths share it).

    gap_nm: air gap between the apex and the top surface of the sample [nm]; a whole
        number of grid cells. With a layer on a metal, the metal-to-metal distance is
        gap_nm plus the (grid-snapped) layer thickness.
    """

    tip: Tip = field(default_factory=Tip)
    sample: Sample = field(default_factory=Sample)
    gap_nm: float = 2.0
    illumination: Illumination = field(default_factory=Illumination)
    numerics: Numerics = field(default_factory=Numerics)
    outputs: Outputs = field(default_factory=Outputs)

    def __post_init__(self):
        if not self.gap_nm > 0:
            raise ValueError("gap_nm must be > 0")
        x0, x1, y0, y1 = self.outputs.near_box_nm
        if y1 <= self.gap_nm:
            raise ValueError("near_box_nm must extend above the apex")

    def with_value(self, path: str, value) -> "Model":
        """Copy with one (possibly nested) field changed, e.g. ``m.with_value("tip.apex_radius_nm", 30)``."""
        return _replace_path(self, path.split("."), value)


def _replace_path(obj, parts, value):
    if not is_dataclass(obj):
        raise TypeError(f"cannot set {'.'.join(parts)} on {type(obj).__name__}")
    name = parts[0]
    if name not in {f.name for f in fields(obj)}:
        raise AttributeError(f"{type(obj).__name__} has no field {name!r}")
    if len(parts) == 1:
        return replace(obj, **{name: value})
    return replace(obj, **{name: _replace_path(getattr(obj, name), parts[1:], value)})
