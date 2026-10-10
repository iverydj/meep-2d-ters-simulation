"""Optical materials as passive pole models in Meep's convention.

Every material is a permittivity model

    eps(f) = eps_inf + sum_lorentz sigma f0^2 / (f0^2 - f^2 - i f gamma)
                     + sum_drude   sigma f0^2 / (-f^2 - i f gamma)

with f the frequency in Meep units (f = 1 / wavelength in um, c = 1) and the time
dependence exp(-i 2 pi f t). This is exactly what ``meep.LorentzianSusceptibility`` and
``meep.DrudeSusceptibility`` implement, so the same object gives the permittivity used by
the analytic reference field (pure Python) and the medium simulated by Meep. With all
sigma, gamma >= 0 the model is passive (Im eps >= 0 for f > 0) and causal.

The user-facing API takes wavelengths in nm. Built-in materials:

* ``AU``: Lorentz-Drude fit of Rakic et al., Appl. Opt. 37, 5271 (1998),
  doi:10.1364/AO.37.005271; the same parameters as ``meep.materials.Au``.
* ``SIO2``: non-dispersive, n = 1.45.
* ``WS2_MONOLAYER``: uniaxial monolayer WS2 (optic axis along the layer normal), passive
  Lorentz fits of the in-plane monolayer data of Ermolaev et al. (2020) and the
  out-of-plane (extraordinary) bulk data of Munkhbat et al. (2022); see
  ``materials/fit_ws2_lorentz.py``.
* ``AIR``.

New materials: ``Material.constant`` (lossless dielectric), ``Material.from_meep`` (any
isotropic entry of ``meep.materials``), ``ters2d.fitting.fit_lorentz`` (passive Lorentz fit
of tabulated n, k data) and ``Material.load_json`` (a saved fit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

EV_TO_INV_UM = 1.0 / 1.23984193  # photon energy [eV] -> frequency [1/um] (Meep's eV_um_scale)
POLE_KINDS = ("lorentz", "drude")


def frequency(wavelength_nm):
    """Meep frequency [1/um] of a vacuum wavelength [nm]."""
    return 1e3 / np.asarray(wavelength_nm, dtype=float)


@dataclass(frozen=True)
class Pole:
    """One susceptibility term.

    kind: "lorentz" or "drude".
    frequency: resonance frequency f0 [1/um] (Meep units; f0 = 1 / resonance wavelength in um).
    gamma: damping rate [1/um] (Meep units).
    sigma: dimensionless strength.
    """

    kind: str
    frequency: float
    gamma: float
    sigma: float

    def __post_init__(self):
        if self.kind not in POLE_KINDS:
            raise ValueError(f"unknown pole kind {self.kind!r}; use one of {POLE_KINDS}")
        if self.frequency <= 0 or self.gamma < 0 or self.sigma < 0:
            raise ValueError(f"pole is not passive (need frequency > 0, gamma >= 0, sigma >= 0): {self}")

    def chi(self, f):
        f = np.asarray(f, dtype=float)
        f0sq = self.frequency**2
        if self.kind == "lorentz":
            return self.sigma * f0sq / (f0sq - f**2 - 1j * f * self.gamma)
        return self.sigma * f0sq / (-(f**2) - 1j * f * self.gamma)


@dataclass(frozen=True)
class Material:
    """Isotropic material: eps_inf plus passive poles.

    name: label used in file names and plots.
    eps_inf: high-frequency permittivity (> 0).
    poles: Lorentz/Drude terms (empty for a non-dispersive, lossless dielectric).
    valid_wavelength_nm: (min, max) wavelength range of the underlying data [nm]; None for
        a model without a stated range (e.g. a constant index). Simulations outside the
        range raise an error.
    reference: data source.
    """

    name: str
    eps_inf: float = 1.0
    poles: tuple[Pole, ...] = ()
    valid_wavelength_nm: tuple[float, float] | None = None
    reference: str = ""

    def __post_init__(self):
        object.__setattr__(self, "poles", tuple(self.poles))
        if self.valid_wavelength_nm is not None:
            object.__setattr__(self, "valid_wavelength_nm", tuple(float(v) for v in self.valid_wavelength_nm))
            lo, hi = self.valid_wavelength_nm
            if not 0 < lo < hi:
                raise ValueError(f"{self.name}: invalid wavelength range {self.valid_wavelength_nm}")
        if not self.eps_inf > 0:
            raise ValueError(f"{self.name}: eps_inf must be > 0")

    @property
    def dispersive(self) -> bool:
        return bool(self.poles)

    def epsilon_at_frequency(self, f):
        """Complex permittivity at Meep frequencies f [1/um]."""
        f = np.asarray(f, dtype=float)
        eps = np.full(f.shape, self.eps_inf, dtype=complex)
        for p in self.poles:
            eps = eps + p.chi(f)
        return eps

    def epsilon(self, wavelength_nm):
        """Complex permittivity at vacuum wavelengths [nm]."""
        return self.epsilon_at_frequency(frequency(wavelength_nm))

    def epsilon_xy(self, wavelength_nm):
        """(in-plane, out-of-plane) permittivity; equal for an isotropic material."""
        e = self.epsilon(wavelength_nm)
        return e, e

    def check_wavelengths(self, wavelength_nm) -> None:
        if self.valid_wavelength_nm is None:
            return
        lo, hi = self.valid_wavelength_nm
        w = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
        bad = w[(w < lo) | (w > hi)]
        if bad.size:
            raise ValueError(f"{self.name}: wavelengths {bad} nm outside the model's range [{lo}, {hi}] nm")

    # ------------------------------------------------------------------ constructors

    @classmethod
    def constant(cls, name: str, index: float | None = None, epsilon: float | None = None) -> "Material":
        """Non-dispersive lossless dielectric, given by its refractive index or permittivity."""
        if (index is None) == (epsilon is None):
            raise ValueError("give exactly one of index and epsilon")
        eps = float(index) ** 2 if index is not None else float(epsilon)
        return cls(name, eps)

    @classmethod
    def from_meep(cls, name: str) -> "Material":
        """Isotropic material of Meep's materials library, e.g. ``Material.from_meep("Ag")``.

        Copies eps_inf, the susceptibilities and the validity range of ``meep.materials.<name>``.
        Requires Meep. Anisotropic or conductive library entries are rejected.
        """
        import meep.materials as mm

        med = getattr(mm, name)
        diag = med.epsilon_diag
        if not (diag.x == diag.y == diag.z) or med.epsilon_offdiag.norm() != 0:
            raise ValueError(f"meep.materials.{name} is not isotropic")
        if med.D_conductivity_diag.norm() != 0 or med.mu_diag.x != 1 or med.H_susceptibilities:
            raise ValueError(f"meep.materials.{name} has a conductivity or magnetic response (not supported)")
        poles = []
        for s in med.E_susceptibilities:
            sd = s.sigma_diag
            if not (sd.x == sd.y == sd.z) or s.sigma_offdiag.norm() != 0:
                raise ValueError(f"meep.materials.{name} has an anisotropic susceptibility")
            kind = "drude" if type(s).__name__ == "DrudeSusceptibility" else "lorentz"
            if type(s).__name__ not in ("DrudeSusceptibility", "LorentzianSusceptibility"):
                raise ValueError(f"meep.materials.{name}: unsupported susceptibility {type(s).__name__}")
            poles.append(Pole(kind, s.frequency, s.gamma, sd.x))
        fr = med.valid_freq_range
        valid = None
        if fr.min > 0 and np.isfinite(fr.max) and fr.max < 1e20:
            valid = (1e3 / fr.max, 1e3 / fr.min)
        return cls(f"{name} (meep.materials)", diag.x, tuple(poles), valid, f"meep.materials.{name}")

    # ------------------------------------------------------------------ JSON

    def to_dict(self) -> dict:
        return {
            "type": "isotropic",
            "name": self.name,
            "eps_inf": self.eps_inf,
            "valid_wavelength_nm": None if self.valid_wavelength_nm is None else list(self.valid_wavelength_nm),
            "reference": self.reference,
            "convention": "eps_inf + sum sigma f0^2/(f0^2 - f^2 - i f gamma) (lorentz), sigma f0^2/(-f^2 - i f gamma) "
            "(drude); f, f0, gamma in 1/um (Meep units, f = 1000 / wavelength_nm)",
            "poles": [{"kind": p.kind, "frequency": p.frequency, "gamma": p.gamma, "sigma": p.sigma} for p in self.poles],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        if d.get("type", "isotropic") != "isotropic":
            raise ValueError("not an isotropic material record")
        poles = tuple(Pole(p["kind"], p["frequency"], p["gamma"], p["sigma"]) for p in d["poles"])
        valid = d["valid_wavelength_nm"]
        return cls(d["name"], float(d["eps_inf"]), poles, None if valid is None else tuple(valid), d.get("reference", ""))

    def save_json(self, path: Path, extra: dict | None = None) -> None:
        d = self.to_dict()
        if extra:
            d.update(extra)
        Path(path).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "Material":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class UniaxialMaterial:
    """Uniaxial material with the optic axis along the layer normal (the y axis, along the tip).

    in_plane: permittivity for fields parallel to the layer (x and z).
    out_of_plane: permittivity for fields along the layer normal (y).
    In the 2D p-polarized problem the field has an in-plane (Ex) and an out-of-plane (Ey)
    component, so both enter; the gap field is mostly out of plane.
    """

    name: str
    in_plane: Material
    out_of_plane: Material

    @property
    def dispersive(self) -> bool:
        return self.in_plane.dispersive or self.out_of_plane.dispersive

    def epsilon_xy(self, wavelength_nm):
        return self.in_plane.epsilon(wavelength_nm), self.out_of_plane.epsilon(wavelength_nm)

    def check_wavelengths(self, wavelength_nm) -> None:
        self.in_plane.check_wavelengths(wavelength_nm)
        self.out_of_plane.check_wavelengths(wavelength_nm)


AnyMaterial = Material | UniaxialMaterial


def is_metallic(material: AnyMaterial, wavelength_nm) -> bool:
    """True if Re(eps) < 0 (in plane) at every wavelength, False if > 0 at every wavelength.

    Raises if the sign changes inside the wavelength set: the boundary treatment differs
    between metals and dielectrics, so such a spectrum must be split.
    """
    ex, _ = material.epsilon_xy(np.atleast_1d(np.asarray(wavelength_nm, dtype=float)))
    neg = np.real(ex) < 0
    if neg.all():
        return True
    if not neg.any():
        return False
    raise ValueError(f"{material.name}: Re(eps) changes sign within the wavelengths {wavelength_nm}; split the spectrum")


# ---------------------------------------------------------------------------- built-ins


def _rakic_au() -> Material:
    # (oscillator strength, resonance energy [eV], damping [eV]); the first term is the Drude term.
    plasma = 9.03 * EV_TO_INV_UM
    terms = [
        (0.760, 0.0, 0.053),
        (0.024, 0.415, 0.241),
        (0.010, 0.830, 0.345),
        (0.071, 2.969, 0.870),
        (0.601, 4.304, 2.494),
        (4.384, 13.32, 2.214),
    ]
    poles = []
    for i, (strength, e0, g) in enumerate(terms):
        if i == 0:
            f0 = 1e-10  # Meep's representation of the Drude term
            poles.append(Pole("drude", f0, g * EV_TO_INV_UM, strength * plasma**2 / f0**2))
        else:
            f0 = e0 * EV_TO_INV_UM
            poles.append(Pole("lorentz", f0, g * EV_TO_INV_UM, strength * plasma**2 / f0**2))
    return Material(
        "Au",
        1.0,
        tuple(poles),
        (247.97, 6199.2),  # the range of meep.materials.Au
        "Lorentz-Drude fit: A. D. Rakic et al., Appl. Opt. 37, 5271 (1998), doi:10.1364/AO.37.005271 (= meep.materials.Au)",
    )


def load_packaged(filename: str) -> Material:
    """Material JSON shipped in ``ters2d/data``."""
    text = resources.files("ters2d").joinpath("data", filename).read_text(encoding="utf-8")
    return Material.from_dict(json.loads(text))


AU = _rakic_au()
SIO2 = Material("SiO2", 1.45**2, reference="non-dispersive, n = 1.45")
AIR = Material("air", 1.0)
WS2_MONOLAYER = UniaxialMaterial(
    "WS2",
    in_plane=load_packaged("ws2_1L_inplane_lorentz.json"),
    out_of_plane=load_packaged("ws2_outofplane_lorentz.json"),
)
