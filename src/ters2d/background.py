"""Analytic tip-free field: a focused p- or s-polarized beam over a layered, uniaxial sample.

The beam is a superposition of plane waves with incidence angles theta_j around theta_0
and a Gaussian angular spectrum exp(-((theta_j - theta_0) / theta_d)^2) (a 2D Gaussian
beam with divergence half-angle theta_d, 1/e^2 of the intensity), cut at +-2 theta_d,
all in phase at the focus. Each plane wave is reflected and transmitted by the sample
(air / layers / substrate) with exact transfer conditions, so the result is an exact
solution of Maxwell's equations for the sample without the tip. It drives the tip
through the scattered-field sources (``sources.py``) and is the reference field of the
enhancement factor.

Conventions: exp(-i omega t), Meep units (um, c = 1, omega = k = 2 pi / wavelength). The
plane of incidence is the xy plane (z is invariant). The sample surface is y = 0, the sample
fills y < 0. Each region may be uniaxial with the optic axis along y: eps_x (in plane, also
the zz component) and eps_y (out of plane).

* p polarization (TM in the thin-film convention: H transverse to the plane of incidence):
  Hz out of plane, Ex = (i / (k eps_x)) dHz/dy, Ey = -(i / (k eps_y)) dHz/dx,
  ky^2 = eps_x (k^2 - kx^2 / eps_y); Hz and Ex are continuous.
* s polarization (TE in the thin-film convention, TM in the photonic-crystal convention of
  Meep's tutorials: E along z, out of the plane): Hx = -(i / k) dEz/dy, Hy = (i / k) dEz/dx,
  ky^2 = eps_x k^2 - kx^2 (only the in-plane permittivity enters); Ez and Hx are continuous.

The field functions return (scalar out-of-plane field, first in-plane component, second
in-plane component): (Hz, Ex, Ey) for p, (Ez, Hx, Hy) for s; see ``FIELD_NAMES``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POLARIZATIONS = ("p", "s")
FIELD_NAMES = {"p": ("Hz", "Ex", "Ey"), "s": ("Ez", "Hx", "Hy")}  # order of the returned arrays
E_NAMES = {"p": ("Ex", "Ey"), "s": ("Ez",)}  # the electric-field components


def e_components(fields, polarization: str) -> list:
    """The electric-field arrays among the (scalar, in-plane, in-plane) fields of a beam."""
    return [fields[1], fields[2]] if polarization == "p" else [fields[0]]


@dataclass(frozen=True)
class Stack:
    """Sample below y = 0.

    substrate: (eps_x, eps_y) of the semi-infinite substrate.
    layers: ((eps_x, eps_y, thickness_um), ...) ordered from the surface downward (the
        first entry is the top layer, directly below y = 0).
    """

    substrate: tuple[complex, complex]
    layers: tuple[tuple[complex, complex, float], ...] = ()

    @classmethod
    def isotropic(cls, eps_substrate: complex) -> "Stack":
        return cls((complex(eps_substrate), complex(eps_substrate)))

    @property
    def thickness(self) -> float:
        return float(sum(t for _, _, t in self.layers))

    def tops(self) -> np.ndarray:
        """y of the top of region m: 0 for the air (m = 0) and the first layer, ..., -thickness for the substrate."""
        t = [0.0] + [lay[2] for lay in self.layers]
        return np.concatenate([[0.0], -np.cumsum(t)])

    def region_eps(self):
        """[(eps_x, eps_y)] for air, the layers (top down) and the substrate."""
        return (
            [(1.0 + 0j, 1.0 + 0j)]
            + [(complex(a), complex(b)) for a, b, _ in self.layers]
            + [(complex(self.substrate[0]), complex(self.substrate[1]))]
        )


@dataclass(frozen=True)
class GaussianBeam:
    """2D Gaussian beam as a plane-wave superposition (lengths in um).

    wavelength: vacuum wavelength [um].
    incidence_deg: angle of the beam axis from the surface normal; the beam travels
        towards +x and -y (it comes from the upper left).
    divergence_deg: theta_d, half-angle of the Gaussian angular spectrum (1/e^2 intensity).
    focus: (x, y) where all incident plane waves are in phase [um].
    n_waves: number of plane waves.
    polarization: "p" (unit incident Hz per plane wave) or "s" (unit incident Ez).
    """

    wavelength: float
    incidence_deg: float
    divergence_deg: float
    focus: tuple[float, float] = (0.0, 0.0)
    n_waves: int = 721
    polarization: str = "p"

    def __post_init__(self):
        if self.polarization not in POLARIZATIONS:
            raise ValueError(f"polarization must be one of {POLARIZATIONS}")

    def angles_weights(self):
        th0 = np.deg2rad(self.incidence_deg)
        td = np.deg2rad(self.divergence_deg)
        th = np.linspace(th0 - 2 * td, th0 + 2 * td, self.n_waves)
        w = np.exp(-(((th - th0) / td) ** 2)) * (th[1] - th[0])
        return th, w


def _ky(eps_x, eps_y, kx, k):
    """Normal wavenumber in a uniaxial medium for TM waves (branch Im >= 0, then Re >= 0)."""
    ky = np.sqrt(eps_x * (k**2 - kx**2 / eps_y) + 0j)
    ky = np.where(ky.imag < 0, -ky, ky)
    ky = np.where((np.abs(ky.imag) < 1e-14) & (ky.real < 0), -ky, ky)
    return ky


def stack_coefficients(stack: Stack, kx, k, polarization: str = "p") -> dict:
    """Plane-wave amplitudes for a unit incident scalar field F = exp(i (kx x - ky0 y)),
    F = Hz for p and F = Ez for s.

    In region m (0 = air, 1..N = layers from the top, N + 1 = substrate) with top z_top[m]
    and bottom z_bot[m]:

        F = down[m] exp(-i ky[m] (y - z_top[m])) + up[m] exp(i ky[m] (y - z_bot[m]))

    (air: z_top = z_bot = 0, down = 1, up = r; substrate: up = 0). F and its normal
    derivative weighted by the admittance (ky / eps_x for p, ky for s) are continuous at every
    interface. Reflection coefficients are built recursively from the substrate up, which is
    stable for thick or lossy layers.
    Returns a dict with r, ky, down, up (lists over regions; arrays over kx), z_top, z_bot.
    """
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization must be one of {POLARIZATIONS}")
    kx = np.atleast_1d(np.asarray(kx, dtype=float))
    eps = stack.region_eps()
    nreg = len(eps)
    if polarization == "p":
        ky = [_ky(ex, ey, kx, k) for ex, ey in eps]
        adm = [ky[m] / eps[m][0] for m in range(nreg)]  # TM admittance ky / eps_x
    else:
        ky = [_ky(ex, ex, kx, k) for ex, _ in eps]  # s sees only the in-plane permittivity
        adm = list(ky)  # TE admittance ky (mu = 1)
    thick = [0.0] + [float(t) for _, _, t in stack.layers] + [0.0]
    tops = stack.tops()
    z_top = list(tops)
    z_bot = [0.0] + list(tops[2:]) + [None]
    rif = [(adm[m] - adm[m + 1]) / (adm[m] + adm[m + 1]) for m in range(nreg - 1)]  # interface m | m+1
    gamma = [None] * nreg  # up / down ratio at the top of region m
    r_bot = [None] * nreg  # up / down ratio at the bottom of region m
    gamma[-1] = np.zeros_like(kx, dtype=complex)
    for m in range(nreg - 2, -1, -1):
        r_bot[m] = (rif[m] + gamma[m + 1]) / (1 + rif[m] * gamma[m + 1])
        gamma[m] = r_bot[m] * np.exp(2j * ky[m] * thick[m])
    down = [None] * nreg
    up = [None] * nreg
    down[0] = np.ones_like(kx, dtype=complex)
    for m in range(nreg - 1):
        d_bot = down[m] * np.exp(1j * ky[m] * thick[m])
        up[m] = r_bot[m] * d_bot
        down[m + 1] = d_bot * (1 + rif[m]) / (1 + rif[m] * gamma[m + 1])
    up[-1] = np.zeros_like(kx, dtype=complex)
    return {
        "r": up[0],
        "ky": ky,
        "down": down,
        "up": up,
        "z_top": z_top,
        "z_bot": z_bot,
        "eps": eps,
        "polarization": polarization,
    }


def _region_index(stack: Stack, y) -> np.ndarray:
    """Region of each y: 0 air (y >= 0), m for layer m, N + 1 substrate."""
    y = np.asarray(y, dtype=float)
    tops = stack.tops()  # [0, 0, -t1, ..., -T]
    idx = np.zeros(y.shape, dtype=int)
    for m in range(1, len(tops)):
        idx = np.where(y < tops[m], m, idx)
    return idx


def _y_factors(c: dict, m: int, kx, k, y):
    """(Hz, Ex, Ey) for p or (Ez, Hx, Hy) for s of every plane wave in region m at heights y:
    arrays (n_waves, n_y)."""
    kym = c["ky"][m][:, None]
    ex_, ey_ = c["eps"][m]
    Y = np.asarray(y, dtype=float)[None, :]
    dn = c["down"][m][:, None] * np.exp(-1j * kym * (Y - c["z_top"][m]))
    if c["z_bot"][m] is None:
        upw = np.zeros_like(dn)
    else:
        upw = c["up"][m][:, None] * np.exp(1j * kym * (Y - c["z_bot"][m]))
    h = dn + upw
    if c["polarization"] == "s":  # here h is Ez
        return h, -(kym / k) * (dn - upw), -(np.asarray(kx)[:, None] / k) * h
    e_x = (kym / (k * ex_)) * (dn - upw)
    e_y = (np.asarray(kx)[:, None] / (k * ey_)) * h
    return h, e_x, e_y


def _phased_waves(beam: GaussianBeam):
    k = 2 * np.pi / beam.wavelength
    th, w = beam.angles_weights()
    kx = k * np.sin(th)
    ky0 = k * np.cos(th)
    fx, fy = beam.focus
    return k, kx, w * np.exp(-1j * kx * fx + 1j * ky0 * fy)  # incident waves in phase at the focus


def beam_fields(beam: GaussianBeam, stack: Stack, x, y, part: str = "total"):
    """(Hz, Ex, Ey) for p or (Ez, Hx, Hy) for s of the tip-free field at the points (x, y).

    part: "total" (incident + reflected in air, transmitted fields in the sample),
    "incident" (the free-space incident beam alone, everywhere) or "reflected" (the
    reflected beam alone, air formula).
    """
    k, kx, w = _phased_waves(beam)
    X = np.asarray(x, dtype=float).ravel()
    Y = np.asarray(y, dtype=float).ravel()
    if part == "incident":
        st, regions = Stack.isotropic(1.0), np.zeros(Y.shape, dtype=int)
    elif part in ("total", "reflected"):
        st, regions = stack, _region_index(stack, Y)
        if part == "reflected":
            regions = np.zeros(Y.shape, dtype=int)
    else:
        raise ValueError(f"unknown part {part!r}")
    c = stack_coefficients(st, kx, k, beam.polarization)
    if part == "incident":
        c["up"] = [np.zeros_like(u) for u in c["up"]]
    if part == "reflected":
        c["down"] = [np.zeros_like(d) for d in c["down"]]
    out = [np.zeros(X.size, complex) for _ in range(3)]
    for m in np.unique(regions):
        sel = np.where(regions == m)[0]
        for s in range(0, sel.size, 4000):
            idx = sel[s : s + 4000]
            px = np.exp(1j * np.outer(kx, X[idx])) * w[:, None]
            for o, fac in zip(out, _y_factors(c, int(m), kx, k, Y[idx])):
                o[idx] = np.sum(px * fac, axis=0)
    shape = np.shape(x)
    return tuple(o.reshape(shape) for o in out)


def beam_fields_grid(beam: GaussianBeam, stack: Stack, xs, ys):
    """(Hz, Ex, Ey) for p or (Ez, Hx, Hy) for s of the total tip-free field on the tensor grid
    xs x ys (arrays [ix, iy]).

    Separable evaluation (field = sum_j X_j(x) Y_j(y)) done as matrix products.
    """
    k, kx, w = _phased_waves(beam)
    c = stack_coefficients(stack, kx, k, beam.polarization)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    Xf = np.exp(1j * np.outer(xs, kx)) * w  # (nx, nj)
    yh = np.zeros((kx.size, ys.size), complex)
    yex = np.zeros_like(yh)
    yey = np.zeros_like(yh)
    regions = _region_index(stack, ys)
    for m in np.unique(regions):
        sel = regions == m
        h, e_x, e_y = _y_factors(c, int(m), kx, k, ys[sel])
        yh[:, sel], yex[:, sel], yey[:, sel] = h, e_x, e_y
    return Xf @ yh, Xf @ yex, Xf @ yey
