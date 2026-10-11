"""Analytic references used by the validation scripts (pure NumPy/SciPy).

Conventions: exp(-i omega t), Meep units (um, c = 1, omega = k = 2 pi / wavelength),
p polarization with Hz out of plane, sample surface at y = 0, light incident from y > 0.
"""

from __future__ import annotations

import numpy as np


def fresnel_rp(eps, sin_theta):
    """Reflection coefficient of the Hz amplitude (p/TM polarization) from air onto eps."""
    eps = np.asarray(eps, dtype=complex)
    s = np.asarray(sin_theta, dtype=float)
    k1 = np.sqrt(1 - s**2 + 0j)
    k2 = np.sqrt(eps - s**2 + 0j)
    k2 = np.where(k2.imag < 0, -k2, k2)
    return (eps * k1 - k2) / (eps * k1 + k2)


def fit_up_down(y, hz, ky):
    """Least-squares split of Hz(y) = a exp(-i ky y) + b exp(i ky y); returns (a, b, residual)."""
    y = np.asarray(y, dtype=float)
    A = np.c_[np.exp(-1j * ky * y), np.exp(1j * ky * y)]
    coef, *_ = np.linalg.lstsq(A, np.asarray(hz, dtype=complex), rcond=None)
    resid = np.linalg.norm(A @ coef - hz) / np.linalg.norm(hz)
    return complex(coef[0]), complex(coef[1]), float(resid)


def mie_cylinder_hz(eps, radius, wavelength, angles, weights, center, focus, x, y, n_max=None):
    """Exact total field (Hz, Ex, Ey) outside a circular cylinder in a plane-wave superposition.

    Incident Hz = sum_j w_j exp(i k u_j . (r - focus)), u_j = (sin th_j, -cos th_j) (the free
    space beam of ``background.GaussianBeam``). Hz parallel to the cylinder axis (in-plane E).
    """
    from scipy.special import h1vp, hankel1, jv, jvp

    k = 2 * np.pi / wavelength
    m = np.sqrt(complex(eps))
    xk = k * radius
    if n_max is None:
        n_max = int(np.ceil(xk + 4 * xk ** (1 / 3) + 10))
    n = np.arange(-n_max, n_max + 1)
    D = jvp(n, m * xk) / (m * jv(n, m * xk))
    a = (D * jv(n, xk) - jvp(n, xk)) / (h1vp(n, xk) - D * hankel1(n, xk))
    X = np.asarray(x, dtype=float) - center[0]
    Y = np.asarray(y, dtype=float) - center[1]
    r = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    if np.any(r < radius):
        raise ValueError("Mie evaluation points must lie outside the cylinder")
    th = np.asarray(angles)
    ux, uy = np.sin(th), -np.cos(th)
    phase = np.asarray(weights) * np.exp(1j * k * (ux * (center[0] - focus[0]) + uy * (center[1] - focus[1])))
    phi_inc = np.arctan2(uy, ux)  # propagation direction angle of each plane wave
    kr = k * r[..., None]
    jn, hn = jv(n, kr), hankel1(n, kr)
    djn, dhn = jvp(n, kr), h1vp(n, kr)
    hz = np.zeros(r.shape, complex)
    dr = np.zeros(r.shape, complex)
    dp = np.zeros(r.shape, complex)
    for pj, pi in zip(phase, phi_inc):
        e = (1j**n) * np.exp(1j * n * (phi[..., None] - pi))
        hz += pj * np.sum(e * (jn + a * hn), axis=-1)
        dr += pj * np.sum(e * k * (djn + a * dhn), axis=-1)
        dp += pj * np.sum(e * 1j * n * (jn + a * hn), axis=-1)
    er = (1j / k) * dp / r
    ephi = -(1j / k) * dr
    ex = er * np.cos(phi) - ephi * np.sin(phi)
    ey = er * np.sin(phi) + ephi * np.cos(phi)
    return hz, ex, ey


def mie_cylinder_ez(eps, radius, wavelength, angles, weights, center, focus, x, y, n_max=None):
    """Exact total field (Ez, Hx, Hy) outside a circular cylinder in an s-polarized plane-wave
    superposition: E parallel to the cylinder axis ("TM" in the photonic-crystal convention,
    TE with respect to the xy plane of incidence).

    Incident Ez = sum_j w_j exp(i k u_j . (r - focus)), u_j = (sin th_j, -cos th_j) (the free
    space s beam of ``background.GaussianBeam``). Ez and dEz/dr are continuous at r = radius.
    """
    from scipy.special import h1vp, hankel1, jv, jve, jvp

    k = 2 * np.pi / wavelength
    m = np.sqrt(complex(eps))
    xk = k * radius
    if n_max is None:
        n_max = int(np.ceil(xk + 4 * xk ** (1 / 3) + 10))
    n = np.arange(-n_max, n_max + 1)
    # D = m J_n'(m x) / J_n(m x), with exponentially scaled Bessel functions (large |m| for metals)
    D = m * 0.5 * (jve(n - 1, m * xk) - jve(n + 1, m * xk)) / jve(n, m * xk)
    a = (D * jv(n, xk) - jvp(n, xk)) / (h1vp(n, xk) - D * hankel1(n, xk))
    X = np.asarray(x, dtype=float) - center[0]
    Y = np.asarray(y, dtype=float) - center[1]
    r = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    if np.any(r < radius):
        raise ValueError("Mie evaluation points must lie outside the cylinder")
    th = np.asarray(angles)
    ux, uy = np.sin(th), -np.cos(th)
    phase = np.asarray(weights) * np.exp(1j * k * (ux * (center[0] - focus[0]) + uy * (center[1] - focus[1])))
    phi_inc = np.arctan2(uy, ux)
    kr = k * r[..., None]
    jn, hn = jv(n, kr), hankel1(n, kr)
    djn, dhn = jvp(n, kr), h1vp(n, kr)
    ez = np.zeros(r.shape, complex)
    dr = np.zeros(r.shape, complex)
    dp = np.zeros(r.shape, complex)
    for pj, pi in zip(phase, phi_inc):
        e = (1j**n) * np.exp(1j * n * (phi[..., None] - pi))
        ez += pj * np.sum(e * (jn + a * hn), axis=-1)
        dr += pj * np.sum(e * k * (djn + a * dhn), axis=-1)
        dp += pj * np.sum(e * 1j * n * (jn + a * hn), axis=-1)
    h_r = -(1j / k) * dp / r  # H = -(i/k) curl(Ez z)
    h_phi = (1j / k) * dr
    hx = h_r * np.cos(phi) - h_phi * np.sin(phi)
    hy = h_r * np.sin(phi) + h_phi * np.cos(phi)
    return ez, hx, hy


def complex_scale(sim, model):
    """Least-squares complex factor c minimizing |sim - c * model|, and the relative residual."""
    sim = np.asarray(sim).ravel()
    model = np.asarray(model).ravel()
    c = np.vdot(model, sim) / np.vdot(model, model)
    resid = np.linalg.norm(sim - c * model) / np.linalg.norm(sim)
    return complex(c), float(resid)


def rel_l2(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def stokes_wavelength_nm(laser_nm: float, shift_cm1: float) -> float:
    return 1.0 / (1.0 / laser_nm - shift_cm1 * 1e-7)


def raman_enhancement(wavelength_nm, ef, laser_nm: float, shift_cm1: float) -> float:
    """EF(laser) * EF(Stokes) from a sampled EF spectrum (linear interpolation)."""
    wl = np.asarray(wavelength_nm, dtype=float)
    ls = stokes_wavelength_nm(laser_nm, shift_cm1)
    if not (wl.min() <= laser_nm <= wl.max() and wl.min() <= ls <= wl.max()):
        raise ValueError("laser or Stokes wavelength outside the spectrum")
    return float(np.interp(laser_nm, wl, ef) * np.interp(ls, wl, ef))
