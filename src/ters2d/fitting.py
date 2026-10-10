"""Passive Lorentz fits of tabulated optical constants (n, k) for use in Meep.

Meep needs a material as eps_inf plus Lorentz/Drude poles (see ``materials``). Tabulated
n, k data are fitted with bounded least squares on the relative complex error of eps;
every strength sigma and width gamma is >= 0, so the fitted model is passive and causal
at all frequencies, not only inside the fitted window. Each pole gets a minimum width
(``min_gamma_um_inv``): a lossless pole rings indefinitely in a pulsed FDTD run and
slows the convergence of the DFT fields.

Typical use (see ``examples/06_add_material_from_nk.py``)::

    wl, n, k = read_nk_csv("my_material.csv")
    mat = fit_lorentz(wl, n, k, [(2.0, 0.05), (2.4, 0.2), (4.0, 1.0)], name="X",
                      fit_window_nm=(450, 950), valid_wavelength_nm=(480, 900))
    print(fit_quality(mat, wl, n, k))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .materials import EV_TO_INV_UM, Material, Pole, frequency


def read_nk_csv(path: Path):
    """Read a CSV with the header ``wavelength_nm,n,k`` (lines starting with # are comments).

    Returns (wavelength_nm, n, k) as arrays sorted by wavelength.
    """
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    header = [h.strip() for h in lines[0].split(",")]
    if header != ["wavelength_nm", "n", "k"]:
        raise ValueError(f"{path}: expected the header 'wavelength_nm,n,k', got {lines[0]!r}")
    arr = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    o = np.argsort(arr[:, 0])
    return arr[o, 0], arr[o, 1], arr[o, 2]


def _poles_from_params(params, n_poles):
    return float(params[0]), tuple(Pole("lorentz", *params[1 + 3 * i : 4 + 3 * i]) for i in range(n_poles))


def fit_lorentz(
    wavelength_nm,
    n,
    k,
    initial_poles_eV,
    *,
    name: str,
    fit_window_nm: tuple[float, float] | None = None,
    valid_wavelength_nm: tuple[float, float] | None = None,
    min_gamma_um_inv: float = 0.02,
    fixed_gamma_um_inv: float | None = None,
    reference: str = "",
    max_nfev: int = 20000,
) -> Material:
    """Fit eps_inf + Lorentz poles to n + i k.

    initial_poles_eV: list of (resonance energy [eV], width [eV]) starting values, one per
        pole. Place poles at the absorption features of the data (excitons, interband
        edges) plus one or two broad poles above the window for the background.
    fit_window_nm: wavelengths used in the fit (default: all).
    valid_wavelength_nm: range stored in the material; simulations outside it raise an
        error (default: the fit window). Choose it slightly inside the fit window.
    min_gamma_um_inv: lower bound of every pole width [1/um] (Meep units).
    fixed_gamma_um_inv: if given, all widths are fixed to this value (for lossless data).

    Bounds: eps_inf in [1, 30]; f0 within +-50 % of its start; gamma in [min_gamma, 5]/um;
    sigma in [0, 200].
    """
    from scipy.optimize import least_squares

    wl = np.asarray(wavelength_nm, dtype=float)
    eps_data = (np.asarray(n, dtype=float) + 1j * np.asarray(k, dtype=float)) ** 2
    if fit_window_nm is not None:
        m = (wl >= fit_window_nm[0]) & (wl <= fit_window_nm[1])
        wl, eps_data = wl[m], eps_data[m]
    if wl.size < 3 * len(initial_poles_eV) + 1:
        raise ValueError("fewer data points than fit parameters")
    f = frequency(wl)
    p0, lo, hi = [1.0], [1.0], [30.0]
    for e0, w in initial_poles_eV:
        f0 = e0 * EV_TO_INV_UM
        if fixed_gamma_um_inv is None:
            p0 += [f0, max(w * EV_TO_INV_UM, min_gamma_um_inv), 0.5]
            lo += [0.5 * f0, min_gamma_um_inv, 0.0]
            hi += [1.5 * f0, 5.0, 200.0]
        else:
            g = fixed_gamma_um_inv
            p0 += [f0, g, 0.5]
            lo += [0.5 * f0, g * (1 - 1e-9), 0.0]
            hi += [1.5 * f0, g * (1 + 1e-9), 200.0]
    p0 = np.clip(p0, lo, hi)
    n_poles = len(initial_poles_eV)

    def resid(p):
        eps_inf, poles = _poles_from_params(p, n_poles)
        r = (Material("fit", eps_inf, poles).epsilon_at_frequency(f) - eps_data) / np.abs(eps_data)
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev)
    if not sol.success:
        raise RuntimeError(f"Lorentz fit of {name} did not converge: {sol.message}")
    eps_inf, poles = _poles_from_params(sol.x, n_poles)
    valid = valid_wavelength_nm if valid_wavelength_nm is not None else (float(wl.min()), float(wl.max()))
    return Material(name, eps_inf, poles, valid, reference)


def fit_quality(material: Material, wavelength_nm, n, k) -> dict:
    """Relative error |eps_fit - eps_data| / |eps_data| inside the material's valid range, and
    the minimum Im(eps) of the fit between 200 nm and 20 um (>= 0 for a passive model)."""
    wl = np.asarray(wavelength_nm, dtype=float)
    eps_data = (np.asarray(n, dtype=float) + 1j * np.asarray(k, dtype=float)) ** 2
    lo, hi = material.valid_wavelength_nm
    m = (wl >= lo) & (wl <= hi)
    rel = np.abs(material.epsilon(wl[m]) - eps_data[m]) / np.abs(eps_data[m])
    f_dense = np.linspace(0.05, 5.0, 20001)
    return {
        "rel_rms": float(np.sqrt(np.mean(rel**2))),
        "rel_max": float(rel.max()),
        "min_im_eps_200nm_to_20um": float(np.min(material.epsilon_at_frequency(f_dense).imag)),
    }
