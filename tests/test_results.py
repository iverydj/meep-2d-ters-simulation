import numpy as np
import pytest

from ters2d.analysis import mie_cylinder_ez, mie_cylinder_hz, raman_enhancement, stokes_wavelength_nm
from ters2d.background import GaussianBeam, Stack, beam_fields
from ters2d.results import enhancement_metrics, interp2


def test_interp2_bilinear_exact_for_linear_field():
    x = np.linspace(-1, 1, 5)
    y = np.linspace(0, 2, 7)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = 2 * X - 3 * Y + 1j * X * 0.5
    assert interp2(x, y, f, 0.13, 1.37) == pytest.approx(2 * 0.13 - 3 * 1.37 + 0.065j)


def _synthetic(gain=10.0, layer=False):
    dx = 0.0005
    x = (np.arange(-20, 20) + 0.5) * dx
    y = (np.arange(-4, 30) + 0.5) * dx
    X, _ = np.meshgrid(x, y, indexing="ij")
    bg = np.ones_like(X, dtype=complex)
    data = {"near_x": x, "near_y": y, "near_bEx": 0 * bg, "near_bEy": bg, "near_Ex": 0 * bg, "near_Ey": gain * bg}
    meta = {
        "wavelength_nm": 633.0,
        "model": {"illumination": {"incidence_deg": 60.0, "divergence_deg": 14.0, "n_waves": 721}},
        "layout": {"probe": [0.0, 0.001], "gap_um": 0.002, "dx_um": dx, "focus": [0.0, 0.0], "layers": []},
    }
    if layer:
        meta["layout"]["layers"] = [[-0.0015, 0.0]]
    return data, meta


def test_enhancement_definitions():
    for layer in (False, True):
        e = enhancement_metrics(*_synthetic(gain=10.0, layer=layer))
        assert e["EF_probe"] == pytest.approx(100.0)
        assert e["EF_gap"] == pytest.approx(100.0)
        assert e["EF4_probe"] == pytest.approx(1e4)
        assert e["EF4_gap"] == pytest.approx(1e4)
        assert ("EF_layer" in e) == layer


def test_mie_reduces_to_incident_field_for_vacuum_cylinder():
    beam = GaussianBeam(0.633, 60, 14)
    th, w = beam.angles_weights()
    x = np.array([0.03, -0.04, 0.0])
    y = np.array([0.01, 0.02, -0.05])
    m = mie_cylinder_hz(1.0 + 0j, 0.02, 0.633, th, w, (0, 0), (0, 0), x, y)
    b = beam_fields(beam, Stack.isotropic(1.0), x, y, part="incident")
    for i in range(3):
        np.testing.assert_allclose(m[i], b[i], rtol=1e-8, atol=1e-10)


def test_mie_small_cylinder_quasistatic_enhancement():
    """|E|^2 at the pole of a small cylinder along the field -> |2 eps/(eps+1)|^2 (2D quasi-static)."""
    eps = -9.8 + 1.96j
    R = 0.002
    beam = GaussianBeam(0.633, 90 - 1e-3, 1e-4, n_waves=3)  # ~ plane wave along +x, E along +y
    th, w = beam.angles_weights()
    _, ex, ey = mie_cylinder_hz(eps, R, 0.633, th, w, (0, 0), (0, 0), np.array([0.0]), np.array([R * 1.0001]))
    _, ex0, ey0 = beam_fields(beam, Stack.isotropic(1.0), np.array([0.0]), np.array([0.0]), part="incident")
    ratio = (abs(ex[0]) ** 2 + abs(ey[0]) ** 2) / (abs(ex0[0]) ** 2 + abs(ey0[0]) ** 2)
    assert ratio == pytest.approx(abs(2 * eps / (eps + 1)) ** 2, rel=0.02)


def test_stokes_and_raman_enhancement():
    assert stokes_wavelength_nm(633.0, 418.0) == pytest.approx(650.2, abs=0.1)
    wl = np.array([600.0, 650.0, 700.0])
    ef = np.array([10.0, 20.0, 30.0])
    assert raman_enhancement(wl, ef, 625.0, 0.0) == pytest.approx(15.0**2)


def test_mie_ez_reduces_to_incident_field_for_vacuum_cylinder():
    beam = GaussianBeam(0.633, 60, 14, polarization="s")
    th, w = beam.angles_weights()
    x = np.array([0.03, -0.04, 0.0])
    y = np.array([0.01, 0.02, -0.05])
    m = mie_cylinder_ez(1.0 + 0j, 0.02, 0.633, th, w, (0, 0), (0, 0), x, y)
    b = beam_fields(beam, Stack.isotropic(1.0), x, y, part="incident")
    for i in range(3):
        np.testing.assert_allclose(m[i], b[i], rtol=1e-8, atol=1e-10)


def test_mie_ez_conductor_limit():
    """E along the axis of a near-perfect conductor: Ez -> 0 on its surface."""
    beam = GaussianBeam(0.633, 60, 14, polarization="s", n_waves=31)
    th, w = beam.angles_weights()
    phi = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    R = 0.05
    rr = R * (1 + 1e-9)
    ez_pec, _, _ = mie_cylinder_ez(-1e8 + 1e4j, R, 0.633, th, w, (0, 0), (0, 0), rr * np.cos(phi), rr * np.sin(phi))
    ez_inc = beam_fields(beam, Stack.isotropic(1.0), rr * np.cos(phi), rr * np.sin(phi), part="incident")[0]
    assert np.max(np.abs(ez_pec)) < 1e-3 * np.max(np.abs(ez_inc))


def test_enhancement_definitions_s():
    dx = 0.0005
    x = (np.arange(-20, 20) + 0.5) * dx
    y = (np.arange(-4, 30) + 0.5) * dx
    X, _ = np.meshgrid(x, y, indexing="ij")
    bg = np.ones_like(X, dtype=complex)
    data = {"near_x": x, "near_y": y, "near_bEz": bg, "near_Ez": 3 * bg}
    meta = {
        "wavelength_nm": 633.0,
        "model": {"illumination": {"incidence_deg": 60.0, "divergence_deg": 14.0, "n_waves": 721, "polarization": "s"}},
        "layout": {"probe": [0.0, 0.001], "gap_um": 0.002, "dx_um": dx, "focus": [0.0, 0.0], "layers": []},
    }
    e = enhancement_metrics(data, meta)
    assert e["EF_probe"] == pytest.approx(9.0) and e["EF_gap"] == pytest.approx(9.0)


def test_mie_ez_matches_boundary_conditions_for_gold():
    """mie_cylinder_ez for one plane wave vs a series whose coefficients come from solving the
    continuity of Ez and dEz/dr at r = R directly (interior c_n J_n(m k r))."""
    from scipy.special import h1vp, hankel1, jv, jvp

    eps, R, wl, th = -9.8 + 2.0j, 0.02, 0.633, np.deg2rad(60.0)
    k = 2 * np.pi / wl
    m = np.sqrt(eps)
    x, y = np.array([0.025, -0.03, 0.0]), np.array([0.0, 0.01, -0.04])
    ez, _, _ = mie_cylinder_ez(eps, R, wl, [th], [1.0], (0, 0), (0, 0), x, y, n_max=25)
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    phi_inc = np.arctan2(-np.cos(th), np.sin(th))
    ref = np.zeros(3, complex)
    for n in range(-25, 26):
        A = np.array([[hankel1(n, k * R), -jv(n, m * k * R)], [h1vp(n, k * R), -m * jvp(n, m * k * R)]])
        a_n, _ = np.linalg.solve(A, -np.array([jv(n, k * R), jvp(n, k * R)]))
        ref += (1j**n) * np.exp(1j * n * (phi - phi_inc)) * (jv(n, k * r) + a_n * hankel1(n, k * r))
    np.testing.assert_allclose(ez, ref, rtol=1e-9)
