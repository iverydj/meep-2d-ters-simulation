import numpy as np
import pytest

from ters2d.analysis import fresnel_rp
from ters2d.background import GaussianBeam, Stack, beam_fields, beam_fields_grid, stack_coefficients
from ters2d.materials import AU, SIO2, WS2_MONOLAYER

WL = 0.633
K = 2 * np.pi / WL
EPS_AU = complex(AU.epsilon(633.0))


def ws2_on_au(t=0.00065):
    ex, ey = WS2_MONOLAYER.epsilon_xy(633.0)
    return Stack((EPS_AU, EPS_AU), ((complex(ex), complex(ey), t),))


def three_layer():
    """Uniaxial lossy layer / thick dielectric / metal film on a dielectric substrate."""
    return Stack(
        (2.25 + 0j, 2.25 + 0j),
        ((6.0 + 1.5j, 4.0 + 0.1j, 0.0007), (2.1 + 0j, 2.1 + 0j, 0.09), (-10 + 1.2j, -10 + 1.2j, 0.012)),
    )


def test_single_interface_reduces_to_fresnel():
    th = np.deg2rad(np.linspace(1, 85, 40))
    for eps in (SIO2.eps_inf, EPS_AU):
        r = stack_coefficients(Stack.isotropic(eps), K * np.sin(th), K)["r"]
        np.testing.assert_allclose(r, fresnel_rp(eps, np.sin(th)), rtol=1e-12, atol=1e-12)


def test_fresnel_limits():
    n = 1.45
    assert fresnel_rp(n**2, 0.0) == pytest.approx((n - 1) / (n + 1))  # normal incidence (Hz amplitude)
    assert abs(fresnel_rp(n**2, np.sin(np.arctan(n)))) < 1e-12  # Brewster angle


def test_zero_thickness_layer_is_transparent():
    th = np.deg2rad([30, 60])
    a = stack_coefficients(Stack.isotropic(EPS_AU), K * np.sin(th), K)["r"]
    b = stack_coefficients(ws2_on_au(0.0), K * np.sin(th), K)["r"]
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_layer_of_substrate_material_is_invisible():
    th = np.deg2rad([20, 60])
    a = stack_coefficients(Stack.isotropic(EPS_AU), K * np.sin(th), K)["r"]
    b = stack_coefficients(Stack((EPS_AU, EPS_AU), ((EPS_AU, EPS_AU, 0.03),)), K * np.sin(th), K)["r"]
    np.testing.assert_allclose(a, b, rtol=1e-12)


def _global_solve(stack, kx, k, polarization="p"):
    """Independent check: all interface conditions as one linear system (amplitudes referenced at y = 0)."""
    eps = stack.region_eps()
    n = len(eps)
    if polarization == "p":
        ky = [np.sqrt(ex * (k**2 - kx[0] ** 2 / ey) + 0j) for ex, ey in eps]
        ky = [q if q.imag >= 0 else -q for q in ky]
        adm = [ky[m] / eps[m][0] for m in range(n)]
    else:
        ky = [np.sqrt(ex * k**2 - kx[0] ** 2 + 0j) for ex, _ in eps]
        ky = [q if q.imag >= 0 else -q for q in ky]
        adm = list(ky)
    z = stack.tops()[1:]  # interface heights: 0, -t1, ...
    # unknowns: r, (U_m, D_m) for layers, T ; field in region m: U e^{i ky y} + D e^{-i ky y}
    nu = 2 * n - 2
    M = np.zeros((nu, nu), complex)
    rhs = np.zeros(nu, complex)

    def col(m, which):  # index of the up/down amplitude of region m
        if m == 0:
            return 0 if which == "U" else None
        if m == n - 1:
            return nu - 1 if which == "D" else None
        return 1 + 2 * (m - 1) + (0 if which == "U" else 1)

    row = 0
    for m in range(n - 1):
        y = z[m]
        for cond in ("H", "E"):
            for reg, sgn in ((m, 1), (m + 1, -1)):
                for which, s in (("U", 1), ("D", -1)):
                    fac = np.exp(1j * s * ky[reg] * y) * (1 if cond == "H" else s * adm[reg])
                    c = col(reg, which)
                    if c is None:
                        if reg == 0 and which == "D":
                            rhs[row] -= sgn * fac  # incident wave, unit amplitude
                        continue
                    M[row, c] += sgn * fac
            row += 1
    sol = np.linalg.solve(M, rhs)
    return sol[0]


@pytest.mark.parametrize("theta", [10.0, 45.0, 60.0, 80.0])
def test_recursion_matches_global_solve(theta):
    kx = np.array([K * np.sin(np.deg2rad(theta))])
    for st in (ws2_on_au(), three_layer()):
        r = stack_coefficients(st, kx, K)["r"][0]
        assert r == pytest.approx(_global_solve(st, kx, K), rel=1e-10)


def test_lossless_energy_conservation_multilayer():
    th = np.deg2rad(np.linspace(5, 80, 16))
    st = Stack((2.25 + 0j, 2.25 + 0j), ((4.0 + 0j, 3.0 + 0j, 0.05), (1.9 + 0j, 1.9 + 0j, 0.2)))
    c = stack_coefficients(st, K * np.sin(th), K)
    R = np.abs(c["r"]) ** 2
    T = np.abs(c["down"][-1]) ** 2 * (c["ky"][-1].real / 2.25) / c["ky"][0].real  # TM power flux of Hz amplitudes
    np.testing.assert_allclose(R + T, 1.0, rtol=1e-12)


@pytest.mark.parametrize("stack_fn", [ws2_on_au, three_layer])
def test_tangential_continuity_at_interfaces(stack_fn):
    st = stack_fn()
    beam = GaussianBeam(WL, 60, 14)
    h = 1e-10
    for y0 in st.tops()[1:]:
        up = beam_fields(beam, st, np.array([0.013]), np.array([y0 + h]))
        dn = beam_fields(beam, st, np.array([0.013]), np.array([y0 - h]))
        for i in (0, 1):  # Hz and Ex are tangential
            assert abs(up[i][0] - dn[i][0]) < 1e-6 * abs(up[i][0])


@pytest.mark.parametrize("y", [0.3, -0.0003, -0.0008, -0.05, -0.2])
def test_maxwell_consistency(y):
    """Ex = (i/(k eps_x)) dHz/dy and Ey = -(i/(k eps_y)) dHz/dx in every region."""
    st = three_layer()
    beam = GaussianBeam(WL, 60, 14)
    h = 1e-7
    x0 = 0.05

    def hz(x, yy):
        return beam_fields(beam, st, np.array([x]), np.array([yy]))[0][0]

    _, ex, ey = beam_fields(beam, st, np.array([x0]), np.array([y]))
    tops = st.tops()
    m = 0 if y >= 0 else int(np.sum(y < tops[1:]))
    exx, eyy = st.region_eps()[m]
    dhy = (hz(x0, y + h) - hz(x0, y - h)) / (2 * h)
    dhx = (hz(x0 + h, y) - hz(x0 - h, y)) / (2 * h)
    assert ex[0] == pytest.approx(1j / (K * exx) * dhy, rel=1e-5)
    assert ey[0] == pytest.approx(-1j / (K * eyy) * dhx, rel=1e-5)


def test_grid_and_pointwise_evaluation_agree():
    beam = GaussianBeam(WL, 60, 14)
    st = three_layer()
    xs = np.linspace(-0.2, 0.2, 9)
    ys = np.array([-0.15, -0.05, -0.0003, 0.0, 0.0009, 0.15])
    g = beam_fields_grid(beam, st, xs, ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    p = beam_fields(beam, st, X, Y)
    for a, b in zip(g, p):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_total_is_incident_plus_reflected_in_air():
    beam = GaussianBeam(WL, 60, 14)
    st = ws2_on_au()
    x, y = np.array([-0.1, 0.2]), np.array([0.05, 0.4])
    tot = beam_fields(beam, st, x, y)
    inc = beam_fields(beam, st, x, y, part="incident")
    ref = beam_fields(beam, st, x, y, part="reflected")
    for t, i, r in zip(tot, inc, ref):
        np.testing.assert_allclose(t, i + r, rtol=1e-12)


def test_beam_is_focused_and_exact_angle():
    beam = GaussianBeam(WL, 60, 14)
    th, w = beam.angles_weights()
    assert np.sum(w * th) / np.sum(w) == pytest.approx(np.deg2rad(60), abs=1e-12)
    xs = np.linspace(-1, 1, 201)
    hz, _, _ = beam_fields(beam, Stack.isotropic(1.0), xs, -xs * np.tan(np.deg2rad(30)), part="incident")
    assert abs(xs[np.argmax(np.abs(hz))]) < 0.011


def test_gaussian_beam_has_no_power_law_tail():
    beam = GaussianBeam(WL, 60, 14)
    th0 = np.deg2rad(60)
    s = np.array([0.0, 2.0, 4.0])  # distance from the axis in units of the waist
    w0 = WL / (np.pi * np.deg2rad(14))
    x, y = s * w0 * np.cos(th0), s * w0 * np.sin(th0)
    _, ex, ey = beam_fields(beam, Stack.isotropic(1.0), x, y, part="incident")
    i = (np.abs(ex) ** 2 + np.abs(ey) ** 2) / (abs(ex[0]) ** 2 + abs(ey[0]) ** 2)
    assert i[1] == pytest.approx(np.exp(-8), rel=0.15)
    assert i[2] < 1e-5  # floor set by the cut of the spectrum at 2 divergence angles


# ----------------------------------------------------------------------------- s polarization


def test_s_single_interface_reduces_to_fresnel():
    th = np.deg2rad(np.linspace(1, 85, 40))
    for eps in (SIO2.eps_inf, EPS_AU):
        r = stack_coefficients(Stack.isotropic(eps), K * np.sin(th), K, "s")["r"]
        k1 = np.cos(th)
        k2 = np.sqrt(eps - np.sin(th) ** 2 + 0j)
        np.testing.assert_allclose(r, (k1 - k2) / (k1 + k2), rtol=1e-12, atol=1e-12)


def test_s_sees_only_the_in_plane_permittivity():
    th = np.deg2rad([20, 60])
    a = Stack((EPS_AU, EPS_AU), ((6.0 + 1.5j, 4.0 + 0.1j, 0.0007),))
    b = Stack((EPS_AU, EPS_AU), ((6.0 + 1.5j, 1.7 + 0.0j, 0.0007),))
    ra = stack_coefficients(a, K * np.sin(th), K, "s")["r"]
    rb = stack_coefficients(b, K * np.sin(th), K, "s")["r"]
    np.testing.assert_allclose(ra, rb, rtol=1e-14)
    assert not np.allclose(stack_coefficients(a, K * np.sin(th), K)["r"], stack_coefficients(b, K * np.sin(th), K)["r"])


@pytest.mark.parametrize("theta", [10.0, 45.0, 60.0, 80.0])
def test_s_recursion_matches_global_solve(theta):
    kx = np.array([K * np.sin(np.deg2rad(theta))])
    for st in (ws2_on_au(), three_layer()):
        r = stack_coefficients(st, kx, K, "s")["r"][0]
        assert r == pytest.approx(_global_solve(st, kx, K, "s"), rel=1e-10)


def test_s_lossless_energy_conservation_multilayer():
    th = np.deg2rad(np.linspace(5, 80, 16))
    st = Stack((2.25 + 0j, 2.25 + 0j), ((4.0 + 0j, 3.0 + 0j, 0.05), (1.9 + 0j, 1.9 + 0j, 0.2)))
    c = stack_coefficients(st, K * np.sin(th), K, "s")
    R = np.abs(c["r"]) ** 2
    T = np.abs(c["down"][-1]) ** 2 * c["ky"][-1].real / c["ky"][0].real  # TE power flux of Ez amplitudes
    np.testing.assert_allclose(R + T, 1.0, rtol=1e-12)


@pytest.mark.parametrize("stack_fn", [ws2_on_au, three_layer])
def test_s_tangential_continuity_at_interfaces(stack_fn):
    st = stack_fn()
    beam = GaussianBeam(WL, 60, 14, polarization="s")
    h = 1e-10
    for y0 in st.tops()[1:]:
        up = beam_fields(beam, st, np.array([0.013]), np.array([y0 + h]))
        dn = beam_fields(beam, st, np.array([0.013]), np.array([y0 - h]))
        for i in (0, 1):  # Ez and Hx are tangential
            assert abs(up[i][0] - dn[i][0]) < 1e-6 * abs(up[i][0])


@pytest.mark.parametrize("y", [0.3, -0.0003, -0.0008, -0.05, -0.2])
def test_s_maxwell_consistency(y):
    """Hx = -(i/k) dEz/dy and Hy = (i/k) dEz/dx in every region (mu = 1)."""
    st = three_layer()
    beam = GaussianBeam(WL, 60, 14, polarization="s")
    h = 1e-7
    x0 = 0.05

    def ez(x, yy):
        return beam_fields(beam, st, np.array([x]), np.array([yy]))[0][0]

    _, hx, hy = beam_fields(beam, st, np.array([x0]), np.array([y]))
    dey = (ez(x0, y + h) - ez(x0, y - h)) / (2 * h)
    dex = (ez(x0 + h, y) - ez(x0 - h, y)) / (2 * h)
    assert hx[0] == pytest.approx(-1j / K * dey, rel=1e-5)
    assert hy[0] == pytest.approx(1j / K * dex, rel=1e-5)


def test_s_helmholtz_in_a_layer():
    """Ez solves (laplacian + k^2 eps_x) Ez = 0 inside a uniaxial layer."""
    st = three_layer()
    beam = GaussianBeam(WL, 60, 14, polarization="s")
    h = 2e-5
    x0, y0 = 0.02, -0.05  # inside the 90 nm dielectric layer (eps = 2.1)

    def ez(x, yy):
        return beam_fields(beam, st, np.array([x]), np.array([yy]))[0][0]

    lap = (ez(x0 + h, y0) + ez(x0 - h, y0) + ez(x0, y0 + h) + ez(x0, y0 - h) - 4 * ez(x0, y0)) / h**2
    assert lap == pytest.approx(-(K**2) * 2.1 * ez(x0, y0), rel=1e-3)


def test_s_grid_pointwise_and_parts():
    beam = GaussianBeam(WL, 60, 14, polarization="s")
    st = three_layer()
    xs = np.linspace(-0.2, 0.2, 9)
    ys = np.array([-0.15, -0.05, -0.0003, 0.0, 0.0009, 0.15])
    g = beam_fields_grid(beam, st, xs, ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for a, b in zip(g, beam_fields(beam, st, X, Y)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    x, y = np.array([-0.1, 0.2]), np.array([0.05, 0.4])
    tot = beam_fields(beam, st, x, y)
    inc = beam_fields(beam, st, x, y, part="incident")
    ref = beam_fields(beam, st, x, y, part="reflected")
    for t, i, r in zip(tot, inc, ref):
        np.testing.assert_allclose(t, i + r, rtol=1e-12)


def test_s_field_is_suppressed_on_gold():
    """Tangential Ez on gold: |Ez_bg|^2 at the surface ~ |t_s|^2 = |2 cos(th) / (cos(th) + sqrt(eps - sin^2 th))|^2
    of the incident intensity (0.09 at 633 nm and 60 deg), for the beam as for a plane wave."""
    beam = GaussianBeam(WL, 60, 14, polarization="s")
    ez = beam_fields(beam, Stack.isotropic(EPS_AU), np.array([0.0]), np.array([0.0]))[0]
    ei = beam_fields(beam, Stack.isotropic(1.0), np.array([0.0]), np.array([0.0]), part="incident")[0]
    c = np.cos(np.deg2rad(60))
    t2 = abs(2 * c / (c + np.sqrt(EPS_AU - (1 - c**2)))) ** 2
    assert abs(ez[0]) ** 2 / abs(ei[0]) ** 2 == pytest.approx(t2, rel=0.1)
    assert t2 < 0.1
