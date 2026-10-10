import numpy as np
import pytest
from conftest import requires_meep

from ters2d.materials import AU, SIO2, WS2_MONOLAYER, Material, Pole, UniaxialMaterial, is_metallic

F_DENSE = np.linspace(0.05, 5.0, 4000)  # 200 nm - 20 um


def test_ws2_models_passive_everywhere():
    for m in (WS2_MONOLAYER.in_plane, WS2_MONOLAYER.out_of_plane):
        assert np.all(m.epsilon_at_frequency(F_DENSE).imag >= 0)
        assert all(p.sigma >= 0 and p.gamma >= 0 for p in m.poles)


def test_au_passive_and_metallic_in_window():
    eps = AU.epsilon(np.linspace(500, 900, 50))
    assert np.all(eps.imag > 0)
    assert np.all(eps.real < -2)
    assert is_metallic(AU, [500, 900]) and not is_metallic(SIO2, [500, 900])


def test_au_matches_literature_value_at_633nm():
    # Rakic Lorentz-Drude model: eps(633 nm) ~ -9.8 + 1.96i
    eps = AU.epsilon(633.0)
    assert eps.real == pytest.approx(-9.81, abs=0.05)
    assert eps.imag == pytest.approx(1.96, abs=0.05)


def test_lorentz_model_kramers_kronig():
    """Re(eps) - eps_inf equals the Kramers-Kronig transform of Im(eps) (causality / sign check)."""
    from scipy.integrate import quad

    ws2 = WS2_MONOLAYER.in_plane
    au_interband = Material("Au interband", 1.0, AU.poles[1:])
    for m in (ws2, au_interband):
        for w0 in (1.2, 1.6, 2.0):
            val, _ = quad(lambda w: w * m.epsilon_at_frequency(w).imag / (w + w0), 0, 400, weight="cauchy", wvar=w0, limit=800)
            assert 2 / np.pi * val == pytest.approx(m.epsilon_at_frequency(w0).real - m.eps_inf, rel=1e-4, abs=1e-5)


def test_validity_window_enforced():
    with pytest.raises(ValueError):
        WS2_MONOLAYER.check_wavelengths([400.0])
    with pytest.raises(ValueError):
        AU.check_wavelengths([10000.0])
    SIO2.check_wavelengths([100.0, 10000.0])  # no stated range


def test_nonpassive_pole_rejected():
    with pytest.raises(ValueError):
        Pole("lorentz", 1.0, 0.1, -0.5)


def test_constant_and_json_round_trip(tmp_path):
    m = Material.constant("n2", index=2.0)
    assert m.epsilon(500.0) == pytest.approx(4.0) and not m.dispersive
    p = tmp_path / "ws2.json"
    WS2_MONOLAYER.in_plane.save_json(p)
    back = Material.load_json(p)
    assert back == WS2_MONOLAYER.in_plane


def test_uniaxial_epsilon_xy():
    u = UniaxialMaterial("u", Material.constant("a", epsilon=4.0), Material.constant("b", epsilon=2.0))
    ex, ey = u.epsilon_xy(600.0)
    assert ex == pytest.approx(4.0) and ey == pytest.approx(2.0)


def test_metal_classification_rejects_sign_change():
    drude = Material("drude", 1.0, (Pole("drude", 1e-10, 0.05, 2.0 / 1e-20),))  # plasma frequency sqrt(2) / um
    assert is_metallic(drude, [1000.0])
    with pytest.raises(ValueError):
        is_metallic(drude, [500.0, 1000.0])


@requires_meep
def test_au_matches_meep_library():
    from meep.materials import Au

    for wl in (500.0, 633.0, 850.0):
        assert complex(np.asarray(Au.epsilon(1e3 / wl))[0, 0]) == pytest.approx(complex(AU.epsilon(wl)), rel=1e-12)
    m = Material.from_meep("Au")
    assert m.poles == AU.poles and m.eps_inf == AU.eps_inf


@requires_meep
def test_meep_media_match_python_models():
    from ters2d.meep_model import meep_medium

    for mat in (AU, SIO2):
        e = np.asarray(meep_medium(mat).epsilon(1e3 / 620.0))
        assert e[0, 0] == pytest.approx(complex(mat.epsilon(620.0)), rel=1e-12)
    e = np.asarray(meep_medium(WS2_MONOLAYER).epsilon(1e3 / 620.0))
    ex, ey = WS2_MONOLAYER.epsilon_xy(620.0)
    assert e[0, 0] == pytest.approx(complex(ex), rel=1e-12)
    assert e[2, 2] == pytest.approx(complex(ex), rel=1e-12)
    assert e[1, 1] == pytest.approx(complex(ey), rel=1e-12)
