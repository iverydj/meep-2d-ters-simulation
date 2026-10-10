from pathlib import Path

import numpy as np
import pytest

from ters2d.fitting import fit_lorentz, fit_quality, read_nk_csv
from ters2d.materials import Material, Pole

ROOT = Path(__file__).resolve().parents[1]


def test_fit_recovers_a_known_model():
    true = Material("t", 2.0, (Pole("lorentz", 1.6, 0.05, 0.4), Pole("lorentz", 3.5, 1.0, 3.0)))
    wl = np.linspace(450, 950, 300)
    nk = np.sqrt(true.epsilon(wl))
    fit = fit_lorentz(wl, nk.real, nk.imag, [(2.0, 0.05), (4.3, 1.2)], name="fit", valid_wavelength_nm=(480, 900))
    q = fit_quality(fit, wl, nk.real, nk.imag)
    assert q["rel_rms"] < 1e-6
    assert q["min_im_eps_200nm_to_20um"] >= 0


def test_read_nk_csv_and_header_check(tmp_path):
    wl, n, k = read_nk_csv(ROOT / "materials" / "nk" / "ws2_monolayer_inplane_ermolaev2020.csv")
    assert wl[0] < wl[-1] and n.size == k.size == wl.size and np.all(k >= 0)
    bad = tmp_path / "bad.csv"
    bad.write_text("lambda,n,k\n500,1,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_nk_csv(bad)
