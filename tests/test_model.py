import pytest

from ters2d import Illumination, Layer, Model, Numerics, Outputs, Sample, Tip
from ters2d.materials import AU, SIO2, WS2_MONOLAYER


def test_with_value_nested():
    m = Model()
    m2 = m.with_value("tip.apex_radius_nm", 30.0)
    assert m2.tip.apex_radius_nm == 30.0 and m.tip.apex_radius_nm == 20.0
    assert m.with_value("gap_nm", 4.0).gap_nm == 4.0
    with pytest.raises(AttributeError):
        m.with_value("tip.radius", 3.0)


def test_validation():
    with pytest.raises(ValueError):
        Tip(apex_radius_nm=0)
    with pytest.raises(ValueError):
        Illumination(incidence_deg=80, divergence_deg=14)
    with pytest.raises(ValueError):
        Numerics(drive_ramp_um=(4.5, 6.0))
    with pytest.raises(ValueError):
        Numerics(chunk_layout="x")
    with pytest.raises(ValueError):
        Model(gap_nm=0)
    with pytest.raises(ValueError):
        Outputs(near_box_nm=(1, 2, -1, 1))
    with pytest.raises(TypeError):
        Tip(material=WS2_MONOLAYER)


def test_sample_label_and_order():
    s = Sample(AU, layers=[Layer(SIO2, 5.0), Layer(WS2_MONOLAYER, 0.65)])
    assert isinstance(s.layers, tuple)
    assert s.label == "WS2/SiO2/Au"


def test_shaft_illumination_check():
    from ters2d.layout import build_layout
    from ters2d.run import check_shaft_illumination

    lay = build_layout(Model(), [633.0])
    assert check_shaft_illumination(lay, [633.0])[633.0] < 1e-4
    steep = build_layout(Model(illumination=Illumination(incidence_deg=45.0)), [633.0])
    with pytest.raises(ValueError):
        check_shaft_illumination(steep, [633.0])
    small = build_layout(Model(numerics=Numerics.small(500)), [633.0])  # check disabled by the preset
    assert check_shaft_illumination(small, [633.0])[633.0] > 0.1


def test_polarization_option():
    from ters2d.layout import build_layout

    with pytest.raises(ValueError):
        Illumination(polarization="x")
    m = Model().with_value("illumination.polarization", "s")
    assert m.illumination.polarization == "s" and Model().illumination.polarization == "p"
    lp, ls = build_layout(Model(), [633.0]), build_layout(m, [633.0])
    assert lp.tip_grid is lp.tip  # p: nothing shifted
    assert ls.tip_grid.apex_y > ls.tip.apex_y and ls.tip_grid.apex_y - ls.tip.apex_y < 1e-2 * ls.dx
