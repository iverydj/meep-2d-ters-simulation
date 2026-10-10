import numpy as np
import pytest

from ters2d import Layer, Model, Numerics, Outputs, Sample, Tip
from ters2d.layout import build_layout, snapped_thickness, tip_shape
from ters2d.materials import AU, SIO2, WS2_MONOLAYER, Material

WL = [633.0]


def test_gap_must_be_integer_cells():
    with pytest.raises(ValueError):
        build_layout(Model(gap_nm=2.0, numerics=Numerics(resolution=1501)), WL)
    build_layout(Model(gap_nm=2.0, numerics=Numerics(resolution=1500)), WL)


def test_layer_snapping():
    assert snapped_thickness(0.00065, 1 / 3000) == pytest.approx(2 / 3000)
    with pytest.raises(ValueError):
        snapped_thickness(0.00065, 1 / 2000)  # 1.3 cells


def test_interfaces_on_grid_lines_and_even_cells():
    for res in (1000, 1500, 2000, 3000):
        for sample in (Sample(AU), Sample(SIO2), Sample(AU, (Layer(WS2_MONOLAYER, 0.65),))):
            if sample.layers and res % 1500:
                continue
            lay = build_layout(Model(sample=sample, numerics=Numerics(resolution=res)), WL)
            vals = [*lay.cell_lo, *lay.cell_hi, lay.gap] + [v for yy in lay.layers for v in yy]
            for v in vals:
                assert abs(v * res - round(v * res)) < 1e-6
            for n in (round(s * res) for s in lay.cell_size):
                assert n % 2 == 0


def test_production_layout_unchanged():
    """The default model reproduces the cell of the reference production runs (26.84 million cells)."""
    lay = build_layout(Model(), WL)
    assert lay.n_cells == 26_840_000
    assert lay.cell_lo == pytest.approx((-2.2, -0.2)) and lay.cell_hi == pytest.approx((2.2, 5.9))
    assert not lay.bottom_pml and lay.side_absorbers
    sio2 = build_layout(Model(sample=Sample(SIO2)), WL)
    assert sio2.n_cells == 28_600_000 and sio2.bottom_pml and not sio2.side_absorbers


def test_tip_shape_is_smooth_union():
    t = tip_shape(0.02, 15.0, 0.002)
    assert t.contains(0.0, 0.002 + 1e-6) and not t.contains(0.0, 0.002 - 1e-6)
    yt = t.tangent_y
    assert t.half_width(yt - 1e-9) == pytest.approx(t.half_width(yt + 1e-9), rel=1e-6)
    assert t.half_width(1.0) - t.half_width(0.5) == pytest.approx(0.5 * np.tan(np.deg2rad(15)))
    outline = t.outline(0.3)
    assert np.all(t.contains(outline[:, 0] * 0.999, np.clip(outline[:, 1], 0.0021, None)))


def test_boundary_rules():
    with pytest.raises(ValueError):  # a dielectric substrate needs a bottom PML
        build_layout(Model(sample=Sample(SIO2), numerics=Numerics(bottom_pml=False)), WL)
    with pytest.raises(ValueError):  # a thin metal substrate without bottom PML is not opaque
        build_layout(Model(numerics=Numerics(substrate_depth_um=0.02)), WL)
    lay = build_layout(Model(numerics=Numerics(bottom_pml=True)), WL)
    assert lay.bottom_pml and lay.substrate_metallic


def test_layers_below_the_surface():
    s = Sample(SIO2, (Layer(Material.constant("spacer", index=1.5), 90.0), Layer(WS2_MONOLAYER, 0.65)))
    lay = build_layout(Model(sample=s, gap_nm=2.0, numerics=Numerics(resolution=1500)), WL)
    (b0, t0), (b1, t1) = lay.layers
    assert t1 == 0.0 and t0 == pytest.approx(b1) and b0 == pytest.approx(-(90.0 + 2 / 3) * 1e-3)
    st = lay.stack(633.0)
    assert st.layers[0][2] == pytest.approx(2 / 3000) and st.layers[1][2] == pytest.approx(0.09)


def test_shaft_must_clear_the_side_absorbers():
    with pytest.raises(ValueError):
        build_layout(Model(tip=Tip(apex_radius_nm=40), numerics=Numerics(side_absorber_start_um=0.6)), WL)
    build_layout(Model(tip=Tip(apex_radius_nm=40)), WL)


def test_probe_default_and_custom():
    assert build_layout(Model(gap_nm=4.0), WL).probe == pytest.approx((0.0, 0.002))
    lay = build_layout(Model(gap_nm=4.0, outputs=Outputs(probe_nm=(5.0, 1.0))), WL)
    assert lay.probe == pytest.approx((0.005, 0.001))
    with pytest.raises(ValueError):
        build_layout(Model(gap_nm=4.0, outputs=Outputs(probe_nm=(0.0, 10.0))), WL)  # inside the tip
