"""End-to-end smoke tests on a tiny, coarse domain (skipped when meep is not installed)."""

from conftest import requires_meep

from ters2d import Layer, Model, Numerics, Sample, field_map, simulate
from ters2d.materials import AU, Material


@requires_meep
def test_tiny_gap_mode_run(tmp_path):
    model = Model(gap_nm=2.0, numerics=Numerics.small(resolution=500))
    r = simulate(model, 633.0, tmp_path, log=lambda *_: None)
    assert r.ef_probe > 10  # a 2 nm Au-Au gap enhances the field strongly
    assert r.meta["dft_converged"]
    fm = field_map(r)
    assert fm.e2_rel.shape == (fm.x_nm.size, fm.y_nm.size)
    again = simulate(model, 633.0, tmp_path, log=lambda *_: None)  # reuses the file
    assert again.ef_probe == r.ef_probe


@requires_meep
def test_tiny_layered_run(tmp_path):
    spacer = Material.constant("spacer", index=2.0)
    model = Model(sample=Sample(AU, (Layer(spacer, 2.0),)), gap_nm=2.0, numerics=Numerics.small(resolution=500))
    r = simulate(model, 633.0, tmp_path, log=lambda *_: None)
    assert r.ef_probe > 10 and "EF_layer" in r.metrics
