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


@requires_meep
def test_s_drive_is_jz_on_ez_points():
    import meep as mp
    import numpy as np

    from ters2d.background import GaussianBeam, Stack, beam_fields_grid
    from ters2d.sources import volume_sources

    class Disk:
        def contains(self, x, y):
            return x**2 + y**2 <= 0.004**2

    dx, eps = 1e-3, -9.8 + 2.0j
    beam = GaussianBeam(0.633, 60, 14, polarization="s")
    src = mp.GaussianSource(1 / 0.633, fwidth=1.0)
    srcs, n = volume_sources(Disk(), (-0.006, 0.006, -0.006, 0.006), dx, beam, Stack.isotropic(1.0), eps, src)
    assert len(srcs) == 1 and srcs[0].component == mp.Ez
    xs = (np.arange(-6, 7)) * dx  # Ez points at whole cells
    ez = beam_fields_grid(beam, Stack.isotropic(1.0), xs, xs)[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    inside = X**2 + Y**2 <= 0.004**2
    expected = np.where(inside, -1j * (2 * np.pi / 0.633) * (eps - 1) * ez, 0)
    np.testing.assert_allclose(srcs[0].amp_data[1:-1, 1:-1, 0], expected, rtol=1e-12, atol=1e-14)
    assert n == int(inside.sum())


@requires_meep
def test_tiny_s_run(tmp_path):
    model = Model(gap_nm=2.0, numerics=Numerics.small(resolution=500)).with_value("illumination.polarization", "s")
    r = simulate(model, 633.0, tmp_path, log=lambda *_: None)
    assert r.meta["dft_converged"] and r.meta["polarization"] == "s"
    # the tip screens the tangential Ez: FEM of this model 0.0040, upstream Meep 0.0032 at 2 nm cells; without the
    # drive the value would be the tip-free 0.091, with the drive sign flipped ~0.4
    assert 0.0025 < r.metrics["I_probe_tip_over_inc"] < 0.0045
