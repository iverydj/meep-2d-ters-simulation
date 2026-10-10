"""The FEM reference reproduces its stored value of the default case (skipped without NGSolve; ~30 s)."""

import pytest
from conftest import requires_ngsolve

from ters2d import Model


@requires_ngsolve
def test_fem_default_case_matches_table():
    from ters2d.fem import FemSettings, solve

    r = solve(Model(), 633.0, FemSettings(threads=4))
    assert r["EF_probe"] == pytest.approx(151.307569, rel=1e-6)  # docs/figures/data/fem_tip_reference.csv
