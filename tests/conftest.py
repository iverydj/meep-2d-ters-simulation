import importlib.util

import pytest

HAS_MEEP = importlib.util.find_spec("meep") is not None
requires_meep = pytest.mark.skipif(not HAS_MEEP, reason="meep is not installed")

HAS_NGSOLVE = importlib.util.find_spec("ngsolve") is not None
requires_ngsolve = pytest.mark.skipif(not HAS_NGSOLVE, reason="ngsolve is not installed")
