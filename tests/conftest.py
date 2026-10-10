import importlib.util

import pytest

HAS_MEEP = importlib.util.find_spec("meep") is not None
requires_meep = pytest.mark.skipif(not HAS_MEEP, reason="meep is not installed")
