import importlib

import pytest

MODULES = [
    "cube.sim",
    "cube.reduce",
    "cube.parity",
    "data.parse_cfop",
    "data.build_dataset",
    "models.graph",
    "models.gnn",
    "search.beam",
    "search.evaluate",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
