"""Keep this project's tests out of pytest runs started from a parent folder.

In arbCube this project lives in nxn_reduction/, next to the team's own tests. Its tests
import cube, data, models and search from this folder, which only works with this
folder's pytest config (pyproject.toml: pythonpath = ["."]), so they run only when
pytest is started here: cd nxn_reduction, then python -m pytest.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent


def pytest_ignore_collect(collection_path, config):
    if Path(config.rootpath).resolve() != HERE:
        return True
    return None
