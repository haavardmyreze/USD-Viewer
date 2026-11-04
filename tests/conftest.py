import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# keep test uploads away from the real data directory
os.environ.setdefault("USDV_DATA_DIR", tempfile.mkdtemp(prefix="usdv-test-"))

from tests.make_scene import build  # noqa: E402


@pytest.fixture(scope="session")
def scene_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("scene")
    build(out)
    return out
