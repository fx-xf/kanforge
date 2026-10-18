"""Every tutorial in examples/ runs (quick mode, a few optimisation steps)."""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("[0-9]*.py"))
NEEDS = {"05_images_kanconv.py": ["sklearn"], "06_transformer_kanify.py": ["transformers"]}


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(path):
    pytest.importorskip("torch")
    for mod in NEEDS.get(path.name, []):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"needs {mod}")
    env = {**os.environ, "KANFORGE_QUICK": "1", "PYTHONWARNINGS": "ignore"}
    res = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip()
