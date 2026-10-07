from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (HERE / "results/round2/metrics.csv").exists(), reason="results absent")
def test_results_notebook_executes():
    nb = nbformat.read(HERE / "notebooks/results.ipynb", as_version=4)
    NotebookClient(nb, timeout=120, resources={"metadata": {"path": str(HERE / "notebooks")}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
