"""Convert the percent workbench to Jupyter and execute its cached full baseline."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
source = root / "paper_figure/notebook/05_settled_free_energy_workbench.py"
target = source.with_suffix(".ipynb")
cells = []
mode, lines = None, []


def flush():
    if mode is None:
        return
    text = "\n".join(lines).strip()
    if mode == "markdown":
        text = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line
                         for line in text.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(text))


for line in source.read_text(encoding="utf-8").splitlines():
    if line.startswith("# %%"):
        flush()
        mode = "markdown" if "[markdown]" in line else "code"
        lines = []
    else:
        lines.append(line)
flush()
notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = dict(display_name="Python 3 (ipykernel)", language="python", name="python3")
notebook.metadata.language_info = dict(name="python")
notebook.metadata.jupytext = dict(formats="ipynb,py:percent", text_representation=dict(
    extension=".py", format_name="percent", format_version="1.3"))
nbformat.write(notebook, target)
os.environ.pop("PMT_NO_SHOW", None)
os.environ["PMT_MINIMAX_PROFILE"] = "full"
os.environ["JUPYTER_RUNTIME_DIR"] = str(root / "tmp/minimax_jupyter")
os.environ["IPYTHONDIR"] = str(root / "tmp/minimax_ipython")
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
client = NotebookClient(notebook, timeout=2400, kernel_name="python3",
                        resources={"metadata": {"path": str(source.parent)}})
client.execute()
nbformat.validate(notebook)
errors = [output for cell in notebook.cells if cell.cell_type == "code"
          for output in cell.get("outputs", []) if output.output_type == "error"]
assert not errors
nbformat.write(notebook, target)
print(f"Executed {sum(c.cell_type=='code' for c in notebook.cells)} code cells: {target}")
