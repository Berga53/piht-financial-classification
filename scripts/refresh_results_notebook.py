"""Execute the results notebook and replace its saved tables and plots."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def refresh(root: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "piht-matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(Path(tempfile.gettempdir()) / "piht-ipython"))
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    path = root / "notebooks/inspect_results.ipynb"
    notebook = json.loads(path.read_text())
    shell = InteractiveShell.instance()
    setup = shell.run_cell(
        "import matplotlib\n"
        "matplotlib.use('module://matplotlib_inline.backend_inline')\n"
        "from matplotlib_inline.backend_inline import configure_inline_support\n"
        "configure_inline_support(get_ipython(), 'module://matplotlib_inline.backend_inline')",
        store_history=False,
    )
    setup.raise_error()
    with tempfile.TemporaryDirectory(prefix="piht-plots-") as directory:
        staged_plots = Path(directory)
        first_code = True
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            with capture_output() as captured:
                result = shell.run_cell("".join(cell["source"]), store_history=True)
            result.raise_error()
            outputs = []
            for name, value in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                if value:
                    # The notebook should report the final artifact location.
                    value = value.replace(str(staged_plots), str(root / "results/notebook_plots"))
                    outputs.append({"output_type": "stream", "name": name, "text": value})
            outputs.extend(
                {"output_type": "display_data", "data": o.data, "metadata": o.metadata}
                for o in captured.outputs
            )
            cell["execution_count"] = result.execution_count
            cell["outputs"] = outputs
            if first_code:
                shell.user_ns["PLOTS_DIR"] = staged_plots
                first_code = False
            print(f"Refreshed notebook cell {index}", flush=True)

        plots = sorted(staged_plots.glob("*.png"))
        if not plots:
            raise RuntimeError("Notebook produced no plots")
        from PIL import Image
        for plot in plots:
            with Image.open(plot) as image:
                image.verify()
        destination = root / "results/notebook_plots"
        destination.mkdir(parents=True, exist_ok=True)
        published = []
        for plot in plots:
            output = destination / plot.name
            temporary = output.with_suffix(".png.tmp")
            temporary.write_bytes(plot.read_bytes())
            temporary.replace(output)
            published.append(output)
        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        temporary.replace(path)
    print(f"Saved tables in {path} and {len(published)} plots in {destination}", flush=True)
    return [path, *published]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    refresh(project_root)
