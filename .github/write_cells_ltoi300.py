"""Write cells_ltoi300.md with kwasm viewers."""

import base64
import inspect
import shutil
import traceback

import kwasm.embed
import matplotlib as mpl
import matplotlib.pyplot as plt
from gdsfactory.serialization import clean_value_json

mpl.use("Agg")

from ltoi300 import _cells as cells
from ltoi300.config import PATH


def _can_resolve(module_path: str, name: str) -> bool:
    """Check if griffe can resolve a cell (skip alias resolution errors)."""
    try:
        import griffe

        loader = griffe.GriffeLoader()
        mod = loader.load(module_path)
        member = mod.members.get(name)
        if member and hasattr(member, "is_alias") and member.is_alias:
            member.resolve(loader)
        return True
    except Exception:
        return False


filepath = PATH.repo / "docs" / "cells_ltoi300.md"
kwasm_dir = PATH.repo / "docs" / "kwasm"
gds_dir = kwasm_dir / "gds"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()


def _setup_kwasm_viewer() -> None:
    """Write the shared kwasm viewer HTML (with empty GDS slot)."""
    if kwasm_dir.exists():
        shutil.rmtree(kwasm_dir)
    gds_dir.mkdir(parents=True)
    template = kwasm.embed._read_artifacts()
    template = template.replace("KWASM_GDS_B64", "")
    lyp_path = PATH.repo / "ltoi300" / "klayout" / "ltoi300.lyp"
    if lyp_path.exists():
        lyp_b64 = base64.b64encode(lyp_path.read_bytes()).decode("ascii")
        template = template.replace("KWASM_LYP_B64", lyp_b64)
    else:
        template = template.replace("KWASM_LYP_B64", "")
    template = template.replace("KWASM_LYRDB_B64", "")
    template = template.replace("KWASM_NETLIST_B64", "")
    (kwasm_dir / "viewer.html").write_text(template)


def _write_gds(name: str, kwargs: str) -> bool:
    """Instantiate cell and write GDS + PNG. Returns True on success."""
    try:
        sig = inspect.signature(cells[name])
        defaults = {}
        for p in sig.parameters:
            v = sig.parameters[p].default
            if isinstance(v, int | float | str | tuple):
                defaults[p] = v
        c = cells[name](**defaults)
        c.write(str(gds_dir / f"{name}.gds"))
        c.plot()
        plt.savefig(str(gds_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception:
        traceback.print_exc()
        plt.close("all")
        return False
    else:
        return True


_setup_kwasm_viewer()

with open(filepath, "w+") as f:
    f.write(
        """

Luxtelligence provides a library of components that have been fabricated in the reference material stack, and whose performance has been tested and validated. Here follows a list of the available parametric cells (gdsfactory.Component objects):


Cells ltoi300
=============================
"""
    )

    for name in sorted(cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(cells[name])
        kwargs = ", ".join(
            [
                f"{p}={clean_value_json(sig.parameters[p].default)!r}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, int | float | str | tuple)
                and p not in skip_settings
            ]
        )
        resolved = _can_resolve("ltoi300.cells", name)
        if name in skip_plot:
            f.write(f"\n\n## {name}\n\n")
            if resolved:
                f.write(f"\n::: ltoi300.cells.{name}\n\n")
        else:
            f.write(f"\n\n## {name}\n\n")
            if resolved:
                f.write(f"\n::: ltoi300.cells.{name}\n\n")
            has_gds = _write_gds(name, kwargs)
            if has_gds:
                f.write('=== "Static"\n\n')
                f.write(f"    ![{name}](kwasm/gds/{name}.png)\n\n")
                f.write('=== "Dynamic"\n\n')
                f.write(
                    f'    <iframe src="kwasm/viewer.html?url=gds/{name}.gds"'
                    f' loading="lazy" width="100%" height="400"'
                    f' style="border:none"></iframe>\n\n'
                )
            f.write(
                f"""```python
import ltoi300

c = ltoi300.cells.{name}({kwargs}).copy()
c.draw_ports()
c.plot()
```
"""
            )

print(f"Wrote {filepath}")
