import inspect

from lnoi400 import _cells as cells
from lnoi400.config import PATH

filepath = PATH.repo / "docs" / "cells.rst"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()


with open(filepath, "w+") as f:
    f.write(
        """

Luxtelligence provides a library of components that have been fabricated in the reference material stack, and whose performance has been tested and validated. Here follows a list of the available parametric cells (gdsfactory.Component objects):


Cells
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
                f"{p}={repr(sig.parameters[p].default)}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, int | float | str | tuple)
                and p not in skip_settings
            ]
        )
        if name in skip_plot:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: lnoi400.cells.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: lnoi400.cells.{name}

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.{name}({kwargs})
  c.plot()

"""
            )
