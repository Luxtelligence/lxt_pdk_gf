import inspect
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import numpy as np
import sax
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import NDArray

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray


########################
# Models dict discovery
########################


def get_models(module: ModuleType) -> dict[str, Callable[..., sax.SDict]]:
    """Get all models defined in a module."""
    models = {}
    for name, func in inspect.getmembers(module):
        if name[0] != "_":
            if not callable(func):
                continue
            _func = func
            while isinstance(_func, partial):
                _func = _func.func
            try:
                sig = inspect.signature(_func)
            except ValueError:
                continue
            if sig.return_annotation == sax.SDict:
                models[name] = func
    return models


####################
# Utility functions
####################


def get_json_data(
    module: ModuleType,
    data_tag: str,
) -> dict:
    """Load data from a json structure."""

    path = Path(module.__file__).parent / "data" / f"{data_tag}.json"
    with open(path) as f:
        data_dict = json.load(f)
    return data_dict


def poly_eval_from_json(
    module: ModuleType,
    wl: np.ndarray,
    data_tag: str,
    key: str,
) -> np.ndarray:
    """Evaluate a polynomial model for frequency-dependent response
    stored in json format."""

    cell_data = get_json_data(module, data_tag)
    if "center_wavelength" in cell_data.keys():
        wl0 = cell_data["center_wavelength"]
    else:
        wl0 = 0.0
    poly_coef = cell_data[key]
    poly_coef.reverse()
    poly_model = Polynomial(poly_coef)
    return poly_model(wl - wl0)


#####################################
# Polynomial built by interpolation
#####################################


def _2port_poly_model(
    module: ModuleType,
    data_tag: str,
    wl: Float = 1.55,
    trans_abs_key: str = "",
    trans_phase_key: str = "",
    refl_abs_key: str = "",
    refl_phase_key: str = "",
) -> sax.SDict:
    s_par = {}
    locs = locals()
    for key in ["trans_abs_key", "trans_phase_key", "refl_abs_key", "refl_phase_key"]:
        if locs[key]:
            s_par[key] = poly_eval_from_json(module, wl, data_tag, locs[key])
        else:
            s_par[key] = np.zeros_like(wl)

    trans = s_par["trans_abs_key"] * jnp.exp(1j * s_par["trans_phase_key"])
    refl = s_par["refl_abs_key"] * jnp.exp(1j * s_par["refl_phase_key"])

    return sax.reciprocal(
        {
            ("o2", "o1"): trans,
            ("o1", "o1"): refl,
            ("o2", "o2"): refl,
        }
    )


def _1in_2out_symmetric_poly_model(
    module: ModuleType,
    data_tag: str,
    wl: Float = 1.55,
    trans_abs_key: str = "",
    trans_phase_key: str = "",
    rin_abs_key: str = "",
    rin_phase_key: str = "",
    rout_abs_key: str = "",
    rout_phase_key: str = "",
    rcross_abs_key: str = "",
    rcross_phase_key: str = "",
) -> sax.SDict:
    s_par = {}
    locs = locals()
    for key in [
        "trans_abs_key",
        "trans_phase_key",
        "rin_abs_key",
        "rin_phase_key",
        "rout_abs_key",
        "rout_phase_key",
        "rcross_abs_key",
        "rcross_phase_key",
    ]:
        if locs[key]:
            s_par[key] = poly_eval_from_json(module, wl, data_tag, locs[key])
        else:
            s_par[key] = np.zeros_like(wl)

    trans = s_par["trans_abs_key"] * jnp.exp(1j * s_par["trans_phase_key"])
    rin = s_par["rin_abs_key"] * jnp.exp(1j * s_par["rin_phase_key"])
    rout = s_par["rout_abs_key"] * jnp.exp(1j * s_par["rout_phase_key"])
    rcross = s_par["rcross_abs_key"] * jnp.exp(1j * s_par["rcross_phase_key"])

    return sax.reciprocal(
        {
            ("o2", "o1"): trans,
            ("o3", "o1"): trans,
            ("o1", "o1"): rin,
            ("o2", "o2"): rout,
            ("o3", "o3"): rout,
            ("o2", "o3"): rcross,
        }
    )


def _2in_2out_symmetric_poly_model(
    module: ModuleType,
    data_tag: str,
    wl: Float = 1.55,
    trans_bar_abs_key: str = "",
    trans_bar_phase_key: str = "",
    trans_cross_abs_key: str = "",
    trans_cross_phase_key: str = "",
    refl_self_abs_key: str = "",
    refl_self_phase_key: str = "",
    refl_cross_abs_key: str = "",
    refl_cross_phase_key: str = "",
) -> sax.SDict:
    s_par = {}
    locs = locals()
    for key in [
        "trans_bar_abs_key",
        "trans_bar_phase_key",
        "trans_cross_abs_key",
        "trans_cross_phase_key",
        "refl_self_abs_key",
        "refl_self_phase_key",
        "refl_cross_abs_key",
        "refl_cross_phase_key",
    ]:
        if locs[key]:
            s_par[key] = poly_eval_from_json(module, wl, data_tag, locs[key])
        else:
            s_par[key] = np.zeros_like(wl)

    bar = s_par["trans_bar_abs_key"] * jnp.exp(1j * s_par["trans_bar_phase_key"])
    cross = s_par["trans_cross_abs_key"] * jnp.exp(1j * s_par["trans_cross_phase_key"])
    refl_self = s_par["refl_self_abs_key"] * jnp.exp(1j * s_par["refl_self_phase_key"])
    refl_cross = s_par["refl_cross_abs_key"] * jnp.exp(
        1j * s_par["refl_cross_phase_key"]
    )

    sdict = {
        ("o1", "o4"): bar,
        ("o2", "o3"): bar,
        ("o1", "o3"): cross,
        ("o2", "o4"): cross,
        ("o1", "o2"): refl_cross,
        ("o3", "o4"): refl_cross,
    }

    for n in range(1, 5):
        port = f"o{n}"
        sdict[(port, port)] = refl_self

    return sax.reciprocal(sdict)


if __name__ == "__main__":
    pass
