import inspect
from collections.abc import Callable
from pathlib import Path
from functools import partial
import lnoi400
import json

import jax.numpy as jnp
import sax
from gdsfactory.pdk import get_cross_section_name
from gdsfactory.typings import CrossSectionSpec
import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from gplugins.sax.models import straight as __straight
from gplugins.sax.models import bend as __bend


import matplotlib.pyplot as plt

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Straights
################

def _straight(
    *,
    wl: Float = 1.55,
    length: Float = 10.0,
    cross_section: CrossSectionSpec = "xs_rwg1000") -> sax.SDict:

    cross_section_name = get_cross_section_name(cross_section)

    if cross_section_name == "xs_rwg1000":
        return __straight(
                wl = wl,
                length = length,
                loss = 2e-5,
                wl0 = 1.55,
                neff = 1.8,
                ng = 2.22,
        )

    if cross_section_name == "xs_swg250":
        return __straight(
            wl = wl,
            length = length,
            loss = 1e-4,
            wl0 = 1.55,
            neff = 1.44,
            ng = 1.7,
        )

    else:
        raise ValueError(f"A model for the specified waveguide cross section {cross_section} is not defined.")
    
straight_rwg1000 = partial(_straight, cross_section="xs_rwg1000")
straight_swg250 = partial(_straight, cross_section="xs_swg250")
    
################
# Bends
################

def _2port_poly_model(
        *,
        wl: Float,
        data_tag: str,
        trans_abs_key: str = "",
        trans_phase_key: str = "",
        refl_abs_key: str = "",
        refl_phase_key: str = "",
) -> sax.SDict:
    
    s_par = {}
    for key in ["trans_abs_key", "trans_phase_key", "refl_abs_key", "refl_phase_key"]:
        locs = locals()
        if locs[key]:
            s_par[key] = poly_eval_from_json(wl, data_tag, locs[key])
        else:
            s_par[key] = np.zeros_like(wl)
    
    trans = s_par["trans_abs_key"]*jnp.exp(1j*s_par["trans_phase_key"])
    refl = s_par["refl_abs_key"]*jnp.exp(1j*s_par["refl_phase_key"])

    return sax.reciprocal(
        {
            ("o2", "o1"): trans,
            ("o1", "o1"): refl,
            ("o2", "o2"): refl,
        }
    )

U_bend_racetrack = partial(_2port_poly_model, 
                           data_tag="UBendRacetrack",
                           trans_abs_key="pol_trans_abs",
                           trans_phase_key="pol_trans_phase",
                           )


################
# Edge couplers
################


double_linear_inverse_taper = partial(_2port_poly_model,
                                      data_tag="edge_coupler_double_linear_taper",
                                      trans_abs_key="pol_trans_abs",
                                      trans_phase_key="pol_trans_phase",
                                      refl_abs_key="pol_refl_abs",
                                      refl_phase_key="pol_refl_phase"
                                      )


################
# MMIs
################

####################
# Utility functions
####################

def get_json_data(
        data_tag: str,
) -> dict:
    """Load data from a json structure."""
    path = Path(lnoi400.__file__).parent / "data" / f"{data_tag}.json"
    with open(path, "r") as f:
        data_dict = json.load(f)
    return data_dict

def poly_eval_from_json(
        wl: np.ndarray,
        data_tag: str,
        key: str,
) -> np.ndarray:
    """Evaluate a polynomial model for frequency-dependent response stored in json format."""
    cell_data = get_json_data(data_tag)
    if "center_wavelength" in cell_data.keys():
        wl0 = cell_data["center_wavelength"]
    else:
        wl0 = 0.0
    poly_coef = cell_data[key]
    poly_coef.reverse()
    poly_model = Polynomial(poly_coef)
    return poly_model(wl - wl0)

################
# Models Dict
################

def get_models() -> dict[str, Callable[..., sax.SDict]]:
    models = {}
    for name, func in list(globals().items()):
        if not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except ValueError:
            continue
        if str(sig.return_annotation) == "sax.SDict":
            models[name] = func
    return models

if __name__ == "__main__":

    wavelengths = np.linspace(1.4, 1.7, 1000)
    model = double_linear_inverse_taper(wl = wavelengths)

    plt.figure(tight_layout = True)
    plt.plot(wavelengths, 20*np.log10(np.abs(model["o1", "o2"])))
    plt.show()

