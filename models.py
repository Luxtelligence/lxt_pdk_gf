import json
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from gplugins.sax.models import straight as __straight
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

import lnoi400

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
    cross_section: str = "xs_rwg1000",
) -> sax.SDict:
    if not isinstance(cross_section, str):
        raise TypeError(
            f"""The cross_section parameter should be a string,
                        received {type(cross_section)} instead."""
        )

    if cross_section == "xs_rwg1000":
        return __straight(
            wl=wl,
            length=length,
            loss=2e-5,
            wl0=1.55,
            neff=1.8,
            ng=2.22,
        )

    if cross_section == "xs_swg250":
        return __straight(
            wl=wl,
            length=length,
            loss=1e-4,
            wl0=1.55,
            neff=1.44,
            ng=1.7,
        )

    else:
        raise ValueError(
            f"""A model for the specified waveguide
                         cross section {cross_section} is not defined."""
        )


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
    locs = locals()
    for key in ["trans_abs_key", "trans_phase_key", "refl_abs_key", "refl_phase_key"]:
        if locs[key]:
            s_par[key] = poly_eval_from_json(wl, data_tag, locs[key])
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


U_bend_racetrack = partial(
    _2port_poly_model,
    data_tag="ubend_racetrack",
    trans_abs_key="pol_trans_abs",
    trans_phase_key="pol_trans_phase",
)


################
# Edge couplers
################


double_linear_inverse_taper = partial(
    _2port_poly_model,
    data_tag="edge_coupler_double_linear_taper",
    trans_abs_key="pol_trans_abs",
    trans_phase_key="pol_trans_phase",
    refl_abs_key="pol_refl_abs",
    refl_phase_key="pol_refl_phase",
)


################
# MMIs
################


def _1in_2out_symmetric_poly_model(
    *,
    wl: Float,
    data_tag: str,
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
            s_par[key] = poly_eval_from_json(wl, data_tag, locs[key])
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
    *,
    wl: Float,
    data_tag: str,
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
            s_par[key] = poly_eval_from_json(wl, data_tag, locs[key])
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


mmi1x2_optimized1550 = partial(
    _1in_2out_symmetric_poly_model,
    data_tag="mmi_1x2_optimized_1550",
    trans_abs_key="pol_trans_abs",
    trans_phase_key="pol_trans_phase",
    rin_abs_key="pol_refl_in_abs",
    rin_phase_key="pol_refl_in_phase",
    rout_abs_key="pol_refl_out_abs",
    rout_phase_key="pol_refl_out_phase",
    rcross_abs_key="pol_refl_cross_abs",
    rcross_phase_key="pol_refl_cross_phase",
)

mmi2x2_optimized1550 = partial(
    _2in_2out_symmetric_poly_model,
    data_tag="mmi_2x2_optimized_1550",
    trans_bar_abs_key="pol_trans_bar_abs",
    trans_bar_phase_key="pol_trans_bar_phase",
    trans_cross_abs_key="pol_trans_cross_abs",
    trans_cross_phase_key="pol_trans_cross_phase",
    refl_self_abs_key="pol_refl_bar_abs",
    refl_self_phase_key="pol_refl_bar_phase",
    refl_cross_abs_key="pol_refl_cross_abs",
    refl_cross_phase_key="pol_refl_cross_phase",
)

####################
# Utility functions
####################


def get_json_data(
    data_tag: str,
) -> dict:
    """Load data from a json structure."""

    path = Path(lnoi400.__file__).parent / "data" / f"{data_tag}.json"
    with open(path) as f:
        data_dict = json.load(f)
    return data_dict


def poly_eval_from_json(
    wl: np.ndarray,
    data_tag: str,
    key: str,
) -> np.ndarray:
    """Evaluate a polynomial model for frequency-dependent response
    stored in json format."""

    cell_data = get_json_data(data_tag)
    if "center_wavelength" in cell_data.keys():
        wl0 = cell_data["center_wavelength"]
    else:
        wl0 = 0.0
    poly_coef = cell_data[key]
    poly_coef.reverse()
    poly_model = Polynomial(poly_coef)
    return poly_model(wl - wl0)


if __name__ == "__main__":
    import gplugins.sax as gs

    gs.plot_model(
        mmi2x2_optimized1550, wavelength_start=1.4, wavelength_stop=1.7, port1="o4"
    )
    plt.show()
