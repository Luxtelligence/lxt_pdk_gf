import inspect
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from gplugins.sax.models import phase_shifter as _phase_shifter
from gplugins.sax.models import straight as __straight
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

import lnoi400

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

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
    wl: Float = 1.55,
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
    wl: Float = 1.55,
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
    wl: Float = 1.55,
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


################
# Modulators
################


def eo_phase_shifter(
    wl: Float = 1.55,
    wl_0: float = 1.55,
    length: float = 7500.0,
    neff_0: float = 1.85,
    ng_0: float = 2.2,
    loss: float = 2e-5,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
) -> sax.SDict:
    # Default V_pi
    if np.isnan(V_pi):
        V_pi = 2 * 3.3e4 / length
    v = V_dc / V_pi

    # Effective index at the operation frequency
    neff = neff_0 - (ng_0 - neff_0) * (wl - wl_0) / wl_0

    ps = _phase_shifter(
        wl=wl,
        neff=neff,
        voltage=v,
        length=length,
        loss=loss,
    )

    return ps


def mzm_unbalanced(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 1000.0,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
) -> sax.SDict:
    mzm, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler": "mmi",
                "top": "ps_top",
                "bot": "ps_bot",
                "dl": "wg_straight",
                "splitter": "mmi",
            },
            "connections": {
                "coupler,o2": "top,o1",
                "coupler,o3": "bot,o1",
                "bot,o2": "dl,o1",
                "splitter,o2": "top,o2",
                "splitter,o3": "dl,o2",
            },
            "ports": {
                "o1": "coupler,o1",
                "o2": "splitter,o1",
            },
        },
        models={
            "mmi": partial(
                mmi1x2_optimized1550,
                wl=wl,
            ),
            "wg_straight": partial(
                _straight,
                wl=wl,
                length=length_imbalance,
                cross_section="xs_rwg1000",
            ),
            "ps_top": partial(
                eo_phase_shifter,
                wl=wl,
                length=modulation_length,
                V_dc=V_dc,
                V_pi=V_pi,
            ),
            "ps_bot": partial(
                eo_phase_shifter,
                wl=wl,
                length=modulation_length,
                V_dc=-V_dc,
                V_pi=V_pi,
            ),
        },
        backend="default",
    )
    return mzm()


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    models = {}
    for name, func in list(globals().items()):
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


if __name__ == "__main__":
    import gplugins.sax as gs

    # print(list(get_models()))
    # for name, model in get_models().items():
    #     try:
    #         print(name, model())
    #     except NotImplementedError:
    #         continue

    for V in [0, 1.0, 2.0, 3.0]:
        mzm = partial(
            mzm_unbalanced,
            V_dc=V,
            modulation_length=7500.0,
            length_imbalance=50.0,
        )

        gs.plot_model(
            mzm,
            wavelength_start=1.4,
            wavelength_stop=1.7,
            port1="o1",
            ports2=("o2",),
        )
    plt.show()
