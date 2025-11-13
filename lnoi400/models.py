import inspect
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import sax
from gplugins.sax.models import phase_shifter as _phase_shifter
from gplugins.sax.models import straight as __straight
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from sax.utils import reciprocal

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
            loss_dB_cm=0.2,
            wl0=1.55,
            neff=1.8,
            ng=2.22,
        )

    if cross_section == "xs_swg250":
        return __straight(
            wl=wl,
            length=length,
            loss_dB_cm=1.0,
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

#####################
# Directional coupler
#####################

directional_coupler_balanced = partial(
    _2in_2out_symmetric_poly_model,
    data_tag="directional_coupler_balanced",
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
    ng_0: float = 2.21,
    loss: float = 2e-5,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
) -> sax.SDict:
    # Default V_pi
    if np.isnan(V_pi):
        V_pi = 2 * 3.3e4 * wl / length / wl_0
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


def to_phase_shifter(
    wl: Float = 1.55,
    wl_0: float = 1.55,
    neff_0: float = 1.8,
    ng_0: float = 2.22,
    loss: float = 2e-5,
    heater_length: float = 700.0,
    heater_width: float = 1.0,
    P_pi: float = np.nan,
    R: float = np.nan,
    V_dc: float = 0.0,
):
    """Model for a thermal phase shifter.

    Args:
        wl: wavelength in um.
        wl_0: center wavelength in um.
        neff_0: effective index at center wavelength.
        ng_0: group index at center wavelength.
        loss: propagation loss (dB/um)
        heater_length: in um.
        heater_width: in um.
        P_pi: dissipated power for a pi phase shift (W).
        R: resistance (Ohm).
        V_dc: static voltage applied to the resistor (V).
    """
    if np.isnan(R):
        R = 25 * heater_length / 700.0 / heater_width
    if np.isnan(P_pi):
        P_pi = 0.075 * heater_width * wl / wl_0

    # Effective index at the operation frequency
    neff = neff_0 - (ng_0 - neff_0) * (wl - wl_0) / wl_0

    P = V_dc**2 / R
    deltaphi = P * jnp.pi / P_pi
    phase = 2 * jnp.pi * neff * heater_length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * heater_length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


def mzm_unbalanced(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 1000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of a Mach-Zehnder modulator with EO and TO phase tuning mechanisms.

    Args:
        wl: wavelength in um.
        length_imbalance: length difference between the MZ branches, in um.
        modulation_length: length of the EO modulation section, in um.
        V_pi: voltage dropped on the EO phase modulation section for a pi phase shift (in V).
        P_pi: power dissipated in the TO element for a pi phase shift (in W).
        V_dc: voltage applied to the EO shifter (in V).
        V_ht: voltage applied to the TO shifter (in V).
        kwargs: to_phase_shifter keyword arguments.
    """
    mzm, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler": "mmi",
                "top_shifter": "ps_top",
                "bot_shifter": "ps_bot",
                "dl": "wg_straight",
                "top_tops": "tops",
                "bot_tops": "dummy_tops",
                "splitter": "mmi",
            },
            "connections": {
                "coupler,o2": "top_shifter,o1",
                "coupler,o3": "bot_shifter,o1",
                "bot_shifter,o2": "dl,o1",
                "dl,o2": "bot_tops,o1",
                "top_shifter,o2": "top_tops,o1",
                "splitter,o2": "top_tops,o2",
                "splitter,o3": "bot_tops,o2",
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
            "tops": partial(
                to_phase_shifter,
                wl=wl,
                P_pi=P_pi,
                V_dc=V_ht,
                **kwargs,
            ),
            "dummy_tops": partial(
                to_phase_shifter,
                wl=wl,
                P_pi=P_pi,
                V_dc=0.0,
                **kwargs,
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
    pass
