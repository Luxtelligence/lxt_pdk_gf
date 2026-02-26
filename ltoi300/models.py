from functools import partial

import jax.numpy as jnp
import numpy as np
import sax
from gplugins.sax.models import phase_shifter as _phase_shifter
from gplugins.sax.models import straight as __straight
from numpy.typing import NDArray
from sax.utils import reciprocal

import ltoi300
from _utils.models import (
    _1in_2out_symmetric_poly_model,
    _2in_2out_symmetric_poly_model,
)

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
    cross_section: str = "xs_rwg900",
) -> sax.SDict:
    if not isinstance(cross_section, str):
        raise TypeError(
            f"""The cross_section parameter should be a string,
                        received {type(cross_section)} instead."""
        )

    if cross_section == "xs_rwg700":
        # O-band single-mode waveguide (700 nm width)
        return __straight(
            wl=wl,
            length=length,
            loss_dB_cm=0.2,  # TODO: Update with actual measured/simulated loss
            wl0=1.31,  # O-band center wavelength
            neff=1.8,  # TODO: Update with actual neff
            ng=2.22,  # TODO: Update with actual ng
        )

    if cross_section == "xs_rwg900":
        # C-band single-mode waveguide (900 nm width)
        return __straight(
            wl=wl,
            length=length,
            loss_dB_cm=0.2,  # TODO: Update with actual measured/simulated loss
            wl0=1.55,  # C-band center wavelength
            neff=1.8,  # TODO: Update with actual neff
            ng=2.22,  # TODO: Update with actual ng
        )

    if cross_section == "xs_rwg2500":
        # O-band multi-mode waveguide (2500 nm width)
        return __straight(
            wl=wl,
            length=length,
            loss_dB_cm=0.2,  # TODO: Update with actual measured/simulated loss
            wl0=1.31,  # O-band center wavelength
            neff=1.8,  # TODO: Update with actual neff
            ng=2.22,  # TODO: Update with actual ng
        )

    else:
        raise ValueError(
            f"""A model for the specified waveguide
                         cross section {cross_section} is not defined."""
        )


straight_rwg700_oband = partial(_straight, cross_section="xs_rwg700")
straight_rwg900_cband = partial(_straight, cross_section="xs_rwg900")
straight_rwg2500_oband = partial(_straight, cross_section="xs_rwg2500")


################
# MMIs - O-band
################

# TODO: Add JSON data files to ltoi300/data/ directory
mmi1x2_oband = partial(
    _1in_2out_symmetric_poly_model,
    module=ltoi300,
    data_tag="mmi_1x2_oband",
    trans_abs_key="pol_trans_abs",
    trans_phase_key="pol_trans_phase",
    rin_abs_key="pol_refl_in_abs",
    rin_phase_key="pol_refl_in_phase",
    rout_abs_key="pol_refl_out_abs",
    rout_phase_key="pol_refl_out_phase",
    rcross_abs_key="pol_refl_cross_abs",
    rcross_phase_key="pol_refl_cross_phase",
)

mmi2x2_oband = partial(
    _2in_2out_symmetric_poly_model,
    module=ltoi300,
    data_tag="mmi_2x2_oband",
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
# MMIs - C-band
################

# TODO: Add JSON data files to ltoi300/data/ directory
mmi1x2_cband = partial(
    _1in_2out_symmetric_poly_model,
    module=ltoi300,
    data_tag="mmi_1x2_cband",
    trans_abs_key="pol_trans_abs",
    trans_phase_key="pol_trans_phase",
    rin_abs_key="pol_refl_in_abs",
    rin_phase_key="pol_refl_in_phase",
    rout_abs_key="pol_refl_out_abs",
    rout_phase_key="pol_refl_out_phase",
    rcross_abs_key="pol_refl_cross_abs",
    rcross_phase_key="pol_refl_cross_phase",
)

mmi2x2_cband = partial(
    _2in_2out_symmetric_poly_model,
    module=ltoi300,
    data_tag="mmi_2x2_cband",
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


################
# Phase Shifters - General Models
################


def _eo_phase_shifter(
    wl: Float = 1.55,
    wl_0: float = 1.55,
    length: float = 5000.0,
    neff_0: float = 1.85,  # TODO: Update with actual neff
    ng_0: float = 2.21,  # TODO: Update with actual ng
    loss: float = 2e-5,  # TODO: Update with actual loss
    V_pi: float = np.nan,
    V_dc: float = 0.0,
) -> sax.SDict:
    """General electro-optic phase shifter model.

    Args:
        wl: wavelength in um.
        wl_0: center wavelength in um.
        length: phase shifter length in um.
        neff_0: effective index at center wavelength.
        ng_0: group index at center wavelength.
        loss: propagation loss (dB/um).
        V_pi: voltage for pi phase shift (V). If NaN, calculated from default.
        V_dc: static voltage applied (V).
    """
    # Default V_pi
    if np.isnan(V_pi):
        V_pi = 2 * 3.3e4 * wl / length / wl_0  # TODO: Update with actual V_pi
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


def _to_phase_shifter(
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
) -> sax.SDict:
    """General model for a thermal phase shifter.

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


################
# Phase Shifters - O-band
################


def eo_phase_shifter_oband(
    wl: Float = 1.31,
    length: float = 5000.0,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Electro-optic phase shifter for O-band operation.

    Args:
        wl: wavelength in um.
        length: phase shifter length in um.
        V_pi: voltage for pi phase shift (V). If NaN, calculated from default.
        V_dc: static voltage applied (V).
        kwargs: Additional arguments passed to _eo_phase_shifter.
    """
    # TODO: Update with actual O-band V_pi default value
    if np.isnan(V_pi):
        V_pi = 2 * 3.3e4 * wl / length / 1.31  # Placeholder - update with actual value

    return _eo_phase_shifter(
        wl=wl,
        wl_0=1.31,
        length=length,
        V_pi=V_pi,
        V_dc=V_dc,
        **kwargs,
    )


def to_phase_shifter_oband(
    wl: Float = 1.31,
    heater_length: float = 700.0,
    heater_width: float = 1.0,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Thermal phase shifter for O-band operation.

    Args:
        wl: wavelength in um.
        heater_length: in um.
        heater_width: in um.
        P_pi: dissipated power for a pi phase shift (W). If NaN, calculated from default.
        V_dc: static voltage applied to the resistor (V).
        kwargs: Additional arguments passed to _to_phase_shifter.
    """
    # TODO: Update with actual O-band P_pi default value
    if np.isnan(P_pi):
        P_pi = (
            0.075 * heater_width * wl / 1.31
        )  # Placeholder - update with actual value

    return _to_phase_shifter(
        wl=wl,
        wl_0=1.31,
        heater_length=heater_length,
        heater_width=heater_width,
        P_pi=P_pi,
        V_dc=V_dc,
        **kwargs,
    )


################
# Phase Shifters - C-band
################


def eo_phase_shifter_cband(
    wl: Float = 1.55,
    length: float = 5000.0,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Electro-optic phase shifter for C-band operation.

    Args:
        wl: wavelength in um.
        length: phase shifter length in um.
        V_pi: voltage for pi phase shift (V). If NaN, calculated from default.
        V_dc: static voltage applied (V).
        kwargs: Additional arguments passed to _eo_phase_shifter.
    """
    # TODO: Update with actual C-band V_pi default value
    if np.isnan(V_pi):
        V_pi = 2 * 3.3e4 * wl / length / 1.55  # Placeholder - update with actual value

    return _eo_phase_shifter(
        wl=wl,
        wl_0=1.55,
        length=length,
        V_pi=V_pi,
        V_dc=V_dc,
        **kwargs,
    )


def to_phase_shifter_cband(
    wl: Float = 1.55,
    heater_length: float = 700.0,
    heater_width: float = 1.0,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Thermal phase shifter for C-band operation.

    Args:
        wl: wavelength in um.
        heater_length: in um.
        heater_width: in um.
        P_pi: dissipated power for a pi phase shift (W). If NaN, calculated from default.
        V_dc: static voltage applied to the resistor (V).
        kwargs: Additional arguments passed to _to_phase_shifter.
    """
    # TODO: Update with actual C-band P_pi default value
    if np.isnan(P_pi):
        P_pi = (
            0.075 * heater_width * wl / 1.55
        )  # Placeholder - update with actual value

    return _to_phase_shifter(
        wl=wl,
        wl_0=1.55,
        heater_length=heater_length,
        heater_width=heater_width,
        P_pi=P_pi,
        V_dc=V_dc,
        **kwargs,
    )


################
# MZMs - O-band with 1x2 MMI
################


def terminated_mzm_1x2mmi_oband(
    wl: Float = 1.31,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of a terminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 1x2 MMIs for splitting/combining in O-band.

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
                mmi1x2_oband,
                wl=wl,
            ),
            "wg_straight": partial(
                _straight,
                wl=wl,
                length=length_imbalance,
                cross_section="xs_rwg700",
            ),
            "ps_top": partial(
                eo_phase_shifter_oband,
                wl=wl,
                length=modulation_length,
                V_dc=V_dc,
                V_pi=V_pi,
            ),
            "ps_bot": partial(
                eo_phase_shifter_oband,
                wl=wl,
                length=modulation_length,
                V_dc=-V_dc,
                V_pi=V_pi,
            ),
            "tops": partial(
                to_phase_shifter_oband,
                wl=wl,
                P_pi=P_pi,
                V_dc=V_ht,
                **kwargs,
            ),
            "dummy_tops": partial(
                to_phase_shifter_oband,
                wl=wl,
                P_pi=P_pi,
                V_dc=0.0,
                **kwargs,
            ),
        },
        backend="default",
    )
    return mzm()


def unterminated_mzm_1x2mmi_oband(
    wl: Float = 1.31,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 1x2 MMIs for splitting/combining in O-band.

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
    # Same structure as terminated, but may have different RF termination behavior
    # For now, using the same model structure
    return terminated_mzm_1x2mmi_oband(
        wl=wl,
        length_imbalance=length_imbalance,
        modulation_length=modulation_length,
        V_pi=V_pi,
        P_pi=P_pi,
        V_dc=V_dc,
        V_ht=V_ht,
        **kwargs,
    )


################
# MZMs - O-band with 2x2 MMI
################


def terminated_mzm_2x2mmi_oband(
    wl: Float = 1.31,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of a terminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 2x2 MMIs for splitting/combining in O-band.

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
                "coupler,o3": "top_shifter,o1",
                "coupler,o4": "bot_shifter,o1",
                "bot_shifter,o2": "dl,o1",
                "dl,o2": "bot_tops,o1",
                "top_shifter,o2": "top_tops,o1",
                "splitter,o3": "top_tops,o2",
                "splitter,o4": "bot_tops,o2",
            },
            "ports": {
                "o1": "coupler,o1",
                "o2": "coupler,o2",
                "o3": "splitter,o1",
                "o4": "splitter,o2",
            },
        },
        models={
            "mmi": partial(
                mmi2x2_oband,
                wl=wl,
            ),
            "wg_straight": partial(
                _straight,
                wl=wl,
                length=length_imbalance,
                cross_section="xs_rwg700",
            ),
            "ps_top": partial(
                eo_phase_shifter_oband,
                wl=wl,
                length=modulation_length,
                V_dc=V_dc,
                V_pi=V_pi,
            ),
            "ps_bot": partial(
                eo_phase_shifter_oband,
                wl=wl,
                length=modulation_length,
                V_dc=-V_dc,
                V_pi=V_pi,
            ),
            "tops": partial(
                to_phase_shifter_oband,
                wl=wl,
                P_pi=P_pi,
                V_dc=V_ht,
                **kwargs,
            ),
            "dummy_tops": partial(
                to_phase_shifter_oband,
                wl=wl,
                P_pi=P_pi,
                V_dc=0.0,
                **kwargs,
            ),
        },
        backend="default",
    )
    return mzm()


def unterminated_mzm_2x2mmi_oband(
    wl: Float = 1.31,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 2x2 MMIs for splitting/combining in O-band.

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
    # Same structure as terminated, but may have different RF termination behavior
    # For now, using the same model structure
    return terminated_mzm_2x2mmi_oband(
        wl=wl,
        length_imbalance=length_imbalance,
        modulation_length=modulation_length,
        V_pi=V_pi,
        P_pi=P_pi,
        V_dc=V_dc,
        V_ht=V_ht,
        **kwargs,
    )


################
# MZMs - C-band with 1x2 MMI
################


def terminated_mzm_1x2mmi_cband(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of a terminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 1x2 MMIs for splitting/combining in C-band.

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
                mmi1x2_cband,
                wl=wl,
            ),
            "wg_straight": partial(
                _straight,
                wl=wl,
                length=length_imbalance,
                cross_section="xs_rwg900",
            ),
            "ps_top": partial(
                eo_phase_shifter_cband,
                wl=wl,
                length=modulation_length,
                V_dc=V_dc,
                V_pi=V_pi,
            ),
            "ps_bot": partial(
                eo_phase_shifter_cband,
                wl=wl,
                length=modulation_length,
                V_dc=-V_dc,
                V_pi=V_pi,
            ),
            "tops": partial(
                to_phase_shifter_cband,
                wl=wl,
                P_pi=P_pi,
                V_dc=V_ht,
                **kwargs,
            ),
            "dummy_tops": partial(
                to_phase_shifter_cband,
                wl=wl,
                P_pi=P_pi,
                V_dc=0.0,
                **kwargs,
            ),
        },
        backend="default",
    )
    return mzm()


def unterminated_mzm_1x2mmi_cband(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 1x2 MMIs for splitting/combining in C-band.

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
    # Same structure as terminated, but may have different RF termination behavior
    # For now, using the same model structure
    return terminated_mzm_1x2mmi_cband(
        wl=wl,
        length_imbalance=length_imbalance,
        modulation_length=modulation_length,
        V_pi=V_pi,
        P_pi=P_pi,
        V_dc=V_dc,
        V_ht=V_ht,
        **kwargs,
    )


################
# MZMs - C-band with 2x2 MMI
################


def terminated_mzm_2x2mmi_cband(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of a terminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 2x2 MMIs for splitting/combining in C-band.

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
                "coupler,o3": "top_shifter,o1",
                "coupler,o4": "bot_shifter,o1",
                "bot_shifter,o2": "dl,o1",
                "dl,o2": "bot_tops,o1",
                "top_shifter,o2": "top_tops,o1",
                "splitter,o3": "top_tops,o2",
                "splitter,o4": "bot_tops,o2",
            },
            "ports": {
                "o1": "coupler,o1",
                "o2": "coupler,o2",
                "o3": "splitter,o1",
                "o4": "splitter,o2",
            },
        },
        models={
            "mmi": partial(
                mmi2x2_cband,
                wl=wl,
            ),
            "wg_straight": partial(
                _straight,
                wl=wl,
                length=length_imbalance,
                cross_section="xs_rwg900",
            ),
            "ps_top": partial(
                eo_phase_shifter_cband,
                wl=wl,
                length=modulation_length,
                V_dc=V_dc,
                V_pi=V_pi,
            ),
            "ps_bot": partial(
                eo_phase_shifter_cband,
                wl=wl,
                length=modulation_length,
                V_dc=-V_dc,
                V_pi=V_pi,
            ),
            "tops": partial(
                to_phase_shifter_cband,
                wl=wl,
                P_pi=P_pi,
                V_dc=V_ht,
                **kwargs,
            ),
            "dummy_tops": partial(
                to_phase_shifter_cband,
                wl=wl,
                P_pi=P_pi,
                V_dc=0.0,
                **kwargs,
            ),
        },
        backend="default",
    )
    return mzm()


def unterminated_mzm_2x2mmi_cband(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    P_pi: float = np.nan,
    V_dc: float = 0.0,
    V_ht: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO and TO phase tuning mechanisms.
    Uses 2x2 MMIs for splitting/combining in C-band.

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
    # Same structure as terminated, but may have different RF termination behavior
    # For now, using the same model structure
    return terminated_mzm_2x2mmi_cband(
        wl=wl,
        length_imbalance=length_imbalance,
        modulation_length=modulation_length,
        V_pi=V_pi,
        P_pi=P_pi,
        V_dc=V_dc,
        V_ht=V_ht,
        **kwargs,
    )


if __name__ == "__main__":
    pass
