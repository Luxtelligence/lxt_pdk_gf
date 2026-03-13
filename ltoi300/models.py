from functools import partial

import jax.numpy as jnp
import numpy as np
import sax
from gplugins.sax.models import phase_shifter as _phase_shifter
from gplugins.sax.models import straight as __straight
from numpy.typing import NDArray

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
            loss_dB_cm=0.5,
            wl0=1.31,  # O-band center wavelength
            neff=1.75,
            ng=2.2,  # TODO: Update with actual ng
        )

    if cross_section == "xs_rwg900":
        # C-band single-mode waveguide (900 nm width)
        return __straight(
            wl=wl,
            length=length,
            loss_dB_cm=0.5,
            wl0=1.55,  # C-band center wavelength
            neff=1.7,
            ng=2.1,
        )

    if cross_section == "xs_rwg2500":
        # O-band multi-mode waveguide (2500 nm width)
        return __straight(
            wl=wl,
            length=length,
            loss_dB_cm=0.2,
            wl0=1.31,  # O-band center wavelength
            neff=1.8,
            ng=2.2,
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

mmi1x2_oband = partial(
    _1in_2out_symmetric_poly_model,
    module=ltoi300,
    data_tag="mmi1x2_oband",
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
    data_tag="mmi2x2_oband",
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

mmi1x2_cband = partial(
    _1in_2out_symmetric_poly_model,
    module=ltoi300,
    data_tag="mmi1x2_cband",
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
    data_tag="mmi2x2_cband",
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


def _eo_phase_shifter(
    wl: Float = 1.55,
    wl_0: float = 1.55,
    length: float = 5000.0,
    neff_0: float = 1.75,
    ng_0: float = 2.1,
    loss: float = 2e-5,
    V_pi: float | None = None,
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
        V_pi: voltage for pi phase shift (V).
        V_dc: static voltage applied (V).
    """
    if V_pi is not None:
        v = V_dc / V_pi
    else:
        raise ValueError("V_pi must be specified.")

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


##########################
# Phase Shifters - O-band
##########################


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

    if np.isnan(V_pi):
        V_pi = 2 * 2.2e4 * wl / length / 1.31

    return _eo_phase_shifter(
        wl=wl,
        wl_0=1.31,
        length=length,
        V_pi=V_pi,
        V_dc=V_dc,
        **kwargs,
    )


##########################
# Phase Shifters - C-band
##########################


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

    if np.isnan(V_pi):
        V_pi = 2 * 2.8e4 * wl / length / 1.55

    return _eo_phase_shifter(
        wl=wl,
        wl_0=1.55,
        length=length,
        V_pi=V_pi,
        V_dc=V_dc,
        **kwargs,
    )


####################
# MZMs - O-band
####################


def unterminated_mzm_1x2mmi_oband(
    wl: Float = 1.31,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO phase modulation mechanism.
    Uses 1x2 MMIs for splitting/combining in O-band.

    Args:
        wl: wavelength in um.
        length_imbalance: length difference between the MZ branches, in um.
        modulation_length: length of the EO modulation section, in um.
        V_pi: voltage dropped on the EO phase modulation section for a pi phase shift (in V).
        V_dc: voltage applied to the EO shifter (in V).
        kwargs: to_phase_shifter keyword arguments.
    """
    mzm, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler": "mmi",
                "top_shifter": "ps_top",
                "bot_shifter": "ps_bot",
                "dl": "wg_straight",
                "splitter": "mmi",
            },
            "connections": {
                "coupler,o2": "top_shifter,o1",
                "coupler,o3": "bot_shifter,o1",
                "bot_shifter,o2": "dl,o1",
                "dl,o2": "splitter,o3",
                "top_shifter,o2": "splitter,o2",
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
        },
        backend="default",
    )
    return mzm()


def unterminated_mzm_2x2mmi_oband(
    wl: Float = 1.31,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO phase modulation mechanism.
    Uses 2x2 MMIs for splitting/combining in O-band.

    Args:
        wl: wavelength in um.
        length_imbalance: length difference between the MZ branches, in um.
        modulation_length: length of the EO modulation section, in um.
        V_pi: voltage dropped on the EO phase modulation section for a pi phase shift (in V).
        V_dc: voltage applied to the EO shifter (in V).
        kwargs: to_phase_shifter keyword arguments.
    """
    mzm, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler": "mmi",
                "top_shifter": "ps_top",
                "bot_shifter": "ps_bot",
                "dl": "wg_straight",
                "splitter": "mmi",
            },
            "connections": {
                "coupler,o3": "top_shifter,o1",
                "coupler,o4": "bot_shifter,o1",
                "bot_shifter,o2": "dl,o1",
                "dl,o2": "splitter,o4",
                "top_shifter,o2": "splitter,o3",
            },
            "ports": {
                "o1": "coupler,o1",
                "o2": "coupler,o2",
                "o3": "splitter,o2",
                "o4": "splitter,o1",
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
        },
        backend="default",
    )
    return mzm()


####################
# MZMs - C-band
####################


def unterminated_mzm_1x2mmi_cband(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO phase modulation mechanism.
    Uses 1x2 MMIs for splitting/combining in C-band.

    Args:
        wl: wavelength in um.
        length_imbalance: length difference between the MZ branches, in um.
        modulation_length: length of the EO modulation section, in um.
        V_pi: voltage dropped on the EO phase modulation section for a pi phase shift (in V).
        V_dc: voltage applied to the EO shifter (in V).
        kwargs: to_phase_shifter keyword arguments.
    """
    mzm, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler": "mmi",
                "top_shifter": "ps_top",
                "bot_shifter": "ps_bot",
                "dl": "wg_straight",
                "splitter": "mmi",
            },
            "connections": {
                "coupler,o2": "top_shifter,o1",
                "coupler,o3": "bot_shifter,o1",
                "bot_shifter,o2": "dl,o1",
                "dl,o2": "splitter,o3",
                "top_shifter,o2": "splitter,o2",
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
        },
        backend="default",
    )
    return mzm()


def unterminated_mzm_2x2mmi_cband(
    wl: Float = 1.55,
    length_imbalance: float = 100.0,
    modulation_length: float = 5000.0,
    V_pi: float = np.nan,
    V_dc: float = 0.0,
    **kwargs,
) -> sax.SDict:
    """Model of an unterminated Mach-Zehnder modulator with EO phase modulation mechanism.
    Uses 2x2 MMIs for splitting/combining in C-band.

    Args:
        wl: wavelength in um.
        length_imbalance: length difference between the MZ branches, in um.
        modulation_length: length of the EO modulation section, in um.
        V_pi: voltage dropped on the EO phase modulation section for a pi phase shift (in V).
        V_dc: voltage applied to the EO shifter (in V).
        kwargs: to_phase_shifter keyword arguments.
    """
    mzm, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler": "mmi",
                "top_shifter": "ps_top",
                "bot_shifter": "ps_bot",
                "dl": "wg_straight",
                "splitter": "mmi",
            },
            "connections": {
                "coupler,o3": "top_shifter,o1",
                "coupler,o4": "bot_shifter,o1",
                "bot_shifter,o2": "dl,o1",
                "dl,o2": "splitter,o4",
                "top_shifter,o2": "splitter,o3",
            },
            "ports": {
                "o1": "coupler,o1",
                "o2": "coupler,o2",
                "o3": "splitter,o2",
                "o4": "splitter,o1",
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
        },
        backend="default",
    )
    return mzm()


########################
# Optical resonators
########################

# TODO: Models for ring resonators are work in progress.


def _test_point_coupler(
    coupling=0.05,
):
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    sdict = sax.reciprocal(
        {
            ("i1", "o1"): tau,
            ("i1", "o2"): 1j * kappa,
            ("i2", "o1"): 1j * kappa,
            ("i2", "o2"): tau,
        }
    )
    return sdict


def _ring_resonator_single_mode_cband(
    wl: Float = 1.55,
    ring_radius: float = 200.0,
    loss_dB_cm: float = 0.5,
    wl0: float = 1.55,
    neff: float = 1.7,
    ng: float = 2.1,
) -> sax.SDict:
    """Model of a single-mode ring resonator in C-band.

    Args:
        wl: wavelength in um.
        radius: radius of the ring in um.
        length: length of the ring in um.
    """

    circumference = 2 * np.pi * ring_radius

    thru_ring, _ = sax.circuit(
        netlist={
            "instances": {
                "pc": "point_coupler",
                "ring_wg": "wg_bent",
            },
            "connections": {"pc,o2": "ring_wg,o1", "ring_wg,o2": "pc,i2"},
            "ports": {
                "o1": "pc,i1",
                "o2": "pc,o1",
            },
        },
        models={
            "point_coupler": partial(
                _test_point_coupler,
            ),
            "wg_bent": partial(
                __straight,
                wl=wl,
                length=circumference,
                loss_dB_cm=loss_dB_cm,
                wl0=wl0,
                neff=neff,
                ng=ng,
            ),
        },
        backend="default",
    )
    return thru_ring()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define a  wavelength grid
    wl_grid = np.linspace(1.5, 1.6, 3000)

    # Test the modulator response
    s_dicts = unterminated_mzm_2x2mmi_oband(wl=wl_grid, length_imbalance=100, V_dc=0.0)

    # Extract the Through port transmission (o1 to o2)
    thru_transmission = s_dicts["o1", "o3"]
    cross_transmission = s_dicts["o1", "o4"]
    intensity_dB = 20 * np.log10(np.abs(thru_transmission) + 1e-12)
    phase_deg = np.degrees(np.unwrap(np.angle(thru_transmission)))
    intensity_dB_cross = 20 * np.log10(np.abs(cross_transmission) + 1e-12)
    phase_deg_cross = np.degrees(np.unwrap(np.angle(cross_transmission)))

    # Plotting
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_top.plot(wl_grid, intensity_dB)
    ax_top.plot(wl_grid, intensity_dB_cross)
    ax_top.set_ylabel("Intensity [dB]")
    ax_top.set_title("Modulator Response")
    ax_top.set_ylim([-30, 0])
    ax_bot.plot(wl_grid, phase_deg)
    ax_bot.plot(wl_grid, phase_deg_cross)
    ax_bot.set_xlabel("Wavelength [um]")
    ax_bot.set_ylabel("Phase [deg]")
    plt.show()

# if False:
#     import matplotlib.pyplot as plt

#     # Define a dense wavelength grid (e.g. 30_000 points in the C-band)
#     wl_grid = np.linspace(1.54, 1.56, 30_000)

#     # Evaluate the ring resonator on the entire grid (SAX evaluates these vectorized)
#     s_dicts = _ring_resonator_single_mode_cband(wl=wl_grid)

#     # Extract the Through port transmission (o1 to o2)
#     thru_transmission = s_dicts["o1", "o2"]
#     intensity_dB = 20 * np.log10(np.abs(thru_transmission) + 1e-12)
#     phase_deg = np.degrees(np.unwrap(np.angle(thru_transmission)))

#     # Plotting
#     fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
#     fig.suptitle(
#         "Ring Resonator Transmission (Single Mode, C-Band)",
#         fontsize=13,
#         fontweight="bold",
#     )

#     ax_top.plot(wl_grid * 1e3, intensity_dB, color="blue", linewidth=1.5)
#     ax_top.set_ylabel("Transmission (dB)")
#     ax_top.grid(True, linestyle="--", alpha=0.5)

#     ax_bot.plot(wl_grid * 1e3, phase_deg, color="orange", linewidth=1.5)
#     ax_bot.set_xlabel("Wavelength (nm)")
#     ax_bot.set_ylabel("Phase (°)")
#     ax_bot.grid(True, linestyle="--", alpha=0.5)

#     fig.tight_layout()
#     plt.show()
