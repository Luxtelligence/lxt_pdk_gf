"""Phase modulator builders for ltoi300.

A phase modulator (PM) is the fundamental building block of an MZM.  It consists
of a single optical waveguide routed through the gap of a CPW transmission line.
Both terminated (one bonding-pad + one RF termination) and unterminated (two
bonding-pads) variants are provided, in O-band and C-band flavours.

The CPW and trail parameters are inherited from the MZM builders so that the two
devices have matching electrical characteristics.
"""

from typing import Any

import gdsfactory as gf

from _utils.gsg_rf import cpw_pad, double_layer_termination, trail_cpw
from _utils.gsg_rf import straight_cpw as _straight_cpw
from ltoi300._builders.mzms import (
    DEFAULT_CPW_PAD_PARAMS,
    DEFAULT_CPW_PARAMS_CBAND,
    DEFAULT_CPW_PARAMS_OBAND,
    DEFAULT_TERMINATION_PARAMS,
    DEFAULT_TRAIL_PARAMS_CBAND,
    DEFAULT_TRAIL_PARAMS_OBAND,
    DEFAULT_TRANSITION_M1_M2_PARAMS,
    DEFAULT_TRANSITION_M2_HR_PARAMS,
    _build_m2_bonding_params,
    _merge,
)
from ltoi300.tech import LAYER, xs_rwg700, xs_rwg900, xs_uni_cpw

# ---------------------------------------------------------------------------
# Default optical-waveguide parameters for a *single-arm* phase modulator.
# The routing section uses the standard single-mode width; the modulation
# section is widened to the same 2.5 µm used inside the MZM.
# ---------------------------------------------------------------------------

DEFAULT_PM_OPTICAL_WG_PARAMS: dict[str, Any] = {
    "taper_length": 100.0,
    "modulation_width": 2.5,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_phase_modulator(
    optical_xs: gf.typings.CrossSectionSpec,
    cpw_params: dict[str, Any],
    trail_params: dict[str, Any],
    modulation_length: float,
    cpw_pad_params: dict[str, Any],
    optical_waveguide_params: dict[str, Any],
    m2_bonding_pad_params: dict[str, Any],
    single_side: bool,
) -> gf.Component:
    """Build a phase modulator composed of one waveguide and a CPW line.

    Args:
        optical_xs: Routing cross-section for the waveguide.
        cpw_params: CPW electrical parameters (gap, widths, type).
        trail_params: T-rail geometry parameters.
        modulation_length: Length of the electro-optic modulation section (µm).
        cpw_pad_params: Parameters for the GSG bonding pad.
        optical_waveguide_params: ``taper_length`` and ``modulation_width``.
        m2_bonding_pad_params: Parameters for the M2 bonding vias on the pad.
        single_side: If True, only one pad is placed (terminated variant); if
            False, two pads are placed (unterminated variant).

    Returns:
        A gdsfactory Component with ports:

        * ``o1``, ``o2`` – optical input/output.
        * ``e1`` – RF bonding-pad GSG port (input side).
        * ``e2`` – RF port on the far side (only for unterminated, single_side=False).
    """
    terminal_xs = gf.get_cross_section(optical_xs)
    taper_length = optical_waveguide_params["taper_length"]
    modulation_width = optical_waveguide_params["modulation_width"]

    _cpw_xs = gf.get_cross_section(
        xs_uni_cpw,
        central_conductor_width=cpw_params["rf_central_conductor_width"],
        gap=cpw_params["rf_gap"],
        ground_planes_width=cpw_params["rf_ground_planes_width"],
    )

    optical_waveguide_dict = {
        "terminal_xs": terminal_xs,
        "modulation_xs": gf.get_cross_section(optical_xs, width=modulation_width),
        "taper_length": taper_length,
    }

    # Build the CPW with a single embedded optical waveguide.
    if cpw_params["type"] == "straight":
        cpw = _straight_cpw(
            cpw_xs=_cpw_xs,
            modulation_length=modulation_length,
            optical_waveguides=optical_waveguide_dict,
            single_waveguide=True,
        )
    elif cpw_params["type"] == "trail":
        cpw = trail_cpw(
            cpw_xs=_cpw_xs,
            modulation_length=modulation_length,
            trail_params=trail_params,
            optical_waveguides=optical_waveguide_dict,
            single_waveguide=True,
        )
    else:
        raise ValueError(
            f"Invalid CPW type: {cpw_params['type']}. Valid types are 'straight' and 'trail'."
        )

    # Build the GSG bonding pad (upper waveguide only).
    pad = cpw_pad(
        cpw_xs=_cpw_xs,
        optical_waveguide_xs=terminal_xs,
        pitch=cpw_pad_params["pitch"],
        length_straight=cpw_pad_params["length_straight"],
        length_tapered=cpw_pad_params["length_tapered"],
        ground_pad_width=cpw_pad_params["ground_pad_width"],
        m2_bonding_pads_params=m2_bonding_pad_params,
        single_waveguide=True,
    )

    # Assemble the phase modulator.
    PM = gf.Component()

    cpw_ref = PM << cpw
    pad1_ref = PM << pad
    pad1_ref.connect("e2", cpw_ref.ports["e1"])

    PM.add_port(name="o1", port=pad1_ref.ports["o1"])
    PM.add_port(name="e1", port=pad1_ref.ports["e1"])

    if single_side:
        PM.add_port(name="e2", port=cpw_ref.ports["e2"])
        PM.add_port(name="o2", port=cpw_ref.ports["o2"])
    else:
        pad2_ref = PM << pad
        pad2_ref.dmirror()
        pad2_ref.connect("e2", cpw_ref.ports["e2"])
        PM.add_port(name="e2", port=pad2_ref.ports["e1"])
        PM.add_port(name="o2", port=pad2_ref.ports["o1"])

    return PM


def _add_pm_ports_with_termination(
    c: gf.Component,
    pm_ref: gf.ComponentReference,
    termination_ref: gf.ComponentReference,
    wg_ext_ref: gf.ComponentReference,
) -> None:
    """Expose ports on a terminated PM assembly.

    Explicitly exposes the primary optical and RF ports, hides the internal
    CPW connection point under a leading underscore, and forwards any other
    auxiliary ports (e.g. heater contacts added by pad/via builders).
    """
    c.add_port(name="_term", port=termination_ref.ports["term"])
    c.add_port(name="o1", port=pm_ref.ports["o1"])
    c.add_port(name="o2", port=wg_ext_ref.ports["o2"])
    c.add_port(name="e1", port=pm_ref.ports["e1"])

    # Forward any auxiliary ports not already handled above.
    hidden = {"o1", "o2", "e1", "e2", "_e2", "_term"}
    for port in pm_ref.ports:
        if port.name not in hidden:
            c.add_port(name=port.name, port=port)

    # Keep the internal CPW port accessible for debugging/probing.
    c.add_port(name="_e2", port=pm_ref.ports["e2"])


def _build_unterminated_pm(
    *,
    optical_xs: gf.typings.CrossSectionSpec,
    default_cpw_params: dict[str, Any],
    default_trail_params: dict[str, Any],
    modulation_length: float,
    cpw_params: dict[str, Any] | None,
    trail_params: dict[str, Any] | None,
    cpw_pad_params: dict[str, Any] | None,
    optical_waveguide_params: dict[str, Any] | None,
    m2_bonding_pad_params: dict[str, Any] | None,
    transition_m1_m2_params: dict[str, Any] | None,
) -> gf.Component:
    """Generic unterminated PM builder shared by O- and C-band variants."""
    _transition_m1_m2_params = _merge(
        DEFAULT_TRANSITION_M1_M2_PARAMS, transition_m1_m2_params
    )
    _m2_bonding_params = _build_m2_bonding_params(
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=_transition_m1_m2_params,
    )
    return _build_phase_modulator(
        optical_xs=optical_xs,
        cpw_params=_merge(default_cpw_params, cpw_params),
        trail_params=_merge(default_trail_params, trail_params),
        modulation_length=modulation_length,
        cpw_pad_params=_merge(DEFAULT_CPW_PAD_PARAMS, cpw_pad_params),
        optical_waveguide_params=_merge(
            DEFAULT_PM_OPTICAL_WG_PARAMS, optical_waveguide_params
        ),
        m2_bonding_pad_params=_m2_bonding_params,
        single_side=False,
    )


def _wrap_pm_with_termination(
    *,
    optical_xs: gf.typings.CrossSectionSpec,
    default_cpw_params: dict[str, Any],
    default_trail_params: dict[str, Any],
    modulation_length: float,
    cpw_params: dict[str, Any] | None,
    trail_params: dict[str, Any] | None,
    cpw_pad_params: dict[str, Any] | None,
    optical_waveguide_params: dict[str, Any] | None,
    m2_bonding_pad_params: dict[str, Any] | None,
    transition_m1_m2_params: dict[str, Any] | None,
    transition_m2_hr_params: dict[str, Any] | None,
    termination_params: dict[str, Any] | None,
) -> gf.Component:
    """Generic terminated PM builder shared by O- and C-band variants."""
    _cpw_params = _merge(default_cpw_params, cpw_params)
    _termination_params = _merge(DEFAULT_TERMINATION_PARAMS, termination_params)
    _cpw_pad_params = _merge(DEFAULT_CPW_PAD_PARAMS, cpw_pad_params)
    _transition_m1_m2_params = _merge(
        DEFAULT_TRANSITION_M1_M2_PARAMS, transition_m1_m2_params
    )
    _transition_m2_hr_params = _merge(
        DEFAULT_TRANSITION_M2_HR_PARAMS, transition_m2_hr_params
    )
    # Use the *merged* M1→M2 params so user overrides propagate into the bonding vias.
    _m2_bonding_params = _build_m2_bonding_params(
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=_transition_m1_m2_params,
    )

    cpw_xs = xs_uni_cpw(
        central_conductor_width=_cpw_params["rf_central_conductor_width"],
        gap=_cpw_params["rf_gap"],
        ground_planes_width=_cpw_params["rf_ground_planes_width"],
    )
    termination = double_layer_termination(
        cpw_xs=cpw_xs,
        termination_layer=LAYER.HRL,
        m2_layer=LAYER.M2,
        m2_pad_length=_termination_params["m2_pad_length"],
        termination_params=_termination_params,
        via_m1_m2_params=_transition_m1_m2_params,
        via_m2_hr_params=_transition_m2_hr_params,
    )

    pm_ref_component = _build_phase_modulator(
        optical_xs=optical_xs,
        cpw_params=_cpw_params,
        trail_params=_merge(default_trail_params, trail_params),
        modulation_length=modulation_length,
        cpw_pad_params=_cpw_pad_params,
        optical_waveguide_params=_merge(
            DEFAULT_PM_OPTICAL_WG_PARAMS, optical_waveguide_params
        ),
        m2_bonding_pad_params=_m2_bonding_params,
        single_side=True,  # the far end connects to the RF termination
    )

    c = gf.Component()
    pm_ref = c << pm_ref_component
    termination_ref = c << termination
    termination_ref.connect("e1", pm_ref.ports["e2"])

    # Extend the optical waveguide through the termination block.
    wg_ext = c << gf.components.straight(
        length=termination_ref.dxsize, cross_section=optical_xs
    )
    wg_ext.connect("o1", pm_ref.ports["o2"])

    _add_pm_ports_with_termination(c, pm_ref, termination_ref, wg_ext)
    return c


# ---------------------------------------------------------------------------
# O-band public builders
# ---------------------------------------------------------------------------


def build_unterminated_eo_phase_shifter_oband(
    modulation_length: float = 5000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
) -> gf.Component:
    """Build an unterminated O-band EO phase modulator.

    The device comprises a single optical waveguide running through the gap of a
    CPW transmission line.  Both ends of the CPW are terminated with bonding pads.

    Args:
        modulation_length: Length of the EO modulation section (µm).
        cpw_params: Overrides for the CPW electrical parameters.
        trail_params: Overrides for the T-rail geometry.
        cpw_pad_params: Overrides for the GSG bonding pad.
        optical_waveguide_params: Overrides for ``taper_length`` / ``modulation_width``.
        m2_bonding_pad_params: Overrides for the M2 bonding via parameters.
        transition_m1_m2_params: Overrides for the M1→M2 transition.

    Returns:
        gf.Component with ports ``o1``, ``o2`` (optical) and ``e1``, ``e2`` (RF).
    """
    return _build_unterminated_pm(
        optical_xs=xs_rwg700,
        default_cpw_params=DEFAULT_CPW_PARAMS_OBAND,
        default_trail_params=DEFAULT_TRAIL_PARAMS_OBAND,
        modulation_length=modulation_length,
        cpw_params=cpw_params,
        trail_params=trail_params,
        cpw_pad_params=cpw_pad_params,
        optical_waveguide_params=optical_waveguide_params,
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
    )


def build_terminated_eo_phase_shifter_oband(
    modulation_length: float = 5000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    termination_params: dict[str, Any] | None = None,
) -> gf.Component:
    """Build a terminated O-band EO phase modulator.

    One end of the CPW carries a bonding pad; the other end carries a
    matched RF termination (double-layer resistive wire).

    Args:
        modulation_length: Length of the EO modulation section (µm).
        cpw_params: Overrides for the CPW electrical parameters.
        trail_params: Overrides for the T-rail geometry.
        cpw_pad_params: Overrides for the GSG bonding pad.
        optical_waveguide_params: Overrides for ``taper_length`` / ``modulation_width``.
        m2_bonding_pad_params: Overrides for the M2 bonding via parameters.
        transition_m1_m2_params: Overrides for the M1→M2 transition.
        transition_m2_hr_params: Overrides for the M2→HRL transition.
        termination_params: Overrides for the RF termination wire parameters.

    Returns:
        gf.Component with ports ``o1``, ``o2`` (optical), ``e1`` (RF pad)
        and ``_term`` (termination probe point, hidden underscore prefix).
    """
    return _wrap_pm_with_termination(
        optical_xs=xs_rwg700,
        default_cpw_params=DEFAULT_CPW_PARAMS_OBAND,
        default_trail_params=DEFAULT_TRAIL_PARAMS_OBAND,
        modulation_length=modulation_length,
        cpw_params=cpw_params,
        trail_params=trail_params,
        cpw_pad_params=cpw_pad_params,
        optical_waveguide_params=optical_waveguide_params,
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
        transition_m2_hr_params=transition_m2_hr_params,
        termination_params=termination_params,
    )


# ---------------------------------------------------------------------------
# C-band public builders
# ---------------------------------------------------------------------------


def build_unterminated_eo_phase_shifter_cband(
    modulation_length: float = 5000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
) -> gf.Component:
    """Build an unterminated C-band EO phase modulator.

    The device comprises a single optical waveguide running through the gap of a
    CPW transmission line.  Both ends of the CPW are terminated with bonding pads.

    Args:
        modulation_length: Length of the EO modulation section (µm).
        cpw_params: Overrides for the CPW electrical parameters.
        trail_params: Overrides for the T-rail geometry.
        cpw_pad_params: Overrides for the GSG bonding pad.
        optical_waveguide_params: Overrides for ``taper_length`` / ``modulation_width``.
        m2_bonding_pad_params: Overrides for the M2 bonding via parameters.
        transition_m1_m2_params: Overrides for the M1→M2 transition.

    Returns:
        gf.Component with ports ``o1``, ``o2`` (optical) and ``e1``, ``e2`` (RF).
    """
    return _build_unterminated_pm(
        optical_xs=xs_rwg900,
        default_cpw_params=DEFAULT_CPW_PARAMS_CBAND,
        default_trail_params=DEFAULT_TRAIL_PARAMS_CBAND,
        modulation_length=modulation_length,
        cpw_params=cpw_params,
        trail_params=trail_params,
        cpw_pad_params=cpw_pad_params,
        optical_waveguide_params=optical_waveguide_params,
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
    )


def build_terminated_eo_phase_shifter_cband(
    modulation_length: float = 5000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    termination_params: dict[str, Any] | None = None,
) -> gf.Component:
    """Build a terminated C-band EO phase modulator.

    One end of the CPW carries a bonding pad; the other end carries a
    matched RF termination (double-layer resistive wire).

    Args:
        modulation_length: Length of the EO modulation section (µm).
        cpw_params: Overrides for the CPW electrical parameters.
        trail_params: Overrides for the T-rail geometry.
        cpw_pad_params: Overrides for the GSG bonding pad.
        optical_waveguide_params: Overrides for ``taper_length`` / ``modulation_width``.
        m2_bonding_pad_params: Overrides for the M2 bonding via parameters.
        transition_m1_m2_params: Overrides for the M1→M2 transition.
        transition_m2_hr_params: Overrides for the M2→HRL transition.
        termination_params: Overrides for the RF termination wire parameters.

    Returns:
        gf.Component with ports ``o1``, ``o2`` (optical), ``e1`` (RF pad)
        and ``_term`` (termination probe point, hidden underscore prefix).
    """
    return _wrap_pm_with_termination(
        optical_xs=xs_rwg900,
        default_cpw_params=DEFAULT_CPW_PARAMS_CBAND,
        default_trail_params=DEFAULT_TRAIL_PARAMS_CBAND,
        modulation_length=modulation_length,
        cpw_params=cpw_params,
        trail_params=trail_params,
        cpw_pad_params=cpw_pad_params,
        optical_waveguide_params=optical_waveguide_params,
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
        transition_m2_hr_params=transition_m2_hr_params,
        termination_params=termination_params,
    )


if __name__ == "__main__":
    pm = build_terminated_eo_phase_shifter_oband()
    pm.show()
