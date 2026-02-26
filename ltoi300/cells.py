import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from _utils.optical_resonators import ring_resonator
from ltoi300._builders.edge_couplers import (
    build_cband_ltoi300_edge_coupler,
    build_oband_ltoi300_edge_coupler,
)
from ltoi300._builders.mmis import (
    build_mmi1x2_cband,
    build_mmi1x2_oband,
    build_mmi2x2_cband,
    build_mmi2x2_oband,
)
from ltoi300._builders.mzms import (
    build_terminated_mzm_1x2mmi_cband,
    build_terminated_mzm_1x2mmi_oband,
    build_terminated_mzm_2x2mmi_cband,
    build_terminated_mzm_2x2mmi_oband,
    build_unterminated_mzm_1x2mmi_cband,
    build_unterminated_mzm_1x2mmi_oband,
    build_unterminated_mzm_2x2mmi_cband,
    build_unterminated_mzm_2x2mmi_oband,
)
from ltoi300._builders.straights import (
    build_straight_rwg700,
    build_straight_rwg900,
    build_straight_rwg2500,
)
from ltoi300.tech import xs_rwg700, xs_rwg900

############################################
############# O-band cells #################
############################################

################
# Straights
################


@gf.cell
def straight_rwg700_oband(length: float = 10.0) -> gf.Component:
    """Returns a standard straight 700 nm-wide single-mode waveguide for O-band propagation.
    Args:
        length: straight length (um).
    """
    return build_straight_rwg700(
        length=length,
    )


@gf.cell
def straight_rwg2500_oband(length: float = 10.0) -> gf.Component:
    """Returns a standard straight 2500 nm-wide multi-mode waveguide for O-band propagation.
    Args:
        length: straight length (um).
    """
    return build_straight_rwg2500(
        length=length,
    )


##########################
# Edge couplers
##########################


@gf.cell
def oband_ltoi300_edge_coupler(
    input_ext: float = 10.0,
    total_taper_length: float = 160.0,
    upper_taper_length: float = 80.0,
) -> gf.Component:
    return build_oband_ltoi300_edge_coupler(
        input_ext=input_ext,
        total_taper_length=total_taper_length,
        upper_taper_length=upper_taper_length,
    )


@gf.cell
def cband_ltoi300_edge_coupler(
    input_ext: float = 10.0,
    total_taper_length: float = 160.0,
    upper_taper_length: float = 80.0,
) -> gf.Component:
    return build_cband_ltoi300_edge_coupler(
        input_ext=input_ext,
        total_taper_length=total_taper_length,
        upper_taper_length=upper_taper_length,
    )


##########################
# 1x2 and 2x2 O-band MMIs
##########################


@gf.cell
def mmi1x2_oband(**kwargs) -> gf.Component:
    r"""Returns a standard 1x2 MMI optimized for low insertion loss on a 300/120 stack in the O-band (at 1310 nm).

    Use with default parameters for standard designs: `mmi = mmi1x2_oband()`. Parameters can be overridden at the user's risk.

    Args:
        width_mmi: Width of the MMI region in micrometers
        length_mmi: Length of the MMI region in micrometers
        width_taper: Width of the taper at the MMI interface in micrometers
        length_taper: Length of the input/output tapers in micrometers
        port_separation: Center-to-center separation between output ports in micrometers
        cross_section: Cross-section specification for the waveguides
        **kwargs: Additional keyword arguments passed to gdsfactory mmi1x2

    Note:
        Default values for all parameters are defined in ltoi300._builders.mmis.build_mmi1x2_oband
    """
    return build_mmi1x2_oband(**kwargs)


@gf.cell
def mmi2x2_oband(**kwargs) -> gf.Component:
    r"""Returns a standard 2x2 MMI optimized for 50% splitting on a 300/120 stack in the O-band (at 1310 nm).

    Use with default parameters for standard designs: `mmi = mmi2x2_oband()`. Parameters can be overridden at the user's risk.

    Args:
        width_mmi: Width of the MMI region in micrometers
        length_mmi: Length of the MMI region in micrometers
        width_taper: Width of the taper at the MMI interface in micrometers
        length_taper: Length of the input/output tapers in micrometers
        port_separation: Center-to-center separation between output ports in micrometers
        cross_section: Cross-section specification for the waveguides
        **kwargs: Additional keyword arguments passed to gdsfactory mmi2x2

    Note:
        Default values for all parameters are defined in ltoi300._builders.mmis.build_mmi2x2_oband
    """
    return build_mmi2x2_oband(**kwargs)


################
# Modulators
################


@gf.cell
def terminated_mzm_1x2mmi_oband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
):
    """Returns a terminated MZM with 1x2 MMI splitter with effective index matching
    for O-band operation."""

    return build_terminated_mzm_1x2mmi_oband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


@gf.cell
def unterminated_mzm_1x2mmi_oband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
):
    """Returns a unterminated MZM with 1x2 MMI splitter with effective index matching
    for O-band operation."""

    return build_unterminated_mzm_1x2mmi_oband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


@gf.cell
def terminated_mzm_2x2mmi_oband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
) -> gf.Component:
    """Returns a terminated MZM with 2x2 MMI splitter with effective index matching
    for O-band operation."""

    return build_terminated_mzm_2x2mmi_oband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


@gf.cell
def unterminated_mzm_2x2mmi_oband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
) -> gf.Component:
    """Returns an unterminated MZM with 2x2 MMI splitter with effective index matching
    for O-band operation."""

    return build_unterminated_mzm_2x2mmi_oband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


@gf.cell
def ring_resonator_oband_single_mode(
    ring_radius: float = 200.0,
    ring_width: float = 0.7,
    gap: float = 1.05,
    bus_length: float | None = None,
) -> gf.Component:
    """Returns a ring resonator with an evanescent coupler for O-band operation.
    The gap is set to 1.05 um to ensure critical coupling.
    Note: the critical coupling condition is loss-specific and
    a different gap may be required for other loss conditions."""
    return ring_resonator(
        gap=gap,
        ring_radius=ring_radius,
        bus_length=bus_length,
        bus_xs=xs_rwg700(),
        ring_xs=xs_rwg700(width=ring_width),
    )


@gf.cell
def ring_resonator_oband_multimode(
    ring_radius: float = 200.0,
    ring_width: float = 1.5,
    gap: float = 0.75,
    bus_length: float | None = None,
) -> gf.Component:
    """Returns a multimode ring resonator with an evanescent coupler for O-band operation.
    The gap is set to 0.75 um to ensure critical coupling.
    Note: the critical coupling condition is loss-specific and
    a different gap may be required for other loss conditions."""
    return ring_resonator(
        gap=gap,
        ring_radius=ring_radius,
        bus_length=bus_length,
        bus_xs=xs_rwg700(),
        ring_xs=xs_rwg700(width=ring_width),
    )


############################################
############# C-band cells #################
############################################


################
# Straights
################


@gf.cell
def straight_rwg900_cband(length: float = 10.0) -> gf.Component:
    """Standard straight single-mode waveguide for C-band propagation."""
    return build_straight_rwg900(
        length=length,
    )


@gf.cell
def straight_arbitrary(
    length: float = 10.0,
    cross_section: CrossSectionSpec = None,
    **kwargs,
) -> gf.Component:
    """An arbitrary R&D straight waveguide with unknown specs.
    Arbitrary cross-section is accepted with the minimal possible waveguide width of 250 nm."""

    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


##########
# Bends
##########


#########################
# 1x2 and 2x2 C-band MMIs
#########################


@gf.cell
def mmi1x2_cband(**kwargs) -> gf.Component:
    r"""Returns a standard 1x2 MMI optimized for low insertion loss on a 300/120 stack in the C-band (at 1550 nm).

    Use with default parameters for standard designs: `mmi = mmi1x2_cband()`. Parameters can be overridden at the user's risk.

    Args:
        width_mmi: Width of the MMI region in micrometers
        length_mmi: Length of the MMI region in micrometers
        width_taper: Width of the taper at the MMI interface in micrometers
        length_taper: Length of the input/output tapers in micrometers
        port_separation: Center-to-center separation between output ports in micrometers
        cross_section: Cross-section specification for the waveguides
        **kwargs: Additional keyword arguments passed to gdsfactory mmi1x2

    Note:
        Default values for all parameters are defined in ltoi300._builders.mmis.build_mmi1x2_cband
    """
    return build_mmi1x2_cband(**kwargs)


@gf.cell
def mmi2x2_cband(**kwargs) -> gf.Component:
    r"""Returns a standard 2x2 MMI optimized for 50% splitting on a 300/120 stack in the C-band (at 1550 nm).

    Use with default parameters for standard designs: `mmi = mmi2x2_cband()`. Parameters can be overridden at the user's risk.

    Args:
        width_mmi: Width of the MMI region in micrometers
        length_mmi: Length of the MMI region in micrometers
        width_taper: Width of the taper at the MMI interface in micrometers
        length_taper: Length of the input/output tapers in micrometers
        port_separation: Center-to-center separation between output ports in micrometers
        cross_section: Cross-section specification for the waveguides
        **kwargs: Additional keyword arguments passed to gdsfactory mmi2x2

    Note:
        Default values for all parameters are defined in ltoi300._builders.mmis.build_mmi2x2_cband
    """
    return build_mmi2x2_cband(**kwargs)


###############
# Modulators
###############


@gf.cell
def terminated_mzm_1x2mmi_cband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
):
    """Returns a terminated MZM with 1x2 MMI splitter with effective index matching
    for C-band operation."""

    return build_terminated_mzm_1x2mmi_cband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


@gf.cell
def unterminated_mzm_1x2mmi_cband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
):
    """Returns a unterminated MZM with 1x2 MMI splitter with effective index matching
    for C-band operation."""

    return build_unterminated_mzm_1x2mmi_cband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


@gf.cell
def terminated_mzm_2x2mmi_cband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
) -> gf.Component:
    """Returns a terminated MZM with 2x2 MMI splitter with effective index matching
    for C-band operation."""

    return build_terminated_mzm_2x2mmi_cband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


@gf.cell
def unterminated_mzm_2x2mmi_cband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    with_heater: bool = True,
) -> gf.Component:
    """Returns an unterminated MZM with 2x2 MMI splitter with effective index matching
    for C-band operation."""

    return build_unterminated_mzm_2x2mmi_cband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
        with_heater=with_heater,
    )


################
# Optical Resonators
################


@gf.cell
def ring_resonator_cband_single_mode(
    ring_radius: float = 200.0,
    ring_width: float = 0.9,
    gap: float = 1.5,
    bus_length: float | None = None,
) -> gf.Component:
    """Returns a single mode ring resonator with an evanescent coupler for C-band operation.
    The gap is set to 1.5 um to ensure critical coupling.
    Note: the critical coupling condition is loss-specific and
    a different gap may be required for other loss conditions."""
    return ring_resonator(
        gap=gap,
        ring_radius=ring_radius,
        bus_length=bus_length,
        bus_xs=xs_rwg900(),
        ring_xs=xs_rwg900(width=ring_width),
    )


@gf.cell
def ring_resonator_cband_multimode(
    ring_radius: float = 200.0,
    ring_width: float = 1.5,
    gap: float = 1.2,
    bus_length: float | None = None,
) -> gf.Component:
    """Returns a multimode ring resonator with an evanescent coupler for C-band operation.
    The gap is set to 1.2 um to ensure critical coupling.
    Note: the critical coupling condition is loss-specific and
    a different gap may be required for other loss conditions."""
    return ring_resonator(
        gap=gap,
        ring_radius=ring_radius,
        bus_length=bus_length,
        bus_xs=xs_rwg900(),
        ring_xs=xs_rwg900(width=ring_width),
    )


if __name__ == "__main__":
    ring_resonator_cband_single_mode().show()
