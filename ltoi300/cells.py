import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from ltoi300._builders.mmis import (
    build_mmi1x2_cband,
    build_mmi1x2_oband,
    build_mmi2x2_cband,
    build_mmi2x2_oband,
)
from ltoi300._builders.mzms import build_terminated_mzm_cband
from ltoi300._builders.straights import (
    build_straight_rwg700,
    build_straight_rwg900,
    build_straight_rwg2500,
)

############################################
############# O-band cells #################
############################################

################
# Straights
################


@gf.cell
def straight_rwg700_oband(length: float = 10.0) -> gf.Component:
    """Standard straight single-mode waveguide for O-band propagation."""
    return build_straight_rwg700(
        length=length,
    )


@gf.cell
def straight_rwg2500_oband(length: float = 10.0) -> gf.Component:
    """Standard straight multi-mode waveguide for O-band propagation."""
    return build_straight_rwg2500(
        length=length,
    )


################
# Modulators
################


# @gf.cell
# def terminated_mzm_cband(
#     modulation_length: float = 5000.0,
#     length_imbalance: float = 100.0,
#     rf_pad_start_width: float = 80.0,
#     rf_ground_planes_width: float = 180.0,
#     rf_pad_length_straight: float = 10.0,
#     rf_pad_length_tapered: float = 300.0,
#     bias_tuning_section_length: float = 700.0,
# ):
#     """Create a routed terminated MZM for wafer-scale testing with edge couplers."""

#     return build_terminated_mzm_cband(
#         modulation_length=modulation_length,
#         length_imbalance=length_imbalance,
#         rf_pad_start_width=rf_pad_start_width,
#         rf_ground_planes_width=rf_ground_planes_width,
#         rf_pad_length_straight=rf_pad_length_straight,
#         rf_pad_length_tapered=rf_pad_length_tapered,
#         bias_tuning_section_length=bias_tuning_section_length,
#     )


#####################
# 1x2 and 2x2 O-band MMIs
#####################


@gf.cell
def mmi1x2_oband() -> gf.Component:
    """1x2 MMI optimized for low insertion loss on a 300/120 stack in the O-band (at 1310 nm)."""

    return build_mmi1x2_oband()


@gf.cell
def mmi2x2_oband() -> gf.Component:
    """1x2 MMI optimized for 50% splitting on a 300/120 stack in the O-band (at 1310 nm)."""

    return build_mmi2x2_oband()


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
def mmi1x2_cband() -> gf.Component:
    """1x2 MMI optimized for low insertion loss on a 300/120 stack in the C-band (at 1550 nm)."""

    return build_mmi1x2_cband()


@gf.cell
def mmi2x2_cband() -> gf.Component:
    """1x2 MMI optimized for 50% splitting on a 300/120 stack in the C-band (at 1550 nm)."""

    return build_mmi2x2_cband()


###############
# Modulators
###############


@gf.cell
def terminated_mzm_cband(
    modulation_length: float = 5000.0,
    length_imbalance: float = 100.0,
    rf_pad_start_width: float = 80.0,
    rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""

    return build_terminated_mzm_cband(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        rf_pad_start_width=rf_pad_start_width,
        rf_ground_planes_width=rf_ground_planes_width,
        rf_pad_length_straight=rf_pad_length_straight,
        rf_pad_length_tapered=rf_pad_length_tapered,
        bias_tuning_section_length=bias_tuning_section_length,
    )


if __name__ == "__main__":
    terminated_mzm_cband().show()
