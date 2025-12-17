import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from ltoi300._builders.mmis import (
    build_mmi1x2_cband,
    build_mmi1x2_oband,
    build_mmi2x2_cband,
    build_mmi2x2_oband,
)
from ltoi300._builders.mzms import build_terminated_mzm_cband
from ltoi300._builders.straights import build_straight_rwg700, build_straight_rwg900

################
# Straights
################


@gf.cell
def straight_rwg900(length: float = 10.0) -> gf.Component:
    """Straight single-mode waveguide for C-band propagation."""
    return build_straight_rwg900(
        length=length,
    )


@gf.cell
def straight_rwg700(length: float = 10.0) -> gf.Component:
    """Straight single-mode waveguide for O-band propagation."""
    return build_straight_rwg700(
        length=length,
    )


@gf.cell
def straight_arbitrary(
    length: float = 10.0,
    cross_section: CrossSectionSpec = None,
    **kwargs,
) -> gf.Component:
    """An arbitrary R&D straight waveguide with unknown specs.
    Only the minimal width of 250 nm is accepted."""

    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


##########
# Bends
##########

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
def terminated_mzm_cband():
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""

    return build_terminated_mzm_cband()


if __name__ == "__main__":
    terminated_mzm_cband().show()
