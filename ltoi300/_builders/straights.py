import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

################
# Straights
################


@gf.cell
def build_straight_rwg900(
    length: float = 10.0, cross_section: CrossSectionSpec = "xs_rwg900"
) -> gf.Component:
    """Straight single-mode waveguide for C-band propagation."""
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
    )


@gf.cell
def build_straight_rwg700(
    length: float = 10.0, cross_section: CrossSectionSpec = "xs_rwg700"
) -> gf.Component:
    """Straight single-mode waveguide for O-band propagation."""
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
    )


@gf.cell
def build_straight_rwg2500(
    length: float = 10.0, cross_section: CrossSectionSpec = "xs_rwg2500"
) -> gf.Component:
    """Straight single-mode waveguide for O-band propagation."""
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
    )
