import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

################
# Straights
################


@gf.cell(tags=["straights"])
def build_straight_rwg900(
    length: float = 10.0, cross_section: CrossSectionSpec = "xs_rwg900", **kwargs
) -> gf.Component:
    """Straight single-mode waveguide for C-band propagation."""
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell(tags=["straights"])
def build_straight_rwg700(
    length: float = 10.0, cross_section: CrossSectionSpec = "xs_rwg700", **kwargs
) -> gf.Component:
    """Straight single-mode waveguide for O-band propagation."""
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell(tags=["straights"])
def build_straight_rwg2500(
    length: float = 10.0, cross_section: CrossSectionSpec = "xs_rwg2500", **kwargs
) -> gf.Component:
    """Straight single-mode waveguide for O-band propagation."""
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )
