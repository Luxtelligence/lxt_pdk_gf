import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

################
# Straights
################


@gf.cell
def _straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_rwg900",
    **kwargs,
) -> gf.Component:
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def straight_rwg900(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight single-mode waveguide for C-band propagation."""
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rwg900"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_rwg700(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight single-mode waveguide for O-band propagation."""
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rwg700"
    return _straight(
        length=length,
        **kwargs,
    )


##########
# Bends
##########


###############
# Modulators
###############


if __name__ == "__main__":
    straight_rwg900().show()
