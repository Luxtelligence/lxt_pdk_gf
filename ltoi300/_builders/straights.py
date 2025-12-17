import gdsfactory as gf

from ltoi300._impl.straight import _straight

################
# Straights
################


@gf.cell
def build_straight_rwg900(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight single-mode waveguide for C-band propagation."""
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rwg900"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def build_straight_rwg700(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight single-mode waveguide for O-band propagation."""
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rwg700"
    return _straight(
        length=length,
        **kwargs,
    )
