import gdsfactory as gf
from gdsfactory.components import mmi1x2, mmi2x2
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

#####################
# 1x2 and 2x2 O-band MMIs
#####################


@gf.cell
def mmi1x2_oband(
    width_mmi: float = 4.5,
    length_mmi: float = 15.8,
    width_taper: float = 1.7,
    length_taper: float = 25.0,
    port_separation: float = 2.45,
    cross_section: CrossSectionSpec = "xs_rwg700",
    **kwargs,
) -> gf.Component:
    """1x2 MMI optimized for low insertion loss on a 300/120 stack in the O-band (at 1310 nm)."""

    gap_mmi = port_separation - width_taper

    return mmi1x2(
        width_mmi=width_mmi,
        length_mmi=length_mmi,
        gap_mmi=gap_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def mmi2x2_oband(
    width_mmi: float = 5.65,
    length_mmi: float = 97.5,
    width_taper: float = 1.75,
    length_taper: float = 25.0,
    port_separation: float = 3.9,
    cross_section: CrossSectionSpec = "xs_rwg700",
    **kwargs,
) -> gf.Component:
    """1x2 MMI optimized for 50% splitting on a 300/120 stack in the O-band (at 1310 nm)."""

    gap_mmi = port_separation - width_taper

    return mmi2x2(
        width_mmi=width_mmi,
        length_mmi=length_mmi,
        gap_mmi=gap_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


#########################
# 1x2 and 2x2 C-band MMIs
#########################


@gf.cell
def mmi1x2_cband(
    width_mmi: float = 4.5,
    length_mmi: float = 13.5,
    width_taper: float = 1.95,
    length_taper: float = 25.0,
    port_separation: float = 2.55,
    cross_section: CrossSectionSpec = "xs_rwg900",
    **kwargs,
) -> gf.Component:
    """1x2 MMI optimized for low insertion loss on a 300/120 stack in the C-band (at 1550 nm)."""

    gap_mmi = port_separation - width_taper

    return mmi1x2(
        width_mmi=width_mmi,
        length_mmi=length_mmi,
        gap_mmi=gap_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def mmi2x2_cband(
    width_mmi: float = 5.15,
    length_mmi: float = 67.5,
    width_taper: float = 1.5,
    length_taper: float = 25.0,
    port_separation: float = 3.65,
    cross_section: CrossSectionSpec = "xs_rwg900",
    **kwargs,
) -> gf.Component:
    """1x2 MMI optimized for 50% splitting on a 300/120 stack in the C-band (at 1550 nm)."""

    gap_mmi = port_separation - width_taper

    return mmi2x2(
        width_mmi=width_mmi,
        length_mmi=length_mmi,
        gap_mmi=gap_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


if __name__ == "__main__":
    straight_rwg900().show()
