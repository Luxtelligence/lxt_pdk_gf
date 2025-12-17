import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec


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
