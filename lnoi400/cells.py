from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from config import PATH
from lnoi400.tech import LAYER, xs_rwg1000, xs_rwg3000, xs_swg250


################
# MMIs
################


@gf.cell
def mmi1x2_optimized1550(
    width_mmi: float = 6.0,
    length_mmi: float = 26.75,
    width_taper: float = 1.5,
    length_taper: float = 25.0,
    port_ratio: float = 0.55,
    cross_section: CrossSectionSpec = "xs_rwg1000",
    **kwargs,
) -> gf.Component:
    """MMI1x2 with layout optimized for maximum transmission at 1550 nm."""
    
    gap_mmi = port_ratio*width_mmi - width_taper # The port ratio is defined as the ratio between the waveguides separation and the MMI width.

    return gf.components.mmi1x2(
        width_mmi = width_mmi,
        length_mmi = length_mmi,
        gap_mmi = gap_mmi,
        length_taper = length_taper,
        width_taper = width_taper,
        cross_section = cross_section,
        **kwargs,
    )

@gf.cell
def mmi2x2optimized1550(
    width_mmi: float = 5.0,
    length_mmi: float = 76.5,
    width_taper: float = 1.5,
    length_taper: float = 25.0,
    port_ratio: float = 0.7,
    cross_section: CrossSectionSpec = "xs_rwg1000",
    **kwargs,
) -> gf.Component:
    """MMI2x2 with layout optimized for maximum transmission at 1550 nm."""


    gap_mmi = port_ratio*width_mmi - width_taper # The port ratio is defined as the ratio between the waveguides separation and the MMI width.

    return gf.components.mmi2x2(
        width_mmi = width_mmi,
        length_mmi = length_mmi,
        gap_mmi = gap_mmi,
        length_taper = length_taper,
        width_taper = width_taper,
        cross_section = cross_section,
        **kwargs,
    )

if __name__ == "__main__":

    mmi = mmi2x2optimized1550()
    mmi.show()
