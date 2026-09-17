import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from _utils.directional_couplers import directional_coupler_base


def build_cband_ltoi300_directional_coupler(
    io_wg_sep: float = 30.6,
    sbend_length: float = 58,
    central_straight_length: float = 38.87 / 2,
    coupl_wg_sep: float = 0.735,
    coup_wg_width: float = 0.7,
    cross_section_io: CrossSectionSpec = "xs_rwg900",
    layer_ridge="LT_RIDGE",
    layer_slab="LT_SLAB",
) -> gf.Component:
    return directional_coupler_base(
        io_wg_sep=io_wg_sep,
        sbend_length=sbend_length,
        central_straight_length=central_straight_length,
        coupl_wg_sep=coupl_wg_sep,
        coup_wg_width=coup_wg_width,
        cross_section_io=cross_section_io,
        layer_ridge=layer_ridge,
        layer_slab=layer_slab,
    )
