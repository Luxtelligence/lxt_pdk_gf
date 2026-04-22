import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from _utils.spline import (
    bend_S_spline_varying_width,
)

#####################
# Directional coupler
#####################


@gf.cell
def directional_coupler_base(
    io_wg_sep: float = 30.6,
    sbend_length: float = 58,
    central_straight_length: float = 16.92,
    coupl_wg_sep: float = 0.8,
    coup_wg_width: float = 0.8,
    cross_section_io: CrossSectionSpec = "xs_rwg1000",
    layer_ridge: str = "LN_RIDGE",
    layer_slab: str = "LN_SLAB",
) -> gf.Component:
    """Returns a directional coupler.

    Args:
        io_wg_sep: Separation of the two straights at the input/output, top-to-top.
        sbend_length: length of the s-bend part.
        central_straight_length: length of the coupling region.
        coupl_wg_sep: Distance between two waveguides in the coupling region (side to side).
        cross_section_io: cross section spec at the i/o (must be defined in tech.py).
        coup_wg_width: waveguide width at the coupling section.
    """

    ridge_layer = gf.get_layer(layer_ridge)
    slab_layer = gf.get_layer(layer_slab)
    s0 = gf.Section(
        width=coup_wg_width,
        offset=0,
        layer=ridge_layer,
        name="RIDGE",
        port_names=("o1", "o2"),
    )
    s1 = gf.Section(width=10.0, offset=0, layer=slab_layer, name="SLAB", simplify=0.03)
    cross_section_coupling = gf.CrossSection(sections=[s0, s1])

    cross_section_io = gf.get_cross_section(cross_section_io)

    s_height = (
        io_wg_sep - coupl_wg_sep - coup_wg_width
    ) / 2  # take into account the width of the waveguide
    size = (sbend_length, s_height)

    # s-bend settings
    settings_s_bend = {
        "size": size,
        "cross_section1": cross_section_coupling,
        "cross_section2": cross_section_io,
        "npoints": 201,
        "layer_ridge": layer_ridge,
        "layer_slab": layer_slab,
    }
    dc = gf.Component()
    # top right branch
    c_tr = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_tr.dmove(
        c_tr.ports["o1"].dcenter,
        (central_straight_length / 2, 0.5 * (coupl_wg_sep + coup_wg_width)),
    )

    # bottom right branch
    c_br = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_br.dmirror_y()
    c_br.dmove(
        c_br.ports["o1"].dcenter,
        (central_straight_length / 2, -0.5 * (coupl_wg_sep + coup_wg_width)),
    )

    # central waveguides
    straight_center_up = dc << gf.components.straight(
        length=central_straight_length, cross_section=cross_section_coupling
    )
    straight_center_up.connect("o2", c_tr.ports["o1"])
    straight_center_down = dc << gf.components.straight(
        length=central_straight_length, cross_section=cross_section_coupling
    )
    straight_center_down.connect("o2", c_br.ports["o1"])

    # top left branch
    c_tl = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_tl.dmirror_x()
    c_tl.dmove(c_tl.ports["o1"].dcenter, straight_center_up.ports["o1"].dcenter)

    # bottom left branch
    c_bl = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_bl.dmirror_x()
    c_bl.dmirror_y()
    c_bl.dmove(c_bl.ports["o1"].dcenter, straight_center_down.ports["o1"].dcenter)

    # Expose the ports
    exposed_ports = [
        ("o1", c_bl.ports["o2"]),
        ("o2", c_tl.ports["o2"]),
        ("o3", c_tr.ports["o2"]),
        ("o4", c_br.ports["o2"]),
    ]

    [dc.add_port(name=name, port=port) for name, port in exposed_ports]
    return dc
