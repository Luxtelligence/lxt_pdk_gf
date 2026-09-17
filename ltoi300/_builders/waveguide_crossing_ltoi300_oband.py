import gdsfactory as gf
import numpy as np
from gdsfactory.typings import CrossSectionSpec

from ltoi300.tech import xs_rwg700


def taper_cs(
    taper_width=1.42,  # Maximum waveguide width in the crossing center (µm)
):
    """
    Cross-section of the central taper.
    """
    s0 = gf.Section(
        width=taper_width,
        offset=0,
        layer="LT_RIDGE",
        name="RIDGE",
        port_names=("o1", "o2"),
    )
    s1 = gf.Section(width=10.0, offset=0, layer="LT_SLAB", name="SLAB", simplify=0.03)
    return gf.CrossSection(sections=[s0, s1])


@gf.cell
def waveguide_crossing_oband(
    cross_section1: CrossSectionSpec = xs_rwg700,
    taper_length: float = 6.5,  # Length of each taper section (µm)
    taper_width: float = 2.0,  # Maximum waveguide width in the crossing center (µm)
    cross_length: float = 9,  # Length of the central wide segment (µm)
    dx_straight: float = 3,  # straight extention of all ports
    turn45: bool = False,  # Turn by 45 deg if true
    **kwargs,
) -> gf.Component:
    """
    An optical waveguide crossing with tapering in the middle.
    """
    cross_rotated = gf.Component()
    cross = gf.Component()
    cross_br_1 = gf.Component()

    wg_taper = gf.components.taper_cross_section(
        cross_section1=xs_rwg700(),
        cross_section2=xs_rwg700(width=taper_width),
        length=taper_length,
    )
    wg_central_wide_segment = gf.components.straight(
        length=cross_length, cross_section=xs_rwg700(width=taper_width)
    )
    taper_1 = cross_br_1 << wg_taper
    wg_cws = cross_br_1 << wg_central_wide_segment
    taper_2 = cross_br_1 << wg_taper
    wg_cws.connect("o1", taper_1.ports["o2"])
    taper_2.connect("o2", wg_cws.ports["o2"])

    # Add ports before referencing cross_br_1 in the parent component
    cross_br_1.add_port(name="o1", port=taper_1.ports["o1"])
    cross_br_1.add_port(name="o2", port=taper_2.ports["o1"])

    # Add the first branch of the cross
    cross_br_1_ref = cross << cross_br_1
    # Add a rotated reference to cross
    cross_br_2_ref = cross << cross_br_1
    cross_br_2_ref.rotate(90)
    total_length = 2 * taper_length + cross_length

    cross_br_2_ref.dmove(
        (total_length / 2, -total_length / 2)
    )  # Optional shift to avoid overlap
    cross.add_port(name="o1", port=cross_br_1_ref.ports["o1"])
    cross.add_port(name="o2", port=cross_br_2_ref.ports["o2"])
    cross.add_port(name="o3", port=cross_br_1_ref.ports["o2"])
    cross.add_port(name="o4", port=cross_br_2_ref.ports["o1"])
    cross_extended = gf.components.extend_ports(
        cross,
        length=dx_straight,
        cross_section=cross_section1,
    )
    cross_rot_ref = cross_rotated << cross_extended
    if turn45:
        cross_rot_ref.rotate(45)
    cross_rot_ref.move(destination=(0, 0), origin=cross_rot_ref.dcenter)
    cross_rotated.add_ports(cross_rot_ref.ports)
    cross_rotated.flatten()
    return cross_rotated


@gf.cell
def waveguide_crossing_x_routed_oband(
    radius: float = 75.0,  # curvature radius for the Euler curve
    taper_length: float = 6.5,  # Length of each taper section (µm)
    taper_width: float = 2.00,  # Maximum waveguide width in the crossing center (µm)
    cross_length: float = 9,  # Length of the central wide segment (µm)
    dx_straight: float = 3,  # straight extention of all ports
    **kwargs,
) -> gf.Component:
    """
    An optical waveguide crossing rotated by 45 degrees and routed to the waveguides aligned along the y-axis (optical axis).
    """
    euler_bend_struct = gf.Component()
    npoints = int(np.round(200 * radius / 80.0))
    eul_bend = gf.components.bend_euler_all_angle(
        radius=radius,
        angle=45,
        p=1,
        with_arc_floorplan=True,
        npoints=npoints,
        cross_section=xs_rwg700,
        **kwargs,
    )
    eul_ref_sw = euler_bend_struct.create_vinst(eul_bend)
    eul_ref_nw = euler_bend_struct.create_vinst(eul_bend)
    eul_ref_ne = euler_bend_struct.create_vinst(eul_bend)
    eul_ref_se = euler_bend_struct.create_vinst(eul_bend)
    cr = waveguide_crossing_oband(
        taper_length=taper_length,
        taper_width=taper_width,
        cross_length=cross_length,
        dx_straight=dx_straight,
        turn45=True,
    )
    tap_ref = euler_bend_struct.create_vinst(cr)

    eul_ref_sw.connect("o1", tap_ref["o1"])
    eul_ref_nw.connect("o2", tap_ref["o2"])
    eul_ref_ne.connect("o1", tap_ref["o3"])
    eul_ref_se.connect("o2", tap_ref["o4"])
    exposed_ports = [
        ("o1", eul_ref_sw.ports["o2"]),
        ("o2", eul_ref_nw["o1"]),
        ("o3", eul_ref_ne["o2"]),
        ("o4", eul_ref_se["o1"]),
    ]

    [euler_bend_struct.add_port(name=name, port=port) for name, port in exposed_ports]
    return euler_bend_struct


if __name__ == "__main__":
    pass
