import gdsfactory as gf
import numpy as np
from gdsfactory.routing import route_quad
from gdsfactory.typings import Layer, CrossSectionSpec
from gdsfactory.cross_section import CrossSection


@gf.cell
def heater_pads_assymm(
    heater_xs: CrossSection,
    routing_xs: CrossSection,
    pad_size: tuple[float, float] = (150.0, 150.0),
    heater_length: float = 700.0,
    pad_pitch: float | None = None,
) -> gf.Component:
    padwidth, _ = pad_size
    if pad_pitch is None:
        pad_pitch = padwidth + 20.0

    c = gf.Component()
    bondpads = gf.components.pad_array(
        pad=gf.components.pad,
        size=pad_size,
        column_pitch=pad_pitch,
        row_pitch=pad_pitch,
        columns=2,
        port_orientation=-90.0,
        layer=routing_xs.layer,
    )
    length = heater_length - padwidth - pad_pitch
    bps = c << bondpads

    def routing():
        c = gf.Component()
        path = gf.path.straight(length=length)
        c = path.extrude(routing_xs)
        return c

    try:
        route = routing()
    except Exception as e:
        print(f"Error routing thermal phase shifter: {e}")
        return None
    route.ports["e2"].dcenter = (
        route.ports["e2"].dcenter[0] - route.ports["e2"].dwidth / 2,
        route.ports["e2"].dcenter[1],
    )
    route.ports["e2"].dangle = -90.0
    rt = c << route
    rt.dxmin = bps.dxmax
    rt.dymin = bps.dymin
    c.add_port(name="e11", port=bps.ports["e11"])
    c.add_port(name="e12", port=rt.ports["e2"])
    c.add_port(name="e2", port=bps.ports["e12"])
    c.flatten()
    return c


@gf.cell
def heater_straight_compact(
    heater_xs: CrossSection,
    routing_xs: CrossSection,
    length: float = 700.0,
    port_contact_width_ratio: float = 3.0,
    pad_size: tuple[float, float] = (150.0, 150.0),
    pad_pitch: float | None = None,
    pad_vert_offset: float = 10.0,
) -> gf.Component:
    """A straight resistive wire used as a low-frequency phase shifter,
    exploiting the thermo-optical effect. The heater is terminated by wide pads
    for probing or bonding."""

    if pad_vert_offset <= 0:
        raise ValueError(
            f"pad_vert_offset must be a positive number, received {pad_vert_offset}."
        )

    if port_contact_width_ratio <= 0:
        raise ValueError(
            f"port_contact_width_ratio must be a positive number, received {port_contact_width_ratio}."
        )

    c = gf.Component()
    bondpads = heater_pads_assymm(
        pad_size=pad_size,
        heater_length=length,
        pad_pitch=pad_pitch,
        heater_xs=heater_xs,
        routing_xs=routing_xs,
    )
    bps = c << bondpads
    bondpads.ports["e12"].orientation = 270.0

    path = gf.path.straight(length=length)
    ht = path.extrude(heater_xs)

    width = heater_xs.width

    # Place the ports along the edge of the wire
    for p in ht.ports:
        if p.orientation == 0.0:
            p.dcenter = (p.dcenter[0] - 0.5 * p.dwidth, p.dcenter[1])
        if p.orientation == 180.0:
            p.dcenter = (p.dcenter[0] + 0.5 * p.dwidth, p.dcenter[1])
        p.orientation = 90.0

    ht_ref = c << ht

    bps.dxmin = ht_ref.dxmin
    bps.dymin = ht_ref.dymax + pad_vert_offset

    port_contact_width = port_contact_width_ratio * width
    ht.ports["e1"].dx += 0.5 * (port_contact_width - width)
    ht.ports["e2"].dx -= 0.5 * (port_contact_width - width)
    routing_params = {
        "width2": port_contact_width,
        "layer": routing_xs.layer,
    }

    # Connect pads and heater wire

    _ = route_quad(
        c,
        port1=bps.ports["e11"],
        port2=ht.ports["e1"],
        **routing_params,
    )

    _ = route_quad(
        c,
        port1=bps.ports["e12"],
        port2=ht.ports["e2"],
        **routing_params,
    )
    c.add_port(
        name="ht_start",
        port=ht.ports["e1"],
    )

    c.add_port(
        name="ht_end",
        port=ht.ports["e2"],
    )

    c.add_port(
        name="e1",
        port=bps.ports["e11"],
    )
    c.add_port(
        name="e2",
        port=bps.ports["e2"],
    )

    c.flatten()

    return c


@gf.cell
def add_heater(
    heater_xs: CrossSection,
    routing_xs: CrossSection,
    interferometer: gf.Component = None,
    heater_on_both_branches: bool = False,
    heater_offset: float = 3.5,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    bias_tuning_section_length: float = 700.0,
    length_imbalance: float = 0.0,
) -> gf.Component:
    """Add heater to the modulator."""
    c = gf.Component()
    ht_cell = heater_straight_compact(
        length=bias_tuning_section_length,
        heater_xs=heater_xs,
        routing_xs=routing_xs,
        pad_size=heater_pad_size,
    )
    ht_ref_1 = c << ht_cell

    if heater_on_both_branches:
        ht_ref_2 = c << ht_cell
    if length_imbalance < 0.0:
        heater_disp_1 = [0, 0.5 * heater_width + heater_offset]
        heater_disp_2 = [0, 0.5 * heater_width + heater_offset]
    else:
        if not heater_on_both_branches:
            ht_ref_1.dmirror_y()
        else:
            ht_ref_2.dmirror_y()
        heater_disp_1 = (
            [0, 0.5 * heater_width + heater_offset]
            if heater_on_both_branches
            else [0, -0.5 * heater_width - heater_offset]
        )
        heater_disp_2 = [0, -0.5 * heater_width - heater_offset]

    ht_ref_1.dmove(
        origin=ht_ref_1.ports["ht_start"].dcenter,
        destination=(
            np.array(interferometer.ports["long_bias_branch_start"].dcenter)
            + heater_disp_1
        ),
    )
    ht_ref_1.dmirror_x(x=(ht_ref_1.xmax + ht_ref_1.xmin) / 2)
    if heater_on_both_branches:
        ht_ref_2.dmove(
            origin=ht_ref_2.ports["ht_start"].dcenter,
            destination=(
                np.array(interferometer.ports["short_bias_branch_start"].dcenter)
                + heater_disp_2
            ),
        )
        ht_ref_2.dmirror_x(x=(ht_ref_2.xmax + ht_ref_2.xmin) / 2)
    c.add_port(name="e1", port=ht_ref_1.ports["e1"])
    c.add_port(name="e2", port=ht_ref_1.ports["e2"])
    c.add_port(name="e3", port=ht_ref_2.ports["e1"])
    c.add_port(name="e4", port=ht_ref_2.ports["e2"])

    return c


if __name__ == "__main__":
    from ltoi300.tech import xs_ht_wire
    heater_xs = xs_ht_wire(width=0.9)
    routing_xs = xs_ht_wire(width=10.0)
    c = heater_straight_compact(
        heater_xs = heater_xs,
        routing_xs = routing_xs,
    )
    c.show()
