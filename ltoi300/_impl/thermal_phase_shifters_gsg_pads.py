import gdsfactory as gf
import numpy as np
from gdsfactory.routing import route_quad
from gdsfactory.typings import Layer

from ltoi300.tech import LAYER, xs_ht_wire


@gf.cell
def heater_resistor(
    path: gf.path.Path | None = None,
    width: float = 0.9,
    offset: float = 0.0,
    ht_layer: Layer = LAYER.HRM,
    length: float = 150.0,
) -> gf.Component:
    """A resistive wire used as a low-frequency phase shifter, exploiting
    the thermo-optical effect."""

    if not path:
        path = gf.path.straight(length=length)

    xs = gf.get_cross_section(xs_ht_wire, width=width, offset=offset, ht_layer=ht_layer)
    c = path.extrude(xs)

    return c


@gf.cell
def heater_pads_assymm(
    pad_size: tuple[float, float] = (150.0, 150.0),
    heater_length: float = 700.0,
    pad_pitch: float | None = None,
    routing_width: float = 30.0,
    ht_layer: Layer = LAYER.HRM,
) -> gf.Component:
    padwidth, padheight = pad_size
    if not pad_pitch:
        pad_pitch = padwidth + 20.0

    c = gf.Component()
    bondpads = gf.components.pad_array(
        pad=gf.components.pad,
        size=pad_size,
        column_pitch=pad_pitch,
        row_pitch=pad_pitch,
        columns=2,
        port_orientation=-90.0,
        layer=ht_layer,
    )
    length = heater_length - padwidth - pad_pitch
    bps = c << bondpads

    def routing():
        c = gf.Component()
        path = gf.path.straight(length=length)
        xs = gf.get_cross_section(xs_ht_wire, width=routing_width, ht_layer=ht_layer)
        c = path.extrude(xs)
        return c

    route = routing()
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
    length: float = 700.0,
    width: float = 0.9,
    offset: float = 0.0,
    port_contact_width_ratio: float = 3.0,
    pad_size: tuple[float, float] = (150.0, 150.0),
    pad_pitch: float | None = None,
    pad_vert_offset: float = 10.0,
    routing_width: float = 30.0,
    ht_layer: Layer = LAYER.HRM,
) -> gf.Component:
    """A straight resistive wire used as a low-frequency phase shifter,
    exploiting the thermo-optical effect. The heater is terminated by wide pads
    for probing or bonding."""

    if pad_vert_offset <= 0:
        raise ValueError(
            "pad_vert_offset must be a positive number,"
            + f"received {pad_vert_offset}."
        )

    if port_contact_width_ratio <= 0:
        raise ValueError(
            "port_contact_width_ratio must be a positive number,"
            + f"received {port_contact_width_ratio}."
        )

    c = gf.Component()
    bondpads = heater_pads_assymm(
        pad_size=pad_size,
        heater_length=length,
        pad_pitch=pad_pitch,
        routing_width=routing_width,
        ht_layer=ht_layer,
    )
    bps = c << bondpads

    ht = heater_resistor(
        path=gf.path.straight(length),
        width=width,
        offset=offset,
        ht_layer=ht_layer,
    )

    # Place the ports along the edge of the wire
    for p in ht.ports:
        if p.orientation == 0.0:
            p.dcenter = (p.dcenter[0] - 0.5 * p.dwidth, p.dcenter[1] + 0.5 * width)
        if p.orientation == 180.0:
            p.dcenter = (p.dcenter[0] + 0.5 * p.dwidth, p.dcenter[1] + 0.5 * width)
        p.orientation = 90.0

    ht_ref = c << ht

    bps.dxmin = ht_ref.dxmin
    bps.dymin = ht_ref.dymax + pad_vert_offset

    port_contact_width = port_contact_width_ratio * width
    ht.ports["e1"].dx += 0.5 * (port_contact_width - width)
    ht.ports["e2"].dx -= 0.5 * (port_contact_width - width)
    routing_params = {
        "width2": port_contact_width,
        "layer": ht_layer,
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
    heater_on_both_branches: bool = False,
    heater_offset: float = 3.5,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    ht_layer: Layer = LAYER.HRM,
    bias_tuning_section_length: float = 700.0,
    length_imbalance: float = 0.0,
    interferometer: gf.Component = None,
) -> gf.Component:
    """Add heater to the modulator."""
    c = gf.Component()
    ht_cell = heater_straight_compact(
        length=bias_tuning_section_length,
        width=heater_width,
        offset=heater_offset,
        pad_size=heater_pad_size,
        ht_layer=ht_layer,
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
        if heater_on_both_branches:
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


@gf.cell
def add_gsg_heater(
    heater_on_both_branches: bool = False,
    heater_offset: float = 3.5,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    ht_layer: Layer = LAYER.HRM,
    bias_tuning_section_length: float = 700.0,
    length_imbalance: float = 0.0,
    interferometer: gf.Component = None,
    m2_layer: Layer = (22, 0),
    shorted_heater_by: float = 80.0,
) -> gf.Component:
    """Add heater to the modulator."""
    c = gf.Component()
    if length_imbalance < 0.0:
        raise NotImplementedError(
            "GSG heater with negative length_imbalance is not implemented"
        )

    ht_cell = heater_straight_compact(
        length=bias_tuning_section_length - shorted_heater_by,
        width=heater_width,
        offset=heater_offset,
        pad_size=heater_pad_size,
        ht_layer=ht_layer,
        pad_pitch=bias_tuning_section_length - shorted_heater_by - 50.0,
    )
    ht_ref_1 = c << ht_cell
    # ht_ref_1.dmirror_y()
    heater_disp_1 = [shorted_heater_by, 0.5 * heater_width - heater_offset]
    ht_ref_1.dmove(
        origin=ht_ref_1.ports["ht_start"].dcenter,
        destination=(
            np.array(interferometer.ports["long_bias_branch_start"].dcenter)
            + heater_disp_1
        ),
    )
    ht_ref_1.dmirror_x(x=(ht_ref_1.xmax + ht_ref_1.xmin) / 2)
    c << add_m2_rectangle_on_e1_and_e2(
        ht_ref_1, heater_pad_size=heater_pad_size, m2_layer=m2_layer
    )
    if heater_on_both_branches:
        ht_ref_2 = c << ht_cell
        ht_ref_2.dmirror_y()
        heater_disp_2 = [shorted_heater_by, -0.5 * heater_width + heater_offset]
        ht_ref_2.dmove(
            origin=ht_ref_2.ports["ht_start"].dcenter,
            destination=(
                np.array(interferometer.ports["short_bias_branch_start"].dcenter)
                + heater_disp_2
            ),
        )
        ht_ref_2.dmirror_x(x=(ht_ref_2.xmax + ht_ref_2.xmin) / 2)
        c << add_m2_rectangle_on_e1_and_e2(
            ht_ref_2, heater_pad_size=heater_pad_size, m2_layer=m2_layer
        )
    c.add_port(name="e1", port=ht_ref_1.ports["e1"])
    c.add_port(name="e2", port=ht_ref_1.ports["e2"])
    c.add_port(name="e3", port=ht_ref_2.ports["e1"])
    c.add_port(name="e4", port=ht_ref_2.ports["e2"])
    heater_routing = c << route_gsg_heater(
        heater_1=ht_ref_1,
        heater_2=ht_ref_2,
        heater_pad_size=heater_pad_size,
        m2_layer=m2_layer,
    )
    # Expose GSG heater routing ports
    c.add_port(name="gsg_e1", port=heater_routing.ports["gsg_e1"])
    c.add_port(name="gsg_e2", port=heater_routing.ports["gsg_e2"])
    c.add_port(name="gsg_e3", port=heater_routing.ports["gsg_e3"])
    c.add_port(name="gsg_e4", port=heater_routing.ports["gsg_e4"])
    c.flatten()
    return c


@gf.cell
def route_gsg_heater(
    heater_1: gf.Component,
    heater_2: gf.Component,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    m2_layer: Layer = (22, 0),
    route_M2_width: float = 25.0,
) -> gf.Component:
    """Route M2 metal between heaters and the 4-pad GSG structure.

    Creates temporary anchor ports near the heaters (offset by half pad height
    for clearance) and bundles routes from GSG pad ports (gsg_e1..gsg_e4)
    to those anchors using Manhattan routing.

    - Metal layer: m2_layer
    - Route width: route_M2_width
    - Uses bend_euler and straight segments.
    """
    heater_routing = gf.Component()
    # Reference to GSG pad array to connect to
    gsg_pads_ref = heater_routing << GSG_heater_pads(
        heater_1=heater_1, heater_2=heater_2, m2_layer=m2_layer
    )

    # Anchor ports aligned to heater pads to simplify routing fanout
    anchor_ports = gf.Component()
    anchor_ports.add_port(
        name="e1",
        center=(
            heater_1.ports["e1"].dcenter[0],
            heater_1.ports["e1"].dcenter[1] - 0.5 * heater_pad_size[1],
        ),
        width=heater_1.ports["e1"].dwidth,
        orientation=-heater_1.ports["e1"].orientation,
        layer=LAYER.M2,
    )
    anchor_ports.add_port(
        name="e2",
        center=(
            heater_1.ports["e2"].dcenter[0],
            heater_1.ports["e2"].dcenter[1] - 0.5 * heater_pad_size[1],
        ),
        width=heater_1.ports["e2"].dwidth,
        orientation=-heater_1.ports["e2"].orientation,
        layer=LAYER.M2,
    )
    anchor_ports.add_port(
        name="e3",
        center=(
            heater_2.ports["e1"].dcenter[0],
            heater_2.ports["e1"].dcenter[1] + 0.5 * heater_pad_size[1],
        ),
        width=heater_2.ports["e1"].dwidth,
        orientation=-heater_2.ports["e1"].orientation,
        layer=LAYER.M2,
    )
    anchor_ports.add_port(
        name="e4",
        center=(
            heater_2.ports["e2"].dcenter[0],
            heater_2.ports["e2"].dcenter[1] + 0.5 * heater_pad_size[1],
        ),
        width=heater_2.ports["e2"].dwidth,
        orientation=-heater_2.ports["e2"].orientation,
        layer=LAYER.M2,
    )

    # Bundle routes from GSG pads to the local heater anchor ports (order reduces crossings)
    gf.routing.route_bundle(
        component=heater_routing,
        ports1=[
            gsg_pads_ref.ports["gsg_e1"],
            gsg_pads_ref.ports["gsg_e2"],
        ],
        ports2=[
            anchor_ports.ports["e1"],
            anchor_ports.ports["e2"],
        ],
        layer=LAYER.M2,
        bend=gf.components.bend_euler,
        straight=gf.components.straight,
        route_width=route_M2_width,
        separation=15,
    )
    gf.routing.route_bundle(
        component=heater_routing,
        ports1=[
            gsg_pads_ref.ports["gsg_e3"],
            gsg_pads_ref.ports["gsg_e4"],
        ],
        ports2=[
            anchor_ports.ports["e4"],
            anchor_ports.ports["e3"],
        ],
        layer=LAYER.M2,
        bend=gf.components.bend_euler,
        straight=gf.components.straight,
        route_width=route_M2_width,
        separation=15,
    )
    # Expose GSG pad ports for external connectivity
    heater_routing.add_port(name="gsg_e1", port=gsg_pads_ref.ports["gsg_e1"])
    heater_routing.add_port(name="gsg_e2", port=gsg_pads_ref.ports["gsg_e2"])
    heater_routing.add_port(name="gsg_e3", port=gsg_pads_ref.ports["gsg_e3"])
    heater_routing.add_port(name="gsg_e4", port=gsg_pads_ref.ports["gsg_e4"])
    return heater_routing


@gf.cell
def GSG_heater_pads(
    heater_1: gf.Component,
    heater_2: gf.Component,
    bp_length: float = 150.0,
    bp_width: float = 100.0,
    m2_layer: Layer = (22, 0),
    x_shift: float = -240.0,
    gap_width: float = 50.0,
    y_shift: float = 0,
) -> gf.Component:
    """Create 4 coplanar metal bond pads (rectangles) for a local GSG access.
    All four pads have the same height equal to `bp_width` and length `bp_length`.

    Order from top to bottom:
    1) Upper pad (height = `bp_width`) → port `gsg_e1`
    2) Second pad (height = `bp_width`) → port `gsg_e2`
    3) Third pad (height = `bp_width`) → port `gsg_e3`
    4) Bottom pad (height = `bp_width`) → port `gsg_e4`

    The stack is centered at the midpoint between `heater_1.e1` and `heater_2.e1`,
    then shifted by (`x_shift`, `y_shift`).
    """
    gsg_pads = gf.Component()

    # Compute stack center at midpoint between heaters and apply shifts
    p1 = np.array(heater_1.ports["e1"].dcenter)
    p2 = np.array(heater_2.ports["e1"].dcenter)
    stack_center = 0.5 * (p1 + p2)
    stack_center = (stack_center[0] - x_shift, stack_center[1] + y_shift)

    # Define identical heights for 4-rectangle stack (top to bottom)
    pad_height = bp_width
    heights = [pad_height, pad_height, pad_height, pad_height]
    total_height = sum(heights) + 3 * gap_width

    # Helper to place a rectangle of given height centered at (cx, cy)
    def place_rect(height: float, center: tuple[float, float]):
        rect = gsg_pads << gf.components.rectangle(
            size=(bp_length, height), layer=m2_layer
        )
        current_center = (
            (rect.xmax + rect.xmin) / 2,
            (rect.ymax + rect.ymin) / 2,
        )
        rect.dmove(origin=current_center, destination=center)
        return rect

    # Compute centers for 4 rectangles from the top down
    cx = stack_center[0]
    y_top_edge = stack_center[1] + total_height / 2.0
    centers = []
    y_cursor = y_top_edge
    for h in heights:
        cy = y_cursor - h / 2.0
        centers.append((cx, cy))
        y_cursor = y_cursor - h - gap_width

    # Create rectangles
    upper_ground_rect = place_rect(heights[0], centers[0])  # top
    intermediate_rect = place_rect(heights[1], centers[1])  # second
    lower_ground_rect = place_rect(heights[2], centers[2])  # third
    bottom_signal_rect = place_rect(heights[3], centers[3])  # bottom

    # Ports on right edges, centered vertically on the corresponding pad
    # gsg_e1 on upper rectangle
    upper_center_final = (
        (upper_ground_rect.xmax + upper_ground_rect.xmin) / 2,
        (upper_ground_rect.ymax + upper_ground_rect.ymin) / 2,
    )
    gsg_pads.add_port(
        name="gsg_e1",
        center=(upper_ground_rect.xmax - bp_length, upper_center_final[1]),
        width=pad_height,
        orientation=90,
        layer=LAYER.M2,
    )

    # gsg_e2 on second rectangle (intermediate)
    intermediate_center_final = (
        (intermediate_rect.xmax + intermediate_rect.xmin) / 2,
        (intermediate_rect.ymax + intermediate_rect.ymin) / 2,
    )
    gsg_pads.add_port(
        name="gsg_e2",
        center=(intermediate_rect.xmax, intermediate_center_final[1]),
        width=pad_height,
        orientation=0,
        layer=LAYER.M2,
    )

    # gsg_e3 on third rectangle (lower ground)
    lower_center_final = (
        (lower_ground_rect.xmax + lower_ground_rect.xmin) / 2,
        (lower_ground_rect.ymax + lower_ground_rect.ymin) / 2,
    )
    gsg_pads.add_port(
        name="gsg_e3",
        center=(lower_ground_rect.xmax, lower_center_final[1]),
        width=pad_height,
        orientation=0,
        layer=LAYER.M2,
    )

    # gsg_e4 on bottom rectangle (signal)
    bottom_signal_center = (
        (bottom_signal_rect.xmax + bottom_signal_rect.xmin) / 2,
        (bottom_signal_rect.ymax + bottom_signal_rect.ymin) / 2,
    )
    gsg_pads.add_port(
        name="gsg_e4",
        center=(bottom_signal_rect.xmax - bp_length, bottom_signal_center[1]),
        width=pad_height,
        orientation=-90,
        layer=LAYER.M2,
    )

    return gsg_pads


@gf.cell
def add_m2_rectangle_on_e1_and_e2(
    heater: gf.Component,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    m2_layer: Layer = (22, 0),
) -> gf.Component:
    """Add M2 rectangle on the e1 and e2 ports of the heater."""
    c = gf.Component()
    pad_rect_e1 = c << gf.components.rectangle(size=heater_pad_size, layer=m2_layer)
    pad_e1_curr_center = (
        (pad_rect_e1.xmax + pad_rect_e1.xmin) / 2,
        (pad_rect_e1.ymax + pad_rect_e1.ymin) / 2,
    )
    pad_e1_target_center = heater.ports["e1"].dcenter
    pad_rect_e1.dmove(origin=pad_e1_curr_center, destination=pad_e1_target_center)
    pad_rect_e2 = c << gf.components.rectangle(size=heater_pad_size, layer=m2_layer)
    pad_e2_curr_center = (
        (pad_rect_e2.xmax + pad_rect_e2.xmin) / 2,
        (pad_rect_e2.ymax + pad_rect_e2.ymin) / 2,
    )
    pad_e2_target_center = heater.ports["e2"].dcenter
    pad_rect_e2.dmove(origin=pad_e2_curr_center, destination=pad_e2_target_center)
    return c
