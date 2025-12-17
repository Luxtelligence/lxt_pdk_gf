from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Layer

from lnoi400.cells import (
    L_turn_bend,
    S_bend_vert,
)
from ltoi300._builders.mmis import (
    build_mmi1x2_oband,
)
from ltoi300._impl.thermal_phase_shifters_gsg_pads import add_gsg_heater, add_heater
from ltoi300.tech import LAYER, xs_rwg700, xs_rwg900, xs_uni_cpw


def draw_gsg_rectangles(
    parent_component,
    width: float,
    center_height: float,
    side_height: float,
    layer,
    x_center: float,
    y_center: float,
    dy: float,
    alias_prefix: str = None,
) -> tuple:
    """
    Generic function to draw three rectangles stacked vertically (GSG pattern).

    Layout Diagram:
    ===============

          |<-- width -->|
          +-------------+
          |             |
          | side_height |  <- Top rectangle
          |             |
          +-------------+
              |
              | dy (center-to-center)
              |
          +-------------+
          |             |
          |center_height|  <- Center rectangle (at y_center)
          |      *      |     * = center point (x_center, y_center)
          +-------------+
              |
              | dy (center-to-center)
              |
          +-------------+
          |             |
          | side_height |  <- Bottom rectangle
          |             |
          +-------------+
                 |
                 |
            x_center (geometric x-center of all rectangles)

    Geometric Parameters:
    - width: Controls the x-size (width) of all three rectangles
    - side_height: Controls the y-size (height) of top and bottom rectangles
    - center_height: Controls the y-size (height) of center rectangle
    - x_center: X-coordinate of the geometric center (middle) of all rectangles
    - y_center: Y-coordinate of the center rectangle's center
    - dy: Center-to-center distance from center rectangle to top/bottom rectangles

    Args:
        parent_component: Parent component to add rectangles to
        width: Width (x-size) of all rectangles
        center_height: Height (y-size) of center rectangle
        side_height: Height (y-size) of top and bottom rectangles
        layer: Layer for the rectangles
        x_center: X-coordinate of the geometric center (middle) of all rectangles
        y_center: Y-center position for center rectangle
        dy: Center-to-center distance from center rectangle to top/bottom rectangles
        alias_prefix: Optional prefix for rectangle aliases

    Returns:
        Tuple of (top_ref, center_ref, bottom_ref) component references
    """
    # Create rectangles
    center_rect = gf.components.rectangle(size=(width, center_height), layer=layer)
    side_rect = gf.components.rectangle(size=(width, side_height), layer=layer)

    # Place rectangles in parent component
    r_center = parent_component << center_rect
    r_center.center = (x_center, y_center)

    r_top = parent_component << side_rect
    r_top.center = (x_center, y_center + dy)

    r_bot = parent_component << side_rect
    r_bot.center = (x_center, y_center - dy)

    # Set aliases if provided
    if alias_prefix:
        r_top.alias = f"{alias_prefix}_top"
        r_center.alias = f"{alias_prefix}_center"
        r_bot.alias = f"{alias_prefix}_bottom"

    return r_top, r_center, r_bot


@gf.cell
def three_rectangles_hr_m2(
    # high_res (layer 23/0)
    term_pad_xsize: float = 14.0,  # updated default
    term_pad_side_ysize: float = 55.0,  # upper & lower high_res
    term_pad_centr_ysize: float = 16.0,  # central high_res
    # M2 (layer 22/0)
    M2_pad_xsize: float = 79.0,
    M2_pad_side_ysize: float = 55.0,  # upper & lower M2
    M2_pad_centr_ysize: float = 16.0,  # central M2
    # placement
    effective_length: float = 100.0,
    straight_ratio: float = 0.5,
    high_res_layer=(23, 0),
    M2_layer=(22, 0),
    M2_offset_x: float = 0.0,  # M2 xmax offset from high_res xmax (left if +)
    # M1_M2 opening layer (40/0) - opening between M1 and M2
    opening_layer=(40, 0),
    opening_offset_from_M2_xmin: float = 2.5,  # used as left + top/bottom offset
    # M2_high_res opening layer - opening between M2 and high_res
    M2_high_res_opening_layer=(41, 0),  # can be same or different layer
    M2_high_res_opening_offset: float = 2.5,  # offset from M2 and high_res borders
    # M1 (virtual) border
    M1_high_r_offset_x: float = None,  # M1 offset from high_res (left if +)
) -> gf.Component:
    """
    Simplified Layout Diagram (M2 and high_res central rectangles):
    ===============================================================

          |<------------------------------------------- M2_pad_xsize -------------------------------------->|
          +-------------------------------------------------------------------------------------------------+
          |                                                                                                 |
          |                                                                                                 |
          |                                             M2_pad_centr_ysize                                  |
          |                                                                                                 |
          |                                                                                                 |
          +-------------------------------------------------------------------------------------------------+


                                                                                                             |<- M2_high_res_opening_offset ------>|

                                                                                                |<-------------- term_pad_xsize ------------------>|
                                                                                                +-------------------------------------  -----------+
                                                                                                |                                                  |
                                                                                                |  |=======|                                       |
                                                                                                |  |       |     term_pad_centr_ysize              |
                                                                                                |  |       |                                       |
                                                                                                |  |====== |                                       |
                                                                                                +------------------------------------   -----------+
    Note: This is a simplified view showing only the central rectangles.
    The full structure has three rectangles (top, center, bottom) for each layer.
    Vertical spacing is controlled by: dy = effective_length * (1 - straight_ratio)
    M2 xmax is offset from high_res xmax by M2_offset_x (shown as non-zero for illustration)

    Parameters:
    - M2_pad_xsize: width of all M2 rectangles
    - M2_pad_centr_ysize: height of center M2 rectangle
    - M2_pad_side_ysize: height of top and bottom M2 rectangles
    - effective_length, straight_ratio: control vertical spacing (dy) between rectangles

    - high_res: 3 rectangles; center has xmax=0, y=0. Top/bottom spaced by dy = effective_length*(1-straight_ratio).
    - M2: same y-centers as high_res; xmax(M2) = 0 - M2_offset_x.
    - M1 layer is not drawn; its (right) border is a vertical line at x = -term_pad_xsize - M1_high_r_offset_x.
    - M1_M2 openings (layer 40/0): 3 rectangles fully inside M2 with:
        * left offset = opening_offset_from_M2_xmin from M2 xmin
        * top/bottom offsets = opening_offset_from_M2_xmin from M2 top/bottom
        * RIGHT edge positioned to keep the same offset from the *M1 border*:
              x_opening_max = x_M1_border - opening_offset_from_M2_xmin
          (opening width is derived from these left/right constraints)
    - M2_high_res openings: 3 rectangles at the intersection between M2 and high_res layers:
        * offset by M2_high_res_opening_offset from M2 and high_res borders
        * positioned at the overlap region between M2 and high_res
    """
    c = gf.Component("three_rectangles_highres_m2_with_openings")

    # Vertical spacing
    dy = effective_length * (1.0 - straight_ratio)

    # --- Place high_res center so xmax = 0 and y = 0 ---
    x_center_high_res = -term_pad_xsize / 2.0  # ensures xmax == 0
    r_high_res_top, r_high_res_center, r_high_res_bot = draw_gsg_rectangles(
        parent_component=c,
        width=term_pad_xsize,
        center_height=term_pad_centr_ysize,
        side_height=term_pad_side_ysize,
        layer=high_res_layer,
        x_center=x_center_high_res,
        y_center=0.0,
        dy=dy,
        alias_prefix="high_res",
    )

    # --- M2 placement from desired xmax offset ---
    xmax_m2 = -M2_offset_x
    x_center_m2 = xmax_m2 - (M2_pad_xsize / 2.0)
    m2_xmin = x_center_m2 - (M2_pad_xsize / 2.0)
    m2_xmax = m2_xmin + M2_pad_xsize

    r_m2_top, r_m2_center, r_m2_bot = draw_gsg_rectangles(
        parent_component=c,
        width=M2_pad_xsize,
        center_height=M2_pad_centr_ysize,
        side_height=M2_pad_side_ysize,
        layer=M2_layer,
        x_center=x_center_m2,
        y_center=r_high_res_center.center[1],
        dy=dy,
        alias_prefix="M2",
    )

    # --- Openings: inside M2 with left/top/bottom offsets = margin,
    #               and RIGHT edge offset from the (virtual) M1 border by the same margin ---
    margin = opening_offset_from_M2_xmin

    # Heights (ensure they remain positive)
    open_h_center = M2_pad_centr_ysize - 2 * margin
    open_h_side = M2_pad_side_ysize - 2 * margin
    if open_h_center <= 0 or open_h_side <= 0:
        raise ValueError(
            "opening_offset_from_M2_xmin is too large: opening heights must be > 0."
        )

    # M1 border (not drawn)
    x_M1_border = -term_pad_xsize - M1_high_r_offset_x

    # Left & right limits for openings
    opening_xmin = m2_xmin + margin
    opening_xmax = x_M1_border - margin

    # Validate geometry
    if opening_xmax <= opening_xmin:
        raise ValueError(
            "Opening has non-positive width: decrease M1_high_r_offset_x and/or margin, "
            "or increase M2_pad_xsize."
        )
    if opening_xmax > m2_xmax:
        raise ValueError(
            "Opening extends beyond M2 to the right. Adjust offsets or sizes."
        )

    # Final width is determined by left/right constraints (opening_xsize acts as a nominal hint)
    opening_width = opening_xmax - opening_xmin

    # Place M1_M2 openings: x-centered between xmin/xmax; y-centered to get equal top/bottom margins
    opening_xcenter = (opening_xmin + opening_xmax) / 2.0

    r_open_top, r_open_center, r_open_bot = draw_gsg_rectangles(
        parent_component=c,
        width=opening_width,
        center_height=open_h_center,
        side_height=open_h_side,
        layer=opening_layer,
        x_center=opening_xcenter,
        y_center=r_m2_center.center[1],
        dy=dy,
        alias_prefix="M1_M2_OPEN",
    )

    # --- M2_high_res openings: at the intersection between M2 and high_res ---
    # Calculate intersection region between M2 and high_res
    high_res_xmin = -term_pad_xsize  # high_res extends from -term_pad_xsize to 0
    high_res_xmax = 0.0
    m2_intersect_xmin = max(high_res_xmin, m2_xmin)
    m2_intersect_xmax = min(high_res_xmax, m2_xmax)

    # Opening dimensions: offset by M2_high_res_opening_offset from both borders
    M2_hr_opening_xmin = m2_intersect_xmin + M2_high_res_opening_offset
    M2_hr_opening_xmax = m2_intersect_xmax - M2_high_res_opening_offset
    M2_hr_opening_width = M2_hr_opening_xmax - M2_hr_opening_xmin

    # Heights: offset from high_res top/bottom by M2_high_res_opening_offset
    M2_hr_open_h_center = term_pad_centr_ysize - 2 * M2_high_res_opening_offset
    M2_hr_open_h_side = term_pad_side_ysize - 2 * M2_high_res_opening_offset

    # Validate geometry
    if M2_hr_opening_width <= 0:
        raise ValueError(
            "M2_high_res opening has non-positive width. Adjust M2_high_res_opening_offset or layer sizes."
        )
    if M2_hr_open_h_center <= 0 or M2_hr_open_h_side <= 0:
        raise ValueError(
            "M2_high_res_opening_offset is too large: opening heights must be > 0."
        )

    # Place M2_high_res openings: x-centered in intersection region
    M2_hr_opening_xcenter = (M2_hr_opening_xmin + M2_hr_opening_xmax) / 2.0

    r_M2_hr_open_top, r_M2_hr_open_center, r_M2_hr_open_bot = draw_gsg_rectangles(
        parent_component=c,
        width=M2_hr_opening_width,
        center_height=M2_hr_open_h_center,
        side_height=M2_hr_open_h_side,
        layer=M2_high_res_opening_layer,
        x_center=M2_hr_opening_xcenter,
        y_center=r_high_res_center.center[1],
        dy=dy,
        alias_prefix="M2_high_res_OPEN",
    )

    # Useful info for downstream inspection
    c.info.update(
        dict(
            x_M1_border=x_M1_border,
            opening_margin=margin,
            opening_width=opening_width,
            opening_xmin=opening_xmin,
            opening_xmax=opening_xmax,
            M2_hr_opening_width=M2_hr_opening_width,
            M2_hr_opening_xmin=M2_hr_opening_xmin,
            M2_hr_opening_xmax=M2_hr_opening_xmax,
        )
    )
    return c


@gf.cell
def CPW_termination_wire(
    resistor_width: float = 1.5,
    resistor_length: float = 190.0 / 2,
    cross_section: CrossSectionSpec = "xs_uni_cpw",
    straight_ratio: float = 0.25,
    # window_bias: float = -5.0,
    # contact_pad_bias: float = -2.5,
    # contact_pad_width: float = 8.0,
    M1_high_r_offset_x: float = 2.5,  # M1 offset from high_res (left if +)
    port_width: float = 21.0,
    termination_layer: LAYER = LAYER.HRM,
    # m2_layer: LAYER = LAYER.M2,
) -> gf.Component:
    """A bondpad with connections to a resistive load."""

    # xs_cpw = gf.get_cross_section(cross_section)

    # Extract the CPW cross sectional parameters

    # sections = xs_cpw.sections
    # signal_section = [s for s in sections if s.name == "signal"][0]
    # signal_width = signal_section.width

    effective_length = resistor_length / (1 + straight_ratio / 2)
    # Pad elements generation

    pad = gf.Component()
    straight_section_shape = [
        (0, resistor_width / 2),
        (0, -resistor_width / 2),
        (effective_length * straight_ratio / 2, -resistor_width / 2),
        (effective_length * straight_ratio / 2, resistor_width / 2),
    ]

    return_section_shape = [
        (0, effective_length * (1 - straight_ratio) - resistor_width / 2),
        (
            effective_length * straight_ratio / 2,
            effective_length * (1 - straight_ratio) - resistor_width / 2,
        ),
        (
            effective_length * straight_ratio / 2,
            effective_length * (1 - straight_ratio) + resistor_width / 2,
        ),
        (0, effective_length * (1 - straight_ratio) + resistor_width / 2),
    ]

    horizontal_shape = [
        (effective_length * straight_ratio / 2, -resistor_width / 2),
        (
            effective_length * straight_ratio / 2,
            effective_length * (1 - straight_ratio) + resistor_width / 2,
        ),
        (
            effective_length * straight_ratio / 2 + resistor_width,
            effective_length * (1 - straight_ratio) + resistor_width / 2,
        ),
        (effective_length * straight_ratio / 2 + resistor_width, -resistor_width / 2),
    ]

    horizontal_shape_2 = [(x, -y) for x, y in horizontal_shape]
    return_section_shape_2 = [(x, -y) for x, y in return_section_shape]

    pad.add_polygon(straight_section_shape, layer=termination_layer)
    pad.add_polygon(return_section_shape, layer=termination_layer)
    pad.add_polygon(horizontal_shape, layer=termination_layer)
    pad.add_polygon(horizontal_shape_2, layer=termination_layer)
    pad.add_polygon(return_section_shape_2, layer=termination_layer)

    # add all necessary things for the routing
    pd_ref = pad << three_rectangles_hr_m2(
        effective_length=effective_length,
        straight_ratio=straight_ratio,
        M1_high_r_offset_x=M1_high_r_offset_x,
    )

    offset = pd_ref.cell.settings["term_pad_xsize"]

    pad.flatten()

    c2 = gf.Component()
    rinner = 1000  # 	The circle radius of inner corners (in database units).
    router = 1000  # 	The circle radius of outer corners (in database units).
    n = 300  # 	The number of points per full circle.

    # round corners for all layers
    for layer, polygons in pad.get_polygons().items():
        for p in polygons:
            p_round = p.round_corners(rinner, router, n)
            c2.add_polygon(p_round, layer=layer)

        # Ports definition
    c2.add_port(
        name="e1",
        center=(-offset - M1_high_r_offset_x, 0.0),
        # width=signal_width,
        width=port_width,
        orientation=180.0,
        port_type="electrical",
        layer="M1",
    )
    c2.add_port(
        name="term",
        center=(effective_length * (straight_ratio / 2), 0),
        width=resistor_width,
        orientation=0,
        port_type="electrical",
        layer="M1",
    )
    return c2


def bonding_pads_curved_path(
    v_offset: float = 25.0,
    h_extent: float = 100.0,
    dx_straight: float = 5.0,
    AR: float = 0.5,
    signal_width: float = 3,
    gap_width: float = 5,
):
    """A bonding pad path based on spline_clamped_path function."""

    t = np.linspace(0, 1, int(np.round(2.5 * h_extent)))
    start = (0.0, 0.0)
    end = (h_extent, v_offset)

    xs = t
    ys = (t**2) * (3 - 2 * t)

    # Rescale to the start and end coordinates
    xs = start[0] + (end[0] - start[0]) * xs
    ys = start[1] + (end[1] - start[1]) * ys
    xs = np.append(xs, xs[-1] + dx_straight)
    ys = np.append(ys, ys[-1])

    # correction to the spline_clamped_path to keep the impedance constant
    dys = (2 * ys - 2 * ys[-1] + signal_width + gap_width) * (1 - AR) / (AR + 1) / 2
    # path for the signal electrode
    path_signal = gf.Path(np.column_stack([xs, ys - dys]))
    path_signal.start_angle = path_signal.end_angle = 0.0
    # path for the ground electrode
    path_ground = gf.Path(np.column_stack([xs, ys + dys]))
    path_ground.start_angle = path_ground.end_angle = 0.0

    return path_signal, path_ground


@gf.cell
def CPW_pad_curved(
    start_width: float = 80.0,
    length_straight: float = 10.0,
    length_tapered: float = 190.0,
    cross_section: CrossSectionSpec = xs_uni_cpw,
    sbend_small_size: tuple[float, float] = (200.0, -45.0),
    sbend_small_straight_extend: float = 5.0,
    add_M2andOpenings: bool = False,
    layer_M2: Layer = (22, 0),
    layer_Openings: Layer = (40, 0),
    m1_opening_offset: float = 2.5,  # Offset from M1 edge to first opening (µm)
    opening_size: float = 12.0,  # Size of the opening rectangle (µm)
    opening_separation: float = 12.0,  # Separation between openings (µm)
    tl_opening_host_width: float = 42.0,  # Width of the TL rectangle needed to host openings (µm)
    m2_pad_length: float = 220,  # Length of M2 pad along x-axis (µm)
) -> gf.Component:
    """RF access line for high-frequency GSG probes with curved electordes
    following the optical waveguides. The probe pad maintains a
    fixed gap/central conductor ratio across its length, to achieve a good
    impedance matching"""
    length_tapered = length_tapered
    xs_cpw = gf.get_cross_section(cross_section)

    # Extract the CPW cross sectional parameters

    sections = xs_cpw.sections
    signal_section = [s for s in sections if s.name == "signal"][0]
    ground_section = [s for s in sections if s.name == "ground_top"][0]
    end_width = signal_section.width
    ground_planes_width = ground_section.width
    end_gap = ground_section.offset - 0.5 * (end_width + ground_planes_width)
    aspect_ratio = end_width / (end_width + 2 * end_gap)

    # Pad elements generation

    pad = gf.Component()
    path_signal, path_ground = bonding_pads_curved_path(
        v_offset=sbend_small_size[1],
        h_extent=sbend_small_size[0],
        dx_straight=sbend_small_straight_extend,
        AR=aspect_ratio,
        signal_width=end_width,
        gap_width=end_gap,
    )

    path_ground_seed = list(
        path_ground.points
    )  # take the points inferred from the s-bend shape and corrected to match the impedance

    ground_path_closed = path_ground_seed
    tmp = np.add(path_ground_seed[-1], (0, ground_planes_width)).tolist()
    ground_path_closed += [(tmp[0], tmp[1])]
    tmp = np.add(path_ground_seed[0], (0, ground_planes_width)).tolist()
    ground_path_closed += [(tmp[0], tmp[1])]
    roundpad = gf.Component()
    roundpad.add_polygon(ground_path_closed, layer=LAYER.M1)

    p1 = pad << roundpad

    p1.dmove(
        (p1.xmax, p1.ymin), (length_straight + length_tapered, end_width / 2 + end_gap)
    )
    p2 = pad << roundpad
    p2.dmove(
        (p2.xmax, p2.ymin), (length_straight + length_tapered, end_width / 2 + end_gap)
    )
    p2.dmirror_y(y=0)

    signal_path_closed = list(path_signal.points)
    converted_path_inv = [
        (x, -y + 2 * sbend_small_size[1] - end_width - end_gap)
        for x, y in path_signal.points
    ]
    signal_path_closed += list(reversed(converted_path_inv))
    centrpad = gf.Component()
    centrpad.add_polygon(signal_path_closed, layer=LAYER.M1)
    pc = pad << centrpad
    pc.dmove((pc.xmin, (pc.ymax + pc.ymin) / 2), (p1.xmin, 0))

    if add_M2andOpenings:
        # Create M2 metal layer component for additional metalization
        central_conductor_end_width = pc.ymax - pc.ymin

        # Calculate opening positions relative to central pad
        pad_center_y = (pc.ymax + pc.ymin) / 2

        # Add signal line extension positioned such that its right edge is at pc.xmin
        signal_line_extension = gf.components.rectangle(
            size=(tl_opening_host_width, central_conductor_end_width),
            layer=LAYER.M1,
            centered=True,
        )

        # Position signal line extension so its right edge aligns with pc.xmin
        signal_line_extension_ref = pad << signal_line_extension
        signal_line_extension_ref.dmove(
            signal_line_extension_ref.ports["e1"].dcenter,
            (pc.xmin - tl_opening_host_width, pad_center_y),
        )

        # Calculate how many openings can fit vertically within central conductor end width
        available_height = central_conductor_end_width - 2 * (m1_opening_offset)
        # Formula: available_height = num_openings * opening_size + (num_openings - 1) * opening_separation
        # Solving for num_openings: num_openings = (available_height + opening_separation) / (opening_size + opening_separation)
        num_openings = int(
            (available_height + opening_separation)
            / (opening_size + opening_separation)
        )

        # Calculate total height needed for all openings with separations
        total_openings_height = (
            num_openings * opening_size + (num_openings - 1) * opening_separation
        )

        # Calculate starting position to center the column of openings
        start_y = pad_center_y - total_openings_height / 2 + opening_size / 2

        # Create first column of opening rectangles positioned relative to signal line extension left edge
        for i in range(num_openings):
            # Create opening rectangle for etching window
            opening_rectangle = gf.components.rectangle(
                size=(opening_size, opening_size), layer=layer_Openings, centered=True
            )

            # Calculate position for this opening relative to signal line extension left edge
            extension_left_edge = pc.xmin - tl_opening_host_width
            x_position = extension_left_edge + m1_opening_offset
            y_position = start_y + i * (opening_size + opening_separation)

            # Position the opening rectangle and add to pad
            opening_reference = pad << opening_rectangle
            opening_reference.dmove(
                opening_reference.ports["e1"].dcenter,
                (x_position, y_position),
            )

        # Create second column of opening rectangles shifted by opening_separation to the right
        for i in range(num_openings):
            # Create opening rectangle for etching window
            opening_rectangle = gf.components.rectangle(
                size=(opening_size, opening_size), layer=layer_Openings, centered=True
            )

            # Calculate position for this opening - same y as first column, shifted right by opening_separation
            extension_left_edge = pc.xmin - tl_opening_host_width
            x_position = (
                extension_left_edge
                + m1_opening_offset
                + opening_size
                + opening_separation
            )
            y_position = start_y + i * (opening_size + opening_separation)

            # Position the opening rectangle and add to pad
            opening_reference = pad << opening_rectangle
            opening_reference.dmove(
                opening_reference.ports["e1"].dcenter,
                (x_position, y_position),
            )

        # Calculate AR to deduce the gap at the end of the pad
        aspect_ratio = end_width / (end_width + 2 * end_gap)
        # Calculate the gap at the end of the pad
        central_conductor_end_gap = (
            central_conductor_end_width * (1 - aspect_ratio) / 2 / aspect_ratio
        )
        mzm_M1_end_width = p1.ymax - p2.ymin
        ground_conductor_end_width = (
            mzm_M1_end_width
            - 2 * central_conductor_end_gap
            - central_conductor_end_width
        ) / 2
        # Add ground line extension
        ground_line_extension = gf.components.rectangle(
            size=(tl_opening_host_width, ground_conductor_end_width),
            layer=LAYER.M1,
            centered=True,
        )
        # Position ground line extension so its right edge aligns with upper ground conductor of the mzm
        ground_line_extension_up_ref = pad << ground_line_extension
        ground_line_extension_up_ref.dmove(
            ground_line_extension_up_ref.ports["e1"].dcenter,
            (pc.xmin - tl_opening_host_width, p1.ymax - ground_conductor_end_width / 2),
        )
        # Position ground line extension so its right edge aligns with lower ground conductor of the mzm
        ground_line_extension_down_ref = pad << ground_line_extension
        ground_line_extension_down_ref.dmove(
            ground_line_extension_down_ref.ports["e1"].dcenter,
            (pc.xmin - tl_opening_host_width, p2.ymin + ground_conductor_end_width / 2),
        )

        # Add openings to upper ground line extension
        # Calculate how many openings can fit vertically within ground conductor end width
        ground_available_height = ground_conductor_end_width - 2 * (m1_opening_offset)
        ground_num_openings = int(
            (ground_available_height + opening_separation)
            / (opening_size + opening_separation)
        )

        # Calculate total height needed for all ground openings with separations
        ground_total_openings_height = (
            ground_num_openings * opening_size
            + (ground_num_openings - 1) * opening_separation
        )

        # Calculate starting position to center the column of ground openings
        ground_center_y = p1.ymax - ground_conductor_end_width / 2
        ground_start_y = (
            ground_center_y - ground_total_openings_height / 2 + opening_size / 2
        )

        # Create first column of openings in upper ground line extension
        for i in range(ground_num_openings):
            # Create opening rectangle for etching window
            opening_rectangle = gf.components.rectangle(
                size=(opening_size, opening_size), layer=layer_Openings, centered=True
            )

            # Calculate position for this opening relative to ground line extension left edge
            extension_left_edge = pc.xmin - tl_opening_host_width
            x_position = extension_left_edge + m1_opening_offset
            y_position = ground_start_y + i * (opening_size + opening_separation)

            # Position the opening rectangle and add to pad
            opening_reference = pad << opening_rectangle
            opening_reference.dmove(
                opening_reference.ports["e1"].dcenter,
                (x_position, y_position),
            )

        # Create second column of openings in upper ground line extension
        for i in range(ground_num_openings):
            # Create opening rectangle for etching window
            opening_rectangle = gf.components.rectangle(
                size=(opening_size, opening_size), layer=layer_Openings, centered=True
            )

            # Calculate position for this opening - same y as first column, shifted right by opening_separation
            extension_left_edge = pc.xmin - tl_opening_host_width
            x_position = (
                extension_left_edge
                + m1_opening_offset
                + opening_size
                + opening_separation
            )
            y_position = ground_start_y + i * (opening_size + opening_separation)

            # Position the opening rectangle and add to pad
            opening_reference = pad << opening_rectangle
            opening_reference.dmove(
                opening_reference.ports["e1"].dcenter,
                (x_position, y_position),
            )

        # Add openings to lower ground line extension
        # Calculate starting position to center the column of ground openings
        ground_center_y = p2.ymin + ground_conductor_end_width / 2
        ground_start_y = (
            ground_center_y - ground_total_openings_height / 2 + opening_size / 2
        )

        # Create first column of openings in lower ground line extension
        for i in range(ground_num_openings):
            # Create opening rectangle for etching window
            opening_rectangle = gf.components.rectangle(
                size=(opening_size, opening_size), layer=layer_Openings, centered=True
            )

            # Calculate position for this opening relative to ground line extension left edge
            extension_left_edge = pc.xmin - tl_opening_host_width
            x_position = extension_left_edge + m1_opening_offset
            y_position = ground_start_y + i * (opening_size + opening_separation)

            # Position the opening rectangle and add to pad
            opening_reference = pad << opening_rectangle
            opening_reference.dmove(
                opening_reference.ports["e1"].dcenter,
                (x_position, y_position),
            )

        # Create second column of openings in lower ground line extension
        for i in range(ground_num_openings):
            # Create opening rectangle for etching window
            opening_rectangle = gf.components.rectangle(
                size=(opening_size, opening_size), layer=layer_Openings, centered=True
            )

            # Calculate position for this opening - same y as first column, shifted right by opening_separation
            extension_left_edge = pc.xmin - tl_opening_host_width
            x_position = (
                extension_left_edge
                + m1_opening_offset
                + opening_size
                + opening_separation
            )
            y_position = ground_start_y + i * (opening_size + opening_separation)

            # Position the opening rectangle and add to pad
            opening_reference = pad << opening_rectangle
            opening_reference.dmove(
                opening_reference.ports["e1"].dcenter,
                (x_position, y_position),
            )

        # Add M2 layer rectangles
        # M2 signal line rectangle - same y-size as signal line extension
        m2_signal_rectangle = gf.components.rectangle(
            size=(m2_pad_length, central_conductor_end_width),
            layer=layer_M2,
            centered=True,
        )
        # Position M2 signal rectangle so its right edge aligns with signal line extension
        m2_signal_ref = pad << m2_signal_rectangle
        m2_signal_ref.dmove(
            m2_signal_ref.ports["e1"].dcenter,
            (pc.xmin - m2_pad_length, pad_center_y),
        )

        # M2 upper ground rectangle - same y-size as upper ground line extension
        m2_ground_up_rectangle = gf.components.rectangle(
            size=(m2_pad_length, ground_conductor_end_width),
            layer=layer_M2,
            centered=True,
        )
        # Position M2 upper ground rectangle so its right edge aligns with upper ground line extension
        m2_ground_up_ref = pad << m2_ground_up_rectangle
        m2_ground_up_ref.dmove(
            m2_ground_up_ref.ports["e1"].dcenter,
            (pc.xmin - m2_pad_length, p1.ymax - ground_conductor_end_width / 2),
        )

        # M2 lower ground rectangle - same y-size as lower ground line extension
        m2_ground_down_rectangle = gf.components.rectangle(
            size=(m2_pad_length, ground_conductor_end_width),
            layer=layer_M2,
            centered=True,
        )
        # Position M2 lower ground rectangle so its right edge aligns with lower ground line extension
        m2_ground_down_ref = pad << m2_ground_down_rectangle
        m2_ground_down_ref.dmove(
            m2_ground_down_ref.ports["e1"].dcenter,
            (pc.xmin - m2_pad_length, p2.ymin + ground_conductor_end_width / 2),
        )

    # Ports definition

    pad.add_port(
        name="e1",
        center=(pad.xmin, 0.0),
        width=start_width,
        orientation=180.0,
        port_type="electrical",
        layer=LAYER.M1,
    )

    pad.add_port(
        name="e2",
        center=(pc.xmax, 0.0),
        width=end_width,
        orientation=0.0,
        port_type="electrical",
        layer=LAYER.M1,
    )
    return pad


@gf.cell()
def trail_cpw_LT(
    length: float = 1000.0,
    signal_width: float = 21,
    gap_width: float = 4,
    th: float = 1.5,
    tl: float = 44.7,
    tw: float = 7.0,
    tt: float = 1.5,
    tc: float = 5.0,
    ground_planes_width: float = 180.0,
    rounding_radius: float = 0.5,
    num_cells: int = 30,
    bondpad: ComponentSpec = "CPW_pad_linear",
    cross_section: CrossSectionSpec = xs_uni_cpw,
    sbend_small_size: tuple[float, float] = (200.0, -45.0),
    sbend_small_straight_extend: float = 5.0,
    terminated: bool = False,
    termination_layer: Layer = LAYER.HRM,
    m2_layer: Layer = (22, 0),
    opening_layer: Layer = (40, 0),  # Layer for etching openings
    add_M2andOpenings: bool = False,  # Enable M2 metal and opening features
    resistor_length: float = 190.0 / 2,  # effective resistor length
) -> gf.Component:
    """A CPW transmission line with periodic T-rails on all electrodes"""

    num_cells = np.floor(length / (tl + tc))
    gap_width_corrected = gap_width + 2 * th + 2 * tt  # total gap width with T-rails

    # redefine cross section of the straight section to include T-rails

    xs_cpw_trail = partial(
        cross_section,
        central_conductor_width=signal_width,
        gap=gap_width_corrected,
        ground_planes_width=ground_planes_width,
    )

    # redefine cross section for bonding pads to include T-rails
    xs_cpw_bonding_pad = partial(
        cross_section,
        central_conductor_width=signal_width + 2 * th + 2 * tt,
        gap=gap_width,
        ground_planes_width=ground_planes_width + th + tt,
    )

    cpw = gf.Component()

    if bondpad["component"] == "CPW_pad_curved":
        bp = CPW_pad_curved(
            cross_section=xs_cpw_bonding_pad,
            sbend_small_size=sbend_small_size,
            sbend_small_straight_extend=sbend_small_straight_extend,
            add_M2andOpenings=add_M2andOpenings,
            layer_M2=m2_layer,
            layer_Openings=opening_layer,
        )
    else:
        bp = gf.get_component(bondpad, cross_section=xs_cpw_bonding_pad)
    strght = cpw << gf.components.straight(length=length, cross_section=xs_cpw_trail)
    dl_tr = 0.5 * (
        length - num_cells * tl - (num_cells - 1) * tc
    )  # t-rail shift to place symmetrically w/r to bond pads
    strght_ter_left = cpw << gf.components.straight(
        length=dl_tr + tl / 2, cross_section=xs_cpw_bonding_pad
    )  # cover the gap formed by th and tt.
    strght_ter_left.dmove(
        strght_ter_left.ports["e1"].dcenter, strght.ports["e1"].dcenter
    )
    strght_ter_left = cpw << gf.components.straight(
        length=dl_tr + tl / 2, cross_section=xs_cpw_bonding_pad
    )  # cover the gap formed by th and tt.
    strght_ter_left.dmove(
        strght_ter_left.ports["e2"].dcenter, strght.ports["e2"].dcenter
    )
    bp1 = cpw << bp
    bp1.connect("e2", strght.ports["e1"], allow_width_mismatch=True)
    if not terminated:
        bp2 = cpw << bp
        bp2.dmirror()
        bp2.connect("e2", strght.ports["e2"], allow_width_mismatch=True)
        cpw.add_port(
            name="bp2",
            port=bp2.ports["e1"],
        )
    else:
        bp2 = cpw << CPW_termination_wire(
            termination_layer=termination_layer,
            # m2_layer=m2_layer,
            resistor_length=resistor_length,
        )
        bp2.connect(
            "e1",
            strght.ports["e2"],
            allow_width_mismatch=True,
            allow_layer_mismatch=True,
        )
    #     cpw.add_port(
    #     name="bp2",
    #     port=bp2.ports["e1"],
    #     )
    cpw.add_ports(strght.ports)

    cpw.add_port(
        name="bp1",
        port=bp1.ports["e1"],
    )

    # Initiate T-rail polygon element. Create a bit more to ensure round corners close to electrodes
    trailpol = gf.kdb.DPolygon(
        [
            (
                tl + 2,
                signal_width / 2,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (
                tl + 2,
                signal_width / 2 - tt,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (
                0 - 2,
                signal_width / 2 - tt,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (
                0 - 2,
                signal_width / 2,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (tl / 2 - tw / 2, signal_width / 2),
            (tl / 2 - tw / 2, signal_width / 2 + th),
            (0, signal_width / 2 + th),
            (0, signal_width / 2 + th + tt),
            (tl, signal_width / 2 + th + tt),
            (tl, signal_width / 2 + th),
            (tl / 2 + tw / 2, signal_width / 2 + th),
            (tl / 2 + tw / 2, signal_width / 2),
        ]
    )

    # Create T-rail component
    trailcomp = gf.Component()
    _ = trailcomp.add_polygon(trailpol, layer=cross_section().layer)

    # Apply roc to the T-rail corners
    trailround = gf.Component()
    rinner = rounding_radius * 1000  # 	The circle radius of inner corners (in nm).
    router = rounding_radius * 1000  # 	The circle radius of outer corners (in nm).
    n = 30  # 	The number of points per full circle.

    for layer, polygons in trailcomp.get_polygons().items():
        for p in polygons:
            p_round = p.round_corners(rinner, router, n)
            trailround.add_polygon(p_round, layer=layer)

    # Create T-rail unit cell
    trail_uc = gf.Component()
    inc_t1 = trail_uc << trailround
    inc_t2 = trail_uc << trailround
    inc_t2.dmirror_y(y=(inc_t2.dymax + inc_t2.dymin) / 2)
    inc_t2.dmovey(gap_width_corrected - th)
    inc_t3 = trail_uc << trailround
    inc_t3.dmirror_y(y=(inc_t3.dymax + inc_t3.dymin) / 2)
    inc_t3.dmovey(-signal_width - th)
    inc_t4 = trail_uc << trailround
    inc_t4.dmovey(-signal_width - gap_width_corrected)

    # Place T-rails symmetrically w/r to bondpads
    [ref.dmovex(dl_tr) for ref in (inc_t1, inc_t2, inc_t3, inc_t4)]
    # [ref.dmovey(300*dl_tr) for ref in (inc_t1, inc_t2, inc_t3, inc_t4)]
    # Duplicate cell
    cpw.add_ref(
        trail_uc,
        columns=num_cells,
        rows=1,
        column_pitch=tl + tc,
    )

    cpw.flatten()

    return cpw


@gf.cell
def eo_phase_shifter_LT(
    rib_core_width_modulator: float = 2.5,
    taper_length: float = 100.0,
    modulation_length: float = 7500.0,
    rf_central_conductor_width: float = 21.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    communication_band: str = "O-band",
    draw_cpw: bool = True,
) -> gf.Component:
    """Phase shifter based on the Pockels effect. The waveguide is located
    within the gap of a CPW transmission line."""
    ps = gf.Component()
    xs_modulator = gf.get_cross_section(xs_rwg700(width=rib_core_width_modulator))
    wg_taper = gf.components.taper(
        length=taper_length,
        width1=xs_rwg700().width
        if communication_band == "O-band"
        else xs_rwg900().width,
        width2=rib_core_width_modulator,
        cross_section=xs_rwg700 if communication_band == "O-band" else xs_rwg900,
        with_two_ports=True,
    )
    wg_phase_modulation = gf.components.straight(
        length=modulation_length - 2 * taper_length, cross_section=xs_modulator
    )

    taper_1 = ps << wg_taper
    wg_pm = ps << wg_phase_modulation
    taper_2 = ps << wg_taper
    taper_2.dmirror_x()
    wg_pm.connect("o1", taper_1.ports["o2"])
    taper_2.dmirror_x()
    taper_2.connect("o2", wg_pm.ports["o2"])

    for name, port in [
        ("o1", taper_1.ports["o1"]),
        ("o2", taper_2.ports["o1"]),
    ]:
        ps.add_port(name=name, port=port)

    # Add the transmission line

    # if draw_cpw:
    #     xs_cpw = gf.partial(
    #         xs_uni_cpw,
    #         central_conductor_width=rf_central_conductor_width,
    #         ground_planes_width=rf_ground_planes_width,
    #         gap=rf_gap,
    #     )

    #     tl = ps << cpw_cell(
    #         length=modulation_length,
    #         cross_section=xs_cpw,
    #         signal_width=rf_central_conductor_width,
    #     )

    #     gap_eff = rf_gap + 2 * np.sum(
    #         [tl.cell.settings[key] for key in ("tt", "th") if key in tl.cell.settings]
    #     )

    #     tl.dmove(
    #         tl.ports["e1"].dcenter,
    #         (0.0, -0.5 * rf_central_conductor_width - 0.5 * gap_eff),
    #     )

    #     for name, port in [
    #         ("e1", tl.ports["bp1"]),
    #         ("e2", tl.ports["bp2"]),
    #     ]:
    #         ps.add_port(name=name, port=port)

    # ps.flatten()

    return ps


@gf.cell
def _mzm_interferometer_LT(
    splitter: ComponentSpec = build_mmi1x2_oband(),
    taper_length: float = 100.0,
    rib_core_width_modulator: float = 2.5,
    modulation_length: float = 7500.0,
    length_imbalance: float = 100.0,
    bias_tuning_section_length: float = 750.0,
    sbend_large_size: tuple[float, float] = (200.0, 50.0),
    sbend_small_size: tuple[float, float] = (200.0, -45.0),
    sbend_small_straight_extend: float = 5.0,
    lbend_tune_arm_reff: float = 75.0,
    lbend_combiner_reff: float = 80.0,
    communication_band: str = "O-band",
) -> gf.Component:
    interferometer = gf.Component()

    sbend_large = S_bend_vert(
        v_offset=sbend_large_size[1],
        h_extent=sbend_large_size[0],
        dx_straight=5.0,
        cross_section="xs_rwg700" if communication_band == "O-band" else "xs_rwg900",
    )

    sbend_small = S_bend_vert(
        v_offset=sbend_small_size[1],
        h_extent=sbend_small_size[0],
        dx_straight=sbend_small_straight_extend,
        cross_section="xs_rwg700" if communication_band == "O-band" else "xs_rwg900",
    )

    def branch_top():
        bt = gf.Component()
        sbend_1 = bt << sbend_large
        sbend_2 = bt << sbend_small

        pm = bt << eo_phase_shifter_LT(
            rib_core_width_modulator=rib_core_width_modulator,
            modulation_length=modulation_length,
            taper_length=taper_length,
            draw_cpw=False,
            communication_band=communication_band,
        )
        sbend_3 = bt << sbend_small
        sbend_2.connect("o1", sbend_1.ports["o2"])
        pm.connect("o1", sbend_2.ports["o2"])
        sbend_3.dmirror_x()
        sbend_3.connect("o1", pm.ports["o2"])

        for name, port in [
            ("o1", sbend_1.ports["o1"]),
            ("o2", sbend_3.ports["o2"]),
            ("taper_start", pm.ports["o1"]),
        ]:
            bt.add_port(name=name, port=port)
        bt.flatten()

        return bt

    def branch_tune_short(straight_unbalance: float = 0.0):
        arm = gf.Component()
        lbend = L_turn_bend(
            radius=lbend_tune_arm_reff,
            cross_section="xs_rwg700"
            if communication_band == "O-band"
            else "xs_rwg900",
        )
        straight_y = gf.components.straight(
            length=20.0 + straight_unbalance,
            cross_section="xs_rwg700"
            if communication_band == "O-band"
            else "xs_rwg900",
        )
        straight_x = gf.components.straight(
            length=bias_tuning_section_length,
            cross_section="xs_rwg700"
            if communication_band == "O-band"
            else "xs_rwg900",
        )
        symbol_to_component = {
            "b": (lbend, "o1", "o2"),
            "L": (straight_y, "o1", "o2"),
            "B": (lbend, "o2", "o1"),
            "_": (straight_x, "o1", "o2"),
        }
        sequence = "bLB_!b!L"
        arm = gf.components.component_sequence(
            sequence=sequence,
            ports_map={"phase_tuning_segment_start": ("_1", "o1")},
            symbol_to_component=symbol_to_component,
        )

        arm.add_port(port=arm.ports["phase_tuning_segment_start"])
        arm.flatten()
        return arm

    def branch_tune_long(straight_unbalance):
        return partial(branch_tune_short, straight_unbalance=straight_unbalance)()

    # splt = mmi1x2_oband() if communication_band == "O-band" else mmi1x2_cband()

    # Uniformly handle the cases of a 1x2 or 2x2 MMI
    splitter = gf.get_component(splitter)
    if len(splitter.ports) == 4:
        out_top = splitter.ports["o3"]
        out_bottom = splitter.ports["o4"]
    elif len(splitter.ports) == 3:
        out_top = splitter.ports["o2"]
        out_bottom = splitter.ports["o3"]
    else:
        raise ValueError(f"Splitter cell {splitter} not supported.")

    def combiner_section():
        comb_section = gf.Component()
        lbend_combiner = L_turn_bend(
            radius=lbend_combiner_reff,
            cross_section=xs_rwg700()
            if communication_band == "O-band"
            else xs_rwg900(),
        )
        lbend_top = comb_section << lbend_combiner
        lbend_bottom = comb_section << lbend_combiner
        lbend_bottom.dmirror_y()
        combiner = comb_section << splitter
        lbend_top.connect("o1", out_top)
        lbend_bottom.connect("o1", out_bottom)

        # comb_section.flatten()

        exposed_ports = [
            ("o2", lbend_top.ports["o2"]),
            ("o1", combiner.ports["o1"]),
            ("o3", lbend_bottom.ports["o2"]),
        ]
        if "2x2" in splitter.name:
            # if "2x2" in splitter:
            exposed_ports.append(
                ("in2", combiner.ports["o2"]),
            )

        for name, port in exposed_ports:
            comb_section.add_port(name=name, port=port)

        return comb_section

    splt_ref = interferometer << splitter
    bt = interferometer << branch_top()
    bb = interferometer << branch_top()
    bs = interferometer << branch_tune_short()
    bl = interferometer << branch_tune_long(abs(0.5 * length_imbalance))
    cs = interferometer << combiner_section()
    bb.dmirror_y()
    bt.connect("o1", out_top)
    bb.connect("o1", out_bottom)
    if length_imbalance >= 0:
        bs.dmirror_y()
        bs.connect("o1", bb.ports["o2"])
        bl.connect("o1", bt.ports["o2"])
    else:
        bs.connect("o1", bt.ports["o2"])
        bl.dmirror_y()
        bl.connect("o1", bb.ports["o2"])
    cs.dmirror_x()
    [
        cs.connect("o2", bl.ports["o2"])
        if length_imbalance >= 0
        else cs.connect("o2", bs.ports["o2"])
    ]

    exposed_ports = [
        ("o1", splt_ref.ports["o1"]),
        ("upper_taper_start", bt.ports["taper_start"]),
        ("short_bias_branch_start", bs.ports["phase_tuning_segment_start"]),
        ("long_bias_branch_start", bl.ports["phase_tuning_segment_start"]),
        ("o2", cs.ports["o1"]),
    ]

    # if "2x2" in splitter:
    if "2x2" in splitter.name:
        exposed_ports.extend(
            [
                ("out2", cs.ports["in2"]),
                ("in2", splt_ref.ports["o2"]),
            ]
        )

    for name, port in exposed_ports:
        interferometer.add_port(name=name, port=port)
    interferometer.flatten()

    return interferometer


@gf.cell
def mzm_unbalanced_LT(
    modulation_length: float = 7500.0,
    length_imbalance: float = 100.0,
    lbend_tune_arm_reff: float = 75.0,
    rf_pad_start_width: float = 80.0,
    rf_central_conductor_width: float = 21.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    cpw_cell: ComponentSpec = trail_cpw_LT,
    rib_core_width_modulator: float = 2.5,
    with_heater: bool = True,
    heater_on_both_branches: bool = False,
    heater_offset: float = 3.5,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    th: float = 1.5,
    tl: float = 44.7,
    tw: float = 7.0,
    tt: float = 1.5,
    tc: float = 5.0,
    communication_band: str = "O-band",
    ht_layer: Layer = LAYER.HRM,
    termination_layer: Layer = LAYER.HRM,
    terminated: bool = False,
    m2_layer: Layer = (22, 0),
    opening_layer: Layer = (40, 0),  # Layer for etching openings
    add_M2andOpenings: bool = False,  # Enable M2 metal and opening features
    resistor_length: float = 190.0 / 2,  # effective resistor length
    gsg_heater: bool = True,
    **kwargs,
) -> gf.Component:
    """Mach-Zehnder modulator based on the Pockels effect with an applied RF electric field.
    The modulator works in a differential push-pull configuration driven by a single GSG line.
    Communication band: O-band centered at 1310 nm and C-band centered at 1550 nm"""

    allowed_bands = ["C-band", "O-band"]
    if communication_band not in allowed_bands:
        raise ValueError(
            f"Invalid communication_band '{communication_band}'. Must be one of {allowed_bands}."
        )

    mzm = gf.Component()

    # Interferometer subcell
    if "splitter" not in kwargs.keys():
        kwargs["splitter"] = (
            "mmi1x2_oband" if communication_band == "O-band" else "mmi1x2_cband"
        )
        splitter = gf.get_component(kwargs["splitter"])

    if "2x2" in kwargs["splitter"]:
        splitter = gf.get_component(kwargs["splitter"])
    if ("oband" in kwargs["splitter"] and communication_band == "C-band") or (
        "cband" in kwargs["splitter"] and communication_band == "O-band"
    ):
        raise ValueError(
            "Communication band and designed band of splitter should be consistent!"
        )

    # splitter = gf.get_component(splitter) # TO DO use this function instead for versatility
    sbend_large_AR = 3.6

    gap_eff = rf_gap + 2 * (tt + th)  # changed as compared to LNOI400

    GS_separation = rf_pad_start_width * rf_gap / rf_central_conductor_width

    sbend_large_v_offset = (
        0.5 * rf_pad_start_width
        + 0.5 * GS_separation
        - 0.5 * splitter.settings["port_separation"]
    )

    sbend_small_straight_length = rf_pad_length_straight * 0.5

    lbend_combiner_reff = (
        0.5 * rf_pad_start_width
        + lbend_tune_arm_reff
        + 0.5 * GS_separation
        - 0.5 * splitter.settings["port_separation"]
    )
    interferometer = (
        mzm
        << partial(
            _mzm_interferometer_LT,
            rib_core_width_modulator=rib_core_width_modulator,
            modulation_length=modulation_length,
            length_imbalance=length_imbalance,
            sbend_large_size=(
                sbend_large_AR * sbend_large_v_offset,
                sbend_large_v_offset,
            ),
            sbend_small_size=(
                rf_pad_length_straight
                + rf_pad_length_tapered
                - 2 * sbend_small_straight_length,
                -0.5
                * (
                    rf_pad_start_width
                    - rf_central_conductor_width
                    + GS_separation
                    - gap_eff
                ),
            ),
            sbend_small_straight_extend=sbend_small_straight_length,
            lbend_tune_arm_reff=lbend_tune_arm_reff,
            lbend_combiner_reff=lbend_combiner_reff,
            bias_tuning_section_length=bias_tuning_section_length,
            communication_band=communication_band,
            **kwargs,
        )()
    )

    interferometer.dmove(
        interferometer.ports["upper_taper_start"].dcenter,
        (0.0, 0.5 * (rf_central_conductor_width + gap_eff)),
    )

    # Add heater for phase tuning
    if with_heater and not gsg_heater:
        ht_ref = mzm << add_heater(
            heater_on_both_branches=heater_on_both_branches,
            heater_offset=heater_offset,
            heater_width=heater_width,
            heater_pad_size=heater_pad_size,
            ht_layer=ht_layer,
            bias_tuning_section_length=bias_tuning_section_length,
            length_imbalance=length_imbalance,
            interferometer=interferometer,
        )
    if with_heater and gsg_heater:
        heater_pad_size = (40.0, 40.0)
        ht_ref = mzm << add_gsg_heater(
            heater_on_both_branches=heater_on_both_branches,
            heater_offset=heater_offset,
            heater_width=heater_width,
            heater_pad_size=heater_pad_size,
            ht_layer=ht_layer,
            bias_tuning_section_length=bias_tuning_section_length,
            length_imbalance=length_imbalance,
            interferometer=interferometer,
            m2_layer=m2_layer,
        )
    # Transmission line subcell

    xs_cpw = gf.partial(
        xs_uni_cpw,
        central_conductor_width=rf_central_conductor_width,
        ground_planes_width=rf_ground_planes_width,
        gap=rf_gap,
    )

    rf_line = mzm << cpw_cell(
        bondpad={
            "component": "CPW_pad_curved",
            "settings": {
                "start_width": rf_pad_start_width,
                "length_straight": rf_pad_length_straight,
                "length_tapered": rf_pad_length_tapered,
            },
        },
        length=modulation_length,
        signal_width=rf_central_conductor_width,
        cross_section=xs_cpw,
        ground_planes_width=rf_ground_planes_width,
        gap_width=rf_gap,
        th=th,
        tl=tl,
        tw=tw,
        tt=tt,
        tc=tc,
        sbend_small_size=(
            rf_pad_length_straight
            + rf_pad_length_tapered
            - 2 * sbend_small_straight_length,
            -0.5
            * (
                rf_pad_start_width
                - rf_central_conductor_width
                + GS_separation
                - gap_eff
            ),
        ),
        sbend_small_straight_extend=sbend_small_straight_length,
        terminated=terminated,
        termination_layer=termination_layer,
        m2_layer=m2_layer,
        opening_layer=opening_layer,
        add_M2andOpenings=add_M2andOpenings,
        resistor_length=resistor_length,
    )

    rf_line.dmove(rf_line.ports["e1"].dcenter, (0.0, 0.0))

    # Expose the ports

    exposed_ports = [
        ("e1", rf_line.ports["bp1"]),
    ]
    if not terminated:
        exposed_ports += [
            ("e2", rf_line.ports["bp2"]),
        ]

    if "1x2" in kwargs["splitter"]:
        exposed_ports.extend(
            [
                ("o1", interferometer.ports["o1"]),
                ("o2", interferometer.ports["o2"]),
            ]
        )
    elif "2x2" in kwargs["splitter"]:
        exposed_ports.extend(
            [
                ("o1", interferometer.ports["o1"]),
                ("o2", interferometer.ports["in2"]),
                ("o3", interferometer.ports["out2"]),
                ("o4", interferometer.ports["o2"]),
            ]
        )

    if with_heater and not gsg_heater:
        exposed_ports += [
            ("e3", ht_ref.ports["e1"]),
            (
                "e4",
                ht_ref.ports["e2"],
            ),
        ]
    if with_heater and heater_on_both_branches and not gsg_heater:
        exposed_ports += [
            ("e5", ht_ref.ports["e3"]),
            (
                "e6",
                ht_ref.ports["e4"],
            ),
        ]
    if with_heater and heater_on_both_branches and gsg_heater:
        exposed_ports += [
            ("gsg_e1", ht_ref.ports["gsg_e1"]),
            ("gsg_e2", ht_ref.ports["gsg_e2"]),
            ("gsg_e3", ht_ref.ports["gsg_e3"]),
            ("gsg_e4", ht_ref.ports["gsg_e4"]),
        ]
    [mzm.add_port(name=name, port=port) for name, port in exposed_ports]

    return mzm
