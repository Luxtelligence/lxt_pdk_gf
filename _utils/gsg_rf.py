import warnings
from typing import Any

import gdsfactory as gf
import numpy as np
from gdsfactory.cross_section import (
    CrossSection,
)
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from _utils.cell_info import copy_info
from _utils.cross_section import get_cpw_from_xs, xs_cpw_single_layer


def _cpw_aspect_ratio(width: float, gap: float) -> float:
    """Aspect ratio AR = width / (width + 2*gap) for impedance matching."""
    return width / (width + 2 * gap)


def _validate_aspect_ratio(
    pad_xs: CrossSection,
    cpw_xs: CrossSection,
    rtol: float = 1e-4,
) -> float:
    """Validate that pad and CPW have matching aspect ratio. Returns AR."""
    pad_width, _, pad_gap, _ = get_cpw_from_xs(pad_xs)
    cpw_width, _, cpw_gap, _ = get_cpw_from_xs(cpw_xs)
    ar_pad = _cpw_aspect_ratio(pad_width, pad_gap)
    ar_cpw = _cpw_aspect_ratio(cpw_width, cpw_gap)
    if not np.isclose(ar_pad, ar_cpw, rtol=rtol):
        raise ValueError(
            f"Aspect ratio mismatch: pad AR={ar_pad:.6f}, CPW AR={ar_cpw:.6f}. "
            "pad_xs and cpw_xs must preserve the same aspect ratio."
        )
    return ar_pad


def _pad_metal_polygon_points(
    length_straight: float,
    length_tapered: float,
    pad_xs: CrossSection,
    cpw_xs: CrossSection,
) -> tuple[tuple[float, float], ...]:
    """Create 6-point polygon outlining the full CPW metal region (both grounds + center).
    Points trace outer boundary: pad end -> straight end -> cpw end, then mirror by x-axis.
    """
    pad_width, pad_ground_width, pad_gap, pad_layer = get_cpw_from_xs(pad_xs)
    cpw_width, cpw_ground_width, cpw_gap, cpw_layer = get_cpw_from_xs(cpw_xs)

    y_pad_outer = pad_width / 2 + pad_gap + pad_ground_width
    y_cpw_outer = cpw_width / 2 + cpw_gap + cpw_ground_width

    x_end = length_straight + length_tapered

    points = [
        (0.0, y_pad_outer),
        (length_straight, y_pad_outer),
        (x_end, y_cpw_outer),
        (x_end, -y_cpw_outer),
        (length_straight, -y_pad_outer),
        (0.0, -y_pad_outer),
    ]
    return points


def _spline_bend_points(
    start: tuple[float, float],
    end: tuple[float, float],
    npoints: int,
) -> np.ndarray:
    """Smooth S-bend using Hermite-like interpolation (t^2)*(3-2*t)."""
    t = np.linspace(0, 1, npoints)
    xs = start[0] + (end[0] - start[0]) * t
    ys = start[1] + (end[1] - start[1]) * (t**2) * (3 - 2 * t)
    return np.column_stack([xs, ys])


def bonding_pads_exclusion_path(
    length_tapered: float,
    length_straight: float,
    pad_xs: CrossSection,
    cpw_xs: CrossSection,
) -> tuple[gf.Path, gf.Path, gf.Section, gf.Section]:
    """Parametric S-bend paths and width functions for CPW bonding pad slots.

    Returns:
        path_upper: gf.Path for upper waveguide/center-line (y >= 0)
        path_lower: gf.Path for lower waveguide/center-line (y <= 0)
        section_gap_upper: gf.Section for upper gap extrusion with parametric width
        section_gap_lower: gf.Section for lower gap extrusion with parametric width
    """
    AR = _validate_aspect_ratio(pad_xs, cpw_xs)

    pad_width, _, pad_gap, pad_layer = get_cpw_from_xs(pad_xs)
    cpw_width, _, cpw_gap, tl_layer = get_cpw_from_xs(cpw_xs)

    # Round to grid: ensure all parameters are grid-aligned
    # Use 0.004 µm (4 DBU) to ensure division by 2 stays grid-aligned
    grid_size = 0.004  # µm
    pad_width = round(pad_width / grid_size) * grid_size
    pad_gap = round(pad_gap / grid_size) * grid_size
    cpw_width = round(cpw_width / grid_size) * grid_size
    cpw_gap = round(cpw_gap / grid_size) * grid_size

    # Waveguide center positions: between central conductor and ground
    y_start_upper = pad_width / 2 + pad_gap / 2
    y_end_upper = cpw_width / 2 + cpw_gap / 2

    npoints = int(np.round(2.5 * length_tapered))

    # Upper path: straight (0, y_start) -> (length_straight, y_start), then S-bend to (length_straight+length_tapered, y_end)
    straight_upper = np.array(
        [
            [0.0, y_start_upper],
            [length_straight, y_start_upper],
        ]
    )
    bend_upper = _spline_bend_points(
        (length_straight, y_start_upper),
        (length_straight + length_tapered, y_end_upper),
        npoints,
    )
    points_upper = np.vstack([straight_upper, bend_upper[1:]])  # skip duplicate
    points_lower = points_upper.copy()
    points_lower[:, 1] = -points_lower[:, 1]

    path_upper = gf.Path(points_upper)
    path_upper.start_angle = path_upper.end_angle = 0.0
    path_lower = gf.Path(points_lower)
    path_lower.start_angle = path_lower.end_angle = 0.0

    # Calculate total path length and segment boundaries
    total_length = length_straight + length_tapered
    t_straight_end = (
        length_straight / total_length
    )  # t value where straight section ends

    def gap_width_upper(t):
        """Width function for upper gap extrusion: gap(y) = 2*|y|*AR/(AR+1)"""
        if np.isscalar(t):
            t = np.array([t])
            scalar_input = True
        else:
            scalar_input = False

        # Map t to y-coordinates along the path
        y_coords = np.zeros_like(t)

        # For straight section (t < t_straight_end): y = y_start_upper
        mask_straight = t <= t_straight_end
        y_coords[mask_straight] = y_start_upper

        # For bend section (t >= t_straight_end): interpolate along the bend
        mask_bend = t > t_straight_end
        if np.any(mask_bend):
            # Map t to normalized bend parameter (0 to 1 within bend)
            t_bend_norm = (t[mask_bend] - t_straight_end) / (1.0 - t_straight_end)
            # Hermite-like interpolation for y-coordinate
            y_coords[mask_bend] = y_start_upper + (y_end_upper - y_start_upper) * (
                t_bend_norm**2
            ) * (3 - 2 * t_bend_norm)

        # Calculate gap width: gap(y) = 2*|y|*(1-AR)/(AR+1)
        gap_widths = 2 * np.abs(y_coords) * (1 - AR) / (AR + 1)

        # Round to grid: widths must be multiples of 0.002 µm (2 DBU)
        # Use integer arithmetic to avoid floating-point precision issues
        grid_size = 0.002  # µm
        gap_widths_dbu = np.round(
            gap_widths / grid_size
        )  # Convert to DBU units (integer)
        gap_widths = gap_widths_dbu * grid_size  # Convert back to µm

        return gap_widths[0] if scalar_input else gap_widths

    def gap_width_lower(t):
        """Width function for lower gap extrusion: same as upper (uses |y|)"""
        # Since we use |y|, the lower path has the same gap width profile
        return gap_width_upper(t)

    # Create sections with parametric width functions
    section_gap_upper = gf.Section(
        layer=tl_layer,
        width=0,
        width_function=gap_width_upper,
        port_names=("o1", "o2"),
    )

    section_gap_lower = gf.Section(
        layer=tl_layer,
        width=0,
        width_function=gap_width_lower,
        port_names=("o1", "o2"),
    )

    return path_upper, path_lower, section_gap_upper, section_gap_lower


@gf.cell
def via_array(
    cpw_xs: CrossSection,
    layer_openings: LayerSpec,
    layer_m2: LayerSpec,
    opening_offset: float = 2.5,
    opening_size: float = 12.0,
    separation: float = 12.0,
    width: float = 45.0,
) -> gf.Component:
    """CPW landing region with opening array for via/electroplating.

    The cell builds short CPW host rectangles on the CPW layer and places a square
    opening array on top of each conductor.
    """
    signal_section = next(
        (section for section in cpw_xs.sections if "signal" in section.name.lower()),
        cpw_xs.sections[0],
    )
    ground_sections = [
        section for section in cpw_xs.sections if "ground" in section.name.lower()
    ]
    ground_section = ground_sections[0] if ground_sections else cpw_xs.sections[0]

    signal_width = signal_section.width
    ground_width = ground_section.width
    signal_layer = signal_section.layer
    gap = abs(ground_section.offset) - 0.5 * (signal_width + ground_width)

    # Round to grid to keep symmetric geometry on-db.
    grid_size = 0.004  # um
    signal_width = round(signal_width / grid_size) * grid_size
    ground_width = round(ground_width / grid_size) * grid_size
    gap = round(gap / grid_size) * grid_size

    signal_center_y = 0.0
    ground_upper_center_y = signal_width / 2 + gap + ground_width / 2
    ground_lower_center_y = -ground_upper_center_y
    conductor_center_x = -width / 2

    c = gf.Component()

    for conductor_name, conductor_width, conductor_center_y in (
        ("signal", signal_width, signal_center_y),
        ("ground_upper", ground_width, ground_upper_center_y),
        ("ground_lower", ground_width, ground_lower_center_y),
    ):
        host_rect = gf.components.rectangle(
            size=(width, conductor_width),
            layer=signal_layer,
            centered=True,
        )
        host_ref = c << host_rect
        host_ref.dmove(host_ref.dcenter, (conductor_center_x, conductor_center_y))

        trgt_rect = gf.components.rectangle(
            size=(width, conductor_width),
            layer=layer_m2,
            centered=True,
        )
        trgt_ref = c << trgt_rect
        trgt_ref.dmove(trgt_ref.dcenter, (conductor_center_x, conductor_center_y))

        available_height = conductor_width - 2 * opening_offset
        available_width = width - 2 * opening_offset
        num_openings_vertical = int(
            (available_height + separation) / (opening_size + separation)
        )
        if num_openings_vertical < 1:
            if available_width > 0 and available_height > 0:
                opening_rect = gf.components.rectangle(
                    size=(available_width, available_height),
                    layer=layer_openings,
                    centered=True,
                )
                opening_ref = c << opening_rect
                opening_ref.dmove(
                    opening_ref.dcenter, (conductor_center_x, conductor_center_y)
                )
                warnings.warn(
                    (
                        f"via_array: '{conductor_name}' conductor cannot fit via rows "
                        f"(num_rows={num_openings_vertical}); using single solid opening."
                    ),
                    stacklevel=2,
                )
            continue

        num_columns = int((available_width + separation) / (opening_size + separation))
        if num_columns < 1:
            opening_width = available_width
            opening_height = available_height
            if opening_width > 0 and opening_height > 0:
                opening_rect = gf.components.rectangle(
                    size=(opening_width, opening_height),
                    layer=layer_openings,
                    centered=True,
                )
                opening_ref = c << opening_rect
                opening_ref.dmove(
                    opening_ref.dcenter, (conductor_center_x, conductor_center_y)
                )
                warnings.warn(
                    (
                        f"via_array: '{conductor_name}' conductor cannot fit via columns "
                        f"(num_columns={num_columns}); using single solid opening."
                    ),
                    stacklevel=2,
                )
            continue

        total_columns_width = (
            num_columns * opening_size + (num_columns - 1) * separation
        )
        total_openings_height = (
            num_openings_vertical * opening_size
            + (num_openings_vertical - 1) * separation
        )
        start_x = conductor_center_x - total_columns_width / 2 + opening_size / 2
        start_y = conductor_center_y - total_openings_height / 2 + opening_size / 2

        for col in range(num_columns):
            x_position = start_x + col * (opening_size + separation)
            for row in range(num_openings_vertical):
                y_position = start_y + row * (opening_size + separation)
                opening_rect = gf.components.rectangle(
                    size=(opening_size, opening_size),
                    layer=layer_openings,
                    centered=True,
                )
                opening_ref = c << opening_rect
                opening_ref.dmove(opening_ref.dcenter, (x_position, y_position))

    # Round only via openings; keep host/pad metals rectangular.
    c_rounded = gf.Component()
    rinner = 1000  # circle radius of inner corners (dbu)
    router = 1000  # circle radius of outer corners (dbu)
    n = 300  # number of points per full circle
    for layer, polygons in c.get_polygons().items():
        for polygon in polygons:
            if layer == layer_openings:
                polygon = polygon.round_corners(rinner, router, n)
            c_rounded.add_polygon(polygon, layer=layer)

    # Port at central conductor center (host region), on CPW signal layer.
    c_rounded.add_port(
        name="e1",
        center=(conductor_center_x - width / 2, signal_center_y),
        width=signal_width,
        orientation=180.0,
        port_type="electrical",
        layer=signal_layer,
    )
    c_rounded.add_port(
        name="e2",
        center=(conductor_center_x + width / 2, signal_center_y),
        width=signal_width,
        orientation=0.0,
        port_type="electrical",
        layer=layer_m2,
    )

    return c_rounded


@gf.cell
def via_solid(
    cpw_xs: CrossSection,
    layer_openings: LayerSpec,
    layer_m2: LayerSpec,
    opening_offset: float = 2.5,
    width: float = 45.0,
) -> gf.Component:
    """CPW landing region with solid rounded openings.

    Similar to `via_array`, but each conductor receives one solid opening rectangle
    inset by `opening_offset` from all sides.
    """
    signal_section = next(
        (section for section in cpw_xs.sections if "signal" in section.name.lower()),
        cpw_xs.sections[0],
    )
    ground_sections = [
        section for section in cpw_xs.sections if "ground" in section.name.lower()
    ]
    ground_section = ground_sections[0] if ground_sections else cpw_xs.sections[0]

    signal_width = signal_section.width
    ground_width = ground_section.width
    signal_layer = signal_section.layer
    gap = abs(ground_section.offset) - 0.5 * (signal_width + ground_width)

    # Round to grid to keep symmetric geometry on-db.
    grid_size = 0.004  # um
    signal_width = round(signal_width / grid_size) * grid_size
    ground_width = round(ground_width / grid_size) * grid_size
    gap = round(gap / grid_size) * grid_size

    signal_center_y = 0.0
    ground_upper_center_y = signal_width / 2 + gap + ground_width / 2
    ground_lower_center_y = -ground_upper_center_y
    conductor_center_x = -width / 2

    c = gf.Component()
    rinner = 1000  # circle radius of inner corners (dbu)
    router = 1000  # circle radius of outer corners (dbu)
    n = 300  # number of points per full circle

    for conductor_width, conductor_center_y in (
        (signal_width, signal_center_y),
        (ground_width, ground_upper_center_y),
        (ground_width, ground_lower_center_y),
    ):
        host_rect = gf.components.rectangle(
            size=(width, conductor_width),
            layer=signal_layer,
            centered=True,
        )
        host_ref = c << host_rect
        host_ref.dmove(host_ref.dcenter, (conductor_center_x, conductor_center_y))

        trgt_rect = gf.components.rectangle(
            size=(width, conductor_width),
            layer=layer_m2,
            centered=True,
        )
        trgt_ref = c << trgt_rect
        trgt_ref.dmove(trgt_ref.dcenter, (conductor_center_x, conductor_center_y))

        opening_width = width - 2 * opening_offset
        opening_height = conductor_width - 2 * opening_offset
        if opening_width <= 0 or opening_height <= 0:
            continue

        opening_rect = gf.components.rectangle(
            size=(opening_width, opening_height),
            layer=layer_openings,
            centered=True,
        )
        opening_ref = c << opening_rect
        opening_ref.dmove(opening_ref.dcenter, (conductor_center_x, conductor_center_y))

    c_rounded = gf.Component()
    for layer, polygons in c.get_polygons().items():
        for polygon in polygons:
            if layer == layer_openings:
                polygon = polygon.round_corners(rinner, router, n)
            c_rounded.add_polygon(polygon, layer=layer)

    # Match via_array external port convention.
    c_rounded.add_port(
        name="e1",
        center=(conductor_center_x - width / 2, signal_center_y),
        width=signal_width,
        orientation=180.0,
        port_type="electrical",
        layer=signal_layer,
    )
    c_rounded.add_port(
        name="e2",
        center=(conductor_center_x + width / 2, signal_center_y),
        width=signal_width,
        orientation=0.0,
        port_type="electrical",
        layer=layer_m2,
    )

    return c_rounded


@gf.cell
def m2_bonding_pads(
    pad_xs: CrossSection,
    layer_m2: LayerSpec,
    layer_openings: LayerSpec,
    m1_opening_offset: float = 2.5,
    opening_size: float = 12.0,
    opening_separation: float = 12.0,
    tl_opening_host_width: float = 45.0,
    m2_pad_length: float = 80.0,
) -> gf.Component:
    """Create M2 metal layer bonding pads with openings for electroplating.

    Creates three M2 rectangles (central conductor + 2 ground planes) with
    via openings matching the CPW pad cross-section parameters.

    Args:
        pad_xs_params: Tuple of (central_width, ground_width, gap) defining pad geometry
        layer_M2: Layer for M2 metal
        layer_Openings: Layer for via openings
        m1_opening_offset: Offset from M1 edge to first opening (µm)
        opening_size: Size of square opening (µm)
        opening_separation: Separation between openings (µm)
        tl_opening_host_width: Width of the transmission line extension hosting openings (µm)
        m2_pad_length: Length of M2 pad along x-axis (µm)

    Returns:
        Component with M2 pads and openings centered at origin
    """

    pad_width, pad_ground_width, pad_gap, tl_layer = get_cpw_from_xs(pad_xs)

    # Round to grid: widths must be multiples of 0.004 µm (4 DBU) for symmetrical cross-sections
    # This ensures that division by 2 (used in position calculations) stays grid-aligned
    grid_size = 0.004  # µm
    pad_width = round(pad_width / grid_size) * grid_size
    pad_ground_width = round(pad_ground_width / grid_size) * grid_size
    pad_gap = round(pad_gap / grid_size) * grid_size

    c = gf.Component()

    m2_pad_xs = xs_cpw_single_layer(
        central_conductor_width=pad_width,
        ground_planes_width=pad_ground_width,
        gap=pad_gap,
        layer=layer_m2,
    )

    m2_pad_ref = c << gf.components.straight(
        length=m2_pad_length, cross_section=m2_pad_xs
    )

    tl_pad_xs = xs_cpw_single_layer(
        central_conductor_width=pad_width,
        ground_planes_width=pad_ground_width,
        gap=pad_gap,
        layer=tl_layer,
    )
    via_ref = c << via_array(
        cpw_xs=tl_pad_xs,
        layer_openings=layer_openings,
        layer_m2=layer_m2,
        opening_offset=m1_opening_offset,
        opening_size=opening_size,
        separation=opening_separation,
        width=tl_opening_host_width,
    )

    m2_pad_ref.connect("e2", via_ref.ports["e2"])

    # Add port at the right edge for connection to CPW pad
    c.add_port(name="e1", port=m2_pad_ref.ports["e1"])

    # Add port at the right edge for connection to CPW pad
    c.add_port(name="e2", port=via_ref.ports["e1"])

    # Add port at the right edge for connection to CPW pad
    c.add_port(
        name="e3",
        center=(0.0, 0.0),
        width=pad_width,
        orientation=0.0,
        port_type="electrical",
        layer=layer_m2,
    )

    return c


@gf.cell
def termination_wire(
    cpw_xs: CrossSection,
    termination_layer: LayerSpec,
    effective_length: float = 48.5,
    resistor_width: float = 1.5,
    hr_layer_offset: float = 2.5,
    pad_length: float = 20.0,
) -> gf.Component:
    """Termination wire for CPW lines.

    Args:
        cpw_xs: CPW cross-section of the line to be terminated
        termination_layer: High-resistivity layer of the termination wire
        effective_length: Effective length of the termination wire corresponding to a single termination resistor equivalent circuit.
        resistor_width: Width of the resistor
        hr_layer_offset: Offset of the high-resistivity layer pads
        pad_length: Length of the high-resistivity layer pad for wire connection
    """

    signal_width, ground_width, gap_width, signal_layer = get_cpw_from_xs(cpw_xs)

    y = signal_width + gap_width

    x = (2 * effective_length - y) / 3

    if x < 10.0:
        x = 10.0
        y = 2 * effective_length - 3 * x

    if y < signal_width / 2 + gap_width + hr_layer_offset:
        raise ValueError("length is too short to fit the termination wire")

    wire = gf.Component()
    straight_section_shape = [
        (0, resistor_width / 2),
        (0, -resistor_width / 2),
        (x, -resistor_width / 2),
        (x, resistor_width / 2),
    ]

    return_section_shape = [
        (0, y + resistor_width / 2),
        (x + resistor_width / 2, y + resistor_width / 2),
        (x + resistor_width / 2, 0),
        (x - resistor_width / 2, 0),
        (x - resistor_width / 2, y - resistor_width / 2),
        (0, y - resistor_width / 2),
    ]

    return_shape_mirror = [(x, -y) for x, y in return_section_shape]

    wire.add_polygon(straight_section_shape, layer=termination_layer)
    wire.add_polygon(return_section_shape, layer=termination_layer)
    wire.add_polygon(return_shape_mirror, layer=termination_layer)

    pad_xs = xs_cpw_single_layer(
        central_conductor_width=signal_width - hr_layer_offset * 2,
        ground_planes_width=ground_width - hr_layer_offset * 2,
        gap=gap_width + hr_layer_offset * 2,
        layer=termination_layer,
    )

    pad_ref = wire << gf.components.straight(length=pad_length, cross_section=pad_xs)
    pad_ref.dmove(origin=pad_ref.ports["e2"].center, destination=(0, 0))

    wire.flatten()

    c = gf.Component()
    rinner = 1000  # circle radius of inner corners (dbu)
    router = 1000  # circle radius of outer corners (dbu)
    n = 300  # number of points per full circle

    for layer, polygons in wire.get_polygons().items():
        for polygon in polygons:
            c.add_polygon(polygon.round_corners(rinner, router, n), layer=layer)

    c.add_port(
        name="e1",
        center=(-pad_length / 2, 0.0),
        cross_section=pad_xs,
        orientation=180.0,
        port_type="electrical",
    )

    c.add_port(
        name="term",
        center=(x, 0),
        width=resistor_width,
        orientation=0,
        port_type="electrical",
        layer=signal_layer,
    )

    return c


@gf.cell
def double_layer_termination(
    cpw_xs: CrossSection,
    termination_layer: LayerSpec,
    m2_layer: LayerSpec,
    m2_pad_length: float = 10.0,
    termination_params: dict[str, Any] | None = None,
    via_m1_m2_params: dict[str, Any] | None = None,
    via_m2_hr_params: dict[str, Any] | None = None,
) -> gf.Component:
    """Double-layer termination for CPW lines. Creates transition from CPW to M2 layer and then to high-resistivity layer resistor wire.

    Args:
        cpw_xs: CPW cross-section of the line to be terminated
        termination_layer: High-resistivity layer of the termination wire
        m2_layer: M2 layer of the termination wire
        effective_length: Effective length of the termination wire corresponding to a single termination resistor equivalent circuit.
    """
    if termination_params is None:
        termination_params = {
            "effective_length": 95.0,
            "resistor_width": 1.5,
            "hr_layer_offset": 0.0,
            "hr_pad_length": 5.0,
        }
    if via_m1_m2_params is None:
        via_m1_m2_params = {
            "type": "array",
            "layer_openings": (40, 0),
            "opening_offset": 2.5,
            "opening_size": 12.0,
            "opening_separation": 12.0,
            "width": 45.0,
        }
    if via_m2_hr_params is None:
        via_m2_hr_params = {
            "type": "solid",
            "layer_openings": (41, 0),
            "opening_offset": 2.5,
            "opening_size": 12.0,
            "opening_separation": 12.0,
            "width": 20.0,
        }

    c = gf.Component()

    signal_width, ground_width, gap_width, signal_layer = get_cpw_from_xs(cpw_xs)

    wire = c << termination_wire(
        cpw_xs=cpw_xs,
        termination_layer=termination_layer,
        effective_length=termination_params["effective_length"],
        resistor_width=termination_params["resistor_width"],
        hr_layer_offset=termination_params["hr_layer_offset"],
        pad_length=termination_params["hr_pad_length"],
    )

    termination_xs = xs_cpw_single_layer(
        central_conductor_width=signal_width,
        ground_planes_width=ground_width,
        gap=gap_width,
        layer=termination_layer,
    )

    m2_pad_xs = xs_cpw_single_layer(
        central_conductor_width=signal_width,
        ground_planes_width=ground_width,
        gap=gap_width,
        layer=m2_layer,
    )

    m2_pad_ref = c << gf.components.straight(
        length=m2_pad_length, cross_section=m2_pad_xs
    )

    if via_m1_m2_params["type"] == "array":
        via_m1_m2 = c << via_array(
            cpw_xs=cpw_xs,
            layer_openings=via_m1_m2_params["layer_openings"],
            layer_m2=m2_layer,
            opening_offset=via_m1_m2_params["opening_offset"],
            opening_size=via_m1_m2_params["opening_size"],
            separation=via_m1_m2_params["opening_separation"],
            width=via_m1_m2_params["width"],
        )
    elif via_m1_m2_params["type"] == "solid":
        via_m1_m2 = c << via_solid(
            cpw_xs=cpw_xs,
            layer_openings=via_m1_m2_params["layer_openings"],
            layer_m2=m2_layer,
            opening_offset=via_m1_m2_params["opening_offset"],
            width=via_m1_m2_params["width"],
        )
    else:
        raise ValueError(f"Invalid via type: {via_m1_m2_params['type']}")

    if via_m2_hr_params["type"] == "array":
        via_m2_hr = c << via_array(
            cpw_xs=termination_xs,
            layer_openings=via_m2_hr_params["layer_openings"],
            layer_m2=m2_layer,
            opening_offset=via_m2_hr_params["opening_offset"],
            opening_size=via_m2_hr_params["opening_size"],
            separation=via_m2_hr_params["opening_separation"],
            width=via_m2_hr_params["width"],
        )
    elif via_m2_hr_params["type"] == "solid":
        via_m2_hr = c << via_solid(
            cpw_xs=termination_xs,
            layer_openings=via_m2_hr_params["layer_openings"],
            layer_m2=m2_layer,
            opening_offset=via_m2_hr_params["opening_offset"],
            width=via_m2_hr_params["width"],
        )
    else:
        raise ValueError(f"Invalid via type: {via_m2_hr_params['type']}")

    m2_pad_ref.connect("e1", via_m1_m2.ports["e2"])

    via_m2_hr.connect("e2", m2_pad_ref.ports["e2"])
    wire.connect("e1", via_m2_hr.ports["e1"])

    c.add_port(name="e1", port=via_m1_m2.ports["e1"])
    c.add_port(name="term", port=wire.ports["term"])

    c.flatten()

    return c


def get_pad_xs(
    cpw_xs: CrossSection,
    pitch: float,
    ground_pad_width: float,
) -> CrossSection:
    """Compute the pad-side CPW cross-section from the terminal CPW cross-section.

    The pad maintains a fixed gap/conductor aspect ratio to achieve impedance
    matching, scaled to the given probe pitch.

    Args:
        cpw_xs: terminal CPW cross-section.
        pitch: probe pitch (distance between optical waveguide entries).
        ground_pad_width: width of the ground pad.
    """
    end_width, _, end_gap, tl_layer = get_cpw_from_xs(cpw_xs)
    aspect_ratio = end_width / (end_width + 2 * end_gap)

    pad_width = pitch * 2 * aspect_ratio / (1 + aspect_ratio)
    pad_gap = pitch * (1 - aspect_ratio) / (1 + aspect_ratio)

    # Round to grid: widths must be multiples of 0.004 µm (4 DBU) for symmetrical cross-sections
    # This ensures that division by 2 (used in path calculations) stays grid-aligned
    grid_size = 0.004  # µm (must be even multiple of DBU for division by 2)

    grid_size = 0.004
    pad_width = round(pad_width / grid_size) * grid_size
    pad_gap = round(pad_gap / grid_size) * grid_size

    return xs_cpw_single_layer(
        central_conductor_width=pad_width,
        ground_planes_width=ground_pad_width,
        gap=pad_gap,
        layer=tl_layer,
    )


def gsg_pad_curved(
    cpw_xs: CrossSection,
    pitch: float,
    length_tapered: float,
    length_straight: float,
    ground_pad_width: float,
) -> gf.Component:
    """Curved pad for CPW lines with curved electrodes following the optical waveguides.
    Helpful for on-slab metalization to avoid additional optical loss.
    The pad maintains a fixed gap/conductor aspect ratio to achieve impedance
    matching, scaled to the given probe pitch.

    Args:
        cpw_xs: CPW cross-section of the line to be terminated
        pitch: probe pitch (distance between optical waveguide entries).
        length_tapered: length of the tapered section of the pad. This will also define optical waveguide curvature depending on the pitch and CPW cross-section.
        length_straight: length of the wide straight section of the pad.
        ground_pad_width: width of the ground pad.
    Returns:
        pad: gf.Component of the pad.
        pad_xs: gf.CrossSection of the pad.
        path_upper: gf.Path of the upper path.
        path_lower: gf.Path of the lower path.
    """

    end_width, ground_planes_width, end_gap, tl_layer = get_cpw_from_xs(cpw_xs)

    grid_size = 0.004
    end_width = round(end_width / grid_size) * grid_size
    end_gap = round(end_gap / grid_size) * grid_size
    ground_planes_width = round(ground_planes_width / grid_size) * grid_size

    pad_xs = get_pad_xs(cpw_xs=cpw_xs, pitch=pitch, ground_pad_width=ground_pad_width)

    # New parametric approach: polygon + path cut + waveguide extrude + gap extrusions
    (
        path_upper,
        path_lower,
        section_gap_upper,
        section_gap_lower,
    ) = bonding_pads_exclusion_path(
        length_tapered=length_tapered,
        length_straight=length_straight,
        pad_xs=pad_xs,
        cpw_xs=cpw_xs,
    )

    # 1. Create upper and lower ground polygons (trapezoids: outer - slot boundary)
    full_pts = _pad_metal_polygon_points(
        length_straight, length_tapered, pad_xs, cpw_xs
    )

    pad_base = gf.Component()
    pad_base.add_polygon(full_pts, layer=tl_layer)

    # 2. Create parametric gap extrusions along the paths
    xs_gap_upper = gf.CrossSection(sections=(section_gap_upper,))
    xs_gap_lower = gf.CrossSection(sections=(section_gap_lower,))

    gap_extrusion_upper = path_upper.extrude(cross_section=xs_gap_upper)
    gap_extrusion_lower = path_lower.extrude(cross_section=xs_gap_lower)

    # 3. Perform boolean subtraction: pad_base - gap_extrusions
    # First subtract upper gap extrusion
    pad_with_upper_cut = gf.boolean(
        A=pad_base,
        B=gap_extrusion_upper,
        operation="not",
        layer=tl_layer,
    )

    # Then subtract lower gap extrusion from the result
    pad_final = gf.boolean(
        A=pad_with_upper_cut,
        B=gap_extrusion_lower,
        operation="not",
        layer=tl_layer,
    )

    c = gf.Component()
    c << pad_final
    c.add_port(
        name="e2",
        cross_section=cpw_xs,
        orientation=0.0,
        center=(length_straight + length_tapered, 0.0),
        port_type="electrical",
    )
    c.add_port(
        name="e1",
        cross_section=pad_xs,
        orientation=180.0,
        center=(0.0, 0.0),
        port_type="electrical",
    )

    return c, pad_xs, path_upper, path_lower


@gf.cell
def cpw_pad(
    cpw_xs: CrossSectionSpec,
    pitch: float = 100.0,
    length_straight: float = 25.0,
    length_tapered: float = 190.0,
    ground_pad_width: float = 150.0,
    optical_waveguide_xs: CrossSectionSpec | None = None,
    m2_bonding_pads_params: dict[str, Any] | None = None,
    single_waveguide: bool = False,
) -> gf.Component:
    """RF access line for high-frequency GSG probes with curved electrodes
    following the optical waveguides. The probe pad maintains a
    fixed gap/central conductor ratio across its length, to achieve a good
    impedance matching.

    Args:
        cpw_xs: terminal CPW cross-section to connect to the pad. Should contain "signal" and "ground" sections.
        pitch: probe pitch - distance between optical waveguide entries.
        length_straight: length of the wide straight section of the pad.
        length_tapered: length of the tapered section of the pad. This will also define optical waveguide curvature depending on the pitch and CPW cross-section.
        ground_pad_width: width of the ground pad.
        optical_waveguide_xs: optical waveguide cross-section.
        m2_bonding_pads_params: optional parameters for `m2_bonding_pads`.
            When provided, `layer_M2` and `layer_Openings` are required keys.
            Other keys are optional and default to `m2_bonding_pads` defaults.
    """

    pad = gf.Component()

    tl_pad, pad_xs, path_upper, path_lower = gsg_pad_curved(
        cpw_xs=cpw_xs,
        pitch=pitch,
        length_tapered=length_tapered,
        length_straight=length_straight,
        ground_pad_width=ground_pad_width,
    )
    p1 = pad << tl_pad

    # 3. Waveguides along path_upper and path_lower (fixed waveguide_xs)
    if optical_waveguide_xs is not None:
        wg_upper = path_upper.extrude(optical_waveguide_xs)
        pad << wg_upper
        pad.add_port(name="o1", port=wg_upper.ports["o1"])
        pad.add_port(name="o2", port=wg_upper.ports["o2"])

        if not single_waveguide:
            wg_lower = path_lower.extrude(optical_waveguide_xs)
            pad << wg_lower
            pad.add_port(name="o3", port=wg_lower.ports["o2"])
            pad.add_port(name="o4", port=wg_lower.ports["o1"])

    if m2_bonding_pads_params is not None:
        required_keys = ("layer_m2", "layer_openings")
        missing_keys = [k for k in required_keys if k not in m2_bonding_pads_params]
        if missing_keys:
            raise ValueError(
                "m2_bonding_pads_params is missing required keys: "
                + ", ".join(missing_keys)
            )

        allowed_optional_keys = {
            "m1_opening_offset",
            "opening_size",
            "opening_separation",
            "tl_opening_host_width",
            "m2_pad_length",
        }
        unknown_keys = {
            k
            for k in m2_bonding_pads_params
            if k not in set(required_keys) | allowed_optional_keys
        }
        if unknown_keys:
            raise ValueError(
                "m2_bonding_pads_params contains unknown keys: "
                + ", ".join(sorted(unknown_keys))
            )

        m2_bonding_pads_component = m2_bonding_pads(
            pad_xs=pad_xs,
            layer_m2=m2_bonding_pads_params["layer_m2"],
            layer_openings=m2_bonding_pads_params["layer_openings"],
            **{
                key: m2_bonding_pads_params[key]
                for key in allowed_optional_keys
                if key in m2_bonding_pads_params
            },
        )

        M2_bonding_pads_ref = pad << m2_bonding_pads_component
        M2_bonding_pads_ref.connect("e2", p1.ports["e1"])
        pad.add_port(
            name="e1",
            port=M2_bonding_pads_ref.ports["e1"],
        )
        pad.add_port(
            name="e3",
            port=p1.ports["e1"],
        )
    else:
        # Add ports for electrical connections
        pad.add_port(name="e1", port=p1.ports["e1"])

    pad.add_port(name="e2", port=p1.ports["e2"])

    return pad


@gf.cell()
def straight_cpw(
    cpw_xs: CrossSectionSpec,
    modulation_length: float = 1000.0,
    optical_waveguides: dict[str, Any] | None = None,
    single_waveguide: bool = False,
) -> gf.Component:
    """A straight CPW transmission line"""
    if optical_waveguides is None:
        optical_waveguides = {
            "terminal_xs": None,
            "modulation_xs": None,
            "taper_length": 100.0,
        }

    cpw = gf.Component()
    strght = cpw << gf.components.straight(
        length=modulation_length, cross_section=cpw_xs
    )
    for s in cpw_xs.sections:
        if s.name == "signal":
            signal_width = s.width
        elif s.name == "ground_top":
            ground_planes_width = s.width
            offset = s.offset
    gap_width = offset - 0.5 * (signal_width + ground_planes_width)

    cpw.add_port(name="e1", port=strght.ports["e1"])
    cpw.add_port(name="e2", port=strght.ports["e2"])

    # Add optical waveguides if provided
    if (
        optical_waveguides is not None
        and optical_waveguides.get("terminal_xs") is not None
    ):
        terminal_xs = optical_waveguides.get("terminal_xs")
        modulation_xs = optical_waveguides.get("modulation_xs")
        taper_length = optical_waveguides.get("taper_length")
        has_modulation_tapers = modulation_xs is not None and taper_length > 0

        y_offset1 = strght.ports["e1"].dcenter[1] + signal_width / 2 + gap_width / 2
        y_offset2 = strght.ports["e2"].dcenter[1] - signal_width / 2 - gap_width / 2
        x_start = strght.ports["e1"].dcenter[0]

        wg_cell = modulation_waveguide(
            modulation_xs=modulation_xs if modulation_xs is not None else terminal_xs,
            terminal_xs=terminal_xs,
            modulation_length=modulation_length,
            taper_length=taper_length if has_modulation_tapers else 0.0,
        )

        # Top waveguide
        wg_top = cpw << wg_cell
        wg_top.move(wg_top.ports["o1"].dcenter, (x_start, y_offset1))
        cpw.add_port(name="o1", port=wg_top.ports["o1"])
        cpw.add_port(name="o2", port=wg_top.ports["o2"])

        if not single_waveguide:
            # Bottom waveguide (keep legacy naming/orientation behavior)
            wg_bot = cpw << wg_cell
            wg_bot.move(wg_bot.ports["o1"].dcenter, (x_start, y_offset2))
            cpw.add_port(name="o4", port=wg_bot.ports["o1"])
            cpw.add_port(name="o3", port=wg_bot.ports["o2"])
            copy_info(cpw, wg_cell)
        else:
            copy_info(cpw, wg_cell)

    cpw.flatten()
    cpw.info["modulation_length"] = modulation_length
    cpw.info["rf_central_conductor_width"] = signal_width
    cpw.info["rf_ground_planes_width"] = ground_planes_width
    cpw.info["rf_gap"] = gap_width
    return cpw


@gf.cell()
def trail_cpw(
    cpw_xs: CrossSectionSpec,
    modulation_length: float = 3000.0,
    trail_params: dict[str, float] | None = None,
    rounding_radius: float = 0.5,
    optical_waveguides: dict[str, Any] | None = None,
    single_waveguide: bool = False,
) -> gf.Component:
    """A CPW transmission line with periodic T-rails on all electrodes"""
    if trail_params is None:
        trail_params = {
            "th": 1.5,
            "tl": 44.7,
            "tw": 7.0,
            "tt": 1.5,
            "tc": 5.0,
        }
    if optical_waveguides is None:
        optical_waveguides = {
            "terminal_xs": None,
            "modulation_xs": None,
            "taper_length": 100.0,
        }

    th = trail_params["th"]
    tl = trail_params["tl"]
    tw = trail_params["tw"]
    tt = trail_params["tt"]
    tc = trail_params["tc"]

    for s in cpw_xs.sections:
        if s.name == "signal":
            signal_width = s.width
        elif s.name == "ground_top":
            ground_planes_width = s.width
            offset = s.offset
    gap_width = offset - 0.5 * (signal_width + ground_planes_width)

    signal_width_corrected = signal_width - 2 * th - 2 * tt
    ground_planes_width_corrected = ground_planes_width - th - tt

    gap_width_corrected = gap_width + 2 * th + 2 * tt  # total gap width with T-rails

    num_cells = np.floor(modulation_length / (tl + tc))

    # redefine cross section of the straight section to include T-rails

    xs_cpw_trail = xs_cpw_single_layer(
        central_conductor_width=signal_width_corrected,
        ground_planes_width=ground_planes_width_corrected,
        gap=gap_width_corrected,
        layer=cpw_xs.layer,
    )

    cpw = gf.Component()

    strght = cpw << gf.components.straight(
        length=modulation_length, cross_section=xs_cpw_trail
    )
    dl_tr = 0.5 * (
        modulation_length - num_cells * tl - (num_cells - 1) * tc
    )  # t-rail shift to place symmetrically w/r to bond pads
    strght_ter_left = cpw << gf.components.straight(
        length=dl_tr + tl / 2, cross_section=xs_cpw_trail
    )  # cover the gap formed by th and tt.
    strght_ter_left.dmove(
        strght_ter_left.ports["e1"].dcenter, strght.ports["e1"].dcenter
    )
    strght_ter_left = cpw << gf.components.straight(
        length=dl_tr + tl / 2, cross_section=xs_cpw_trail
    )  # cover the gap formed by th and tt.
    strght_ter_left.dmove(
        strght_ter_left.ports["e2"].dcenter, strght.ports["e2"].dcenter
    )

    # Initiate T-rail polygon element. Create a bit more to ensure round corners close to electrodes
    trailpol = gf.kdb.DPolygon(
        [
            (
                tl + 2,
                signal_width_corrected / 2,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (
                tl + 2,
                signal_width_corrected / 2 - tt,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (
                0 - 2,
                signal_width_corrected / 2 - tt,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (
                0 - 2,
                signal_width_corrected / 2,
            ),  # additional 2 microns to ensure smooth corners for slotted
            (tl / 2 - tw / 2, signal_width_corrected / 2),
            (tl / 2 - tw / 2, signal_width_corrected / 2 + th),
            (0, signal_width_corrected / 2 + th),
            (0, signal_width_corrected / 2 + th + tt),
            (tl, signal_width_corrected / 2 + th + tt),
            (tl, signal_width_corrected / 2 + th),
            (tl / 2 + tw / 2, signal_width_corrected / 2 + th),
            (tl / 2 + tw / 2, signal_width_corrected / 2),
        ]
    )

    # Create T-rail component
    trailcomp = gf.Component()
    _ = trailcomp.add_polygon(trailpol, layer=cpw_xs.layer)

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
    inc_t3.dmovey(-signal_width_corrected - th)
    inc_t4 = trail_uc << trailround
    inc_t4.dmovey(-signal_width_corrected - gap_width_corrected)

    # Place T-rails symmetrically w/r to bondpads
    [ref.dmovex(dl_tr) for ref in (inc_t1, inc_t2, inc_t3, inc_t4)]
    # [ref.dmovey(300*dl_tr) for ref in (inc_t1, inc_t2, inc_t3, inc_t4)]

    str_left = cpw << gf.components.straight(
        length=tl / 2 + dl_tr, cross_section=cpw_xs
    )
    str_left.dmove(str_left.ports["e1"].dcenter, strght.ports["e1"].dcenter)
    str_right = cpw << gf.components.straight(
        length=tl / 2 + dl_tr, cross_section=cpw_xs
    )
    str_right.dmove(str_right.ports["e2"].dcenter, strght.ports["e2"].dcenter)
    cpw.add_port(name="e1", port=str_left.ports["e1"])
    cpw.add_port(name="e2", port=str_right.ports["e2"])

    # Here we expect waveguide_dict with keys "terminal_xs", "modulation_xs", and "taper_length"
    if (
        optical_waveguides is not None
        and optical_waveguides.get("terminal_xs") is not None
    ):
        terminal_xs = optical_waveguides.get("terminal_xs")
        modulation_xs = optical_waveguides.get("modulation_xs")
        taper_length = optical_waveguides.get("taper_length")
        has_modulation_tapers = modulation_xs is not None and taper_length > 0

        y_offset1 = (
            str_left.ports["e1"].dcenter[1]
            + signal_width_corrected / 2
            + gap_width_corrected / 2
        )
        y_offset2 = (
            str_right.ports["e1"].dcenter[1]
            - signal_width_corrected / 2
            - gap_width_corrected / 2
        )
        x_start = str_left.ports["e1"].dcenter[0]

        wg_cell = modulation_waveguide(
            modulation_xs=modulation_xs if modulation_xs is not None else terminal_xs,
            terminal_xs=terminal_xs,
            modulation_length=modulation_length,
            taper_length=taper_length if has_modulation_tapers else 0.0,
        )

        # Top waveguide
        wg_top = cpw << wg_cell
        wg_top.move(wg_top.ports["o1"].dcenter, (x_start, y_offset1))
        cpw.add_port(name="o1", port=wg_top.ports["o1"])
        cpw.add_port(name="o2", port=wg_top.ports["o2"])

        if not single_waveguide:
            # Bottom waveguide (keep legacy naming/orientation behavior)
            wg_bot = cpw << wg_cell
            wg_bot.move(wg_bot.ports["o1"].dcenter, (x_start, y_offset2))
            cpw.add_port(name="o4", port=wg_bot.ports["o1"])
            cpw.add_port(name="o3", port=wg_bot.ports["o2"])
            copy_info(cpw, wg_cell)
        else:
            copy_info(cpw, wg_cell)

    # Duplicate cell
    cpw.add_ref(
        trail_uc,
        columns=num_cells,
        rows=1,
        column_pitch=tl + tc,
    )

    cpw.flatten()
    cpw.info["modulation_length"] = modulation_length
    cpw.info["rf_central_conductor_width"] = signal_width_corrected
    cpw.info["rf_ground_planes_width"] = ground_planes_width_corrected
    cpw.info["rf_gap"] = gap_width
    for parameter, value in trail_params.items():
        cpw.info[f"trail_{parameter}"] = value

    return cpw


@gf.cell()
def modulation_waveguide(
    modulation_xs: CrossSection,
    terminal_xs: CrossSection,
    modulation_length: float,
    taper_length: float = 100.0,
) -> gf.Component:
    """A modulation waveguide"""
    from _utils.cross_section import _to_xs_spec

    wg = gf.Component()

    if modulation_xs is not None and terminal_xs is not None and taper_length > 0:
        xs_spec = _to_xs_spec(modulation_xs)
        # Taper-in (terminal_xs -> modulation_xs)
        taper_in = gf.components.taper(
            length=taper_length,
            width1=terminal_xs.width,
            width2=modulation_xs.width,
            cross_section=xs_spec,
        )
        taper_in_ref = wg << taper_in

        # Modulation straight
        straight = gf.components.straight(
            length=modulation_length - 2 * taper_length, cross_section=modulation_xs
        )
        straight_ref = wg << straight
        straight_ref.connect("o1", taper_in_ref.ports["o2"])

        # Taper-out (modulation_xs -> terminal_xs)
        taper_out = gf.components.taper(
            length=taper_length,
            width1=modulation_xs.width,
            width2=terminal_xs.width,
            cross_section=xs_spec,
        )

        taper_out_ref = wg << taper_out
        taper_out_ref.connect("o1", straight_ref.ports["o2"])

        wg.add_port(name="o1", port=taper_in_ref.ports["o1"])
        wg.add_port(name="o2", port=taper_out_ref.ports["o2"])
    else:
        # Simple straight with terminal_xs
        straight = gf.components.straight(
            length=modulation_length, cross_section=terminal_xs
        )
        straight_ref = wg << straight
        wg.add_port(name="o1", port=straight_ref.ports["o1"])
        wg.add_port(name="o2", port=straight_ref.ports["o2"])

    wg.flatten()
    wg.info["modulation_length"] = modulation_length
    wg.info["taper_length"] = taper_length
    wg.info["terminal_width"] = terminal_xs.width
    wg.info["modulation_width"] = (
        modulation_xs.width if modulation_xs is not None else terminal_xs.width
    )
    return wg


if __name__ == "__main__":
    from ltoi300.tech import xs_rwg700, xs_rwg2500, xs_uni_cpw

    c = gf.Component()
    cpw_xs = xs_uni_cpw(central_conductor_width=21, gap=6, ground_planes_width=50.0)
    optical_waveguides = {
        "terminal_xs": xs_rwg700(),
        "modulation_xs": xs_rwg2500(),
        "taper_length": 100.0,
    }
    cpw = straight_cpw(
        cpw_xs=cpw_xs, modulation_length=2000.0, optical_waveguides=optical_waveguides
    )
    print(cpw.info)
    cpw_ref = c << cpw
    pad = cpw_pad(
        cpw_xs=cpw_xs,
        optical_waveguide_xs=xs_rwg700(),
        m2_bonding_pads_params={
            "layer_m2": (22, 0),
            "layer_openings": (40, 0),
        },
    )
    pad_ref = c << pad
    pad_ref.connect("e2", cpw.ports["e1"])

    pad_width = pad_ref.ports["e1"].width
    pad_xs = xs_cpw_single_layer(
        central_conductor_width=pad_width,
        ground_planes_width=50,
        gap=pad_width / cpw_xs.width * 6.0,
        layer=(20, 0),
    )

    termination3 = c << double_layer_termination(
        cpw_xs=cpw_xs, termination_layer=(23, 0), m2_layer=(22, 0)
    )
    termination3.connect("e1", cpw_ref.ports["e2"])

    c.show()
