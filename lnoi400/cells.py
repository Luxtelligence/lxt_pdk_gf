from functools import partial
import numpy as np

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
    **kwargs) -> gf.Component:
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

################
# Bends
################

@gf.cell
def L_turn_bend(
    radius: float = 80.0,
    p: float = 1.0,
    with_arc_floorplan: bool = True,
    direction: str = "ccw",
    cross_section: CrossSectionSpec = "xs_rwg1000",
    **kwargs) -> gf.Component:
    """A 90-degrees bend following an Euler path, with linearly-varying curvature (increasing and decreasing)."""

    npoints = int(np.round(200 * radius/80.))
    angle = 90.

    return gf.components.bend_euler(
        radius = radius,
        angle = angle,
        p = p,
        with_arc_floorplan = with_arc_floorplan,
        npoints = npoints,
        direction = direction,
        cross_section = cross_section,
        **kwargs)

#TODO: inquire about meaning of bend_points_distance in relation with Euler bends

@gf.cell
def U_bend_racetrack(
    v_offset: float = 90.,
    p: float = 1.0,
    with_arc_floorplan: bool = True,
    direction: str = "ccw",
    cross_section: CrossSectionSpec = "xs_rwg3000",
    **kwargs) -> gf.Component:
    """A U-bend with fixed cross-section and dimensions, suitable for building a low-loss racetrack resonator."""

    radius = 0.5*v_offset
    
    npoints = int(np.round(600 * radius/90.))
    angle = 180.

    return gf.components.bend_euler(
        radius = radius,
        angle = angle,
        p = p,
        with_arc_floorplan = with_arc_floorplan,
        npoints = npoints,
        direction = direction,
        cross_section = cross_section,
        **kwargs)

@gf.cell
def S_bend_vert(
    v_offset: float = 25.,
    h_extent: float = 100.,
    cross_section: CrossSectionSpec = "xs_rwg1000") -> gf.Component:
    """A spline bend that bridges a vertical displacement."""

    if np.abs(v_offset) < 10.0:
        raise ValueError(f"The vertical distance bridged by the S-bend ({v_offset}) is too small.")
    
    if np.abs(h_extent/v_offset) < 3.5 or h_extent < 90.0:
        raise ValueError(f"The bend would be too tight. Increase h_extent from its current value of {h_extent}.")

    S_bend = gf.components.bend_s(
        size = (h_extent, v_offset),
        npoints = int(np.round(2.5*h_extent)),
        cross_section = cross_section)

    return S_bend

################
# Edge couplers
################

@gf.cell
def double_linear_inverse_taper(
    cross_section_start: CrossSectionSpec = "xs_swg250",
    cross_section_end: CrossSectionSpec = "xs_rwg1000",
    lower_taper_length: float = 120.,
    lower_taper_end_width: float = 2.05,
    upper_taper_start_width: float = 0.25,
    upper_taper_length: float = 240.) -> gf.Component:
    """Inverse taper with two layers, starting from a wire waveguide at the facet and transitioning to a rib waveguide. The tapering profile is linear in both layers."""

    lower_taper_start_width = gf.get_cross_section(cross_section_start).width
    upper_taper_end_width = gf.get_cross_section(cross_section_end).width

    xs_taper_lower_end = gf.get_cross_section(cross_section_start, 
                                              width = lower_taper_start_width + (lower_taper_end_width - lower_taper_start_width) * (1 + upper_taper_length/lower_taper_length))
    xs_taper_upper_start = gf.get_cross_section(cross_section_end, width = upper_taper_start_width)

    xs_taper_upper_start = partial(
        gf.cross_section.strip,
        layer = LAYER.LN_STRIP,
        width = upper_taper_start_width)
    
    xs_taper_upper_end = partial(xs_taper_upper_start, width = upper_taper_end_width)

    taper_lower = gf.components.taper_cross_section(
        cross_section1 = cross_section_start,
        cross_section2 = xs_taper_lower_end,
        length = lower_taper_length + upper_taper_length,
        linear = True)

    taper_upper = gf.components.taper_cross_section(
        cross_section1 = xs_taper_upper_start,
        cross_section2 = xs_taper_upper_end,
        length = upper_taper_length,
        linear = True)

    # Place the two partial tapers
    
    double_taper = gf.Component()
    ltref = double_taper << taper_lower
    utref = double_taper << taper_upper
    utref.movex(lower_taper_length)

    # Define the input and output optical ports

    double_taper.add_port(port = ltref.ports["o1"])
    double_taper.add_port(port = utref.ports["o2"])

    return double_taper


if __name__ == "__main__":

    taper = double_linear_inverse_taper()
    taper.show()
    print(taper.ports)