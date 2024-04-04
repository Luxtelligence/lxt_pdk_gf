from functools import partial
import numpy as np

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, ComponentSpec

from lnoi400.tech import LAYER


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

##########
# Bends
##########

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

    # Place the two tapers on the different layers
    
    double_taper = gf.Component()
    ltref = double_taper << taper_lower
    utref = double_taper << taper_upper
    utref.movex(lower_taper_length)

    # Define the input and output optical ports

    double_taper.add_port(port = ltref.ports["o1"])
    double_taper.add_port(port = utref.ports["o2"])

    return double_taper.flatten()

###################
# GSG bonding pad
###################

@gf.cell
def CPW_pad_linear(
    start_width: float = 80.0,
    length_straight: float = 30.0,
    length_tapered: float = 100.0,
    cross_section: CrossSectionSpec = "xs_uni_cpw"
    ) -> gf.Component:
    """RF access line for high-frequency GSG probes. The probe pad maintains a 
    fixed gap/center conductor ratio across its length, to achieve a good 
    impedance matching"""

    xs_cpw = gf.get_cross_section(cross_section)

    # Extract the CPW cross sectional parameters

    sections = xs_cpw.sections
    signal_section = [s for s in sections if s.name == "signal"][0]
    ground_section = [s for s in sections if s.name == "ground_top"][0]
    end_width = signal_section.width
    ground_planes_width = ground_section.width
    end_gap = ground_section.offset - 0.5*(end_width + ground_planes_width)
    aspect_ratio = end_width/(end_width + 2*end_gap)

    # Pad elements generation
    
    pad = gf.Component()

    start_gap = 0.5*(aspect_ratio**(-1) - 1)*start_width

    central_conductor_shape = [
        (0.0, start_width/2.),
        (length_straight, start_width/2.),
        (length_straight + length_tapered, end_width/2.),
        (length_straight + length_tapered, -end_width/2.),
        (length_straight, -start_width/2.),
        (0.0, -start_width/2.)]
    
    ground_plane_shape = [
        (0.0, start_width/2. + start_gap),
        (length_straight, start_width/2. + start_gap),
        (length_straight + length_tapered, end_width/2. + end_gap),
        (length_straight + length_tapered, end_width/2. + end_gap + ground_planes_width),
        (0.0, end_width/2. + end_gap + ground_planes_width)]

    pad.add_polygon(central_conductor_shape, layer = "TL")
    pad.add_polygon(ground_plane_shape, layer = "TL")
    G_bot = pad.add_polygon(ground_plane_shape, layer = "TL")
    G_bot.mirror((0, 0), (1, 0))

    # Ports definition

    pad.add_port(
        name = "e1",
        center = (length_straight, 0.),
        width = start_width,
        port_type = "electrical",
        layer = "TL"
    )

    pad.add_port(
        name = "e2",
        center = (length_straight + length_tapered, 0.),
        width = end_width,
        orientation = 0.,
        port_type = "electrical",
        layer = "TL"
    )

    return pad

####################
# Transmission lines
####################

@gf.cell()
def uni_cpw_straight(
    length: float = 3000.,
    cross_section: CrossSectionSpec = "xs_uni_cpw",
    bondpad: ComponentSpec = "CPW_pad_linear",
    ) -> gf.Component:
    """A CPW transmission line for microwaves, with a uniform cross section."""

    cpw = gf.Component()
    bp = gf.get_component(bondpad, cross_section = cross_section)

    tl = cpw << gf.components.straight(length = length, cross_section = cross_section)
    bp1 = cpw << bp
    bp2 = cpw << bp

    bp1.connect("e2", tl.ports["e1"])
    bp2.mirror()
    bp2.connect("e2", tl.ports["e2"])

    return cpw.flatten()

###############
# Modulators
###############



##################
# Chip floorplan
##################

@gf.cell
def chip_frame(
    size: tuple[float, float] = (10_000, 5000),
    exclusion_zone_width: float = 50,
    center: float = None) -> gf.Component:
    """Provide the chip extent and the exclusion zone around the chip frame. 
    In the exclusion zone, only the edge couplers routing to the chip facet should be placed.
    Allowed chip dimensions (in either direction): 5000 um, 10000 um, 20000 um."""

    # Check that the chip dimensions have the admissible values.

    snapped_size = []

    if size[0] <= 5050 and size[1] <= 5050:
        
        raise(ValueError(f"The chip frame size {size} is not supported."))
    
    else:

        for s in size:                
            if abs(s - 5000.) <= 50.:
                snapped_size.append(4950.)
            elif abs(s - 10000.) <= 100.:
                snapped_size.append(10000)
            elif abs(s - 20000.) <= 200:
                snapped_size.append(20100)
            else:
                raise(ValueError(f"The chip frame size {size} is not supported."))
            
    if not(center):
        center = (.5*snapped_size[0] + exclusion_zone_width, .5*snapped_size[1] + exclusion_zone_width)

    # Chip frame elements
        
    inner_box = gf.components.rectangle(
        size = snapped_size,
        layer = LAYER.CHIP_CONTOUR,
        centered = True,
    )

    outer_box = gf.components.rectangle(
        size = [s + 2*exclusion_zone_width for s in snapped_size],
        layer = LAYER.CHIP_EXCLUSION_ZONE,
        centered = True,
    )

    c = gf.Component()
    in_box = c << inner_box
    out_box = c << outer_box

    in_box.move(destination = center)
    out_box.move(destination = center)

    return c.flatten()

if __name__ == "__main__":

    pass

