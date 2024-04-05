from functools import partial
import numpy as np

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, ComponentSpec

from lnoi400.tech import LAYER, uni_cpw


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
    length_straight: float = 10.0,
    length_tapered: float = 190.0,
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

    cpw.add_ports(tl.ports)
    cpw.add_ports({"bp1": bp1.ports["e1"],
                   "bp2": bp2.ports["e1"]})

    return cpw.flatten()

###############
# Modulators
###############

@gf.cell()
def _mzm_interferometer(splitter: ComponentSpec = "mmi1x2_optimized1550",
                        taper_length: float = 100.,
                        rib_core_width_modulator: float = 2.5,
                        modulation_length: float = 7500.,
                        length_imbalance: float = 100.,
                        bias_tuning_section_length: float = 750.,
                        sbend_large_size: tuple[float, float] = [200., 50.],
                        sbend_small_size: tuple[float, float] = [200., -45.],
                        sbend_small_straight_extend: float = 5.0,
                        lbend_combiner_reff: float = 80.0
                        ) -> gf.Component:
    interferometer = gf.Component()

    sbend_large = gf.components.extend_ports(S_bend_vert(v_offset = sbend_large_size[1],
                                                         h_extent = sbend_large_size[0]), length = 5.0).flatten()
    
    sbend_small = gf.components.extend_ports(S_bend_vert(v_offset = sbend_small_size[1],
                                                         h_extent = sbend_small_size[0]),
                                                         length = sbend_small_straight_extend)
    
    xs_modulator = gf.get_cross_section("xs_rwg1000", width = rib_core_width_modulator)

    wg_taper = gf.components.taper_cross_section(cross_section1 = "xs_rwg1000",
                                                 cross_section2 = xs_modulator,
                                                 length = taper_length)
    
    wg_phase_modulation = gf.components.straight(length = modulation_length - 2*taper_length, cross_section = xs_modulator)
        
    @gf.cell
    def branch_top():
        bt = gf.Component()
        sbend_1 = bt << sbend_large
        sbend_2 = bt << sbend_small
        taper_1 = bt << wg_taper
        wg_pm = bt << wg_phase_modulation
        taper_2 = bt << wg_taper
        sbend_3 = bt << sbend_small

        sbend_2.connect("o1", sbend_1.ports["o2"])
        taper_1.connect("o1", sbend_2.ports["o2"])
        wg_pm.connect("o1", taper_1.ports["o2"])
        taper_2.mirror_x()
        taper_2.connect("o2", wg_pm.ports["o2"])
        sbend_3.mirror_x()
        sbend_3.connect("o2", taper_2.ports["o1"])

        bt.add_ports({"o1": sbend_1.ports["o1"],
                      "o2": sbend_3.ports["o1"],
                      "taper_start": taper_1.ports["o1"]})        

        return bt.flatten()
    
    @gf.cell
    def branch_tune_short(
        straight_unbalance: float = 0.
    ):
        arm = gf.Component()
        lbend = L_turn_bend()
        straight_y = gf.components.straight(length = 20. + straight_unbalance, cross_section = "xs_rwg1000")
        straight_x = gf.components.straight(length = bias_tuning_section_length, cross_section = "xs_rwg1000")
        symbol_to_component = {
            "b": (lbend, "o1", "o2"),
            "L": (straight_y, "o1", "o2"),
            "B": (lbend, "o2", "o1"),
            "_": (straight_x, "o1", "o2"),
        }
        sequence = "bLB_!b!L"
        arm = gf.components.component_sequence(sequence = sequence, symbol_to_component = symbol_to_component)
        arm.auto_rename_ports()
        return arm.flatten()
    
    @gf.cell
    def branch_tune_long(straight_unbalance):
        return partial(branch_tune_short, straight_unbalance = straight_unbalance)()
    
    splt = gf.get_component(splitter)

    @gf.cell
    def combiner_section():
        comb_section = gf.Component()
        lbend_combiner = L_turn_bend(radius = lbend_combiner_reff)        
        lbend_top = comb_section << lbend_combiner
        lbend_bottom = comb_section << lbend_combiner
        lbend_bottom.mirror_y()
        combiner = comb_section << splt        
        lbend_top.connect("o1", combiner.ports["o2"])
        lbend_bottom.connect("o1", combiner.ports["o3"])

        comb_section = comb_section.flatten()
        
        comb_section.add_ports({"o2": lbend_top.ports["o2"],
                                "o1": combiner.ports["o1"],
                                "o3": lbend_bottom.ports["o2"]})

        return comb_section    

    splt_ref = interferometer << splt
    bt = interferometer << branch_top()
    bb = interferometer << branch_top()
    bs = interferometer << branch_tune_short()
    bl = interferometer << branch_tune_long(abs(0.5*length_imbalance))
    cs = interferometer << combiner_section()
    bb.mirror_y()
    bt.connect("o1", splt_ref.ports["o2"])
    bb.connect("o1", splt_ref.ports["o3"])
    if length_imbalance >= 0:
        bs.mirror_y()
        bs.connect("o1", bb.ports["o2"])
        bl.connect("o1", bt.ports["o2"])
    else:
        bs.connect("o1", bt.ports["o2"])
        bl.mirror_y()
        bl.connect("o1", bb.ports["o2"])
    cs.mirror_x()
    [cs.connect("o2", bl.ports["o2"]) if length_imbalance >= 0 else cs.connect("o2", bs.ports["o2"])]

    interferometer.add_ports({"o1": splt_ref.ports["o1"],
                              "upper_taper_start": bt.ports["taper_start"],
                              "o2": cs.ports["o1"]})

    return interferometer    

@gf.cell()
def mzm_unbalanced(
    modulation_length: float = 7500.,
    rf_pad_start_width: float = 80.,
    rf_central_conductor_width: float = 10.,
    rf_ground_planes_width: float = 150.,
    rf_gap: float = 4.0,
    rf_pad_length_straight: float = 10.,
    rf_pad_length_tapered: float = 190.,
    **kwargs) -> gf.Component:
    """Mach-Zehnder modulator based on the Pockels effect with an applied RF electric field.
    The modulator works in a differential push-pull configuration driven by a single GSG line."""

    mzm = gf.Component()

    # Transmission line subcell

    xs_cpw = gf.partial(uni_cpw,
                        central_conductor_width = rf_central_conductor_width,
                        ground_planes_width = rf_ground_planes_width,
                        gap = rf_gap)

    rf_line = mzm << uni_cpw_straight(bondpad = {"component": "CPW_pad_linear",
                                                 "settings": {"start_width": rf_pad_start_width,
                                                 "length_straight": rf_pad_length_straight,
                                                 "length_tapered": rf_pad_length_tapered}}, 
                                    length = modulation_length, cross_section = xs_cpw())
    
    rf_line.move(rf_line.ports["e1"], (0., 0.))
    
    # Interferometer subcell

    if "splitter" not in kwargs.keys():
        splitter = "mmi1x2_optimized1550"
        kwargs["splitter"] = splitter

    splitter = gf.get_component(splitter)
    lbend = gf.get_component("L_turn_bend")

    sbend_large_AR = 3.6
    GS_separation = rf_pad_start_width*rf_gap/rf_central_conductor_width

    sbend_large_v_offset = .5*rf_pad_start_width + .5*GS_separation -.5*splitter.settings['port_ratio']*splitter.settings['width_mmi']    
    
    sbend_small_straight_length= rf_pad_length_straight*0.5

    lbend_combiner_reff = (.5*rf_pad_start_width + lbend.settings['radius'] + .5*GS_separation - 
                           .5*splitter.settings['port_ratio']*splitter.settings['width_mmi'])
    
    interferometer = mzm << partial(_mzm_interferometer,
                                    modulation_length = modulation_length,
                                    sbend_large_size = (sbend_large_AR*sbend_large_v_offset, sbend_large_v_offset),
                                    sbend_small_size = (rf_pad_length_straight + rf_pad_length_tapered - 2*sbend_small_straight_length, -0.5*(rf_pad_start_width - rf_central_conductor_width + GS_separation - rf_gap)),
                                    sbend_small_straight_extend = sbend_small_straight_length,
                                    lbend_combiner_reff = lbend_combiner_reff,
                                    **kwargs,)()
    
    interferometer.move(interferometer.ports["upper_taper_start"], (0., 0.5*(rf_central_conductor_width + rf_gap)))
        
    return mzm
    

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

    mzm = mzm_unbalanced(rf_pad_length_tapered = 300.)
    mzm.show()
    print(mzm.references[1].ports)

    for component in mzm.get_dependencies(recursive=True):
        if not component._locked:
            print(
                f"Component {component.name!r} was NOT properly locked. "
                "You need to write it into a function that has the @cell decorator."
            )