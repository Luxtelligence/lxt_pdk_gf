from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.routing import route_quad
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from lnoi400.spline import (
    bend_S_spline,
    bend_S_spline_varying_width,
    spline_clamped_path,
)
from lnoi400.tech import LAYER, xs_uni_cpw

################
# Straights
################


@gf.cell
def _straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_rwg1000",
    **kwargs,
) -> gf.Component:
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def straight_rwg1000(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight single-mode waveguide."""
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rwg1000"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_rwg3000(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight multimode waveguide."""
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rwg3000"
    return _straight(
        length=length,
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
    cross_section: CrossSectionSpec = "xs_rwg1000",
    **kwargs,
) -> gf.Component:
    """
    A 90-degrees bend following an Euler path, with linearly-varying curvature
    (increasing and decreasing).
    """

    npoints = int(np.round(200 * radius / 80.0))
    angle = 90.0

    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        cross_section=cross_section,
        **kwargs,
    )


# TODO: inquire about meaning of bend_points_distance in relation with Euler bends


@gf.cell
def U_bend_racetrack(
    v_offset: float = 90.0,
    p: float = 1.0,
    with_arc_floorplan: bool = True,
    cross_section: CrossSectionSpec = "xs_rwg3000",
    **kwargs,
) -> gf.Component:
    """A U-bend with fixed cross-section and dimensions, suitable for building a low-loss racetrack resonator."""

    radius = 0.5 * v_offset

    npoints = int(np.round(600 * radius / 90.0))
    angle = 180.0

    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def S_bend_vert(
    v_offset: float = 25.0,
    h_extent: float = 100.0,
    dx_straight: float = 5.0,
    cross_section: CrossSectionSpec = "xs_rwg1000",
) -> gf.Component:
    """A spline bend that bridges a vertical displacement."""

    if np.abs(v_offset) < 10.0:
        raise ValueError(
            f"The vertical distance bridged by the S-bend ({v_offset}) is too small."
        )

    if np.abs(h_extent / v_offset) < 3.5 or h_extent < 90.0:
        raise ValueError(
            f"The bend would be too tight. Increase h_extent from its current value of {h_extent}."
        )

    S_bend = gf.components.extend_ports(
        bend_S_spline(
            size=(h_extent, v_offset),
            cross_section=cross_section,
            npoints=int(np.round(2.5 * h_extent)),
            path_method=spline_clamped_path,
        ),
        length=dx_straight,
        cross_section=cross_section,
    )

    bend_cell = gf.Component()
    bend_ref = bend_cell << S_bend
    bend_ref.dmove(bend_ref.ports["o1"].dcenter, (0.0, 0.0))
    bend_cell.add_port(name="o1", port=bend_ref.ports["o1"])
    bend_cell.add_port(name="o2", port=bend_ref.ports["o2"])
    bend_cell.flatten()

    return bend_cell


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

    gap_mmi = (
        port_ratio * width_mmi - width_taper
    )  # The port ratio is defined as the ratio between the waveguides separation and the MMI width.

    return gf.components.mmi1x2(
        width_mmi=width_mmi,
        length_mmi=length_mmi,
        gap_mmi=gap_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
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
    **kwargs,
) -> gf.Component:
    """MMI2x2 with layout optimized for maximum transmission at 1550 nm."""

    gap_mmi = (
        port_ratio * width_mmi - width_taper
    )  # The port ratio is defined as the ratio between the waveguides separation and the MMI width.

    return gf.components.mmi2x2(
        width_mmi=width_mmi,
        length_mmi=length_mmi,
        gap_mmi=gap_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


mmi2x2_optimized1550 = mmi2x2optimized1550

#####################
# Directional coupler
#####################


@gf.cell
def directional_coupler_balanced(
    io_wg_sep: float = 30.6,
    sbend_length: float = 58,
    central_straight_length: float = 16.92,
    coupl_wg_sep: float = 0.8,
    coup_wg_width: float = 0.8,
    cross_section_io: CrossSectionSpec = "xs_rwg1000",
) -> gf.Component:
    """Returns a 50-50 directional coupler. Default parameters give a 50/50 splitting at 1550 nm.

    Args:
        io_wg_sep: Separation of the two straights at the input/output, top-to-top.
        sbend_length: length of the s-bend part.
        central_straight_length: length of the coupling region.
        coupl_wg_sep: Distance between two waveguides in the coupling region (side to side).
        cross_section_io: cross section spec at the i/o (must be defined in tech.py).
        coup_wg_width: waveguide width at the coupling section.
    """

    s0 = gf.Section(
        width=coup_wg_width,
        offset=0,
        layer="LN_RIDGE",
        name="_default",
        port_names=("o1", "o2"),
    )
    s1 = gf.Section(width=10.0, offset=0, layer="LN_SLAB", name="slab", simplify=0.03)
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


################
# Edge couplers
################


@gf.cell
def double_linear_inverse_taper(
    cross_section_start: CrossSectionSpec = "xs_swg250",
    cross_section_end: CrossSectionSpec = "xs_rwg1000",
    lower_taper_length: float = 120.0,
    lower_taper_end_width: float = 2.05,
    upper_taper_start_width: float = 0.25,
    upper_taper_length: float = 240.0,
    slab_removal_width: float = 20.0,
    input_ext: float = 0.0,
) -> gf.Component:
    """Inverse taper with two layers, starting from a wire waveguide at the facet
    and transitioning to a rib waveguide. The tapering profile is linear in both layers."""

    lower_taper_start_width = gf.get_cross_section(cross_section_start).width
    upper_taper_end_width = gf.get_cross_section(cross_section_end).width

    xs_taper_lower_end = partial(
        gf.cross_section.strip,
        width=lower_taper_start_width
        + (lower_taper_end_width - lower_taper_start_width)
        * (1 + upper_taper_length / lower_taper_length),
        layer="LN_SLAB",
    )

    xs_taper_upper_start = partial(
        gf.cross_section.strip, layer=LAYER.LN_RIDGE, width=upper_taper_start_width
    )

    xs_taper_upper_end = partial(xs_taper_upper_start, width=upper_taper_end_width)

    taper_lower = gf.components.taper_cross_section(
        cross_section1=cross_section_start,
        cross_section2=xs_taper_lower_end,
        length=lower_taper_length + upper_taper_length,
        linear=True,
    )

    taper_upper = gf.components.taper_cross_section(
        cross_section1=xs_taper_upper_start,
        cross_section2=xs_taper_upper_end,
        length=upper_taper_length,
        linear=True,
    )

    if input_ext:
        straight_ext = gf.components.straight(
            cross_section=cross_section_start,
            length=input_ext,
        )

    # Place the two tapers on the different layers

    double_taper = gf.Component()
    if input_ext:
        sref = double_taper << straight_ext
        sref.dmovex(-input_ext)
    ltref = double_taper << taper_lower
    utref = double_taper << taper_upper
    utref.dmovex(lower_taper_length)

    # Define the input and output optical ports

    double_taper.add_port(
        port=sref.ports["o1"]
    ) if input_ext else double_taper.add_port(port=ltref.ports["o1"])
    double_taper.add_port(port=utref.ports["o2"])

    # Place the tone inversion box for the slab etch

    if slab_removal_width:
        bn = gf.components.rectangle(
            size=(
                double_taper.ports["o2"].dcenter[0]
                - double_taper.ports["o1"].dcenter[0],
                slab_removal_width,
            ),
            centered=True,
            layer=LAYER.SLAB_NEGATIVE,
        )
        bnref = double_taper << bn
        bnref.dmovex(
            origin=bnref.dxmin,
            destination=-input_ext,
        )
    double_taper.flatten()

    return double_taper


###################
# GSG bonding pad
###################


@gf.cell
def CPW_pad_linear(
    start_width: float = 80.0,
    length_straight: float = 10.0,
    length_tapered: float = 190.0,
    cross_section: CrossSectionSpec = "xs_uni_cpw",
) -> gf.Component:
    """RF access line for high-frequency GSG probes. The probe pad maintains a
    fixed gap/central conductor ratio across its length, to achieve a good
    impedance matching."""

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

    start_gap = 0.5 * (aspect_ratio ** (-1) - 1) * start_width

    central_conductor_shape = [
        (0.0, start_width / 2.0),
        (length_straight, start_width / 2.0),
        (length_straight + length_tapered, end_width / 2.0),
        (length_straight + length_tapered, -end_width / 2.0),
        (length_straight, -start_width / 2.0),
        (0.0, -start_width / 2.0),
    ]

    ground_plane_shape = [
        (0.0, start_width / 2.0 + start_gap),
        (length_straight, start_width / 2.0 + start_gap),
        (length_straight + length_tapered, end_width / 2.0 + end_gap),
        (
            length_straight + length_tapered,
            end_width / 2.0 + end_gap + ground_planes_width,
        ),
        (0.0, end_width / 2.0 + end_gap + ground_planes_width),
    ]

    bottom_ground_shape = [(p[0], -p[1]) for p in ground_plane_shape]

    pad.add_polygon(central_conductor_shape, layer="TL")
    pad.add_polygon(ground_plane_shape, layer="TL")
    pad.add_polygon(bottom_ground_shape, layer="TL")

    # Ports definition

    pad.add_port(
        name="e1",
        center=(length_straight, 0.0),
        width=start_width,
        orientation=180.0,
        port_type="electrical",
        layer="TL",
    )

    pad.add_port(
        name="e2",
        center=(length_straight + length_tapered, 0.0),
        width=end_width,
        orientation=0.0,
        port_type="electrical",
        layer="TL",
    )

    return pad


####################
# Transmission lines
####################


@gf.cell()
def uni_cpw_straight(
    length: float = 1000.0,
    cross_section: CrossSectionSpec = "xs_uni_cpw",
    signal_width: float = 10.0,
    gap_width: float = 4.0,
    ground_planes_width: float = 250.0,
    bondpad: ComponentSpec = "CPW_pad_linear",
) -> gf.Component:
    """A CPW transmission line for microwaves, with a uniform cross section."""

    cpw_xs = gf.get_cross_section(
        cross_section,
        central_conductor_width=signal_width,
        gap=gap_width,
        ground_planes_width=ground_planes_width,
    )
    cpw = gf.Component()
    bp = gf.get_component(bondpad, cross_section=cpw_xs)

    tl = cpw << gf.components.straight(length=length, cross_section=cpw_xs)
    bp1 = cpw << bp
    bp2 = cpw << bp

    bp1.connect("e2", tl.ports["e1"])
    bp2.dmirror()
    bp2.connect("e2", tl.ports["e2"])

    cpw.add_ports(tl.ports)
    cpw.add_port(
        name="bp1",
        port=bp1.ports["e1"],
    )
    cpw.add_port(
        name="bp2",
        port=bp2.ports["e1"],
    )
    cpw.flatten()

    return cpw


@gf.cell()
def trail_cpw(
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
    bondpad: ComponentSpec = "CPW_pad_linear",
    cross_section: CrossSectionSpec = xs_uni_cpw,
) -> gf.Component:
    """A CPW transmission line with periodic T-rails on all electrodes."""

    num_cells = np.floor(length / (tl + tc))
    gap_width_corrected = gap_width + 2 * th + 2 * tt  # total gap width with T-rails

    # redefine cross section to include T-rails
    xs_cpw_trail = partial(
        cross_section,
        central_conductor_width=signal_width,
        gap=gap_width_corrected,
        ground_planes_width=ground_planes_width,
    )

    cpw = gf.Component()
    bp = gf.get_component(bondpad, cross_section=xs_cpw_trail)
    strght = cpw << gf.components.straight(length=length, cross_section=xs_cpw_trail)
    bp1 = cpw << bp
    bp2 = cpw << bp
    bp1.connect("e2", strght.ports["e1"])
    bp2.dmirror()
    bp2.connect("e2", strght.ports["e2"])
    cpw.add_ports(strght.ports)

    cpw.add_port(
        name="bp1",
        port=bp1.ports["e1"],
    )
    cpw.add_port(
        name="bp2",
        port=bp2.ports["e1"],
    )

    # Initiate T-rail polygon element. Create a bit more to ensure round corners close to electrodes
    trailpol = gf.kdb.DPolygon(
        [
            (tl, signal_width / 2),
            (tl, signal_width / 2 - tt),
            (0, signal_width / 2 - tt),
            (0, signal_width / 2),
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
    inc_t2.dmovey(gap_width_corrected - th)
    inc_t3 = trail_uc << trailround
    inc_t3.dmovey(-signal_width - th)
    inc_t4 = trail_uc << trailround
    inc_t4.dmovey(-signal_width - gap_width_corrected)

    # Place T-rails symmetrically w/r to bondpads

    dl_tr = 0.5 * (length - num_cells * tl - (num_cells - 1) * tc)

    [ref.dmovex(dl_tr) for ref in (inc_t1, inc_t2, inc_t3, inc_t4)]

    # Duplicate cell
    cpw.add_ref(
        trail_uc,
        columns=num_cells,
        rows=1,
        column_pitch=tl + tc,
    )

    cpw.flatten()

    return cpw


###################
# Thermal shifters
###################


@gf.cell
def heater_resistor(
    path: gf.path.Path | None = None,
    width: float = 0.9,
    offset: float = 0.0,
) -> gf.Component:
    """A resistive wire used as a low-frequency phase shifter, exploiting
    the thermo-optical effect."""

    if not path:
        path = gf.path.straight(length=150.0)

    xs = gf.get_cross_section("xs_ht_wire", width=width, offset=offset)
    c = path.extrude(xs)

    return c


@gf.cell
def heater_straight_single(
    length: float = 150.0,
    width: float = 0.9,
    offset: float = 0.0,
    port_contact_width_ratio: float = 3.0,
    pad_size: tuple[float, float] = (100.0, 100.0),
    pad_pitch: float | None = None,
    pad_vert_offset: float = 10.0,
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

    if not pad_pitch:
        pad_pitch = length

    c = gf.Component()
    bondpads = gf.components.pad_array(
        pad=gf.components.pad,
        size=pad_size,
        column_pitch=pad_pitch,
        row_pitch=pad_pitch,
        columns=2,
        port_orientation=-90.0,
        layer=LAYER.HT,
    )
    bps = c << bondpads

    ht = heater_resistor(
        path=gf.path.straight(length),
        width=width,
        offset=offset,
    )

    # Place the ports along the edge of the wire
    for p in ht.ports:
        if p.orientation == 0.0:
            p.dcenter = (p.dcenter[0] - 0.5 * p.dwidth, p.dcenter[1] + 0.5 * width)
        if p.orientation == 180.0:
            p.dcenter = (p.dcenter[0] + 0.5 * p.dwidth, p.dcenter[1] + 0.5 * width)
        p.orientation = 90.0

    ht_ref = c << ht

    bps.dcenter = ht_ref.dcenter
    bps.dymin = ht_ref.dymax + pad_vert_offset

    port_contact_width = port_contact_width_ratio * width
    ht.ports["e1"].dx += 0.5 * (port_contact_width - width)
    ht.ports["e2"].dx -= 0.5 * (port_contact_width - width)

    routing_params = {
        "width2": port_contact_width,
        "layer": LAYER.HT,
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
        port=bps.ports["e12"],
    )

    c.flatten()

    return c


###############
# Modulators
###############


@gf.cell
def eo_phase_shifter(
    rib_core_width_modulator: float = 2.5,
    taper_length: float = 100.0,
    modulation_length: float = 7500.0,
    rf_central_conductor_width: float = 10.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    cpw_cell: ComponentSpec = uni_cpw_straight,
    draw_cpw: bool = True,
) -> gf.Component:
    """Phase shifter based on the Pockels effect. The waveguide is located
    within the gap of a CPW transmission line."""
    ps = gf.Component()
    xs_modulator = gf.get_cross_section("xs_rwg1000", width=rib_core_width_modulator)
    wg_taper = gf.components.taper_cross_section(
        cross_section1="xs_rwg1000", cross_section2=xs_modulator, length=taper_length
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

    if draw_cpw:
        xs_cpw = gf.partial(
            xs_uni_cpw,
            central_conductor_width=rf_central_conductor_width,
            ground_planes_width=rf_ground_planes_width,
            gap=rf_gap,
        )
        tl = ps << cpw_cell(
            length=modulation_length,
            cross_section=xs_cpw,
            gap_width=rf_gap,
            signal_width=rf_central_conductor_width,
            ground_planes_width=rf_ground_planes_width,
        )

        gap_eff = rf_gap + 2 * np.sum(
            [tl.cell.settings[key] for key in ("tt", "th") if key in tl.cell.settings]
        )

        tl.dmove(
            tl.ports["e1"].dcenter,
            (0.0, -0.5 * rf_central_conductor_width - 0.5 * gap_eff),
        )

        for name, port in [
            ("e1", tl.ports["bp1"]),
            ("e2", tl.ports["bp2"]),
        ]:
            ps.add_port(name=name, port=port)

    ps.flatten()

    return ps


@gf.cell
def eo_phase_shifter_high_speed(**kwargs) -> gf.Component:
    """High-speed phase shifter based on the Pockels effect. The waveguide is located
    within the gap of a CPW transmission line.
    Note: The base variant (eo_phase_shifter) uses a default central conductor width of 10.0,
    while this high-speed variant explicitly passes 21.0 for rf_central_conductor_width to achieve the desired high-speed properties.
    Pass the parameter set of eo_phase_shifter to modify.
    """
    kwargs.setdefault("rf_central_conductor_width", 21.0)
    kwargs.setdefault("cpw_cell", trail_cpw)
    ps = eo_phase_shifter(**kwargs)
    ps.info["additional_settings"] = dict(ps.settings)
    return ps


@gf.cell
def _mzm_interferometer(
    splitter: ComponentSpec = "mmi1x2_optimized1550",
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
) -> gf.Component:
    interferometer = gf.Component()

    sbend_large = S_bend_vert(
        v_offset=sbend_large_size[1], h_extent=sbend_large_size[0], dx_straight=5.0
    )

    sbend_small = S_bend_vert(
        v_offset=sbend_small_size[1],
        h_extent=sbend_small_size[0],
        dx_straight=sbend_small_straight_extend,
    )

    def branch_top():
        bt = gf.Component()
        sbend_1 = bt << sbend_large
        sbend_2 = bt << sbend_small
        pm = bt << eo_phase_shifter(
            rib_core_width_modulator=rib_core_width_modulator,
            modulation_length=modulation_length,
            taper_length=taper_length,
            draw_cpw=False,
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
        lbend = L_turn_bend(radius=lbend_tune_arm_reff)
        straight_y = gf.components.straight(
            length=20.0 + straight_unbalance, cross_section="xs_rwg1000"
        )
        straight_x = gf.components.straight(
            length=bias_tuning_section_length, cross_section="xs_rwg1000"
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

    splt = gf.get_component(splitter)

    # Uniformly handle the cases of a 1x2 or 2x2 MMI

    if len(splt.ports) == 4:
        out_top = splt.ports["o3"]
        out_bottom = splt.ports["o4"]
    elif len(splt.ports) == 3:
        out_top = splt.ports["o2"]
        out_bottom = splt.ports["o3"]
    else:
        raise ValueError(f"Splitter cell {splitter} not supported.")

    def combiner_section():
        comb_section = gf.Component()
        lbend_combiner = L_turn_bend(radius=lbend_combiner_reff)
        lbend_top = comb_section << lbend_combiner
        lbend_bottom = comb_section << lbend_combiner
        lbend_bottom.dmirror_y()
        combiner = comb_section << splt
        lbend_top.connect("o1", out_top)
        lbend_bottom.connect("o1", out_bottom)

        # comb_section.flatten()

        exposed_ports = [
            ("o2", lbend_top.ports["o2"]),
            ("o1", combiner.ports["o1"]),
            ("o3", lbend_bottom.ports["o2"]),
        ]

        if "2x2" in splitter:
            exposed_ports.append(
                ("in2", combiner.ports["o2"]),
            )

        for name, port in exposed_ports:
            comb_section.add_port(name=name, port=port)

        return comb_section

    splt_ref = interferometer << splt
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

    if "2x2" in splitter:
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
def mzm_unbalanced(
    modulation_length: float = 7500.0,
    length_imbalance: float = 100.0,
    lbend_tune_arm_reff: float = 75.0,
    rf_pad_start_width: float = 80.0,
    rf_central_conductor_width: float = 10.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    cpw_cell: ComponentSpec = uni_cpw_straight,
    with_heater: bool = False,
    heater_offset: float = 1.2,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    **kwargs,
) -> gf.Component:
    """Mach-Zehnder modulator based on the Pockels effect with an applied RF electric field.
    The modulator works in a differential push-pull configuration driven by a single GSG line."""

    mzm = gf.Component()

    # Transmission line subcell

    xs_cpw = gf.partial(
        xs_uni_cpw,
        central_conductor_width=rf_central_conductor_width,
        ground_planes_width=rf_ground_planes_width,
        gap=rf_gap,
    )

    rf_line = mzm << cpw_cell(
        bondpad={
            "component": "CPW_pad_linear",
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
    )

    rf_line.dmove(rf_line.ports["e1"].dcenter, (0.0, 0.0))

    # Interferometer subcell

    if "splitter" not in kwargs.keys():
        kwargs["splitter"] = "mmi1x2_optimized1550"
    splitter = kwargs["splitter"]

    splitter = gf.get_component(splitter)

    sbend_large_AR = 3.6

    gap_eff = rf_gap + 2 * np.sum(
        [
            rf_line.cell.settings[key]
            for key in ("tt", "th")
            if key in rf_line.cell.settings
        ]
    )

    GS_separation = rf_pad_start_width * gap_eff / rf_central_conductor_width

    sbend_large_v_offset = (
        0.5 * rf_pad_start_width
        + 0.5 * GS_separation
        - 0.5 * splitter.settings["port_ratio"] * splitter.settings["width_mmi"]
    )

    sbend_small_straight_length = rf_pad_length_straight * 0.5

    lbend_combiner_reff = (
        0.5 * rf_pad_start_width
        + lbend_tune_arm_reff
        + 0.5 * GS_separation
        - 0.5 * splitter.settings["port_ratio"] * splitter.settings["width_mmi"]
    )

    interferometer = (
        mzm
        << partial(
            _mzm_interferometer,
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
            **kwargs,
        )()
    )

    interferometer.dmove(
        interferometer.ports["upper_taper_start"].dcenter,
        (0.0, 0.5 * (rf_central_conductor_width + gap_eff)),
    )

    # Add heater for phase tuning

    if with_heater:
        ht_ref = mzm << heater_straight_single(
            length=bias_tuning_section_length,
            width=heater_width,
            offset=heater_offset,
            pad_size=heater_pad_size,
        )

        if length_imbalance < 0.0:
            heater_disp = [0, 0.5 * heater_width + heater_offset]
        else:
            ht_ref.dmirror_y()
            heater_disp = [0, -0.5 * heater_width - heater_offset]

        ht_ref.dmove(
            origin=ht_ref.ports["ht_start"].dcenter,
            destination=(
                np.array(interferometer.ports["long_bias_branch_start"].dcenter)
                + heater_disp
            ),
        )

    # Expose the ports

    exposed_ports = [
        ("e1", rf_line.ports["bp1"]),
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

    if with_heater:
        exposed_ports += [
            ("e3", ht_ref.ports["e1"]),
            (
                "e4",
                ht_ref.ports["e2"],
            ),
        ]

    [mzm.add_port(name=name, port=port) for name, port in exposed_ports]
    return mzm


@gf.cell
def mzm_unbalanced_high_speed(**kwargs) -> gf.Component:
    """High-speed Mach-Zehnder modulator based on the Pockels effect with an applied RF electric field.
    The modulator works in a differential push-pull configuration driven by a single GSG line.
    Note: The base variant (mzm_unbalanced) uses a default central conductor width of 10.0,
    while this high-speed variant explicitly passes 21.0 for rf_central_conductor_width to achieve the desired high-speed properties.
    Pass the parameter set of mzm_unbalanced to modify.
    """
    kwargs.setdefault("rf_central_conductor_width", 21.0)
    kwargs.setdefault("cpw_cell", trail_cpw)
    mzm = mzm_unbalanced(**kwargs)
    mzm.info["additional_settings"] = dict(mzm.settings)
    return mzm


##################
# Chip floorplan
##################


@gf.cell
def chip_frame(
    size: tuple[float, float] = (10_000, 5000),
    exclusion_zone_width: float = 50,
    center: tuple[float, float] = None,
) -> gf.Component:
    """Provide the chip extent and the exclusion zone around the chip frame.
    In the exclusion zone, only the edge couplers routing to the chip facet should be placed.
    Allowed chip dimensions (in either direction): 5000 um, 10000 um, 20000 um."""

    # Check that the chip dimensions have the admissible values.

    snapped_size = []

    if size[0] <= 5050 and size[1] <= 5050:
        raise (ValueError(f"The chip frame size {size} is not supported."))

    if size[0] > 20200 or size[1] > 20200:
        raise (ValueError(f"The chip frame size {size} is not supported."))

    else:
        for s in size:
            if abs(s - 5000.0) <= 50.0:
                snapped_size.append(4950.0)
            elif abs(s - 10000.0) <= 100.0:
                snapped_size.append(10000)
            elif abs(s - 20000.0) <= 200:
                snapped_size.append(20100)
            else:
                raise (ValueError(f"The chip frame size {size} is not supported."))

    # Chip frame elements

    inner_box = gf.components.rectangle(
        size=tuple(snapped_size),
        layer=LAYER.CHIP_CONTOUR,
        centered=True,
    )

    outer_box = gf.components.rectangle(
        size=tuple(s + 2 * exclusion_zone_width for s in snapped_size),
        layer=LAYER.CHIP_EXCLUSION_ZONE,
        centered=True,
    )

    c = gf.Component()
    ib = c << inner_box
    ob = c << outer_box

    if center:
        ib.dmove(origin=(0.0, 0.0), destination=center)
        ob.dmove(origin=(0.0, 0.0), destination=center)

    c.flatten()

    return c


if __name__ == "__main__":
    pass
