from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from lnoi400.spline import (
    bend_S_spline,
    bend_S_spline_extr_transition,
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
) -> gf.Component:
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
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
    impedance matching"""

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

    bottom_ground_shape = gf.Path(ground_plane_shape).dmirror((0, 0), (1, 0))

    pad.add_polygon(central_conductor_shape, layer="TL")
    pad.add_polygon(ground_plane_shape, layer="TL")
    pad.add_polygon(bottom_ground_shape.points, layer="TL")

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
    length: float = 3000.0,
    cross_section: CrossSectionSpec = "xs_uni_cpw",
    bondpad: ComponentSpec = "CPW_pad_linear",
) -> gf.Component:
    """A CPW transmission line for microwaves, with a uniform cross section."""

    cpw = gf.Component()
    bp = gf.get_component(bondpad, cross_section=cross_section)

    tl = cpw << gf.components.straight(length=length, cross_section=cross_section)
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

        tl = ps << gf.components.straight(
            length=modulation_length, cross_section=xs_cpw
        )

        tl.dmove(
            tl.ports["e1"].dcenter,
            (0.0, -0.5 * rf_central_conductor_width - 0.5 * rf_gap),
        )

        for name, port in [
            ("e1", tl.ports["e1"]),
            ("e2", tl.ports["e2"]),
        ]:
            ps.add_port(name=name, port=port)

    ps.flatten()

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
            sequence=sequence, symbol_to_component=symbol_to_component
        )
        arm.flatten()
        return arm

    def branch_tune_long(straight_unbalance):
        return partial(branch_tune_short, straight_unbalance=straight_unbalance)()

    splt = gf.get_component(splitter)

    def combiner_section():
        comb_section = gf.Component()
        lbend_combiner = L_turn_bend(radius=lbend_combiner_reff)
        lbend_top = comb_section << lbend_combiner
        lbend_bottom = comb_section << lbend_combiner
        lbend_bottom.dmirror_y()
        combiner = comb_section << splt
        lbend_top.connect("o1", combiner.ports["o2"])
        lbend_bottom.connect("o1", combiner.ports["o3"])

        # comb_section.flatten()

        exposed_ports = [
            ("o2", lbend_top.ports["o2"]),
            ("o1", combiner.ports["o1"]),
            ("o3", lbend_bottom.ports["o2"]),
        ]

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
    bt.connect("o1", splt_ref.ports["o2"])
    bb.connect("o1", splt_ref.ports["o3"])
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
        ("o2", cs.ports["o1"]),
    ]

    for name, port in exposed_ports:
        interferometer.add_port(name=name, port=port)
    interferometer.flatten()

    return interferometer


@gf.cell
def mzm_unbalanced(
    modulation_length: float = 7500.0,
    lbend_tune_arm_reff: float = 75.0,
    rf_pad_start_width: float = 80.0,
    rf_central_conductor_width: float = 10.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 190.0,
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

    rf_line = mzm << uni_cpw_straight(
        bondpad={
            "component": "CPW_pad_linear",
            "settings": {
                "start_width": rf_pad_start_width,
                "length_straight": rf_pad_length_straight,
                "length_tapered": rf_pad_length_tapered,
            },
        },
        length=modulation_length,
        cross_section=xs_cpw(),
    )

    rf_line.dmove(rf_line.ports["e1"].dcenter, (0.0, 0.0))

    # Interferometer subcell

    if "splitter" not in kwargs.keys():
        splitter = "mmi1x2_optimized1550"
        kwargs["splitter"] = splitter

    splitter = gf.get_component(splitter)

    sbend_large_AR = 3.6
    GS_separation = rf_pad_start_width * rf_gap / rf_central_conductor_width

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
                    - rf_gap
                ),
            ),
            sbend_small_straight_extend=sbend_small_straight_length,
            lbend_tune_arm_reff=lbend_tune_arm_reff,
            lbend_combiner_reff=lbend_combiner_reff,
            **kwargs,
        )()
    )

    interferometer.dmove(
        interferometer.ports["upper_taper_start"].dcenter,
        (0.0, 0.5 * (rf_central_conductor_width + rf_gap)),
    )

    # Expose the ports

    exposed_ports = [
        ("o1", interferometer.ports["o1"]),
        ("o2", interferometer.ports["o2"]),
        ("e1", rf_line.ports["e1"]),
        ("e2", rf_line.ports["e2"]),
    ]

    [mzm.add_port(name=name, port=port) for name, port in exposed_ports]

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


#####################
# Directional coupler
#####################


@gf.cell
def dir_coupl(
    io_wg_sep: float = 30.6,
    sbend_length: float = 58,
    central_straight_length: float = 16.92,
    wg_sep: float = 0.8,
    cross_section_io="xs_rwg1000",
    cross_section_coupling="xs_rwg800",
    **kwargs,
) -> gf.Component:
    """Returns directional coupler.
    Design of s-bends is based on spline

    Args:
        io_wg_sep: Separation of the two straights at the input/output, top-to-top.
        sbend_length: length of the s-bend part.
        central_straight_length: length of the coupling region.
        wg_sep: Distance between two waveguides in the coupling region.
        cross_section_io: cross section width of the i/o (must be in tech.py).
        cross_section_coupling: cross section width of the coupling section (must be in tech.py).
    """

    dc = gf.Component()

    # top right branch
    c_tr = dc << bend_S_spline_extr_transition(
        io_wg_sep=io_wg_sep,
        sbend_length=sbend_length,
        wg_sep=wg_sep,
        cross_section1=cross_section_coupling,
        cross_section2=cross_section_io,
        npoints=201,
    )
    cs_central = gf.get_cross_section(cross_section_coupling)
    c_tr.dmove(
        c_tr.ports["o1"].dcenter,
        (central_straight_length/2, 0.5 * (wg_sep + cs_central.sections[0].width)),
    )

    # bottom right branch
    c_br = dc << bend_S_spline_extr_transition(
        io_wg_sep=io_wg_sep,
        sbend_length=sbend_length,
        wg_sep=wg_sep,
        cross_section1=cross_section_coupling,
        cross_section2=cross_section_io,
        npoints=201,
    )
    c_br.dmirror_y()
    c_br.dmove(
        c_br.ports["o1"].dcenter,
        (central_straight_length/2, -0.5 * (wg_sep + cs_central.sections[0].width)),
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
    c_tl = dc << bend_S_spline_extr_transition(
        io_wg_sep=io_wg_sep,
        sbend_length=sbend_length,
        wg_sep=wg_sep,
        cross_section1=cross_section_coupling,
        cross_section2=cross_section_io,
        npoints=201,
    )
    c_tl.dmirror_x()
    c_tl.dmove(c_tl.ports["o1"].dcenter, straight_center_up.ports["o1"].dcenter)

    # bottom left branch
    c_bl = dc << bend_S_spline_extr_transition(
        io_wg_sep=io_wg_sep,
        sbend_length=sbend_length,
        wg_sep=wg_sep,
        cross_section1=cross_section_coupling,
        cross_section2=cross_section_io,
        npoints=201,
    )
    c_bl.dmirror_x()
    c_bl.dmirror_y()
    c_bl.dmove(c_bl.ports["o1"].dcenter, straight_center_down.ports["o1"].dcenter)
    # dc.flatten()
    return dc


if __name__ == "__main__":
    pass
