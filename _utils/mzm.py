from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import CrossSectionSpec

from _utils.bends import L_turn_bend, bend_S_spline
from _utils.cross_section import get_cpw_from_xs
from _utils.gsg_rf import cpw_pad, get_pad_xs, straight_cpw, trail_cpw, rectangular_cpw_pad


@gf.cell
def optical_combiner_direct(
    optical_xs: CrossSection,
    cpw_xs: CrossSection,
    mmi_cell: gf.Component,
    imbalance_length: float,
    heater_section_length: float = 0.0,
    mmi_connection_length: float = 10.0,
    cpw_connection_length: float = 25.0,
    sbend_ratio: float = 3.5,
    roc: float = 60.0,
) -> gf.Component:
    """
    Optical combiner for the MZM.
    Args:
        optical_xs: Optical cross section.
        cpw_xs: CPW cross section.
        mmi_cell: MMI cell.
        imbalance_length: Length of the imbalance section.
        heater_section_length: Length of the heater section.
        mmi_connection_length: Length of the MMI connection section.
        cpw_connection_length: Length of the CPW connection section.
        sbend_ratio: Ratio of the SBEND length to the Y-axis length.
        roc: Radius of the L-turn bends.
    Returns:
        combiner: Optical combiner cell with ports o1 and o2 for the up and down arms at the MMI
        (port o2 does not exist for 1x2 MMI), and ports o3 and o4 for the CPW connection.
        If heater_section_length > 0.0, ports ht1_1 and ht1_2 for the up arm and ports ht2_1 and ht2_2 for the down arm are added.
        If imbalance_length = 0.0, the CPW connection is directly connected to the SBEND ports.
        If imbalance_length > 0.0, the CPW connection is connected to the SBEND ports via L-turn bends with total length imbalance_length.
    """

    combiner = gf.Component()
    mmi = combiner << mmi_cell
    mmi_cell_name = mmi.name

    if "2x2" in mmi_cell_name:
        combiner.add_port(
            name="o1",
            port=mmi.ports["o2"],
        )
        combiner.add_port(
            name="o2",
            port=mmi.ports["o1"],
        )
        up_port = mmi.ports["o3"]
        down_port = mmi.ports["o4"]
    elif "1x2" in mmi_cell_name:
        combiner.add_port(
            name="o1",
            port=mmi.ports["o1"],
        )
        up_port = mmi.ports["o2"]
        down_port = mmi.ports["o3"]

    mmi_connection = gf.components.straight(
        length=mmi_connection_length,
        cross_section=optical_xs,
    )
    mmi_connection_up = combiner << mmi_connection
    mmi_connection_down = combiner << mmi_connection
    mmi_connection_up.connect("o1", up_port)
    mmi_connection_down.connect("o1", down_port)

    # Calculate vertical offset to the CPW connection
    signal_width, _, gap, _ = get_cpw_from_xs(cpw_xs)

    if "2x2" in mmi_cell_name:
        mmi_dy = abs(mmi.ports["o3"].dcenter[1] - mmi.ports["o4"].dcenter[1])
    elif "1x2" in mmi_cell_name:
        mmi_dy = abs(mmi.ports["o2"].dcenter[1] - mmi.ports["o3"].dcenter[1])
    else:
        raise ValueError(
            f"Invalid MMI cell name: {mmi_cell_name}. Valid names are '2x2' and '1x2'."
        )

    y = signal_width / 2 + gap / 2 - mmi_dy / 2
    x = sbend_ratio * y

    sbend = bend_S_spline(
        size=(x, y),
        cross_section=optical_xs,
        npoints=201,
    )

    sbend_up = combiner << sbend
    sbend_down = combiner << sbend
    sbend_down.dmirror_y()
    sbend_up.connect("o1", mmi_connection_up.ports["o2"])
    sbend_down.connect("o1", mmi_connection_down.ports["o2"])
    L_turn = L_turn_bend(
        radius=roc,
        cross_section=optical_xs,
    )

    heater_segment = gf.components.straight(
        length=heater_section_length,
        cross_section=optical_xs,
    )

    cpw_connection = gf.components.straight(
        length=cpw_connection_length,
        cross_section=optical_xs,
    )

    vertical_straight_75 = gf.components.straight(
        length=75.0,
        cross_section=optical_xs,
    )

    symbol_to_component = {
        "c": (cpw_connection, "o1", "o2"),
        "L": (L_turn, "o1", "o2"),
        "V": (vertical_straight_75, "o1", "o2"),
    }

    if heater_section_length == 0.0 and imbalance_length == 0.0:
        cpw_connection_up = combiner << cpw_connection
        cpw_connection_up.connect("o1", sbend_up.ports["o2"])
        cpw_connection_down = combiner << cpw_connection
        cpw_connection_down.connect("o1", sbend_down.ports["o2"])
    else:
        if imbalance_length > 0.0:
            imbalance_segment = gf.components.straight(
                length=imbalance_length / 2,
                cross_section=optical_xs,
            )
            symbol_to_component["_"] = (imbalance_segment, "o1", "o2")
            arm_up_seq = "L!L"
            arm_down_seq = "!L_VL"
        elif imbalance_length < 0.0:
            imbalance_segment = gf.components.straight(
                length=-imbalance_length / 2,
                cross_section=optical_xs,
            )
            symbol_to_component["_"] = (imbalance_segment, "o1", "o2")
            arm_up_seq = "L_!L"
            arm_down_seq = "!LVL"
        else:
            # imbalance_length == 0.0 but heater_section_length > 0:
            # combiner1 thermal: both arms are symmetric (no extra 75 µm V-segment).
            # Path compensation for the structural asymmetry is handled by combiner2.
            arm_up_seq = "L!L"
            arm_down_seq = "!LL"

        arm_up = gf.components.component_sequence(
            sequence=arm_up_seq,
            symbol_to_component=symbol_to_component,
        )
        # arm_up.flatten()
        arm_up_ref = combiner << arm_up
        arm_up_ref.connect("o1", sbend_up.ports["o2"])

        heater_segment_up = combiner << heater_segment
        heater_segment_up.connect("o1", arm_up_ref.ports["o2"])

        arm_up_right = combiner << arm_up
        arm_up_right.dmirror_x()
        arm_up_right.connect("o2", heater_segment_up.ports["o2"])

        arm_down = gf.components.component_sequence(
            sequence=arm_down_seq,
            symbol_to_component=symbol_to_component,
        )
        # arm_down.flatten()
        arm_down_ref = combiner << arm_down
        arm_down_ref.connect("o1", sbend_down.ports["o2"])

        heater_segment_down = combiner << heater_segment
        heater_segment_down.connect("o1", arm_down_ref.ports["o2"])
        arm_down_right = combiner << arm_down
        arm_down_right.dmirror_x()
        arm_down_right.connect("o2", heater_segment_down.ports["o2"])

        cpw_connection_up = combiner << cpw_connection
        cpw_connection_up.connect("o1", arm_up_right.ports["o1"])
        cpw_connection_down = combiner << cpw_connection
        cpw_connection_down.connect("o1", arm_down_right.ports["o1"])

    if heater_section_length > 0.0:
        combiner.add_port(
            name="ht1_1",
            port=heater_segment_up.ports["o1"],
        )
        combiner.add_port(
            name="ht1_2",
            port=heater_segment_up.ports["o2"],
        )
        combiner.add_port(
            name="ht2_1",
            port=heater_segment_down.ports["o1"],
        )
        combiner.add_port(
            name="ht2_2",
            port=heater_segment_down.ports["o2"],
        )
    combiner.add_port(
        name="o3",
        port=cpw_connection_up.ports["o2"],
    )
    combiner.add_port(
        name="o4",
        port=cpw_connection_down.ports["o2"],
    )

    #combiner.flatten()

    return combiner


@gf.cell
def base_mzm(
    optical_xs: CrossSectionSpec,
    cpw_xs: CrossSectionSpec,
    modulation_length: float,
    mmi_cell: gf.Component,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    termination: gf.Component | None = None,
) -> gf.Component:
    if cpw_params is None:
        cpw_params = {
            "type": "trail",
            "rf_gap": 6.0,
            "rf_ground_planes_width": 50.0,
            "rf_central_conductor_width": 21.0,
        }
    if trail_params is None:
        trail_params = {
            "th": 1.5,
            "tl": 44.7,
            "tw": 7.0,
            "tt": 1.5,
            "tc": 5.0,
        }
    if cpw_pad_params is None:
        cpw_pad_params = {
            "single_side": False,
            "pitch": 100.0,
            "length_straight": 25.0,
            "length_tapered": 190.0,
            "ground_pad_width": 150.0,
        }
    if optical_waveguide_params is None:
        optical_waveguide_params = {
            "taper_length": 100.0,
            "modulation_width": 2.0,
            "terminal_width": None,
            "roc": 60.0,
            "imbalance_length": 100.0,
            "heater_section_length": 100.0,
            "mmi_connection_length": 10.0,
            "cpw_connection_length": 50.0,
            "vertical_offset": 0.0,
            "horizontal_offset": 0.0,
            "thermal_phase_shifter_node": True,
            "electric_phase_shifter_node": False,
            "electric_phase_shifter_length": 2000.0,
            "dc_phase_shifter_node": False,
            "dc_phase_shifter_length": 2000.0,
            "folding": False,
        }
    if m2_bonding_pad_params is None:
        m2_bonding_pad_params = {
            "layer_m2": (22, 0),
            "layer_openings": (40, 0),
            "m1_opening_offset": 2.5,
            "opening_size": 12.0,
            "opening_separation": 12.0,
            "tl_opening_host_width": 45.0,
            "m2_pad_length": 80.0,
        }

    if "length_imbalance" in optical_waveguide_params:
        optical_waveguide_params["imbalance_length"] = optical_waveguide_params.pop(
            "length_imbalance"
        )
    if "sbend_length" not in optical_waveguide_params:
        optical_waveguide_params["sbend_length"] = 150.0
    if "sbend_offset" not in optical_waveguide_params:
        optical_waveguide_params["sbend_offset"] = 40.0
    if "horizontal_offset" not in optical_waveguide_params:
        optical_waveguide_params["horizontal_offset"] = 0.0
    if "thermal_phase_shifter_node" not in optical_waveguide_params:
        optical_waveguide_params["thermal_phase_shifter_node"] = True
    if "electric_phase_shifter_node" not in optical_waveguide_params:
        optical_waveguide_params["electric_phase_shifter_node"] = False
    if "electric_phase_shifter_length" not in optical_waveguide_params:
        optical_waveguide_params["electric_phase_shifter_length"] = 2000.0
    if "dc_phase_shifter_node" not in optical_waveguide_params:
        optical_waveguide_params["dc_phase_shifter_node"] = False
    if "dc_phase_shifter_length" not in optical_waveguide_params:
        optical_waveguide_params["dc_phase_shifter_length"] = 2000.0

    if optical_waveguide_params.get("electric_phase_shifter_node") or optical_waveguide_params.get("dc_phase_shifter_node"):
        optical_waveguide_params["thermal_phase_shifter_node"] = False

    thermal_phase_shifter_node = optical_waveguide_params["thermal_phase_shifter_node"]
    horizontal_offset = optical_waveguide_params["horizontal_offset"]
    electric_phase_shifter_node = optical_waveguide_params["electric_phase_shifter_node"]
    electric_phase_shifter_length = optical_waveguide_params["electric_phase_shifter_length"]
    dc_phase_shifter_node = optical_waveguide_params["dc_phase_shifter_node"]
    dc_phase_shifter_length = optical_waveguide_params["dc_phase_shifter_length"]

    vertical_offset = optical_waveguide_params.get("vertical_offset", 0.0)
    folding = optical_waveguide_params.get("folding", False)
    if vertical_offset == 0.0 and (electric_phase_shifter_node or dc_phase_shifter_node):
        if horizontal_offset == 0.0 or horizontal_offset is None:
            horizontal_offset = 270.0

    if electric_phase_shifter_node and dc_phase_shifter_node:
        raise ValueError("Cannot enable both electric_phase_shifter_node and dc_phase_shifter_node simultaneously.")

    if dc_phase_shifter_node:
        cpw_params["type"] = "straight"

    detour_diff = 0.0
    imbalance_length_param = 150.0
    MZM = gf.Component()
    path_up_compensation = 0.0
    path_down_compensation = 0.0

    terminal_xs = (
        gf.get_cross_section(
            optical_xs, width=optical_waveguide_params["terminal_width"]
        )
        if optical_waveguide_params["terminal_width"] is not None
        else gf.get_cross_section(optical_xs)
    )
    # Define optical waveguides for the modulation region
    optical_waveguides = {
        "terminal_xs": terminal_xs,
        "modulation_xs": gf.get_cross_section(
            optical_xs, width=optical_waveguide_params["modulation_width"]
        ),
        "taper_length": optical_waveguide_params["taper_length"],
    }

    # Define CPW transmission line
    _cpw_xs = gf.get_cross_section(
        cpw_xs,
        central_conductor_width=cpw_params["rf_central_conductor_width"],
        gap=cpw_params["rf_gap"],
        ground_planes_width=cpw_params["rf_ground_planes_width"],
    )

    if cpw_params["type"] == "straight":
        cpw = straight_cpw(
            cpw_xs=_cpw_xs,
            modulation_length=modulation_length,
            optical_waveguides=optical_waveguides,
        )
    elif cpw_params["type"] == "trail":
        cpw = trail_cpw(
            cpw_xs=_cpw_xs,
            modulation_length=modulation_length,
            trail_params=trail_params,
            optical_waveguides=optical_waveguides,
        )
    else:
        raise ValueError(
            f"Invalid CPW type: {cpw_params['type']}. Valid types are 'straight' and 'trail'."
        )

    # Define CPW pad
    pad = cpw_pad(
        cpw_xs=_cpw_xs,
        optical_waveguide_xs=optical_waveguides["terminal_xs"],
        pitch=cpw_pad_params["pitch"],
        length_straight=cpw_pad_params["length_straight"],
        length_tapered=cpw_pad_params["length_tapered"],
        ground_pad_width=cpw_pad_params["ground_pad_width"],
        m2_bonding_pads_params=m2_bonding_pad_params,
    )
    pad_xs = get_pad_xs(
        cpw_xs=_cpw_xs,
        pitch=cpw_pad_params["pitch"],
        ground_pad_width=cpw_pad_params["ground_pad_width"],
    )

    if dc_phase_shifter_node:
        top_pad = rectangular_cpw_pad(
            cpw_xs=_cpw_xs,
            optical_waveguide_xs=optical_waveguides["terminal_xs"],
            pitch=cpw_pad_params["pitch"],
            length_straight=cpw_pad_params["length_straight"],
            length_tapered=cpw_pad_params["length_tapered"],
            ground_pad_width=cpw_pad_params.get("dc_ground_width", cpw_pad_params["ground_pad_width"]),
            m2_bonding_pads_params=m2_bonding_pad_params,
            dc_pad_width=cpw_pad_params.get("dc_pad_width", 80.0),
        )
    else:
        top_pad = pad

    if electric_phase_shifter_node or dc_phase_shifter_node:
        _active_length = electric_phase_shifter_length if electric_phase_shifter_node else dc_phase_shifter_length
        if cpw_params["type"] == "straight":
            top_cpw = straight_cpw(
                cpw_xs=_cpw_xs,
                modulation_length=_active_length,
                optical_waveguides=optical_waveguides,
            )
        elif cpw_params["type"] == "trail":
            top_cpw = trail_cpw(
                cpw_xs=_cpw_xs,
                modulation_length=_active_length,
                trail_params=trail_params,
                optical_waveguides=optical_waveguides,
            )
        else:
            raise ValueError(
                f"Invalid CPW type: {cpw_params['type']}. Valid types are 'straight' and 'trail'."
            )

    cpw_ref = MZM << cpw

    if cpw_pad_params["single_side"]:
        # Define optical combiner
        if thermal_phase_shifter_node or electric_phase_shifter_node or dc_phase_shifter_node:
            combiner1 = optical_combiner_direct(
                optical_xs=optical_waveguides["terminal_xs"],
                cpw_xs=_cpw_xs if (electric_phase_shifter_node or dc_phase_shifter_node) else pad_xs,
                mmi_cell=mmi_cell,
                heater_section_length=optical_waveguide_params["heater_section_length"] if thermal_phase_shifter_node else 0.0,
                mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
                cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
                imbalance_length=0.0,
                roc=optical_waveguide_params["roc"],
            )
        else:
            combiner1 = mmi_cell
        combiner2 = None
        pad1_ref = MZM << pad
        pad1_ref.connect("e2", cpw_ref.ports["e1"])

        vertical_offset = optical_waveguide_params.get("vertical_offset", 0.0)
        sbend_length = optical_waveguide_params.get("sbend_length", 150.0)
        sbend_offset = optical_waveguide_params.get("sbend_offset", 0.0)

        # Compute dynamic S-bend offset from the pitch difference
        # between the pad waveguide ports and the combiner output ports.
        # Use max(pitch_diff, sbend_offset) so that even when pitches match
        # (e.g. thermal-only case), sbend_offset still provides routing clearance.
        _combiner_upper_port = "o3" if (thermal_phase_shifter_node or electric_phase_shifter_node or dc_phase_shifter_node) else "o2"
        pad_upper_y = abs(pad.ports["o1"].dcenter[1])
        combiner_upper_y = abs(combiner1.ports[_combiner_upper_port].dcenter[1])
        pitch_diff = pad_upper_y - combiner_upper_y
        if not thermal_phase_shifter_node and not dc_phase_shifter_node and not electric_phase_shifter_node:
            target_spacing = 10.0 if sbend_offset == 40.0 or sbend_offset == 0.0 else max(sbend_offset, 5.0)
            computed_offset = pad_upper_y - (target_spacing / 2)
        else:
            computed_offset = max(pitch_diff, sbend_offset)
        computed_sbend_length = 3.5 * computed_offset if computed_offset > 0.0 else 0.0
        if vertical_offset > 0.0 and folding:
            combiner1_ref = MZM << combiner1
            combiner1_ref.dmirror_x()

            if thermal_phase_shifter_node or electric_phase_shifter_node or dc_phase_shifter_node:
                port_out_top_name = "o3"
                port_out_bottom_name = "o4"
            else:
                port_out_top_name = "o2"
                port_out_bottom_name = "o3"

            if electric_phase_shifter_node or dc_phase_shifter_node:
                if electric_phase_shifter_node and termination is None:
                    raise ValueError(
                        "termination component must be provided to base_mzm "
                        "when electric_phase_shifter_node is True and single_side is True."
                    )
                top_cpw_ref = MZM << top_cpw
                top_pad1_ref = MZM << top_pad
                
                # Connect chain: combiner1_ref -> top_cpw_ref -> top_pad1_ref
                top_cpw_ref.connect("o2", combiner1_ref.ports[port_out_top_name])
                top_pad1_ref.connect("e2", top_cpw_ref.ports["e1"])
                
                if electric_phase_shifter_node:
                    top_termination_ref = MZM << termination
                    top_termination_ref.connect("e1", top_cpw_ref.ports["e2"])

                # Position the whole top row chain
                dx = pad1_ref.ports["o1"].dcenter[0] - top_pad1_ref.ports["o4"].dcenter[0] + horizontal_offset
                dy = pad1_ref.ports["o1"].dcenter[1] + vertical_offset - top_pad1_ref.ports["o4"].dcenter[1]
                combiner1_ref.dmove((dx, dy))
                top_cpw_ref.dmove((dx, dy))
                top_pad1_ref.dmove((dx, dy))
                if electric_phase_shifter_node:
                    top_termination_ref.dmove((dx, dy))

                if computed_offset > 0.0:
                    top_sb_top = MZM << bend_S_spline(
                        size=(computed_sbend_length, computed_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom = MZM << bend_S_spline(
                        size=(computed_sbend_length, computed_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom.dmirror_y()

                    top_sb_top.connect("o1", top_pad1_ref.ports["o1"])
                    top_sb_bottom.connect("o1", top_pad1_ref.ports["o4"])

                    port_target_top = top_sb_bottom.ports["o2"]
                    port_target_bottom = top_sb_top.ports["o2"]
                else:
                    port_target_top = top_pad1_ref.ports["o4"]
                    port_target_bottom = top_pad1_ref.ports["o1"]
            else:
                # Position combiner1_ref at the vertical offset
                dx = pad1_ref.ports["o1"].dcenter[0] - combiner1_ref.ports[port_out_bottom_name].dcenter[0] + horizontal_offset
                dy = pad1_ref.ports["o1"].dcenter[1] + vertical_offset - combiner1_ref.ports[port_out_bottom_name].dcenter[1]
                combiner1_ref.dmove((dx, dy))

                if computed_offset > 0.0:
                    top_sb_top = MZM << bend_S_spline(
                        size=(computed_sbend_length, computed_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom = MZM << bend_S_spline(
                        size=(computed_sbend_length, computed_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom.dmirror_y()

                    if not thermal_phase_shifter_node:
                        top_sb_top.connect("o1", combiner1_ref.ports[port_out_bottom_name])
                        top_sb_bottom.connect("o1", combiner1_ref.ports[port_out_top_name])

                        port_target_top = top_sb_top.ports["o2"]
                        port_target_bottom = top_sb_bottom.ports["o2"]
                    else:
                        top_sb_top.connect("o1", combiner1_ref.ports[port_out_top_name])
                        top_sb_bottom.connect("o1", combiner1_ref.ports[port_out_bottom_name])

                        port_target_top = top_sb_bottom.ports["o2"]
                        port_target_bottom = top_sb_top.ports["o2"]
                else:
                    port_target_top = combiner1_ref.ports[port_out_bottom_name]
                    port_target_bottom = combiner1_ref.ports[port_out_top_name]


            # Connect S-bends to the CPW pad ports
            if computed_offset > 0.0:
                sb_top = MZM << bend_S_spline(
                    size=(computed_sbend_length, computed_offset),
                    cross_section=optical_waveguides["terminal_xs"]
                )
                sb_bottom = MZM << bend_S_spline(
                    size=(computed_sbend_length, computed_offset),
                    cross_section=optical_waveguides["terminal_xs"]
                )
                sb_bottom.dmirror_y()

                sb_top.connect("o1", pad1_ref.ports["o1"])
                sb_bottom.connect("o1", pad1_ref.ports["o4"])

                port_top = sb_top.ports["o2"]
                port_bottom = sb_bottom.ports["o2"]
            else:
                port_top = pad1_ref.ports["o1"]
                port_bottom = pad1_ref.ports["o4"]

            # Route the loop-backs using route_single
            route_top = gf.routing.route_single(
                MZM,
                port1=port_top,
                port2=port_target_top,
                cross_section=optical_waveguides["terminal_xs"],
                bend=gf.components.bend_euler,
                straight="straight_rwg700_oband",
                radius=60.0,
                start_straight_length=5.0,
            )
            route_bottom = gf.routing.route_single(
                MZM,
                port1=port_bottom,
                port2=port_target_bottom,
                cross_section=optical_waveguides["terminal_xs"],
                bend=gf.components.bend_euler,
                straight="straight_rwg700_oband",
                radius=60.0,
                start_straight_length=15.0,
            )
        else:
            combiner1_ref = MZM << combiner1
            port_conn_name = "o3" if (thermal_phase_shifter_node or electric_phase_shifter_node or dc_phase_shifter_node) else "o2"
            if electric_phase_shifter_node or dc_phase_shifter_node:
                if electric_phase_shifter_node and termination is None:
                    raise ValueError(
                        "termination component must be provided to base_mzm "
                        "when electric_phase_shifter_node is True and single_side is True."
                    )
                top_cpw_ref = MZM << top_cpw
                top_pad1_ref = MZM << top_pad

                # Reverse-ordered connect for unfolded chain:
                # 1. Connect top_pad1_ref to pad1_ref
                top_pad1_ref.connect("o1", pad1_ref.ports["o1"])
                # 2. Connect top_cpw_ref to top_pad1_ref
                top_cpw_ref.connect("e2", top_pad1_ref.ports["e2"])
                
                if electric_phase_shifter_node:
                    top_termination_ref = MZM << termination
                    # 3. Connect top_termination_ref to top_cpw_ref
                    top_termination_ref.connect("e1", top_cpw_ref.ports["e1"])
                
                # 4. Connect combiner1_ref to top_cpw_ref
                combiner1_ref.connect(port_conn_name, top_cpw_ref.ports["o1"])

                # Shift upper arm elements westward by horizontal_offset.
                # When dc_phase_shifter_node=True and vertical_offset==0, compute dy_offset
                # so that top_pad1_ref.o4 aligns exactly with pad1_ref.o1 in Y.
                # This makes route_bundle emit straight horizontal waveguides (same-row layout).
                # For all other cases, dy_offset == vertical_offset (unchanged behaviour).
                if dc_phase_shifter_node and vertical_offset == 0.0:
                    dy_offset = pad1_ref.ports["o1"].dcenter[1] - top_pad1_ref.ports["o4"].dcenter[1]
                else:
                    dy_offset = vertical_offset
                top_pad1_ref.dmove((-horizontal_offset, dy_offset))
                top_cpw_ref.dmove((-horizontal_offset, dy_offset))
                combiner1_ref.dmove((-horizontal_offset, dy_offset))
                if electric_phase_shifter_node:
                    top_termination_ref.dmove((-horizontal_offset, dy_offset))

                # Connect the staggered upper pad outputs to the lower pad inputs
                gf.routing.route_bundle(
                    MZM,
                    ports1=[top_pad1_ref.ports["o4"], top_pad1_ref.ports["o1"]],
                    ports2=[pad1_ref.ports["o1"], pad1_ref.ports["o4"]],
                    cross_section=optical_waveguides["terminal_xs"],
                    bend=gf.components.bend_euler,
                    straight="straight_rwg700_oband",
                    radius=60.0,
                )
            else:
                if vertical_offset > 0.0 or horizontal_offset > 0.0:
                    combiner1_ref.connect(port_conn_name, pad1_ref.ports["o1"])
                    # Identify the top and bottom output ports of the combiner
                    port_top_name = "o3"
                    port_bottom_name = "o4" if thermal_phase_shifter_node else "o2"
                    
                    # Get coordinates before the shift
                    y_top_placed = combiner1_ref.ports[port_top_name].dcenter[1]
                    y_bottom_placed = combiner1_ref.ports[port_bottom_name].dcenter[1]
                    y_center_placed = (y_top_placed + y_bottom_placed) / 2.0
                    
                    pad_top_y = pad1_ref.ports["o1"].dcenter[1]
                    pad_bottom_y = pad1_ref.ports["o4"].dcenter[1]
                    pad_center_y = (pad_top_y + pad_bottom_y) / 2.0
                    
                    # Shift combiner1_ref to center it at vertical_offset
                    dy_shift = pad_center_y + vertical_offset - y_center_placed
                    combiner1_ref.dmove((-horizontal_offset, dy_shift))
                    
                    # Get final coordinates of the combiner output ports
                    y_top_final = y_top_placed + dy_shift
                    y_bottom_final = y_bottom_placed + dy_shift
                    
                    # Calculate S-bend vertical sizes based on connectivity
                    if thermal_phase_shifter_node:
                        # Parallel S-bend connections
                        dy_top = pad_top_y - y_top_final
                        dy_bottom = pad_bottom_y - y_bottom_final
                    else:
                        # Crossed S-bend connections
                        dy_top = pad_top_y - y_bottom_final
                        dy_bottom = pad_bottom_y - y_top_final

                    # Instantiate and place S-bends directly
                    sb_top = MZM << bend_S_spline(
                        size=(horizontal_offset, dy_top),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    sb_bottom = MZM << bend_S_spline(
                        size=(horizontal_offset, dy_bottom),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    
                    sb_top.connect("o2", pad1_ref.ports["o1"])
                    sb_bottom.connect("o2", pad1_ref.ports["o4"])
                else:
                    # Compute required S-bend offset from the pitch difference
                    # between the pad waveguide ports and the combiner output ports
                    pad_upper_y = abs(pad.ports["o1"].dcenter[1])
                    combiner_upper_y = abs(combiner1.ports[port_conn_name].dcenter[1])
                    computed_offset = pad_upper_y - combiner_upper_y

                    if computed_offset > 0.0:
                        sbend_ratio = 3.5
                        computed_length = sbend_ratio * computed_offset
                        sb_top = MZM << bend_S_spline(
                            size=(computed_length, computed_offset),
                            cross_section=optical_waveguides["terminal_xs"]
                        )
                        sb_bottom = MZM << bend_S_spline(
                            size=(computed_length, computed_offset),
                            cross_section=optical_waveguides["terminal_xs"]
                        )
                        sb_bottom.dmirror_y()

                        sb_top.connect("o2", pad1_ref.ports["o1"])
                        sb_bottom.connect("o2", pad1_ref.ports["o4"])

                        combiner1_ref.connect(port_conn_name, sb_top.ports["o1"])
                    else:
                        combiner1_ref.connect(port_conn_name, pad1_ref.ports["o1"])

        # ==========================================
        # Optical Path Matching Section for combiner2
        # ==========================================
        import numpy as np
        from _utils.spline import spline_clamped_path

        mmi_cell_name = mmi_cell.name
        if "2x2" in mmi_cell_name:
            mmi_len = float(abs(mmi_cell.ports["o3"].dcenter[0] - mmi_cell.ports["o2"].dcenter[0]))
        else:
            mmi_len = float(abs(mmi_cell.ports["o2"].dcenter[0] - mmi_cell.ports["o1"].dcenter[0]))

        signal_width, _, gap, _ = get_cpw_from_xs(pad_xs)
        if "2x2" in mmi_cell_name:
            mmi_dy = abs(mmi_cell.ports["o3"].dcenter[1] - mmi_cell.ports["o4"].dcenter[1])
        elif "1x2" in mmi_cell_name:
            mmi_dy = abs(mmi_cell.ports["o2"].dcenter[1] - mmi_cell.ports["o3"].dcenter[1])
        else:
            mmi_dy = 0.0
        # Combiner-internal S-bend length (MMI pitch → CPW/pad pitch)
        y_offset = signal_width / 2 + gap / 2 - mmi_dy / 2
        sbend_ratio = 3.5
        x_offset = sbend_ratio * y_offset
        t = np.linspace(0, 1, 201)
        path = spline_clamped_path(t, start=(0.0, 0.0), end=(x_offset, y_offset))
        sbend_len = float(path.length())

        # Routing S-bend length (using computed_offset for loop-back clearance)
        if computed_offset > 0.0:
            routing_x = 3.5 * computed_offset
            routing_path = spline_clamped_path(t, start=(0.0, 0.0), end=(routing_x, computed_offset))
            routing_sbend_len = float(routing_path.length())
        else:
            routing_sbend_len = 0.0

        mmi_conn_len_param = optical_waveguide_params.get("mmi_connection_length", 0.0)
        heater_len = optical_waveguide_params.get("heater_section_length", 0.0)
        cpw_conn_len = optical_waveguide_params.get("cpw_connection_length", 0.0)

        # combiner1 has imbalance_length = 0.0
        if thermal_phase_shifter_node:
            L_turn_len = np.pi / 2 * optical_waveguide_params.get("roc", 60.0)
            # combiner1 is now symmetric: both arms use 2 L-turns (no V-segment).
            # The 75 µm asymmetry that was previously in combiner1 is now
            # compensated entirely inside combiner2 via imbalance_length_param.
            arm_up_len1 = 2 * L_turn_len
            arm_down_len1 = 2 * L_turn_len

            path_up_input1 = mmi_len + mmi_conn_len_param + sbend_len + arm_up_len1
            path_down_input1 = mmi_len + mmi_conn_len_param + sbend_len + arm_down_len1

            L_comb1_up = path_up_input1 + heater_len + arm_up_len1 + cpw_conn_len
            L_comb1_down = path_down_input1 + heater_len + arm_down_len1 + cpw_conn_len
        elif electric_phase_shifter_node or dc_phase_shifter_node:
            L_comb1_up = mmi_len + mmi_conn_len_param + sbend_len + cpw_conn_len
            L_comb1_down = mmi_len + mmi_conn_len_param + sbend_len + cpw_conn_len
        else:
            L_comb1_up = mmi_len
            L_comb1_down = mmi_len

        r_top_len = float(route_top.length) * 0.001 if ('route_top' in locals() or 'route_top' in globals()) and hasattr(route_top, 'length') else 0.0
        r_bottom_len = float(route_bottom.length) * 0.001 if ('route_bottom' in locals() or 'route_bottom' in globals()) and hasattr(route_bottom, 'length') else 0.0

        # Bottom-row routing S-bend lengths (sb_top / sb_bottom)
        s_top_len = float(sb_top.info["length"]) if ('sb_top' in locals() or 'sb_top' in globals()) and hasattr(sb_top, 'info') and 'length' in sb_top.info else 0.0
        s_bottom_len = float(sb_bottom.info["length"]) if ('sb_bottom' in locals() or 'sb_bottom' in globals()) and hasattr(sb_bottom, 'info') and 'length' in sb_bottom.info else 0.0

        # Top-row routing S-bend lengths (top_sb_top / top_sb_bottom)
        ts_top_len = float(top_sb_top.info["length"]) if ('top_sb_top' in locals() or 'top_sb_top' in globals()) and hasattr(top_sb_top, 'info') and 'length' in top_sb_top.info else 0.0
        ts_bottom_len = float(top_sb_bottom.info["length"]) if ('top_sb_bottom' in locals() or 'top_sb_bottom' in globals()) and hasattr(top_sb_bottom, 'info') and 'length' in top_sb_bottom.info else 0.0

        if electric_phase_shifter_node or dc_phase_shifter_node:
            top_pad_len = cpw_pad_params["length_straight"] + cpw_pad_params["length_tapered"]
            _active_len = electric_phase_shifter_length if electric_phase_shifter_node else dc_phase_shifter_length
            top_row_active_len = _active_len + top_pad_len
        else:
            top_row_active_len = 0.0

        main_pad_len = cpw_pad_params["length_straight"] + cpw_pad_params["length_tapered"]
        bottom_row_active_len = main_pad_len + modulation_length

        if vertical_offset > 0.0 and folding:
            path_up_compensation = L_comb1_up + ts_top_len + top_row_active_len + r_bottom_len + s_bottom_len + bottom_row_active_len
            path_down_compensation = L_comb1_down + ts_bottom_len + top_row_active_len + r_top_len + s_top_len + bottom_row_active_len
        else:
            path_up_compensation = L_comb1_up + s_top_len + bottom_row_active_len
            path_down_compensation = L_comb1_down + s_bottom_len + bottom_row_active_len

        imbalance_length = optical_waveguide_params.get("imbalance_length", 0.0)
        imbalance_length_param = (path_up_compensation - path_down_compensation) + imbalance_length - 150.0

        combiner2 = optical_combiner_direct(
            optical_xs=optical_waveguides["terminal_xs"],
            cpw_xs=_cpw_xs,
            mmi_cell=mmi_cell,
            heater_section_length=0.0,
            mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
            cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
            imbalance_length=imbalance_length_param,
            roc=optical_waveguide_params["roc"],
        )

        combiner2_ref = MZM << combiner2
        combiner2_ref.connect("o4", cpw_ref.ports["o2"])

        MZM.add_port(
            name="e1",
            port=pad1_ref.ports["e1"],
        )
        MZM.add_port(
            name="e2",
            port=cpw_ref.ports["e2"],
        )

    else:
        # Define optical combiner
        if thermal_phase_shifter_node or electric_phase_shifter_node or dc_phase_shifter_node:
            combiner1 = optical_combiner_direct(
                optical_xs=optical_waveguides["terminal_xs"],
                cpw_xs=_cpw_xs if (electric_phase_shifter_node or dc_phase_shifter_node) else pad_xs,
                mmi_cell=mmi_cell,
                mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
                cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
                heater_section_length=optical_waveguide_params["heater_section_length"] if thermal_phase_shifter_node else 0.0,
                imbalance_length=0.0,
                roc=optical_waveguide_params["roc"],
            )
        else:
            combiner1 = mmi_cell
        combiner2 = None
        pad1_ref = MZM << pad
        pad1_ref.connect("e2", cpw_ref.ports["e1"])
        pad2_ref = MZM << pad
        pad2_ref.connect("e2", cpw_ref.ports["e2"])

        vertical_offset = optical_waveguide_params.get("vertical_offset", 0.0)
        sbend_length = optical_waveguide_params.get("sbend_length", 150.0)
        sbend_offset = optical_waveguide_params.get("sbend_offset", 0.0)
        pad_upper_y = abs(pad.ports["o1"].dcenter[1])
        if not thermal_phase_shifter_node and not dc_phase_shifter_node and not electric_phase_shifter_node:
            target_spacing = 10.0 if sbend_offset == 40.0 or sbend_offset == 0.0 else max(sbend_offset, 5.0)
            sbend_offset = pad_upper_y - (target_spacing / 2)
            sbend_length = 3.5 * sbend_offset if sbend_offset > 0.0 else 0.0
        if vertical_offset > 0.0 and folding:
            combiner1_ref = MZM << combiner1
            combiner1_ref.dmirror_x()

            if thermal_phase_shifter_node or electric_phase_shifter_node or dc_phase_shifter_node:
                port_out_top_name = "o3"
                port_out_bottom_name = "o4"
            else:
                port_out_top_name = "o2"
                port_out_bottom_name = "o3"

            if electric_phase_shifter_node or dc_phase_shifter_node:
                top_cpw_ref = MZM << top_cpw
                top_pad1_ref = MZM << top_pad
                
                if electric_phase_shifter_node:
                    top_pad2_ref = MZM << top_pad

                    # Connect chain: combiner1_ref -> top_pad2_ref -> top_cpw_ref -> top_pad1_ref
                    top_pad2_ref.connect("o2", combiner1_ref.ports[port_out_top_name])
                    top_cpw_ref.connect("o2", top_pad2_ref.ports["o1"])
                    top_pad1_ref.connect("e2", top_cpw_ref.ports["e1"])
                else:
                    # Connect chain: combiner1_ref -> top_cpw_ref -> top_pad1_ref
                    top_cpw_ref.connect("o2", combiner1_ref.ports[port_out_top_name])
                    top_pad1_ref.connect("e2", top_cpw_ref.ports["e1"])

                # Position the whole top row chain
                dx = pad1_ref.ports["o1"].dcenter[0] - top_pad1_ref.ports["o4"].dcenter[0] + horizontal_offset
                dy = pad1_ref.ports["o1"].dcenter[1] + vertical_offset - top_pad1_ref.ports["o4"].dcenter[1]
                combiner1_ref.dmove((dx, dy))
                top_cpw_ref.dmove((dx, dy))
                top_pad1_ref.dmove((dx, dy))
                if electric_phase_shifter_node:
                    top_pad2_ref.dmove((dx, dy))

                if sbend_length > 0.0 and sbend_offset > 0.0:
                    top_sb_top = MZM << bend_S_spline(
                        size=(sbend_length, sbend_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom = MZM << bend_S_spline(
                        size=(sbend_length, sbend_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom.dmirror_y()

                    top_sb_top.connect("o1", top_pad1_ref.ports["o1"])
                    top_sb_bottom.connect("o1", top_pad1_ref.ports["o4"])

                    port_target_top = top_sb_bottom.ports["o2"]
                    port_target_bottom = top_sb_top.ports["o2"]
                else:
                    port_target_top = top_pad1_ref.ports["o4"]
                    port_target_bottom = top_pad1_ref.ports["o1"]
            else:
                # Position combiner1_ref at the vertical offset
                dx = pad1_ref.ports["o1"].dcenter[0] - combiner1_ref.ports[port_out_bottom_name].dcenter[0] + horizontal_offset
                dy = pad1_ref.ports["o1"].dcenter[1] + vertical_offset - combiner1_ref.ports[port_out_bottom_name].dcenter[1]
                combiner1_ref.dmove((dx, dy))

                if sbend_length > 0.0 and sbend_offset > 0.0:
                    top_sb_top = MZM << bend_S_spline(
                        size=(sbend_length, sbend_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom = MZM << bend_S_spline(
                        size=(sbend_length, sbend_offset),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    top_sb_bottom.dmirror_y()

                    if not thermal_phase_shifter_node:
                        top_sb_top.connect("o1", combiner1_ref.ports[port_out_bottom_name])
                        top_sb_bottom.connect("o1", combiner1_ref.ports[port_out_top_name])

                        port_target_top = top_sb_top.ports["o2"]
                        port_target_bottom = top_sb_bottom.ports["o2"]
                    else:
                        top_sb_top.connect("o1", combiner1_ref.ports[port_out_top_name])
                        top_sb_bottom.connect("o1", combiner1_ref.ports[port_out_bottom_name])

                        port_target_top = top_sb_bottom.ports["o2"]
                        port_target_bottom = top_sb_top.ports["o2"]
                else:
                    port_target_top = combiner1_ref.ports[port_out_bottom_name]
                    port_target_bottom = combiner1_ref.ports[port_out_top_name]


            # Connect S-bends to the CPW pad ports if sbend_offset > 0
            if sbend_length > 0.0 and sbend_offset > 0.0:
                sb_top = MZM << bend_S_spline(
                    size=(sbend_length, sbend_offset),
                    cross_section=optical_waveguides["terminal_xs"]
                )
                sb_bottom = MZM << bend_S_spline(
                    size=(sbend_length, sbend_offset),
                    cross_section=optical_waveguides["terminal_xs"]
                )
                sb_bottom.dmirror_y()

                sb_top.connect("o1", pad1_ref.ports["o1"])
                sb_bottom.connect("o1", pad1_ref.ports["o4"])

                port_top = sb_top.ports["o2"]
                port_bottom = sb_bottom.ports["o2"]
            else:
                port_top = pad1_ref.ports["o1"]
                port_bottom = pad1_ref.ports["o4"]

            # Route the loop-backs using route_single
            route_top = gf.routing.route_single(
                MZM,
                port1=port_top,
                port2=port_target_top,
                cross_section=optical_waveguides["terminal_xs"],
                bend=gf.components.bend_euler,
                straight="straight_rwg700_oband",
                radius=60.0,
                start_straight_length=5.0,
            )
            route_bottom = gf.routing.route_single(
                MZM,
                port1=port_bottom,
                port2=port_target_bottom,
                cross_section=optical_waveguides["terminal_xs"],
                bend=gf.components.bend_euler,
                straight="straight_rwg700_oband",
                radius=60.0,
                start_straight_length=15.0,
            )
        else:
            combiner1_ref = MZM << combiner1
            port_conn_name = "o3" if (thermal_phase_shifter_node or electric_phase_shifter_node or dc_phase_shifter_node) else "o2"
            if electric_phase_shifter_node or dc_phase_shifter_node:
                top_cpw_ref = MZM << top_cpw
                top_pad1_ref = MZM << top_pad

                # Reverse-ordered connect for unfolded chain:
                # 1. Connect top_pad1_ref to pad1_ref
                top_pad1_ref.connect("o1", pad1_ref.ports["o1"])
                # 2. Connect top_cpw_ref to top_pad1_ref
                top_cpw_ref.connect("e2", top_pad1_ref.ports["e2"])
                
                if electric_phase_shifter_node:
                    top_pad2_ref = MZM << top_pad
                    # 3. Connect top_pad2_ref to top_cpw_ref
                    top_pad2_ref.connect("e2", top_cpw_ref.ports["e1"])
                    # 4. Connect combiner1_ref to top_pad2_ref
                    combiner1_ref.connect(port_conn_name, top_pad2_ref.ports["o1"])
                else:
                    combiner1_ref.connect(port_conn_name, top_cpw_ref.ports["o1"])

                # Shift upper arm elements westward by horizontal_offset.
                # When dc_phase_shifter_node=True and vertical_offset==0, compute dy_offset
                # so that top_pad1_ref.o4 aligns exactly with pad1_ref.o1 in Y.
                # This makes route_bundle emit straight horizontal waveguides (same-row layout).
                # For all other cases, dy_offset == vertical_offset (unchanged behaviour).
                if dc_phase_shifter_node and vertical_offset == 0.0:
                    dy_offset = pad1_ref.ports["o1"].dcenter[1] - top_pad1_ref.ports["o4"].dcenter[1]
                else:
                    dy_offset = vertical_offset
                top_pad1_ref.dmove((-horizontal_offset, dy_offset))
                top_cpw_ref.dmove((-horizontal_offset, dy_offset))
                combiner1_ref.dmove((-horizontal_offset, dy_offset))
                if electric_phase_shifter_node:
                    top_pad2_ref.dmove((-horizontal_offset, dy_offset))

                # Connect the staggered upper pad outputs to the lower pad inputs
                gf.routing.route_bundle(
                    MZM,
                    ports1=[top_pad1_ref.ports["o4"], top_pad1_ref.ports["o1"]],
                    ports2=[pad1_ref.ports["o1"], pad1_ref.ports["o4"]],
                    cross_section=optical_waveguides["terminal_xs"],
                    bend=gf.components.bend_euler,
                    straight="straight_rwg700_oband",
                    radius=60.0,
                )
            else:
                if vertical_offset > 0.0 or horizontal_offset > 0.0:
                    # Identify the top and bottom output ports of the combiner
                    port_top_name = "o3"
                    port_bottom_name = "o4" if thermal_phase_shifter_node else "o2"
                    
                    # Get coordinates before the shift
                    y_top_placed = combiner1_ref.ports[port_top_name].dcenter[1]
                    y_bottom_placed = combiner1_ref.ports[port_bottom_name].dcenter[1]
                    y_center_placed = (y_top_placed + y_bottom_placed) / 2.0
                    
                    pad_top_y = pad1_ref.ports["o1"].dcenter[1]
                    pad_bottom_y = pad1_ref.ports["o4"].dcenter[1]
                    pad_center_y = (pad_top_y + pad_bottom_y) / 2.0
                    
                    # Shift combiner1_ref to center it at vertical_offset
                    dy_shift = pad_center_y + vertical_offset - y_center_placed
                    combiner1_ref.dmove((-horizontal_offset, dy_shift))
                    
                    # Get final coordinates of the combiner output ports
                    y_top_final = y_top_placed + dy_shift
                    y_bottom_final = y_bottom_placed + dy_shift
                    
                    # Calculate S-bend vertical sizes based on connectivity
                    if thermal_phase_shifter_node:
                        # Parallel S-bend connections
                        dy_top = pad_top_y - y_top_final
                        dy_bottom = pad_bottom_y - y_bottom_final
                    else:
                        # Crossed S-bend connections
                        dy_top = pad_top_y - y_bottom_final
                        dy_bottom = pad_bottom_y - y_top_final

                    # Instantiate and place S-bends directly
                    sb_top = MZM << bend_S_spline(
                        size=(horizontal_offset, dy_top),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    sb_bottom = MZM << bend_S_spline(
                        size=(horizontal_offset, dy_bottom),
                        cross_section=optical_waveguides["terminal_xs"]
                    )
                    
                    sb_top.connect("o2", pad1_ref.ports["o1"])
                    sb_bottom.connect("o2", pad1_ref.ports["o4"])
                else:
                    combiner1_ref.connect(port_conn_name, pad1_ref.ports["o1"])

        # ==========================================
        # Optical Path Matching Section for combiner2
        # ==========================================
        import numpy as np
        from _utils.spline import spline_clamped_path

        mmi_cell_name = mmi_cell.name
        if "2x2" in mmi_cell_name:
            mmi_len = float(abs(mmi_cell.ports["o3"].dcenter[0] - mmi_cell.ports["o2"].dcenter[0]))
        else:
            mmi_len = float(abs(mmi_cell.ports["o2"].dcenter[0] - mmi_cell.ports["o1"].dcenter[0]))

        signal_width, _, gap, _ = get_cpw_from_xs(pad_xs)
        if "2x2" in mmi_cell_name:
            mmi_dy = abs(mmi_cell.ports["o3"].dcenter[1] - mmi_cell.ports["o4"].dcenter[1])
        elif "1x2" in mmi_cell_name:
            mmi_dy = abs(mmi_cell.ports["o2"].dcenter[1] - mmi_cell.ports["o3"].dcenter[1])
        else:
            mmi_dy = 0.0
        y_offset = signal_width / 2 + gap / 2 - mmi_dy / 2
        sbend_ratio = 3.5
        x_offset = sbend_ratio * y_offset
        t = np.linspace(0, 1, 201)
        path = spline_clamped_path(t, start=(0.0, 0.0), end=(x_offset, y_offset))
        sbend_len = float(path.length())

        mmi_conn_len_param = optical_waveguide_params.get("mmi_connection_length", 0.0)
        heater_len = optical_waveguide_params.get("heater_section_length", 0.0)
        cpw_conn_len = optical_waveguide_params.get("cpw_connection_length", 0.0)

        # combiner1 has imbalance_length = 0.0
        if thermal_phase_shifter_node:
            L_turn_len = np.pi / 2 * optical_waveguide_params.get("roc", 60.0)
            # combiner1 is now symmetric: both arms use 2 L-turns (no V-segment).
            # The 75 µm asymmetry that was previously in combiner1 is now
            # compensated entirely inside combiner2 via imbalance_length_param.
            arm_up_len1 = 2 * L_turn_len
            arm_down_len1 = 2 * L_turn_len

            path_up_input1 = mmi_len + mmi_conn_len_param + sbend_len + arm_up_len1
            path_down_input1 = mmi_len + mmi_conn_len_param + sbend_len + arm_down_len1

            L_comb1_up = path_up_input1 + heater_len + arm_up_len1 + cpw_conn_len
            L_comb1_down = path_down_input1 + heater_len + arm_down_len1 + cpw_conn_len
        elif electric_phase_shifter_node or dc_phase_shifter_node:
            L_comb1_up = mmi_len + mmi_conn_len_param + sbend_len + cpw_conn_len
            L_comb1_down = mmi_len + mmi_conn_len_param + sbend_len + cpw_conn_len
        else:
            L_comb1_up = mmi_len
            L_comb1_down = mmi_len

        r_top_len = float(route_top.length) * 0.001 if ('route_top' in locals() or 'route_top' in globals()) and hasattr(route_top, 'length') else 0.0
        r_bottom_len = float(route_bottom.length) * 0.001 if ('route_bottom' in locals() or 'route_bottom' in globals()) and hasattr(route_bottom, 'length') else 0.0

        s_top_len = float(sb_top.info["length"]) if ('sb_top' in locals() or 'sb_top' in globals()) and hasattr(sb_top, 'info') and 'length' in sb_top.info else 0.0
        s_bottom_len = float(sb_bottom.info["length"]) if ('sb_bottom' in locals() or 'sb_bottom' in globals()) and hasattr(sb_bottom, 'info') and 'length' in sb_bottom.info else 0.0

        if electric_phase_shifter_node or dc_phase_shifter_node:
            top_pad_len = cpw_pad_params["length_straight"] + cpw_pad_params["length_tapered"]
            _active_len = electric_phase_shifter_length if electric_phase_shifter_node else dc_phase_shifter_length
            if electric_phase_shifter_node:
                top_row_active_len = _active_len + 2 * top_pad_len
            else:
                top_row_active_len = _active_len + top_pad_len
        else:
            top_row_active_len = 0.0

        # Top-row routing S-bend lengths (top_sb_top / top_sb_bottom)
        ts_top_len = float(top_sb_top.info["length"]) if ('top_sb_top' in locals() or 'top_sb_top' in globals()) and hasattr(top_sb_top, 'info') and 'length' in top_sb_top.info else 0.0
        ts_bottom_len = float(top_sb_bottom.info["length"]) if ('top_sb_bottom' in locals() or 'top_sb_bottom' in globals()) and hasattr(top_sb_bottom, 'info') and 'length' in top_sb_bottom.info else 0.0

        main_pad_len = cpw_pad_params["length_straight"] + cpw_pad_params["length_tapered"]
        bottom_row_active_len = main_pad_len + modulation_length

        if vertical_offset > 0.0 and folding:
            path_up_compensation = L_comb1_up + ts_top_len + top_row_active_len + r_bottom_len + s_bottom_len + bottom_row_active_len
            path_down_compensation = L_comb1_down + ts_bottom_len + top_row_active_len + r_top_len + s_top_len + bottom_row_active_len
        else:
            path_up_compensation = L_comb1_up + s_top_len + bottom_row_active_len
            path_down_compensation = L_comb1_down + s_bottom_len + bottom_row_active_len

        imbalance_length = optical_waveguide_params.get("imbalance_length", 0.0)
        imbalance_length_param = (path_up_compensation - path_down_compensation) + imbalance_length - 150.0

        combiner2 = optical_combiner_direct(
            optical_xs=optical_waveguides["terminal_xs"],
            cpw_xs=pad_xs,
            mmi_cell=mmi_cell,
            heater_section_length=0.0,
            mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
            cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
            imbalance_length=imbalance_length_param,
            roc=optical_waveguide_params["roc"],
        )

        combiner2_ref = MZM << combiner2
        combiner2_ref.connect("o3", pad2_ref.ports["o1"])

        MZM.add_port(
            name="e1",
            port=pad1_ref.ports["e1"],
        )
        MZM.add_port(
            name="e2",
            port=pad2_ref.ports["e1"],
        )

    MZM.add_port(
        name="o1",
        port=combiner1_ref.ports["o1"],
    )
    if "2x2" in mmi_cell.name:
        MZM.add_port(
            name="o2",
            port=combiner1_ref.ports["o2"],
        )
        MZM.add_port(
            name="o3",
            port=combiner2_ref.ports["o2"],
        )
        MZM.add_port(
            name="o4",
            port=combiner2_ref.ports["o1"],
        )
    else:
        MZM.add_port(
            name="o2",
            port=combiner2_ref.ports["o1"],
        )

    if thermal_phase_shifter_node and optical_waveguide_params["heater_section_length"] > 0.0:
        MZM.add_port(
            name="ht1_1",
            port=combiner1_ref.ports["ht1_1"],
        )
        MZM.add_port(
            name="ht1_2",
            port=combiner1_ref.ports["ht1_2"],
        )
        MZM.add_port(
            name="ht2_1",
            port=combiner1_ref.ports["ht2_1"],
        )
        MZM.add_port(
            name="ht2_2",
            port=combiner1_ref.ports["ht2_2"],
        )

    if electric_phase_shifter_node or dc_phase_shifter_node:
        MZM.add_port(
            name="e_top1",
            port=top_pad1_ref.ports["e1"],
        )
        if electric_phase_shifter_node:
            if not cpw_pad_params["single_side"]:
                MZM.add_port(
                    name="e_top2",
                    port=top_pad2_ref.ports["e1"],
                )
            else:
                MZM.add_port(
                    name="_term_top",
                    port=top_termination_ref.ports["term"],
                )

    # Expose internal section ports at the top-level Component
    if "combiner1_ref" in locals() and combiner1_ref is not None:
        MZM.add_ports(combiner1_ref.ports, prefix="combiner1_")
    if "combiner2_ref" in locals() and combiner2_ref is not None:
        MZM.add_ports(combiner2_ref.ports, prefix="combiner2_")
    if "top_cpw_ref" in locals() and top_cpw_ref is not None:
        MZM.add_ports(top_cpw_ref.ports, prefix="top_cpw_")
    if "cpw_ref" in locals() and cpw_ref is not None:
        MZM.add_ports(cpw_ref.ports, prefix="bottom_cpw_")
    if "pad1_ref" in locals() and pad1_ref is not None:
        MZM.add_ports(pad1_ref.ports, prefix="pad1_")
    if "pad2_ref" in locals() and pad2_ref is not None:
        MZM.add_ports(pad2_ref.ports, prefix="pad2_")
    if "top_pad1_ref" in locals() and top_pad1_ref is not None:
        MZM.add_ports(top_pad1_ref.ports, prefix="top_pad1_")
    if "top_pad2_ref" in locals() and top_pad2_ref is not None:
        MZM.add_ports(top_pad2_ref.ports, prefix="top_pad2_")
    if "top_termination_ref" in locals() and top_termination_ref is not None:
        MZM.add_ports(top_termination_ref.ports, prefix="top_termination_")
    if "sb_top" in locals() and sb_top is not None:
        MZM.add_ports(sb_top.ports, prefix="sb_top_")
    if "sb_bottom" in locals() and sb_bottom is not None:
        MZM.add_ports(sb_bottom.ports, prefix="sb_bottom_")

    # Calculate path_up and path_down lengths incorporating the Optical Path Matching Section
    # detour inside combiner2
    L_turn_comb2 = np.pi / 2 * optical_waveguide_params.get("roc", 60.0)
    if imbalance_length_param > 0.0:
        upper_detour = 4 * L_turn_comb2
        lower_detour = 4 * L_turn_comb2 + imbalance_length_param + 150.0
    elif imbalance_length_param < 0.0:
        upper_detour = 4 * L_turn_comb2 - imbalance_length_param
        lower_detour = 4 * L_turn_comb2 + 150.0
    else:
        upper_detour = 4 * L_turn_comb2
        lower_detour = 4 * L_turn_comb2 + 150.0

    # Path Up connects to upper_detour inside combiner2 (due to 180-degree rotation), and Path Down connects to lower_detour inside combiner2
    path_up_length = path_up_compensation + upper_detour
    path_down_length = path_down_compensation + lower_detour
    propagation_difference = float(abs(path_up_length - path_down_length))

    MZM.info["path_up_length"] = path_up_length
    MZM.info["path_down_length"] = path_down_length
    MZM.info["propagation_difference"] = propagation_difference
    MZM.info["path_up_compensation"] = path_up_compensation
    MZM.info["path_down_compensation"] = path_down_compensation

    return MZM


if __name__ == "__main__":
    from ltoi300.cells import mmi2x2_oband
    from ltoi300.tech import xs_rwg700, xs_uni_cpw

    mmi = mmi2x2_oband()
    mzm = base_mzm(
        optical_xs=xs_rwg700,
        cpw_xs=xs_uni_cpw,
        modulation_length=2000.0,
        mmi_cell=mmi,
    )
    mzm.show()
