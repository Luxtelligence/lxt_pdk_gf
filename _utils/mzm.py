from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import CrossSectionSpec

from _utils.bends import L_turn_bend, bend_S_spline
from _utils.cross_section import get_cpw_from_xs
from _utils.gsg_rf import cpw_pad, get_pad_xs, straight_cpw, trail_cpw


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
    # If the imbalance length is 0, the CPW connection is directly connected to the SBEND ports
    if imbalance_length == 0.0:
        heater_segment = gf.components.straight(
            length=heater_section_length,
            cross_section=optical_xs,
        )

        heater_segment_up = combiner << heater_segment
        heater_segment_down = combiner << heater_segment
        heater_segment_up.connect("o1", sbend_up.ports["o2"])
        heater_segment_down.connect("o1", sbend_down.ports["o2"])

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

        cpw_connection = gf.components.straight(
            length=cpw_connection_length,
            cross_section=optical_xs,
        )
        cpw_connection_up = combiner << cpw_connection
        cpw_connection_down = combiner << cpw_connection
        cpw_connection_up.connect("o1", heater_segment_up.ports["o2"])
        cpw_connection_down.connect("o1", heater_segment_down.ports["o2"])

        combiner.add_port(
            name="o3",
            port=cpw_connection_up.ports["o2"],
        )
        combiner.add_port(
            name="o4",
            port=cpw_connection_down.ports["o2"],
        )

    else:
        L_turn = L_turn_bend(
            radius=roc,
            cross_section=optical_xs,
        )

        heater_segment = gf.components.straight(
            length=heater_section_length,
            cross_section=optical_xs,
        )

        imbalance_segment = gf.components.straight(
            length=imbalance_length / 2,
            cross_section=optical_xs,
        )

        cpw_connection = gf.components.straight(
            length=cpw_connection_length,
            cross_section=optical_xs,
        )

        symbol_to_component = {
            "c": (cpw_connection, "o1", "o2"),
            "L": (L_turn, "o1", "o2"),
            "_": (imbalance_segment, "o1", "o2"),
        }
        arm_up = gf.components.component_sequence(
            sequence="L_!L",
            symbol_to_component=symbol_to_component,
        )
        arm_up.flatten()
        arm_up_ref = combiner << arm_up
        arm_up_ref.connect("o1", sbend_up.ports["o2"])

        heater_segment_up = combiner << heater_segment
        heater_segment_up.connect("o1", arm_up_ref.ports["o2"])

        arm_up_right = combiner << arm_up
        arm_up_right.dmirror_x()
        arm_up_right.connect("o2", heater_segment_up.ports["o2"])

        arm_down = gf.components.component_sequence(
            sequence="!LL",
            symbol_to_component=symbol_to_component,
        )
        arm_down.flatten()
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

    combiner.flatten()

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
            "left_optical_branch": "mmi",
            "right_optical_branch": "mmi",
            "left_rf_pad": "probe",
            "right_rf_pad": "probe",
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

    MZM = gf.Component()

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

    cpw_ref = MZM << cpw

    if (
        cpw_pad_params["left_rf_pad"] == "probe"
        and cpw_pad_params["right_rf_pad"] == "termination"
        and optical_waveguide_params["left_optical_branch"] == "mmi"
        and optical_waveguide_params["right_optical_branch"] == "mmi"
    ):
        # Define optical combiner
        combiner1 = optical_combiner_direct(
            optical_xs=optical_waveguides["terminal_xs"],
            cpw_xs=pad_xs,
            mmi_cell=mmi_cell,
            heater_section_length=optical_waveguide_params["heater_section_length"],
            mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
            cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
            imbalance_length=optical_waveguide_params["imbalance_length"],
            roc=optical_waveguide_params["roc"],
        )
        # Define optical combiner
        combiner2 = optical_combiner_direct(
            optical_xs=optical_waveguides["terminal_xs"],
            cpw_xs=_cpw_xs,
            mmi_cell=mmi_cell,
            heater_section_length=0.0,
            mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
            cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
            imbalance_length=0.0,
            roc=optical_waveguide_params["roc"],
        )
        pad1_ref = MZM << pad
        pad1_ref.connect("e2", cpw_ref.ports["e1"])
        combiner1_ref = MZM << combiner1
        combiner1_ref.connect("o3", pad1_ref.ports["o1"])
        combiner2_ref = MZM << combiner2
        combiner2_ref.connect("o3", cpw_ref.ports["o3"])

        MZM.add_port(
            name="e1",
            port=pad1_ref.ports["e1"],
        )
        MZM.add_port(
            name="e2",
            port=cpw_ref.ports["e2"],
        )
        if "1x2" in mmi_cell.name:
            MZM.add_port(
                name="o1",
                port=combiner1_ref.ports["o1"],
            )
            MZM.add_port(
                name="o2",
                port=combiner2_ref.ports["o1"],
            )
        else:
            MZM.add_port(
                name="o2",
                port=combiner1_ref.ports["o1"],
            )
            MZM.add_port(
                name="o1",
                port=combiner1_ref.ports["o2"],
            )
            MZM.add_port(
                name="o4",
                port=combiner2_ref.ports["o1"],
            )
            MZM.add_port(
                name="o3",
                port=combiner2_ref.ports["o2"],
            )

    if (
        cpw_pad_params["left_rf_pad"] == "probe"
        and cpw_pad_params["right_rf_pad"] == "termination"
        and optical_waveguide_params["left_optical_branch"] == "mmi"
        and optical_waveguide_params["right_optical_branch"] == "open"
    ):
        # Define optical combiner
        combiner1 = optical_combiner_direct(
            optical_xs=optical_waveguides["terminal_xs"],
            cpw_xs=pad_xs,
            mmi_cell=mmi_cell,
            heater_section_length=optical_waveguide_params["heater_section_length"],
            mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
            cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
            imbalance_length=optical_waveguide_params["imbalance_length"],
            roc=optical_waveguide_params["roc"],
        )
        pad1_ref = MZM << pad
        pad1_ref.connect("e2", cpw_ref.ports["e1"])
        combiner1_ref = MZM << combiner1
        combiner1_ref.connect("o3", pad1_ref.ports["o1"])

        MZM.add_port(
            name="e1",
            port=pad1_ref.ports["e1"],
        )
        MZM.add_port(
            name="e2",
            port=cpw_ref.ports["e2"],
        )
        if "1x2" in mmi_cell.name:
            MZM.add_port(
                name="o1",
                port=combiner1_ref.ports["o1"],
            )
            MZM.add_port(
                name="o2",
                port=cpw_ref.ports["o2"],
            )
            MZM.add_port(
                name="o3",
                port=cpw_ref.ports["o3"],
            )
        else:
            MZM.add_port(
                name="o2",
                port=combiner1_ref.ports["o1"],
            )
            MZM.add_port(
                name="o1",
                port=combiner1_ref.ports["o2"],
            )
            MZM.add_port(
                name="o4",
                port=cpw_ref.ports["o3"],
            )
            MZM.add_port(
                name="o3",
                port=cpw_ref.ports["o2"],
            )

    if (
        cpw_pad_params["left_rf_pad"] == "probe"
        and cpw_pad_params["right_rf_pad"] == "termination"
        and optical_waveguide_params["left_optical_branch"] == "open"
        and optical_waveguide_params["right_optical_branch"] == "open"
    ):
        pad1_ref = MZM << pad
        pad1_ref.connect("e2", cpw_ref.ports["e1"])

        MZM.add_port(
            name="e1",
            port=pad1_ref.ports["e1"],
        )
        MZM.add_port(
            name="e2",
            port=cpw_ref.ports["e2"],
        )
        MZM.add_port(
            name="o2",
            port=pad1_ref.ports["o1"],
        )
        MZM.add_port(
            name="o3",
            port=cpw_ref.ports["o2"],
        )
        MZM.add_port(
            name="o4",
            port=cpw_ref.ports["o3"],
        )
        MZM.add_port(
            name="o1",
            port=pad1_ref.ports["o4"],
        )

    if (
        cpw_pad_params["left_rf_pad"] == "probe"
        and cpw_pad_params["right_rf_pad"] == "probe"
        and optical_waveguide_params["left_optical_branch"] == "mmi"
        and optical_waveguide_params["right_optical_branch"] == "mmi"
    ):
        # Define optical combiner
        combiner1 = optical_combiner_direct(
            optical_xs=optical_waveguides["terminal_xs"],
            cpw_xs=pad_xs,
            mmi_cell=mmi_cell,
            mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
            cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
            heater_section_length=optical_waveguide_params["heater_section_length"],
            imbalance_length=optical_waveguide_params["imbalance_length"],
            roc=optical_waveguide_params["roc"],
        )
        # Define optical combiner
        combiner2 = optical_combiner_direct(
            optical_xs=optical_waveguides["terminal_xs"],
            cpw_xs=pad_xs,
            mmi_cell=mmi_cell,
            mmi_connection_length=optical_waveguide_params["mmi_connection_length"],
            cpw_connection_length=optical_waveguide_params["cpw_connection_length"],
            heater_section_length=0.0,
            imbalance_length=0.0,
            roc=optical_waveguide_params["roc"],
        )
        pad1_ref = MZM << pad
        pad1_ref.connect("e2", cpw_ref.ports["e1"])
        pad2_ref = MZM << pad
        pad2_ref.connect("e2", cpw_ref.ports["e2"])
        combiner1_ref = MZM << combiner1
        combiner1_ref.connect("o3", pad1_ref.ports["o1"])
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
        if "1x2" in mmi_cell.name:
            MZM.add_port(
                name="o1",
                port=combiner1_ref.ports["o1"],
            )
            MZM.add_port(
                name="o2",
                port=combiner2_ref.ports["o1"],
            )
        else:
            MZM.add_port(
                name="o2",
                port=combiner1_ref.ports["o1"],
            )
            MZM.add_port(
                name="o1",
                port=combiner1_ref.ports["o2"],
            )
            MZM.add_port(
                name="o4",
                port=combiner2_ref.ports["o1"],
            )
            MZM.add_port(
                name="o3",
                port=combiner2_ref.ports["o2"],
            )

    if (
        optical_waveguide_params["heater_section_length"] > 0.0
        and optical_waveguide_params["left_optical_branch"] == "mmi"
    ):
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
