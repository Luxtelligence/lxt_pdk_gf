from typing import Any

import gdsfactory as gf

from _utils.gsg_rf import double_layer_termination, m2_transition
from _utils.mzm import base_mzm
from _utils.thermal_phase_shifters import heater_straight_compact
from ltoi300.tech import LAYER, xs_ht_wire, xs_rwg700, xs_rwg900, xs_uni_cpw

###############################################
#### Default parameters - band-independent ####
###############################################

DEFAULT_CPW_PAD_PARAMS: dict[str, Any] = {
    "left_rf_pad": "probe",
    "right_rf_pad": "probe",
    "left_optical_branch": "mmi",
    "right_optical_branch": "mmi",
    "pitch": 100.0,
    "length_straight": 25.0,
    "length_tapered": 150.0,
    "ground_pad_width": 150.0,
}
DEFAULT_OPTICAL_WG_PARAMS: dict[str, Any] = {
    "taper_length": 100.0,
    "modulation_width": 2.5,
    "terminal_width": None,
    "roc": 60.0,
    "imbalance_length": 100.0,
    "heater_section_length": 100.0,
    "mmi_connection_length": 10.0,
    "cpw_connection_length": 75.0,
}
DEFAULT_HEATER_PARAMS: dict[str, Any] = {
    "length": 1000.0,
    "width": 2.5,
    "routing_width": 6.0,
    "offset": 0.0,
    "both_arms": True,
    "port_contact_width_ratio": 3.0,
    "pad_size": (150.0, 150.0),
    "pad_pitch": None,
    "pad_vert_offset": 10.0,
}
DEFAULT_M2_BONDING_PAD_PARAMS: dict[str, Any] = {
    "m2_pad_length": 80.0,
}
DEFAULT_TERMINATION_PARAMS: dict[str, Any] = {
    "m2_pad_length": 10.0,
    "effective_length": 58.112,
    "resistor_width": 1.5,
    "hr_layer_offset": 0.0,
    "hr_pad_length": 5.0,
}
DEFAULT_TRANSITION_M1_M2_PARAMS: dict[str, Any] = {
    "type": "array",  # or "solid"
    "layer_openings": LAYER.VIA_M1_M2,
    "layer_m1": LAYER.M1,
    "layer_m2": LAYER.M2,
    "opening_offset": 2.5,
    "opening_size": 12.0,
    "opening_separation": 12.0,
    "width": 45.0,
}
DEFAULT_TRANSITION_M2_HR_PARAMS: dict[str, Any] = {
    "type": "solid",  # or "array"
    "layer_openings": LAYER.VIA_M2_HRL,
    "layer_hr": LAYER.HRL,
    "layer_m2": LAYER.M2,
    "opening_offset": 2.5,
    "opening_size": 12.0,
    "opening_separation": 12.0,
    "width": 20.0,
}

############################################
########### O-band builders ################
############################################

DEFAULT_CPW_PARAMS_OBAND: dict[str, Any] = {
    "type": "trail",
    "rf_gap": 5.5,
    "rf_ground_planes_width": 50.0,
    "rf_central_conductor_width": 20.0,  # For Trails central conductor shrinks by th+tt at the modulation region.
}
DEFAULT_TRAIL_PARAMS_OBAND: dict[str, Any] = {
    "tl": 53.0,
    "tw": 53.0,
    "th": 2.5,
    "tt": 2.5,
    "tc": 5.0,
}


def build_unterminated_mzm_oband(
    mmi_cell: gf.Component,
    modulation_length: float = 2000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    heater_params: dict[str, Any] | None = None,
    **base_mzm_kwargs: Any,
) -> gf.Component:
    forbidden = {"optical_xs", "cpw_xs"}
    bad = forbidden.intersection(base_mzm_kwargs)
    if bad:
        raise ValueError(f"Do not override fixed keys in O-band wrapper: {sorted(bad)}")

    _m2_bonding_params = _build_m2_bonding_params(
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
    )

    c = gf.Component()

    _heater_params = _merge(DEFAULT_HEATER_PARAMS, heater_params)
    if _heater_params["length"] > 0.0:
        _optical_waveguide_params = _merge(
            DEFAULT_OPTICAL_WG_PARAMS, optical_waveguide_params
        )

        if (
            _optical_waveguide_params["heater_section_length"]
            < _heater_params["length"]
        ):
            _optical_waveguide_params["heater_section_length"] = _heater_params[
                "length"
            ]

    mzm_ref = c << base_mzm(
        optical_xs=xs_rwg700,  # fixed by wrapper
        cpw_xs=xs_uni_cpw,  # fixed by wrapper
        modulation_length=modulation_length,
        mmi_cell=mmi_cell,
        cpw_params=_merge(DEFAULT_CPW_PARAMS_OBAND, cpw_params),
        trail_params=_merge(DEFAULT_TRAIL_PARAMS_OBAND, trail_params),
        cpw_pad_params=_merge(DEFAULT_CPW_PAD_PARAMS, cpw_pad_params),
        optical_waveguide_params=_optical_waveguide_params,
        m2_bonding_pad_params=_m2_bonding_params,
        **base_mzm_kwargs,
    )

    if _heater_params["length"] > 0.0:
        _transition_m2_hr_params = _merge(
            DEFAULT_TRANSITION_M2_HR_PARAMS, transition_m2_hr_params
        )
        heater = heater_straight_compact(
            heater_xs=xs_ht_wire(width=_heater_params["width"]),
            routing_xs=xs_ht_wire(width=_heater_params["routing_width"]),
            length=_heater_params["length"],
            port_contact_width_ratio=_heater_params["port_contact_width_ratio"],
            pad_size=_heater_params["pad_size"],
            pad_pitch=_heater_params["pad_pitch"],
            pad_vert_offset=_heater_params["pad_vert_offset"],
            transition_m2_hr_params=_transition_m2_hr_params,
        )
        heater_ref_1 = c << heater
        heater_ref_1.dmove(
            origin=heater_ref_1.ports["ht_start"].dcenter,
            destination=mzm_ref.ports["ht1_1"].dcenter + (0, _heater_params["offset"]),
        )
        c.add_port(name="e3", port=heater_ref_1.ports["e1"])
        c.add_port(name="e4", port=heater_ref_1.ports["e2"])
        if _heater_params["both_arms"]:
            heater_ref_2 = c << heater
            heater_ref_2.dmirror_y()
            heater_ref_2.dmove(
                origin=heater_ref_2.ports["ht_start"].dcenter,
                destination=mzm_ref.ports["ht2_1"].dcenter
                + (0, -_heater_params["offset"]),
            )
            c.add_port(name="e5", port=heater_ref_2.ports["e1"])
            c.add_port(name="e6", port=heater_ref_2.ports["e2"])

    utility_ports = ["ht1_1", "ht1_2", "ht2_1", "ht2_2"]
    for port in mzm_ref.ports:
        if port.name in utility_ports:
            c.add_port(name=f"_{port.name}", port=port)
        else:
            c.add_port(name=port.name, port=port)
    return c


def build_terminated_mzm_oband(
    mmi_cell: gf.Component,
    modulation_length: float = 2000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    termination_params: dict[str, Any] | None = None,
    heater_params: dict[str, Any] | None = None,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()

    _cpw_params = _merge(DEFAULT_CPW_PARAMS_OBAND, cpw_params)
    _termination_params = _merge(DEFAULT_TERMINATION_PARAMS, termination_params)
    _cpw_pad_params = _merge(DEFAULT_CPW_PAD_PARAMS, cpw_pad_params)
    _cpw_pad_params["right_rf_pad"] = "termination"
    _cpw_pad_params["left_rf_pad"] = "probe"
    _cpw_pad_params["right_optical_branch"] = "mmi"
    _cpw_pad_params["left_optical_branch"] = "mmi"
    _transition_m1_m2_params = _merge(
        DEFAULT_TRANSITION_M1_M2_PARAMS, transition_m1_m2_params
    )
    _transition_m2_hr_params = _merge(
        DEFAULT_TRANSITION_M2_HR_PARAMS, transition_m2_hr_params
    )

    cpw_xs = xs_uni_cpw(
        central_conductor_width=_cpw_params["rf_central_conductor_width"],
        gap=_cpw_params["rf_gap"],
        ground_planes_width=_cpw_params["rf_ground_planes_width"],
    )
    termination = double_layer_termination(
        cpw_xs=cpw_xs,
        termination_layer=LAYER.HRL,
        m2_layer=LAYER.M2,
        m2_pad_length=_termination_params["m2_pad_length"],
        termination_params=_termination_params,
        via_m1_m2_params=_transition_m1_m2_params,
        via_m2_hr_params=_transition_m2_hr_params,
    )
    mzm_ref = c << build_unterminated_mzm_oband(
        mmi_cell=mmi_cell,
        modulation_length=modulation_length,
        cpw_params=_cpw_params,
        trail_params=trail_params,
        cpw_pad_params=_cpw_pad_params,
        optical_waveguide_params=optical_waveguide_params,
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
        transition_m2_hr_params=transition_m2_hr_params,
        heater_params=heater_params,
    )
    termination_ref = c << termination
    termination_ref.connect("e1", mzm_ref.ports["e2"])
    c.add_port(name="_term", port=termination_ref.ports["term"])
    utility_ports = ["ht1_1", "ht1_2", "ht2_1", "ht2_2", "e2"]
    for port in mzm_ref.ports:
        if port.name in utility_ports:
            c.add_port(name=f"_{port.name}", port=port)
        else:
            c.add_port(name=port.name, port=port)
    return c


############################################
########### C-band builders ################
############################################

DEFAULT_CPW_PARAMS_CBAND: dict[str, Any] = {
    "type": "trail",
    "rf_gap": 5.5,
    "rf_ground_planes_width": 50.0,
    "rf_central_conductor_width": 16.0,  # For Trails central conductor shrinks by th+tt at the modulation region.
}

DEFAULT_TRAIL_PARAMS_CBAND: dict[str, Any] = {
    "tl": 53.0,
    "tw": 53.0,
    "th": 1.5,
    "tt": 1.5,
    "tc": 5.0,
}


def build_unterminated_mzm_cband(
    mmi_cell: gf.Component,
    modulation_length: float = 2000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    heater_params: dict[str, Any] | None = None,
    **base_mzm_kwargs: Any,
) -> gf.Component:
    forbidden = {"optical_xs", "cpw_xs"}
    bad = forbidden.intersection(base_mzm_kwargs)
    if bad:
        raise ValueError(f"Do not override fixed keys in C-band wrapper: {sorted(bad)}")

    _m2_bonding_params = _build_m2_bonding_params(
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
    )

    c = gf.Component()

    _heater_params = _merge(DEFAULT_HEATER_PARAMS, heater_params)
    if _heater_params["length"] > 0.0:
        _optical_waveguide_params = _merge(
            DEFAULT_OPTICAL_WG_PARAMS, optical_waveguide_params
        )

        if (
            _optical_waveguide_params["heater_section_length"]
            < _heater_params["length"]
        ):
            _optical_waveguide_params["heater_section_length"] = _heater_params[
                "length"
            ]

    mzm_ref = c << base_mzm(
        optical_xs=xs_rwg900,  # fixed by wrapper
        cpw_xs=xs_uni_cpw,  # fixed by wrapper
        modulation_length=modulation_length,
        mmi_cell=mmi_cell,
        cpw_params=_merge(DEFAULT_CPW_PARAMS_CBAND, cpw_params),
        trail_params=_merge(DEFAULT_TRAIL_PARAMS_CBAND, trail_params),
        cpw_pad_params=_merge(DEFAULT_CPW_PAD_PARAMS, cpw_pad_params),
        optical_waveguide_params=_optical_waveguide_params,
        m2_bonding_pad_params=_m2_bonding_params,
        **base_mzm_kwargs,
    )

    if (
        _heater_params["length"] > 0.0
        and optical_waveguide_params["left_optical_branch"] == "mmi"
    ):
        _transition_m2_hr_params = _merge(
            DEFAULT_TRANSITION_M2_HR_PARAMS, transition_m2_hr_params
        )
        heater = heater_straight_compact(
            heater_xs=xs_ht_wire(width=_heater_params["width"]),
            routing_xs=xs_ht_wire(width=_heater_params["routing_width"]),
            length=_heater_params["length"],
            port_contact_width_ratio=_heater_params["port_contact_width_ratio"],
            pad_size=_heater_params["pad_size"],
            pad_pitch=_heater_params["pad_pitch"],
            pad_vert_offset=_heater_params["pad_vert_offset"],
            transition_m2_hr_params=_transition_m2_hr_params,
        )
        heater_ref_1 = c << heater
        heater_ref_1.dmove(
            origin=heater_ref_1.ports["ht_start"].dcenter,
            destination=mzm_ref.ports["ht1_1"].dcenter + (0, _heater_params["offset"]),
        )
        c.add_port(name="e3", port=heater_ref_1.ports["e1"])
        c.add_port(name="e4", port=heater_ref_1.ports["e2"])
        if _heater_params["both_arms"]:
            heater_ref_2 = c << heater
            heater_ref_2.dmirror_y()
            heater_ref_2.dmove(
                origin=heater_ref_2.ports["ht_start"].dcenter,
                destination=mzm_ref.ports["ht2_1"].dcenter
                + (0, -_heater_params["offset"]),
            )
            c.add_port(name="e5", port=heater_ref_2.ports["e1"])
            c.add_port(name="e6", port=heater_ref_2.ports["e2"])
    utility_ports = ["ht1_1", "ht1_2", "ht2_1", "ht2_2"]
    for port in mzm_ref.ports:
        if port.name in utility_ports:
            c.add_port(name=f"_{port.name}", port=port)
        else:
            c.add_port(name=port.name, port=port)
    return c


def build_terminated_mzm_cband(
    mmi_cell: gf.Component,
    modulation_length: float = 2000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    termination_params: dict[str, Any] | None = None,
    heater_params: dict[str, Any] | None = None,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()

    _cpw_params = _merge(DEFAULT_CPW_PARAMS_CBAND, cpw_params)
    _termination_params = _merge(DEFAULT_TERMINATION_PARAMS, termination_params)
    _cpw_pad_params = _merge(DEFAULT_CPW_PAD_PARAMS, cpw_pad_params)
    _cpw_pad_params["left_optical_branch"] = "mmi"
    _cpw_pad_params["right_optical_branch"] = "mmi"
    _cpw_pad_params["right_rf_pad"] = "termination"
    _cpw_pad_params["left_rf_pad"] = "probe"
    _transition_m1_m2_params = _merge(
        DEFAULT_TRANSITION_M1_M2_PARAMS, transition_m1_m2_params
    )
    _transition_m2_hr_params = _merge(
        DEFAULT_TRANSITION_M2_HR_PARAMS, transition_m2_hr_params
    )

    cpw_xs = xs_uni_cpw(
        central_conductor_width=_cpw_params["rf_central_conductor_width"],
        gap=_cpw_params["rf_gap"],
        ground_planes_width=_cpw_params["rf_ground_planes_width"],
    )
    termination = double_layer_termination(
        cpw_xs=cpw_xs,
        termination_layer=LAYER.HRL,
        m2_layer=LAYER.M2,
        m2_pad_length=_termination_params["m2_pad_length"],
        termination_params=_termination_params,
        via_m1_m2_params=_transition_m1_m2_params,
        via_m2_hr_params=_transition_m2_hr_params,
    )
    mzm_ref = c << build_unterminated_mzm_cband(
        mmi_cell=mmi_cell,
        modulation_length=modulation_length,
        cpw_params=_cpw_params,
        trail_params=trail_params,
        cpw_pad_params=_cpw_pad_params,
        optical_waveguide_params=optical_waveguide_params,
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
        transition_m2_hr_params=transition_m2_hr_params,
        heater_params=heater_params,
    )
    termination_ref = c << termination
    termination_ref.connect("e1", mzm_ref.ports["e2"])
    c.add_port(name="_term", port=termination_ref.ports["term"])
    utility_ports = ["ht1_1", "ht1_2", "ht2_1", "ht2_2", "e2"]
    for port in mzm_ref.ports:
        if port.name in utility_ports:
            c.add_port(name=f"_{port.name}", port=port)
        else:
            c.add_port(name=port.name, port=port)
    return c


def build_phase_shifter_modular_cband(
    mmi_cell: gf.Component,
    modulation_length: float = 2000.0,
    cpw_params: dict[str, Any] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_pad_params: dict[str, Any] | None = None,
    optical_waveguide_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    termination_params: dict[str, Any] | None = None,
    heater_params: dict[str, Any] | None = None,
):
    """Create a phase-shifter modular C-band component.

    Args:
        cpw_pad_params: Overrides for CPW pad parameters. The keys
            ``left_optical_branch`` and ``right_optical_branch`` accept:
            - ``"mmi"``  – terminate the optical branch with an MMI coupler.
            - ``"open"`` – leave the waveguide unterminated (bare port).
    """
    c = gf.Component()

    _cpw_params = _merge(DEFAULT_CPW_PARAMS_CBAND, cpw_params)
    _termination_params = _merge(DEFAULT_TERMINATION_PARAMS, termination_params)
    _cpw_pad_params = _merge(DEFAULT_CPW_PAD_PARAMS, cpw_pad_params)
    _transition_m1_m2_params = _merge(
        DEFAULT_TRANSITION_M1_M2_PARAMS, transition_m1_m2_params
    )
    _transition_m2_hr_params = _merge(
        DEFAULT_TRANSITION_M2_HR_PARAMS, transition_m2_hr_params
    )

    cpw_xs = xs_uni_cpw(
        central_conductor_width=_cpw_params["rf_central_conductor_width"],
        gap=_cpw_params["rf_gap"],
        ground_planes_width=_cpw_params["rf_ground_planes_width"],
    )
    termination = double_layer_termination(
        cpw_xs=cpw_xs,
        termination_layer=LAYER.HRL,
        m2_layer=LAYER.M2,
        m2_pad_length=_termination_params["m2_pad_length"],
        termination_params=_termination_params,
        via_m1_m2_params=_transition_m1_m2_params,
        via_m2_hr_params=_transition_m2_hr_params,
    )
    mzm_ref = c << build_unterminated_mzm_cband(
        mmi_cell=mmi_cell,
        modulation_length=modulation_length,
        cpw_params=_cpw_params,
        trail_params=trail_params,
        cpw_pad_params=_cpw_pad_params,
        optical_waveguide_params=optical_waveguide_params,
        m2_bonding_pad_params=m2_bonding_pad_params,
        transition_m1_m2_params=transition_m1_m2_params,
        transition_m2_hr_params=transition_m2_hr_params,
        heater_params=heater_params,
    )
    if _cpw_pad_params["right_rf_pad"] == "termination":
        termination_ref = c << termination
        termination_ref.connect("e1", mzm_ref.ports["e2"])
        c.add_port(name="_term", port=termination_ref.ports["term"])

    if _cpw_pad_params["left_rf_pad"] == "bend_connection":
        cpw_xs = xs_uni_cpw(
            central_conductor_width=_cpw_params["rf_central_conductor_width"],
            gap=_cpw_params["rf_gap"],
            ground_planes_width=_cpw_params["rf_ground_planes_width"],
        )

        m2_transition_cell = m2_transition(
            cpw_xs=cpw_xs,
            m2_layer=LAYER.M2,
            termination_params=_termination_params,
            via_m1_m2_params=_transition_m1_m2_params,
        )
        m2_transition_ref_left = c << m2_transition_cell
        m2_transition_ref_left.connect("e2", mzm_ref.ports["e1"], allow_width_mismatch=True, allow_layer_mismatch=True
        )

        wg_length = _transition_m1_m2_params["width"]
        straight_o1_left = c << gf.components.straight(length=wg_length, cross_section=xs_rwg900)
        straight_o1_left.connect("o1", mzm_ref.ports["o1"])
        straight_o2_left = c << gf.components.straight(length=wg_length, cross_section=xs_rwg900)
        straight_o2_left.connect("o1", mzm_ref.ports["o2"])


    if _cpw_pad_params["right_rf_pad"] == "bend_connection":
        cpw_xs = xs_uni_cpw(
            central_conductor_width=_cpw_params["rf_central_conductor_width"],
            gap=_cpw_params["rf_gap"],
            ground_planes_width=_cpw_params["rf_ground_planes_width"],
        )

        m2_transition_cell = m2_transition(
            cpw_xs=cpw_xs,
            m2_layer=LAYER.M2,
            termination_params=_termination_params,
            via_m1_m2_params=_transition_m1_m2_params,
        )
        m2_transition_ref_right = c << m2_transition_cell
        m2_transition_ref_right.connect("e1", mzm_ref.ports["e2"], allow_width_mismatch=True, allow_layer_mismatch=True
        )

        # Port numbering on the right side depends on how many ports the left
        # optical branch contributes: "mmi" → 2 left ports (o1,o2) so right is
        # o3/o4; "open" → 1 left port (o1) so right is o2/o3.
        _left_is_mmi = optical_waveguide_params["left_optical_branch"] == "mmi"
        _right_p1, _right_p2 = ("o2", "o3") if _left_is_mmi else ("o3", "o4")

        wg_length = _transition_m1_m2_params["width"]
        straight_o1_right = c << gf.components.straight(length=wg_length, cross_section=xs_rwg900)
        straight_o1_right.connect("o1", mzm_ref.ports[_right_p1])
        straight_o2_right = c << gf.components.straight(length=wg_length, cross_section=xs_rwg900)
        straight_o2_right.connect("o1", mzm_ref.ports[_right_p2])

    utility_ports = ["ht1_1", "ht1_2", "ht2_1", "ht2_2"]
    if _cpw_pad_params["left_rf_pad"] == "bend_connection":
        utility_ports.extend(["e1", "o1", "o2"])
    if _cpw_pad_params["right_rf_pad"] == "bend_connection" and _left_is_mmi:
        utility_ports.extend(["e2", _right_p1, _right_p2, "o4"])
    if _cpw_pad_params["right_rf_pad"] == "bend_connection" and not _left_is_mmi:
        utility_ports.extend(["e2", _right_p1, _right_p2])
    for port in mzm_ref.ports:
        if port.name in utility_ports:
            c.add_port(name=f"_{port.name}", port=port)
        else:
            c.add_port(name=port.name, port=port)
    if _cpw_pad_params["left_rf_pad"] == "bend_connection":
        c.add_port(name="e1", port=m2_transition_ref_left.ports["e1"])
        c.add_port(name="o1", port=straight_o1_left.ports["o2"])
        c.add_port(name="o2", port=straight_o2_left.ports["o2"])
    if _cpw_pad_params["right_rf_pad"] == "bend_connection":
        c.add_port(name="e2", port=m2_transition_ref_right.ports["e2"])
        c.add_port(name=_right_p1, port=straight_o1_right.ports["o2"])
        c.add_port(name=_right_p2, port=straight_o2_right.ports["o2"])
    return c


############################################
########### Helper functions ###############
############################################


def _build_m2_bonding_params(
    m2_bonding_pad_params: dict[str, Any] | None,
    transition_m1_m2_params: dict[str, Any] | None,
) -> dict[str, Any]:
    pad = _merge(DEFAULT_M2_BONDING_PAD_PARAMS, m2_bonding_pad_params)
    tr = _merge(DEFAULT_TRANSITION_M1_M2_PARAMS, transition_m1_m2_params)

    return {
        "layer_m2": LAYER.M2,  # or keep base default if you prefer
        "layer_openings": tr["layer_openings"],
        "m1_opening_offset": tr["opening_offset"],
        "opening_size": tr["opening_size"],
        "opening_separation": tr["opening_separation"],
        "tl_opening_host_width": tr["width"],
        "m2_pad_length": pad["m2_pad_length"],
    }


def _merge(
    defaults: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


if __name__ == "__main__":
    from ltoi300.cells import mmi1x2_oband

    mzm = build_terminated_mzm_oband(mmi_cell=mmi1x2_oband())
    mzm.show()
