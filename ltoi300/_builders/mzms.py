from typing import Any

import gdsfactory as gf

from _utils.gsg_rf import double_layer_termination, straight_cpw, trail_cpw
from _utils.mzm import base_mzm
from _utils.thermal_phase_shifters import heater_straight_compact
from ltoi300.tech import LAYER, xs_ht_wire, xs_rwg700, xs_rwg900, xs_uni_cpw

from _utils.bends import S_bend_vert, get_s_bend_length
from _utils.Phase_shifters import (
    EO_Phase_shifter as _EO_Phase_shifter,
    TO_phase_shifter as _TO_phase_shifter,
    rectangular_cpw_pad,
)
from ltoi300._builders.mmis import (
    build_mmi1x2_oband,
    build_mmi1x2_cband,
    build_mmi2x2_oband,
    build_mmi2x2_cband,
)

###############################################
#### Default parameters - band-independent ####
###############################################

DEFAULT_CPW_PAD_PARAMS: dict[str, Any] = {
    "single_side": False,
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
    _cpw_pad_params["single_side"] = True
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
    _cpw_pad_params["single_side"] = True
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


def _safe_s_bend_vert(
    v_offset: float,
    h_extent: float,
    dx_straight: float = 5.0,
    cross_section: Any = "xs_rwg700",
) -> gf.Component:
    """A spline bend that bridges a vertical displacement without minimum displacement limits.
    
    Why: Bypasses the default S_bend_vert's validation check (which enforces vertical offset >= 10.0). 
    This enables compact vertical routing adjustments and C-band alignment loops to compile correctly.
    """
    import numpy as np
    from _utils.spline import bend_S_spline, spline_clamped_path

    if abs(v_offset) < 1e-4:
        return gf.components.straight(length=h_extent, cross_section=cross_section)

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


def build_terminated_mzm_folded(
    mmi_cell: gf.Component,
    band: str = "oband",
    modulation_length: float = 5000.0,
    modulation_width: float = 2.5,
    taper_length: float = 100.0,
    rf_gap: float = 5.5,
    rf_central_conductor_width: float | None = None,
    RF_ground_width : float = 150.0,
    gsg_pitch: float = 100.0,
    dc_pad_width: float = 80.0,
    dc_ground_width: float = 150.0,
    length_straight: float = 25.0,
    length_tapered: float = 150.0,
    length_imbalance: float = 0.0,
    compensation_length: float | None = None,
    bias_tuning_section_length: float = 700.0,
    dc_phase_shifter_length: float = 2000.0,
    thermal_phase_shifter_node: bool = True,
    dc_phase_shifter_node: bool = False,
    vertical_offset: float | None = None,
    horizontal_offset: float | None = None,
    uturn_radius: float | None = None,
    uturn_separation: float = 10.0,
    pads_on_same_y: bool = False,
    pad_size: tuple[float, float] = (150.0, 150.0),
    pad_size_aligned: tuple[float, float] = (150.0, 150.0),
    pad_group_vertical_offset: float = 320.0,
    pad_group_horizontal_offset: float = 150.0,
    pad_group_spacing: float | None = None,
    routing_dx_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    routing_dy_offsets: tuple[float, ...] | None = None,
    trail_params: dict[str, Any] | None = None,
    cpw_params: dict[str, Any] | None = None,
    termination_params: dict[str, Any] | None = None,
    transition_m1_m2_params: dict[str, Any] | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    m2_bonding_pad_params: dict[str, Any] | None = None,
) -> gf.Component:
    """Returns a routed folded terminated MZM supporting both O-band and C-band.
    
    Contains 2 horizontal rows connected via a West U-turn:
      - Row 2 (Top, y=vertical_offset): MMI splitter -> S-bends -> Phase Shifter (TO/EO) -> extensions.
      - Row 1 (Bottom, y=0): GSG pad -> CPW modulator -> termination -> extensions -> S-bends -> MMI combiner.
    """
    import math
    c = gf.Component()

    xs_func = xs_rwg700 if band == "oband" else xs_rwg900
    terminal_xs = xs_func()
    straight_name = "straight_rwg700_oband" if band == "oband" else "straight_rwg900_cband"

    # Default central conductor width if not specified
    if rf_central_conductor_width is None:
        rf_central_conductor_width = 20.0 if band == "oband" else 16.0

    # Calculate baseline vertical and horizontal offsets dynamically
    default_vertical_offset = 600.0 if pads_on_same_y else 500.0
    default_horizontal_offset = 110.0 if pads_on_same_y else 150.0

    # User-specified offsets are treated as relative additions to the baseline defaults
    vertical_offset = default_vertical_offset + (vertical_offset or 0.0)
    horizontal_offset = default_horizontal_offset + (horizontal_offset or 0.0)

    # Increase default pad group vertical offset (ref vertical offset) by 100 um when pads_on_same_y is True
    if pads_on_same_y and pad_group_vertical_offset == 320.0:
        pad_group_vertical_offset = 420.0

    # Calculate loopback routing difference dynamically using dummy straight sections
    dummy_c = gf.Component()
    p_space = rf_central_conductor_width + rf_gap
        
    d_r2_down = dummy_c << gf.components.straight(length=1.0, cross_section=terminal_xs)
    d_r2_down.dmove((0, vertical_offset))
    d_r2_up = dummy_c << gf.components.straight(length=1.0, cross_section=terminal_xs)
    d_r2_up.dmove((0, vertical_offset + p_space))
    
    d_r1_down = dummy_c << gf.components.straight(length=1.0, cross_section=terminal_xs)
    d_r1_up = dummy_c << gf.components.straight(length=1.0, cross_section=terminal_xs)
    d_r1_up.dmove((0, p_space))
    
    # Both EO (physically parallel overall) and TO (physically crossed overall) U-turns
    # require concentric (non-crossing) physical paths inside the West loop-back.
    # Therefore, we match the ports in a crossed configuration (upper-to-lower, lower-to-upper)
    # to avoid routing collisions in route_bundle.
    ports1 = [d_r2_up.ports["o1"], d_r2_down.ports["o1"]]
    ports2 = [d_r1_down.ports["o1"], d_r1_up.ports["o1"]]
    
    actual_uturn_radius = uturn_radius if uturn_radius is not None else terminal_xs.radius
    dummy_routes = gf.routing.route_bundle(
        dummy_c,
        ports1=ports1,
        ports2=ports2,
        cross_section=terminal_xs,
        bend=gf.components.bend_euler,
        straight=straight_name,
        radius=actual_uturn_radius,
        separation=uturn_separation,
    )
    
    r_top_len_dummy = float(dummy_routes[1].length) * 0.001
    r_bottom_len_dummy = float(dummy_routes[0].length) * 0.001
    routing_diff = abs(r_top_len_dummy - r_bottom_len_dummy)

    if compensation_length is None:
        if dc_phase_shifter_node:
            compensation_length = routing_diff + 40.0
        else:
            compensation_length = routing_diff

    # Common parameters
    _cpw_xs = xs_uni_cpw(
        central_conductor_width=rf_central_conductor_width,
        gap=rf_gap,
        ground_planes_width=50.0,
    )
    
    optical_waveguides = {
        "terminal_xs": terminal_xs,
        "modulation_xs": xs_func(width=modulation_width),
        "taper_length": taper_length,
    }

    # Merge bonding pads, transition, and termination parameters
    _m2_bonding_pad_params = _build_m2_bonding_params(m2_bonding_pad_params, transition_m1_m2_params)
    _termination_params = _merge(DEFAULT_TERMINATION_PARAMS, termination_params)
    _transition_m1_m2_params = _merge(DEFAULT_TRANSITION_M1_M2_PARAMS, transition_m1_m2_params)
    _transition_m2_hr_params = _merge(DEFAULT_TRANSITION_M2_HR_PARAMS, transition_m2_hr_params)

    # ==========================================================================
    # 1. Row 2 (Top Row, y = vertical_offset) - Phase shifter section propagating East to West
    # ==========================================================================
    mmi_in = c << mmi_cell
    mmi_in.drotate(180)
    mmi_in.dmove((0, vertical_offset))

    # Identify output ports of splitter (after 180 rotation, the vertical order is flipped)
    if "2x2" in mmi_cell.name:
        up_port = mmi_in.ports["o4"]
        down_port = mmi_in.ports["o3"]
    else:  # 1x2 MMI
        up_port = mmi_in.ports["o3"]
        down_port = mmi_in.ports["o2"]

    y_out_ps = (rf_central_conductor_width + rf_gap) / 2
    y_mmi_port = abs(up_port.dcenter[1] - vertical_offset)
    v_offset_1 = y_out_ps - y_mmi_port
    h_extent_1 = max(90.0, 3.5 * abs(v_offset_1) + 10.0)
    
    sb_in_up = c << _safe_s_bend_vert(v_offset=v_offset_1, h_extent=h_extent_1, cross_section=terminal_xs)
    sb_in_down = c << _safe_s_bend_vert(v_offset=-v_offset_1, h_extent=h_extent_1, cross_section=terminal_xs)
    sb_in_up.connect("o2", down_port)
    sb_in_down.connect("o2", up_port)

    # Instantiate Phase Shifter in native orientation (no rotation/mirroring)
    if dc_phase_shifter_node:
        ps = c << _EO_Phase_shifter(
            length=dc_phase_shifter_length,
            rf_gap=rf_gap,
            rf_central_conductor_width=rf_central_conductor_width,
            dc_pad_width=dc_pad_width,
            dc_ground_width=dc_ground_width,
            compensation_length=compensation_length,
            length_imbalance=length_imbalance,
            band=band,
        )
        ps.connect("o3", sb_in_down.ports["o1"])
        ps_in_upper_port = ps.ports["o1"]
        ps_in_lower_port = ps.ports["o2"]
    elif thermal_phase_shifter_node:
        ps = c << _TO_phase_shifter(
            length=bias_tuning_section_length,
            spacing=rf_central_conductor_width + rf_gap,
            compensation_length=compensation_length,
            length_imbalance=length_imbalance,
            pads_on_same_y=pads_on_same_y,
            pad_group_vertical_offset=pad_group_vertical_offset,
            pad_group_horizontal_offset=pad_group_horizontal_offset,
            pad_group_spacing=pad_group_spacing,
            pad_size=pad_size,
            pad_size_aligned=pad_size_aligned,
            routing_dx_offsets=routing_dx_offsets,
            routing_dy_offsets=routing_dy_offsets,
            band=band,
        )
        ps.connect("o3", sb_in_up.ports["o1"])
        ps_in_upper_port = ps.ports["o4"]
        ps_in_lower_port = ps.ports["o1"]
    else:
        raise ValueError("Either dc_phase_shifter_node or thermal_phase_shifter_node must be True.")

    # 125 um straight waveguide extensions between EO/TO and S-bends (extended by horizontal_offset)
    ext_ps_up = c << gf.components.straight(length=125.0 + horizontal_offset, cross_section=terminal_xs)
    ext_ps_down = c << gf.components.straight(length=125.0 + horizontal_offset, cross_section=terminal_xs)
    ext_ps_up.connect("o2", ps_in_upper_port)
    ext_ps_down.connect("o2", ps_in_lower_port)

    # Calculate S-bend parameters for CPW pad transitions
    v_offset_2 = y_out_ps - (gsg_pitch / 2)
    h_extent_2 = max(90.0, 3.5 * abs(v_offset_2) + 10.0)

    # West S-bends (connecting straight extensions to West U-turn - only for EO RF pads)
    sb_pad_up = None
    sb_pad_down = None
    if dc_phase_shifter_node:
        sb_pad_up = c << _safe_s_bend_vert(v_offset=v_offset_2, h_extent=h_extent_2, cross_section=terminal_xs)
        sb_pad_down = c << _safe_s_bend_vert(v_offset=-v_offset_2, h_extent=h_extent_2, cross_section=terminal_xs)
        sb_pad_up.dmirror_x()
        sb_pad_down.dmirror_x()
        sb_pad_up.connect("o2", ext_ps_up.ports["o1"])
        sb_pad_down.connect("o2", ext_ps_down.ports["o1"])

    # Expose electrical ports after any horizontal translations are completed
    if dc_phase_shifter_node:
        c.add_port(name="e3", port=ps.ports["e1"])
        c.add_port(name="e4", port=ps.ports["e2"])
    elif thermal_phase_shifter_node:
        if pads_on_same_y:
            c.add_port(name="port_E_TO_1", port=ps.ports["port_E_TO_1"])
            c.add_port(name="port_E_TO_2", port=ps.ports["port_E_TO_2"])
            c.add_port(name="port_E_TO_3", port=ps.ports["port_E_TO_3"])
            c.add_port(name="port_E_TO_4", port=ps.ports["port_E_TO_4"])
        else:
            c.add_port(name="e3", port=ps.ports["e1"])
            c.add_port(name="e4", port=ps.ports["e2"])
            c.add_port(name="e5", port=ps.ports["e3"])
            c.add_port(name="e6", port=ps.ports["e4"])

    # ==========================================================================
    # 2. Row 1 (Bottom Row, y = 0) - Modulator section propagating West to East
    # ==========================================================================
    if band == "oband":
        cpw_mod = c << straight_cpw(
            cpw_xs=_cpw_xs,
            modulation_length=modulation_length,
            optical_waveguides=optical_waveguides,
        )
    else:  # cband
        _trail_params = _merge(DEFAULT_TRAIL_PARAMS_CBAND, trail_params)
        cpw_mod = c << trail_cpw(
            cpw_xs=_cpw_xs,
            modulation_length=modulation_length,
            trail_params=_trail_params,
            optical_waveguides=optical_waveguides,
        )

    # GSG pad for modulator
    pad_mod = c << rectangular_cpw_pad(
        cpw_xs=_cpw_xs,
        optical_waveguide_xs=terminal_xs,
        pitch=gsg_pitch,
        length_straight=length_straight,
        length_tapered=length_tapered,
        ground_pad_width=RF_ground_width,
        m2_bonding_pads_params=_m2_bonding_pad_params,
        dc_pad_width=dc_pad_width,
    )
    pad_mod.connect("e2", cpw_mod.ports["e1"])

    # Resistive termination
    termination = double_layer_termination(
        cpw_xs=_cpw_xs,
        termination_layer=LAYER.HRL,
        m2_layer=LAYER.M2,
        m2_pad_length=_termination_params["m2_pad_length"],
        termination_params=_termination_params,
        via_m1_m2_params=_transition_m1_m2_params,
        via_m2_hr_params=_transition_m2_hr_params,
    )
    term_ref = c << termination
    term_ref.connect("e1", cpw_mod.ports["e2"])

    # Waveguide extensions on the West (GSG pad side, extending Westward)
    ext_pad_up = c << gf.components.straight(length=75.0, cross_section=terminal_xs)
    ext_pad_down = c << gf.components.straight(length=75.0, cross_section=terminal_xs)
    ext_pad_up.connect("o2", pad_mod.ports["o1"])
    ext_pad_down.connect("o2", pad_mod.ports["o4"])

    # West S-bends on Row 1 (connecting modulator extensions to West U-turn)
    sb_pad_mod_up = c << _safe_s_bend_vert(v_offset=v_offset_2, h_extent=h_extent_2, cross_section=terminal_xs)
    sb_pad_mod_down = c << _safe_s_bend_vert(v_offset=-v_offset_2, h_extent=h_extent_2, cross_section=terminal_xs)
    sb_pad_mod_up.dmirror_x()
    sb_pad_mod_down.dmirror_x()
    sb_pad_mod_up.connect("o2", ext_pad_up.ports["o1"])
    sb_pad_mod_down.connect("o2", ext_pad_down.ports["o1"])

    # Waveguide extensions on the East (termination side, extending Eastward)
    ext_out_up = c << gf.components.straight(length=75.0, cross_section=terminal_xs)
    ext_out_down = c << gf.components.straight(length=75.0, cross_section=terminal_xs)
    ext_out_up.connect("o1", cpw_mod.ports["o2"])
    ext_out_down.connect("o1", cpw_mod.ports["o3"])

    # Output Combiner (MMI, on the East end, rotated 180 so o1/outputs match combiner roles)
    mmi_out = c << mmi_cell

    # Identify input ports of combiner
    if "2x2" in mmi_cell.name:
        comb_up_port = mmi_out.ports["o4"]
        comb_down_port = mmi_out.ports["o3"]
    else:  # 1x2 MMI
        comb_up_port = mmi_out.ports["o3"]
        comb_down_port = mmi_out.ports["o2"]

    # East S-bends to combiner
    y_out_mod = rf_central_conductor_width / 2 + rf_gap / 2
    y_mmi_out_port = abs(comb_up_port.dcenter[1])
    v_offset_3 = y_mmi_out_port - y_out_mod
    h_extent_3 = max(90.0, 3.5 * abs(v_offset_3) + 10.0)
    sb_out_up = c << _safe_s_bend_vert(v_offset=v_offset_3, h_extent=h_extent_3, cross_section=terminal_xs)
    sb_out_down = c << _safe_s_bend_vert(v_offset=-v_offset_3, h_extent=h_extent_3, cross_section=terminal_xs)
    sb_out_up.connect("o1", ext_out_up.ports["o2"])
    sb_out_down.connect("o1", ext_out_down.ports["o2"])

    # Connect combiner input ports to S-bend outputs
    mmi_out.connect(comb_up_port.name, sb_out_up.ports["o2"])

    # List of all references on Row 1 (used for alignment shift)
    row1_refs = [
        cpw_mod, pad_mod, term_ref, 
        ext_pad_up, ext_pad_down, 
        sb_pad_mod_up, sb_pad_mod_down,
        ext_out_up, ext_out_down, 
        sb_out_up, sb_out_down, mmi_out
    ]

    # ==========================================================================
    # 3. Horizontal & Vertical Alignment of Row 1 relative to Row 2
    # ==========================================================================
    if vertical_offset > 0.0:
        if dc_phase_shifter_node:
            last_point_upper_x = sb_pad_up.ports["o1"].dcenter[0]
            last_point_upper_y = sb_pad_up.ports["o1"].dcenter[1]
        else:
            last_point_upper_x = ext_ps_down.ports["o1"].dcenter[0]
            last_point_upper_y = ext_ps_down.ports["o1"].dcenter[1]
        
        dx = last_point_upper_x - sb_pad_mod_up.ports["o1"].dcenter[0]
        dy = last_point_upper_y - vertical_offset - sb_pad_mod_up.ports["o1"].dcenter[1]
        
        for ref in row1_refs:
            ref.dmove((dx, dy))

    # ==========================================================================
    # 4. West U-Turn Loop-Back Routing
    # ==========================================================================
    routes = None
    if vertical_offset > 0.0:
        actual_uturn_radius = uturn_radius if uturn_radius is not None else terminal_xs.radius
        ports1_uturn = (
            [sb_pad_up.ports["o1"], sb_pad_down.ports["o1"]]
            if dc_phase_shifter_node
            else [ext_ps_down.ports["o1"], ext_ps_up.ports["o1"]]
        )
        routes = gf.routing.route_bundle(
            c,
            ports1=ports1_uturn,
            ports2=[sb_pad_mod_down.ports["o1"], sb_pad_mod_up.ports["o1"]],
            cross_section=terminal_xs,
            bend=gf.components.bend_euler,
            straight=straight_name,
            radius=actual_uturn_radius,
            separation=uturn_separation,
        )

    # ==========================================================================
    # 5. Path Length & Propagation Difference Calculation
    # ==========================================================================
    sb_in_up_len = get_s_bend_length(v_offset_1, h_extent_1)
    sb_in_down_len = sb_in_up_len
    
    ext_ps_len = 125.0 + horizontal_offset
    
    roc_ps = 60.0 if band == "oband" else 50.0
    comp_len = compensation_length
    imb_len = length_imbalance
    L_extra = comp_len + imb_len
    
    if dc_phase_shifter_node:
        top_pad_len = length_straight + length_tapered
        L_EO_active = top_pad_len + dc_phase_shifter_length + 2 * taper_length
        L_EO_comp_up = 60.0 + 2 * math.pi * roc_ps
        L_EO_comp_down = L_extra + 20.0 + 2 * math.pi * roc_ps
        
        # EO crossed layout detour assignments: Upper arm gets straight, Lower arm gets detour
        L_ps_up = L_EO_active + L_EO_comp_up
        L_ps_down = L_EO_active + L_EO_comp_down
    else:
        H_base = 20.0
        L_TO_up = 20.0 + 2 * H_base + bias_tuning_section_length + 2 * math.pi * roc_ps
        L_TO_down = L_TO_up + L_extra
        
        # TO layout detour assignments
        L_ps_up = L_TO_down
        L_ps_down = L_TO_up

    v_offset_2_val = y_out_ps - (gsg_pitch / 2)
    h_extent_2_val = max(90.0, 3.5 * abs(v_offset_2_val) + 10.0)
    sb_pad_mod_len = get_s_bend_length(v_offset_2_val, h_extent_2_val)

    if dc_phase_shifter_node:
        sb_pad_up_len = sb_pad_mod_len
        sb_pad_down_len = sb_pad_up_len
    else:
        sb_pad_up_len = 0.0
        sb_pad_down_len = 0.0

    r_top_len = float(routes[1].length) * 0.001 if (vertical_offset > 0.0 and routes is not None) else 0.0
    r_bottom_len = float(routes[0].length) * 0.001 if (vertical_offset > 0.0 and routes is not None) else 0.0

    sb_out_up_len = get_s_bend_length(v_offset_3, h_extent_3)
    sb_out_down_len = sb_out_up_len

    bottom_active_len = 75.0 + (length_straight + length_tapered) + (modulation_length + 2 * taper_length) + 75.0

    if vertical_offset > 0.0:
        if dc_phase_shifter_node:
            path_up_length = ext_ps_len + sb_pad_up_len + r_bottom_len + sb_pad_mod_len + 75.0
            path_down_length = ext_ps_len + sb_pad_down_len + r_top_len + sb_pad_mod_len + 75.0
            
            path_up_total = sb_in_up_len + L_ps_up + path_up_length - ext_ps_len - sb_pad_up_len + bottom_active_len + sb_out_down_len
            path_down_total = sb_in_down_len + L_ps_down + path_down_length - ext_ps_len - sb_pad_down_len + bottom_active_len + sb_out_up_len
        else:
            path_up_length = ext_ps_len + r_top_len + sb_pad_mod_len + 75.0
            path_down_length = ext_ps_len + r_bottom_len + sb_pad_mod_len + 75.0
            
            path_up_total = sb_in_up_len + L_ps_up + path_up_length - ext_ps_len + bottom_active_len + sb_out_up_len
            path_down_total = sb_in_down_len + L_ps_down + path_down_length - ext_ps_len + bottom_active_len + sb_out_down_len
    else:
        path_up_length = ext_ps_len + sb_pad_up_len + 75.0
        path_down_length = ext_ps_len + sb_pad_down_len + 75.0
        
        path_up_total = sb_in_up_len + L_ps_up + path_up_length - ext_ps_len - sb_pad_up_len + bottom_active_len + sb_out_up_len
        path_down_total = sb_in_down_len + L_ps_down + path_down_length - ext_ps_len - sb_pad_down_len + bottom_active_len + sb_out_down_len

    propagation_difference = float(abs(path_up_total - path_down_total))

    c.info["path_up_length"] = path_up_length
    c.info["path_down_length"] = path_down_length
    c.info["propagation_difference"] = propagation_difference
    c.info["path_up_compensation"] = path_up_total
    c.info["path_down_compensation"] = path_down_total
    c.info["vertical_offset"] = vertical_offset
    c.info["horizontal_offset"] = horizontal_offset

    # Expose optical top-level ports
    c.add_port(name="o1", port=mmi_in.ports["o1"])
    if "2x2" in mmi_cell.name:
        c.add_port(name="o2", port=mmi_in.ports["o2"])
        c.add_port(name="o3", port=mmi_out.ports["o2"])
        c.add_port(name="o4", port=mmi_out.ports["o1"])
    else:
        c.add_port(name="o2", port=mmi_out.ports["o1"])

    # Expose main RF pads and internal termination port
    c.add_port(name="e1", port=pad_mod.ports["e1"])
    c.add_port(name="e2", port=cpw_mod.ports["e2"])
    c.add_port(name="_term", port=term_ref.ports["term"])

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
