from typing import Any
import warnings

import gdsfactory as gf
import numpy as np
from gdsfactory.cross_section import CrossSection
from gdsfactory.routing import route_quad


# ---------------------------------------------------------------------------
# Module-level constants                                          (R4)
# ---------------------------------------------------------------------------

# Common kwargs shared by every pad_array call in this module.
_PAD_DEFAULTS: dict[str, Any] = {
    "pad": gf.components.pad,
    "port_orientation": -90.0,
}


# ---------------------------------------------------------------------------
# Private helpers — pure utilities, not @gf.cell
# ---------------------------------------------------------------------------

def _resolve_transition_params(params: dict) -> None:        # R1
    """Fill missing keys in a *transition_m2_hr_params* dict with warned defaults.

    Mutates *params* in-place so callers can read keys immediately after.
    """
    _DEFAULTS: dict[str, Any] = {
        "layer_openings": (41, 0),
        "layer_m2":       (22, 0),
        "opening_offset": 2.5,
    }
    for key, default in _DEFAULTS.items():
        if key not in params:
            warnings.warn(
                f'"{key}" key not found in transition_m2_hr_params. '
                f"Using default value {default!r}.",
                stacklevel=2,
            )
            params[key] = default


def _add_m2_port(                                             # R3
    c: gf.Component,
    name: str,
    cx: float,
    cy: float,
    width: float,
    orientation: float,
    layer: gf.typings.LayerSpec,
) -> None:
    """Add a single M2 electrical port to *c*."""
    c.add_port(
        name=name,
        center=(cx, cy),
        width=width,
        orientation=orientation,
        port_type="electrical",
        layer=layer,
    )


def _place_m2_rect(                                           # R6
    tmp: gf.Component,
    cx: float,
    cy: float,
    size: tuple[float, float],
    layer: gf.typings.LayerSpec,
) -> None:
    """Place a centered M2 rectangle at *(cx, cy)* inside *tmp*."""
    ref = tmp << gf.components.rectangle(size=size, layer=layer, centered=True)
    ref.dmove(ref.dcenter, (cx, cy))


def _pick_west_port(                                          # R5
    component: gf.Component,
    name_a: str,
    name_b: str,
    fallback: str,
) -> str:
    """Return the name of the westernmost of *name_a* / *name_b*.

    Falls back to *fallback* when neither port exists on *component*.
    """
    if name_a in component.ports and name_b in component.ports:
        return (
            name_a
            if component.ports[name_a].dcenter[0] < component.ports[name_b].dcenter[0]
            else name_b
        )
    return fallback


# ---------------------------------------------------------------------------
# Public cell functions
# ---------------------------------------------------------------------------

@gf.cell
def heater_pads_assymm(
    routing_xs: CrossSection,
    pad_size: tuple[float, float] = (150.0, 150.0),
    heater_length: float = 700.0,
    pad_pitch: float | None = None,
) -> gf.Component:
    padwidth, _ = pad_size
    if pad_pitch is None:
        pad_pitch = heater_length - padwidth

    c = gf.Component()
    bondpads = gf.components.pad_array(
        **_PAD_DEFAULTS,                                      # R4
        size=pad_size,
        column_pitch=pad_pitch,
        row_pitch=pad_pitch,
        columns=2,
        layer=routing_xs.layer,
    )
    bps = c << bondpads

    c.add_port(name="e1", port=bps.ports["e11"])
    c.add_port(name="e2", port=bps.ports["e12"])
    c.flatten()
    return c


@gf.cell
def m2_hrl_via(
    size: tuple[float, float] = (20.0, 20.0),
    layer_hrl: gf.typings.LayerSpec = (23, 0),
    layer_via: gf.typings.LayerSpec = (41, 0),
    layer_m2: gf.typings.LayerSpec = (22, 0),
) -> gf.Component:
    """20um x 20um M2-HRL VIA contact pad (M2-HRL_VIA) connecting HRL to M2 metal."""
    c = gf.Component()
    w, h = size
    c << gf.components.rectangle(size=(w, h), layer=layer_hrl, centered=True)
    c << gf.components.rectangle(size=(w, h), layer=layer_via, centered=True)
    c << gf.components.rectangle(size=(w, h), layer=layer_m2,  centered=True)
    _add_m2_port(c, "e1", 0.0, 0.0, w, 90.0, layer_m2)      # R3
    c.flatten()
    return c


@gf.cell
def heater_wire(
    heater_xs: CrossSection,
    length: float = 700.0,
    port_contact_width_ratio: float = 3.0,
    via_size: tuple[float, float] = (20, 20),
    via_offset_y: float = 30,
    layer_via: gf.typings.LayerSpec = (41, 0),
    layer_m2: gf.typings.LayerSpec = (22, 0),
) -> gf.Component:
    """Resistive heater wire terminated with M2-HRL_VIA contact pads at both ends.

    Creates an HRL wire with via contact pads (HRL, VIA_M2_HRL, M2) offset
    vertically by `via_offset_y` away from the optical core.

    The HRL stub connecting the wire (y=0) to the via (y=via_offset_y) is a
    tapered quadrilateral (via ``route_quad``) that widens from
    ``port_contact_width_ratio * heater_xs.width`` at the wire contact up to the
    full via width ``via_size[0]`` at the via pad.
    ``port_contact_width_ratio`` controls the HRL footprint at the wire-end of
    the taper (default 3x the wire width).
    """
    c = gf.Component()

    # 1. HRL heater wire
    path = gf.path.straight(length=length)
    ht   = path.extrude(heater_xs)
    c << ht

    # 2. M2-HRL_VIA contact pads with vertical offset
    via_cell = m2_hrl_via(
        size=via_size,
        layer_hrl=heater_xs.layer,
        layer_via=layer_via,
        layer_m2=layer_m2,
    )

    vw, _              = via_size
    port_contact_width = port_contact_width_ratio * heater_xs.width

    for x_pos, port_name in [(0.0, "ht_end"), (length, "ht_start")]:
        via_ref = c << via_cell
        via_ref.dmove((x_pos, via_offset_y))

        # Taper from thin wire contact to full via width using route_quad.
        # port1 (wire side): narrow, oriented toward the via.
        # port2 (via  side): full via width, oriented back toward the wire.
        if abs(via_offset_y) > 0:
            wire_port = gf.Port(
                name=f"_wire_{port_name}",
                center=(x_pos, 0.0),
                width=port_contact_width,
                orientation=90.0 if via_offset_y > 0 else 270.0,
                port_type="electrical",
                layer=heater_xs.layer,
            )
            via_port = gf.Port(
                name=f"_via_{port_name}",
                center=(x_pos, via_offset_y),
                width=vw,
                orientation=270.0 if via_offset_y > 0 else 90.0,
                port_type="electrical",
                layer=heater_xs.layer,
            )
            route_quad(
                c,
                port1=wire_port,
                port2=via_port,
                width2=vw,
                layer=heater_xs.layer,
            )

        # Electrical port at the via centre, exposed on M2
        orientation = 90.0 if via_offset_y >= 0 else 270.0
        _add_m2_port(c, port_name, x_pos, via_offset_y, vw, orientation, layer_m2)  # R3

    c.flatten()
    return c


@gf.cell
def heater_straight_compact(
    heater_xs: CrossSection,
    routing_xs: CrossSection,
    length: float = 700.0,
    port_contact_width_ratio: float = 3.0,
    pad_size: tuple[float, float] = (150.0, 150.0),
    pad_pitch: float | None = None,
    pad_vert_offset: float = 10.0,
    transition_m2_hr_params: dict[str, Any] | None = None,
) -> gf.Component:
    """A straight resistive wire used as a low-frequency phase shifter,
    exploiting the thermo-optical effect. The heater is terminated by wide pads
    for probing or bonding."""

    if pad_vert_offset <= 0:
        raise ValueError(
            f"pad_vert_offset must be a positive number, received {pad_vert_offset}."
        )
    if port_contact_width_ratio <= 0:
        raise ValueError(
            f"port_contact_width_ratio must be a positive number, received {port_contact_width_ratio}."
        )

    c = gf.Component()
    bondpads = heater_pads_assymm(
        pad_size=pad_size,
        heater_length=length,
        pad_pitch=pad_pitch,
        routing_xs=routing_xs,
    )
    bps = c << bondpads

    path  = gf.path.straight(length=length)
    ht    = path.extrude(heater_xs)
    width = heater_xs.width

    # Rotate wire ports to face upward
    for p in ht.ports:
        if p.orientation == 0.0:
            p.dcenter = (p.dcenter[0] - 0.5 * p.dwidth, p.dcenter[1])
        if p.orientation == 180.0:
            p.dcenter = (p.dcenter[0] + 0.5 * p.dwidth, p.dcenter[1])
        p.orientation = 90.0

    ht_ref = c << ht
    bps.dxmin = ht_ref.dxmin
    bps.dymin = ht_ref.dymax + pad_vert_offset

    port_contact_width = port_contact_width_ratio * width
    ht.ports["e1"].dx += 0.5 * (port_contact_width - width)
    ht.ports["e2"].dx -= 0.5 * (port_contact_width - width)

    # Connect pads and heater wire
    for pad_port, ht_port in [
        (bps.ports["e1"], ht.ports["e1"]),
        (bps.ports["e2"], ht.ports["e2"]),
    ]:
        _ = gf.routing.route_single_electrical(
            c, port1=pad_port, port2=ht_port, cross_section=routing_xs,
        )

    c.add_port(name="ht_start", port=ht.ports["e1"])
    c.add_port(name="ht_end",   port=ht.ports["e2"])

    if transition_m2_hr_params is not None:
        _resolve_transition_params(transition_m2_hr_params)   # R1
        layer_m2     = transition_m2_hr_params["layer_m2"]
        pad_w, pad_h = pad_size

        tmp = gf.Component()
        for port in [bps.ports["e1"], bps.ports["e2"]]:
            _place_m2_rect(tmp, port.dcenter[0], port.dcenter[1], (pad_w, pad_h), layer_m2)  # R6

        for lyr, polygons in tmp.get_polygons().items():
            for polygon in polygons:
                c.add_polygon(polygon, layer=lyr)

        for name, port in [("e1", bps.ports["e1"]), ("e2", bps.ports["e2"])]:
            _add_m2_port(                                      # R3
                c, name,
                port.dcenter[0],
                port.dcenter[1] + pad_h / 2,
                pad_w, 90.0, layer_m2,
            )
    else:
        c.add_port(name="e1", port=bps.ports["e1"])
        c.add_port(name="e2", port=bps.ports["e2"])

    c.flatten()
    return c


@gf.cell
def add_heater(
    heater_xs: CrossSection,
    routing_xs: CrossSection,
    interferometer: gf.Component = None,
    heater_on_both_branches: bool = False,
    heater_offset: float = 0.0,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    bias_tuning_section_length: float = 700.0,
    length_imbalance: float = 0.0,
    port_contact_width_ratio: float = 3.0,
    via_size: tuple[float, float] = (80.0, 80.0),
    via_offset_y: float = 60,
    pad_e1_heater_offset: tuple[float, float] | None = None,
    pad_e2_heater_offset: tuple[float, float] | None = None,
    pad_e3_heater_offset: tuple[float, float] | None = None,
    pad_e4_heater_offset: tuple[float, float] | None = None,
    pad_e1_heater_route: list[tuple[float, float]] | None = None,
    pad_e2_heater_route: list[tuple[float, float]] | None = None,
    pad_e3_heater_route: list[tuple[float, float]] | None = None,
    pad_e4_heater_route: list[tuple[float, float]] | None = None,
    align_pads_same_y: bool = False,
    transition_m2_hr_params: dict[str, Any] | None = None,
) -> gf.Component:
    """Add heater(s) to the modulator with individually configured pad offsets.

    The 4 bond pads can have independent longitudinal and vertical offsets.
    The routing is wired to prevent crossed connections.
    """
    if interferometer is None:
        raise ValueError("`interferometer` must be provided.")

    pad_width, _ = heater_pad_size
    pad_pitch    = pad_width + 25.0
    if align_pads_same_y:
        if pad_e1_heater_offset is None:
            pad_e1_heater_offset = (bias_tuning_section_length + pad_pitch,     pad_width)
        if pad_e2_heater_offset is None:
            pad_e2_heater_offset = (2 * pad_pitch,                              pad_width)
        if pad_e3_heater_offset is None:
            pad_e3_heater_offset = (bias_tuning_section_length + 4 * pad_pitch, pad_width)
        if pad_e4_heater_offset is None:
            pad_e4_heater_offset = (3 * pad_pitch,                              pad_width)
    else:
        # dx=0: pad directly above/below via -> route_quad gives a clean straight
        # taper with equal length for all four connections.
        if pad_e1_heater_offset is None:
            pad_e1_heater_offset = (0.0,  100.0)
        if pad_e2_heater_offset is None:
            pad_e2_heater_offset = (0.0,  100.0)
        if pad_e3_heater_offset is None:
            pad_e3_heater_offset = (0.0, -100.0)
        if pad_e4_heater_offset is None:
            pad_e4_heater_offset = (0.0, -100.0)

    c = gf.Component()

    # R7: factor out common heater_wire kwargs; only via_offset_y sign differs
    _wire_kwargs: dict[str, Any] = dict(
        heater_xs=heater_xs,
        length=bias_tuning_section_length,
        port_contact_width_ratio=port_contact_width_ratio,
        via_size=via_size,
    )
    wire_cell_up   = heater_wire(**_wire_kwargs, via_offset_y=+via_offset_y)
    wire_cell_down = heater_wire(**_wire_kwargs, via_offset_y=-via_offset_y)

    ht_ref_1 = c << wire_cell_up
    ht_ref_2 = c << wire_cell_down if heater_on_both_branches else None

    # Displacement vectors (heater_offset == 0.0 -> wire center is on waveguide axis)
    heater_disp_1 = [0.0,  heater_offset]
    heater_disp_2 = [0.0, -heater_offset]

    # R5: resolve interferometer port names with helper
    port_up_name   = _pick_west_port(interferometer, "ht1_1", "ht1_2", "long_bias_branch_start")
    port_down_name = _pick_west_port(interferometer, "ht2_1", "ht2_2", "short_bias_branch_start")

    # Align heater wire center axis (y=0) to the optical port destination
    ht_ref_1.dmove(
        origin=(
            ht_ref_1.ports["ht_end"].dcenter[0],
            ht_ref_1.ports["ht_end"].dcenter[1] - via_offset_y,
        ),
        destination=np.array(interferometer.ports[port_up_name].dcenter) + heater_disp_1,
    )

    if heater_on_both_branches:
        ht_ref_2.dmove(
            origin=(
                ht_ref_2.ports["ht_end"].dcenter[0],
                ht_ref_2.ports["ht_end"].dcenter[1] + via_offset_y,
            ),
            destination=np.array(interferometer.ports[port_down_name].dcenter) + heater_disp_2,
        )

    # Place individual bond pads
    single_pad = gf.components.pad_array(
        **_PAD_DEFAULTS,                                      # R4
        size=heater_pad_size,
        columns=1,
        layer=routing_xs.layer,
    )

    # R1: resolve transition params once; expose only what is actually used
    layer_m2        = None
    pad_w = pad_h   = None
    tmp_transitions = None
    if transition_m2_hr_params is not None:
        _resolve_transition_params(transition_m2_hr_params)
        layer_m2        = transition_m2_hr_params["layer_m2"]
        pad_w, pad_h    = heater_pad_size
        tmp_transitions = gf.Component()

    def _place_and_route(
        wire_port,
        offset,
        port_name,
        use_route_quad=False,
        start_straight_length=None,
        end_straight_length=None,
        port_orientation=90.0,
        route_points=None,
        mirror_pad_y=False,
    ):
        pad_ref = c << single_pad
        pad_ref.dcenter = (
            wire_port.dcenter[0] + offset[0],
            wire_port.dcenter[1] + offset[1],
        )

        if mirror_pad_y:
            pad_ref.dmirror_y(y=pad_ref.dcenter[1])

        if use_route_quad:
            # M2 taper from full via width (wire_port.width == vw) at the via contact
            # down to the pad port width -- fully covers the via.
            route_quad(
                c,
                port1=wire_port,
                port2=pad_ref.ports["e11"],
                width2=pad_ref.ports["e11"].dwidth,
                layer=routing_xs.layer,
            )
        elif route_points:
            p_start = np.array(wire_port.dcenter)
            p_end   = np.array(pad_ref.ports["e11"].dcenter)
            pts = [p_start] + [p_start + np.array(pt) for pt in route_points] + [p_end]
            path = gf.Path(pts)
            _ = c << path.extrude(routing_xs)
        else:
            routing_port = wire_port.copy()
            routing_port.orientation = port_orientation

            route_kwargs: dict[str, Any] = {}
            if start_straight_length is not None:
                route_kwargs["start_straight_length"] = start_straight_length
            if end_straight_length is not None:
                route_kwargs["end_straight_length"] = end_straight_length

            _ = gf.routing.route_single_electrical(
                c,
                port1=pad_ref.ports["e11"],
                port2=routing_port,
                cross_section=routing_xs,
                **route_kwargs,
            )

        if transition_m2_hr_params is not None:
            _place_m2_rect(                                   # R6
                tmp_transitions,
                pad_ref.dcenter[0], pad_ref.dcenter[1],
                (pad_w, pad_h), layer_m2,
            )
            y_off             = -pad_h / 2 if mirror_pad_y else pad_h / 2
            added_orientation = -90.0      if mirror_pad_y else 90.0
            _add_m2_port(                                     # R3
                c, port_name,
                pad_ref.ports["e11"].dcenter[0],
                pad_ref.ports["e11"].dcenter[1] + y_off,
                pad_w, added_orientation, layer_m2,
            )
        else:
            c.add_port(name=port_name, port=pad_ref.ports["e11"])

        return pad_ref

    # Dynamic calculation of vertical Y-ports for staggering
    pad_h_local  = heater_pad_size[1]
    y_port_upper = pad_e1_heater_offset[1] - 0.5 * pad_h_local

    if heater_on_both_branches:
        wire_gap_y = (
            ht_ref_1.ports["ht_end"].dcenter[1]
            - ht_ref_2.ports["ht_end"].dcenter[1]
        )
        if align_pads_same_y:
            y_port_lower         = pad_e1_heater_offset[1] - 0.5 * pad_h_local + wire_gap_y
            resolved_pad3_offset = (pad_e3_heater_offset[0], pad_e1_heater_offset[1] + wire_gap_y)
            resolved_pad4_offset = (pad_e4_heater_offset[0], pad_e2_heater_offset[1] + wire_gap_y)
        else:
            y_port_lower         = pad_e3_heater_offset[1] + 0.5 * pad_h_local
            resolved_pad3_offset = pad_e3_heater_offset
            resolved_pad4_offset = pad_e4_heater_offset
    else:
        resolved_pad3_offset = None
        resolved_pad4_offset = None

    # Default routes for align_pads_same_y=True only.
    # When align_pads_same_y=False, route_quad is used directly (no waypoints needed).
    if align_pads_same_y:
        if pad_e1_heater_route is None:
            pad_e1_heater_route = [(0.0, pad_width), (pad_e1_heater_offset[0], pad_width)]
        if pad_e2_heater_route is None:
            pad_e2_heater_route = [(0.0, 30.0),      (pad_e2_heater_offset[0], 30.0)]
        if heater_on_both_branches:
            if pad_e3_heater_route is None:
                pad_e3_heater_route = [(0.0, -120.0), (resolved_pad3_offset[0], -120.0)]
            if pad_e4_heater_route is None:
                pad_e4_heater_route = [(0.0,  -30.0), (resolved_pad4_offset[0],  -30.0)]

    # Upper arm: ports exit UP (orientation 90 deg) above the upper optical waveguide
    _place_and_route(
        ht_ref_1.ports["ht_end"],
        pad_e1_heater_offset,
        "e1_heater",
        use_route_quad=not align_pads_same_y,
        port_orientation=90.0,
        start_straight_length=30.0,
        end_straight_length=y_port_upper - 30.0,
        route_points=pad_e1_heater_route,
    )
    _place_and_route(
        ht_ref_1.ports["ht_start"],
        pad_e2_heater_offset,
        "e2_heater",
        use_route_quad=not align_pads_same_y,
        port_orientation=90.0,
        start_straight_length=10.0,
        end_straight_length=y_port_upper - 10.0,
        route_points=pad_e2_heater_route,
    )

    if heater_on_both_branches:
        lower_port_orientation = 270.0
        mirror_lower           = not align_pads_same_y

        _place_and_route(
            ht_ref_2.ports["ht_end"],
            resolved_pad3_offset,
            "e3_heater",
            use_route_quad=not align_pads_same_y,
            port_orientation=lower_port_orientation,
            start_straight_length=30.0,
            end_straight_length=y_port_lower + 30.0,
            route_points=pad_e3_heater_route,
            mirror_pad_y=mirror_lower,
        )
        _place_and_route(
            ht_ref_2.ports["ht_start"],
            resolved_pad4_offset,
            "e4_heater",
            use_route_quad=not align_pads_same_y,
            port_orientation=lower_port_orientation,
            start_straight_length=10.0,
            end_straight_length=y_port_lower + 10.0,
            route_points=pad_e4_heater_route,
            mirror_pad_y=mirror_lower,
        )

    if transition_m2_hr_params is not None:
        for lyr, polygons in tmp_transitions.get_polygons().items():
            for polygon in polygons:
                c.add_polygon(polygon, layer=lyr)

    c.flatten()
    return c


if __name__ == "__main__":
    from ltoi300.tech import xs_ht_wire

    heater_xs  = xs_ht_wire(width=0.9)
    routing_xs = xs_ht_wire(width=10.0)
    c = heater_straight_compact(
        heater_xs=heater_xs,
        routing_xs=routing_xs,
    )
    c.show()
