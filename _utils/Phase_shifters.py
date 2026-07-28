from typing import Any
import gdsfactory as gf
import numpy as np

# Import required cells, tech parameters, and helper functions directly from the PDK
from ltoi300.tech import LayerMapLTOI300, LAYER, xs_rwg700, xs_rwg900, xs_uni_cpw
from _utils.cross_section import get_cpw_from_xs, xs_cpw_single_layer
from _utils.bends import L_turn_bend
from _utils.gsg_rf import (
    _spline_bend_points,
    via_array,
    m2_bonding_pads,
    straight_cpw,
)
from _utils.thermal_phase_shifters import heater_straight_compact

# ==============================================================================
# 1. Custom Rectangular CPW Pad (Used in EO Phase Shifter)
# ==============================================================================
@gf.cell
def rectangular_cpw_pad(
    cpw_xs,
    pitch: float = 100.0,
    length_straight: float = 25.0,
    length_tapered: float = 190.0,
    ground_pad_width: float = 150.0,
    optical_waveguide_xs = None,
    m2_bonding_pads_params: dict[str, Any] | None = None,
    single_waveguide: bool = False,
    dc_pad_width: float = 80.0,
) -> gf.Component:
    """RF access line for high-frequency GSG probes with rectangular pads and
    curved electrodes following the optical waveguides.
    """
    pad = gf.Component()
    _cpw_xs = gf.get_cross_section(cpw_xs)
    end_width, end_ground_width, end_gap, tl_layer = get_cpw_from_xs(_cpw_xs)

    grid_size = 0.004
    dc_pad_width = round(dc_pad_width / grid_size) * grid_size
    ground_pad_width = round(ground_pad_width / grid_size) * grid_size
    end_width = round(end_width / grid_size) * grid_size
    end_ground_width = round(end_ground_width / grid_size) * grid_size
    end_gap = round(end_gap / grid_size) * grid_size

    total_length = length_straight + length_tapered
    y_start_upper = pitch / 2
    y_end_upper = end_width / 2 + end_gap / 2
    npoints = int(np.round(2.5 * length_tapered))

    straight_upper = np.array([[0.0, y_start_upper], [length_straight, y_start_upper]])
    bend_upper = _spline_bend_points((length_straight, y_start_upper), (total_length, y_end_upper), npoints)
    points_upper = np.vstack([straight_upper, bend_upper[1:]])
    points_lower = points_upper.copy()
    points_lower[:, 1] = -points_lower[:, 1]

    path_upper = gf.Path(points_upper)
    path_upper.start_angle = path_upper.end_angle = 0.0
    path_lower = gf.Path(points_lower)
    path_lower.start_angle = path_lower.end_angle = 0.0

    gap_straight = pitch - dc_pad_width
    gap_straight = round(gap_straight / grid_size) * grid_size
    t_straight_end = length_straight / total_length

    def gap_width_func(t):
        if np.isscalar(t):
            t = np.array([t])
            scalar_input = True
        else:
            scalar_input = False

        widths = np.zeros_like(t)
        mask_straight = t <= t_straight_end
        widths[mask_straight] = gap_straight

        mask_bend = t > t_straight_end
        if np.any(mask_bend):
            t_bend_norm = (t[mask_bend] - t_straight_end) / (1.0 - t_straight_end)
            widths[mask_bend] = gap_straight + (end_gap - gap_straight) * (t_bend_norm**2) * (3 - 2 * t_bend_norm)

        grid = 0.002
        widths_dbu = np.round(widths / grid)
        widths = widths_dbu * grid
        return widths[0] if scalar_input else widths

    section_gap_upper = gf.Section(layer=tl_layer, width=0, width_function=gap_width_func, port_names=("o1", "o2"))
    section_gap_lower = gf.Section(layer=tl_layer, width=0, width_function=gap_width_func, port_names=("o1", "o2"))

    y_pad_outer = dc_pad_width / 2 + gap_straight + ground_pad_width
    y_cpw_outer = end_width / 2 + end_gap + end_ground_width
    y_pad_outer = round(y_pad_outer / grid_size) * grid_size
    y_cpw_outer = round(y_cpw_outer / grid_size) * grid_size

    full_pts = [
        (0.0, y_pad_outer), (length_straight, y_pad_outer), (total_length, y_cpw_outer),
        (total_length, -y_cpw_outer), (length_straight, -y_pad_outer), (0.0, -y_pad_outer),
    ]

    pad_base = gf.Component()
    pad_base.add_polygon(full_pts, layer=tl_layer)

    xs_gap_upper = gf.CrossSection(sections=(section_gap_upper,))
    xs_gap_lower = gf.CrossSection(sections=(section_gap_lower,))
    gap_extrusion_upper = path_upper.extrude(cross_section=xs_gap_upper)
    gap_extrusion_lower = path_lower.extrude(cross_section=xs_gap_lower)

    pad_with_upper_cut = gf.boolean(A=pad_base, B=gap_extrusion_upper, operation="not", layer=tl_layer)
    pad_final = gf.boolean(A=pad_with_upper_cut, B=gap_extrusion_lower, operation="not", layer=tl_layer)

    xs_straight = xs_cpw_single_layer(central_conductor_width=dc_pad_width, ground_planes_width=ground_pad_width, gap=gap_straight, layer=tl_layer)
    pad_final.add_port(name="e1", cross_section=xs_straight, orientation=180.0, center=(0.0, 0.0), port_type="electrical")
    pad_final.add_port(name="e2", cross_section=_cpw_xs, orientation=0.0, center=(total_length, 0.0), port_type="electrical")

    p1 = pad << pad_final

    if optical_waveguide_xs is not None:
        wg_upper = path_upper.extrude(optical_waveguide_xs)
        pad << wg_upper
        pad.add_port(name="o1", port=wg_upper.ports["o1"])
        pad.add_port(name="o2", port=wg_upper.ports["o2"])

        if not single_waveguide:
            wg_lower = path_lower.extrude(optical_waveguide_xs)
            pad << wg_lower
            pad.add_port(name="o3", port=wg_lower.ports["o2"])
            pad.add_port(name="o4", port=wg_lower.ports["o1"])

    if m2_bonding_pads_params is not None:
        required_keys = ("layer_m2", "layer_openings")
        missing_keys = [k for k in required_keys if k not in m2_bonding_pads_params]
        if missing_keys:
            raise ValueError("m2_bonding_pads_params is missing required keys: " + ", ".join(missing_keys))

        allowed_optional_keys = {"m1_opening_offset", "opening_size", "opening_separation", "tl_opening_host_width", "m2_pad_length"}
        m2_bonding_pads_component = m2_bonding_pads(
            pad_xs=xs_straight, layer_m2=m2_bonding_pads_params["layer_m2"], layer_openings=m2_bonding_pads_params["layer_openings"],
            **{key: m2_bonding_pads_params[key] for key in allowed_optional_keys if key in m2_bonding_pads_params},
        )
        M2_bonding_pads_ref = pad << m2_bonding_pads_component
        M2_bonding_pads_ref.connect("e2", p1.ports["e1"])
        pad.add_port(name="e1", port=M2_bonding_pads_ref.ports["e1"])
        pad.add_port(name="e3", port=p1.ports["e1"])
    else:
        pad.add_port(name="e1", port=p1.ports["e1"])

    pad.add_port(name="e2", port=p1.ports["e2"])
    return pad

# ==============================================================================
# 2. Compensation Section (Used in EO Phase Shifter)
# ==============================================================================
@gf.cell
def compensation_section(
    optical_xs: gf.CrossSection,
    compensation_length: float = 0.0,
    length_imbalance: float = 0.0,
    roc: float = 60.0,
    cpw_spacing: float = 25.5,
) -> gf.Component:
    """Symmetric path-length compensation section with dual loops.
    
    The upper arm has a detour loop going North.
    The lower arm has a detour loop going South.
    """
    L_extra = compensation_length + length_imbalance
    c = gf.Component()

    if L_extra > 0:
        bend = L_turn_bend(radius=roc, cross_section=optical_xs)

        # Build Path Up
        b_up1 = c << bend
        b_up1.move(b_up1.ports["o1"].dcenter, (0, cpw_spacing / 2))

        H_base = 20
        v_up1 = c << gf.components.straight(length=H_base, cross_section=optical_xs)
        v_up1.connect("o1", b_up1.ports["o2"])

        b_up2 = c << bend 
        b_up2.dmirror_y()
        b_up2.connect("o1", v_up1.ports["o2"])

        v_up2 = c << gf.components.straight(length=H_base, cross_section=optical_xs)
        v_up2.connect("o1", b_up2.ports["o2"])

        b_up3 = c << bend
        b_up3.dmirror_y()
        b_up3.connect("o1", v_up2.ports["o2"])

        h_up = c << gf.components.straight(length=H_base, cross_section=optical_xs)
        h_up.connect("o1", b_up3.ports["o2"])

        b_up4 = c << bend 
        b_up4.connect("o1", h_up.ports["o2"])

        # Build Path Down
        b_down1 = c << bend
        b_down1.dmirror_y()
        b_down1.move(b_down1.ports["o1"].dcenter, (0, -cpw_spacing / 2))

        v_down1 = c << gf.components.straight(length=0.5 * L_extra, cross_section=optical_xs)
        v_down1.connect("o1", b_down1.ports["o2"])

        b_down2 = c << bend
        b_down2.connect("o1", v_down1.ports["o2"])

        v_down2 = c << gf.components.straight(length=H_base, cross_section=optical_xs)
        v_down2.connect("o1", b_down2.ports["o2"])

        b_down3 = c << bend 
        b_down3.connect("o1", v_down2.ports["o2"])

        v_down3 = c << gf.components.straight(length=0.5 * L_extra, cross_section=optical_xs)
        v_down3.connect("o1", b_down3.ports["o2"]) 

        b_down4 = c << bend 
        b_down4.dmirror_y()
        b_down4.connect("o1", v_down3.ports["o2"])

        # Expose ports
        c.add_port(name="o_in_up", port=b_up1.ports["o1"])
        c.add_port(name="o_in_down", port=b_down1.ports["o1"])
        c.add_port(name="o_out_up", port=b_up4.ports["o2"])
        c.add_port(name="o_out_down", port=b_down4.ports["o2"])

        c.flatten()

    return c

# ==============================================================================
# 3. Electro-Optic (EO) Phase Shifter Top-Level Wrapper
# ==============================================================================
@gf.cell
def EO_Phase_shifter(
    length: float = 2000.0,
    rf_gap: float = 5.5,
    rf_central_conductor_width: float = 20.0,
    gsg_pitch: float = 100.0,
    dc_pad_width: float = 80.0,
    dc_ground_width: float = 150.0,
    length_straight: float = 25.0,
    length_tapered: float = 150.0,
    taper_length: float = 100.0,
    modulation_width: float = 2.5,
    length_imbalance: float = 20.0,
    compensation_length: float = 100.0,
    roc: float | None = None,
    band: str = "oband",
) -> gf.Component:
    """Standalone EO Phase Shifter cell combining a rectangular GSG pad, 
    straight active CPW, path-length compensation section, and 1x2 MMI combiner.
    """
    c = gf.Component()

    if roc is None:
        roc = 60.0 if band == "oband" else 50.0

    # 1. Build local cross-sections
    _cpw_xs = xs_uni_cpw(
        central_conductor_width=rf_central_conductor_width,
        gap=rf_gap,
        ground_planes_width=50.0,
    )
    xs_func = xs_rwg700 if band == "oband" else xs_rwg900
    terminal_xs = xs_func()
    modulation_xs = xs_func(width=modulation_width)
    optical_waveguides = {
        "terminal_xs": terminal_xs,
        "modulation_xs": modulation_xs,
        "taper_length": taper_length,
    }

    # Use default M2 metal bonding pads configuration
    m2_bonding_pad_params = {
        "layer_m2": (22, 0),
        "layer_openings": (40, 0),
        "m1_opening_offset": 2.5,
        "opening_size": 12.0,
        "opening_separation": 12.0,
        "tl_opening_host_width": 45.0,
        "m2_pad_length": 80.0,
    }

    # 2. Instantiate straight_cpw and rectangular_cpw_pad
    top_pad = rectangular_cpw_pad(
        cpw_xs=_cpw_xs,
        optical_waveguide_xs=terminal_xs,
        pitch=gsg_pitch,
        length_straight=length_straight,
        length_tapered=length_tapered,
        ground_pad_width=dc_ground_width,
        m2_bonding_pads_params=m2_bonding_pad_params,
        dc_pad_width=dc_pad_width,
    )
    top_cpw = straight_cpw(
        cpw_xs=_cpw_xs,
        modulation_length=length,
        optical_waveguides=optical_waveguides,
    )

    # 3. Add to component and align references
    top_pad_ref = c << top_pad
    top_cpw_ref = c << top_cpw

    top_pad_ref.connect("e2", top_cpw_ref.ports["e1"])

    # 4. Instantiate and connect compensation section if path imbalance/compensation is present
    L_extra = compensation_length + length_imbalance
    comp_ref = None
    if L_extra > 0:
        comp_cell = compensation_section(
            optical_xs=terminal_xs,
            compensation_length=compensation_length,
            length_imbalance=length_imbalance,
            roc=roc,
            cpw_spacing=rf_central_conductor_width + rf_gap,
        )
        comp_ref = c << comp_cell
        comp_ref.connect("o_in_up", top_cpw_ref.ports["o2"])

    # 5. Expose external ports
    c.add_port(name="o1", port=top_pad_ref.ports["o1"])
    c.add_port(name="o2", port=top_pad_ref.ports["o4"])
    if comp_ref is not None:
        c.add_port(name="o3", port=comp_ref.ports["o_out_up"])
        c.add_port(name="o4", port=comp_ref.ports["o_out_down"])
    else:
        c.add_port(name="o3", port=top_cpw_ref.ports["o2"])
        c.add_port(name="o4", port=top_cpw_ref.ports["o3"])
    c.add_port(name="e1", port=top_pad_ref.ports["e1"])
    c.add_port(name="e2", port=top_cpw_ref.ports["e2"])

    return c

# ==============================================================================
# 4. Thermo-Optic (TO) Dual-Heater Active Section Cell Wrapper Definition
# ==============================================================================
@gf.cell
def heater(
    length: float = 700.0,
    heater_width: float = 0.9,
    routing_width: float = 10.0,
    pad_size: tuple[float, float] = (150.0, 150.0),
    pad_vert_offset: float = 10.0,
    optical_xs = None,
    layer_hrl = (23, 0),
    layer_m2 = (22, 0),
    spacing: float = 26.087,
    H_base: float = 20.0,
    compensation_length: float = 100.0,
    length_imbalance: float = 20.0,
    roc: float | None = None,
    transition_m2_hr_params: dict[str, Any] | None = None,
    band: str = "oband",
) -> gf.Component:
    """Thermo-Optic dual-heater active section cell with integrated path-length compensation.
    
    Ports:
      - o1, o2: Upper waveguide West input / East output
      - o3, o4: Lower waveguide East output / West input
      - e1, e2: Upper heater Left / Right electrical pads
      - e3, e4: Lower heater Left / Right electrical pads
    """
    c = gf.Component()
    
    if roc is None:
        roc = 60.0 if band == "oband" else 50.0

    if optical_xs is None:
        xs_func = xs_rwg700 if band == "oband" else xs_rwg900
        optical_xs = xs_func()
        
    L_extra = compensation_length + length_imbalance
    bend = L_turn_bend(radius=roc, cross_section=optical_xs)
    
    heater_xs = gf.cross_section.cross_section(
        width=heater_width,
        layer=layer_hrl,
        port_names=gf.cross_section.port_names_electrical,
        port_types=gf.cross_section.port_types_electrical,
    )
    routing_xs = gf.cross_section.cross_section(
        width=routing_width,
        layer=layer_hrl,
        port_names=gf.cross_section.port_names_electrical,
        port_types=gf.cross_section.port_types_electrical,
    )
    
    if transition_m2_hr_params is None:
        transition_m2_hr_params = {
            "layer_openings": (41, 0),
            "layer_m2": layer_m2,
            "opening_offset": 2.5,
        }
        
    heater_cell = heater_straight_compact(
        heater_xs=heater_xs,
        routing_xs=routing_xs,
        length=length,
        pad_size=pad_size,
        pad_vert_offset=pad_vert_offset,
        transition_m2_hr_params=transition_m2_hr_params,
    )
    
    # Path Up
    ext_up_in = c << gf.components.straight(length=10.0, cross_section=optical_xs)
    ext_up_in.move(ext_up_in.ports["o1"].dcenter, (0, spacing / 2))
    
    b_up1 = c << bend
    b_up1.connect("o1", ext_up_in.ports["o2"])
    
    v_up1 = c << gf.components.straight(length=H_base, cross_section=optical_xs)
    v_up1.connect("o1", b_up1.ports["o2"])
    
    b_up2 = c << bend
    b_up2.dmirror_y()
    b_up2.connect("o1", v_up1.ports["o2"])
    
    wg_up_active = c << gf.components.straight(length=length, cross_section=optical_xs)
    wg_up_active.connect("o1", b_up2.ports["o2"])
    
    ht_up = c << heater_cell
    ht_up.dmove(ht_up.ports["ht_start"].dcenter, b_up2.ports["o2"].dcenter)
    
    b_up3 = c << bend
    b_up3.dmirror_y()
    b_up3.connect("o1", wg_up_active.ports["o2"])
    
    v_up2 = c << gf.components.straight(length=H_base, cross_section=optical_xs)
    v_up2.connect("o1", b_up3.ports["o2"])
    
    b_up4 = c << bend
    b_up4.connect("o1", v_up2.ports["o2"])
    
    ext_up_out = c << gf.components.straight(length=10.0, cross_section=optical_xs)
    ext_up_out.connect("o1", b_up4.ports["o2"])
    
    # Path Down
    ext_down_in = c << gf.components.straight(length=10.0, cross_section=optical_xs)
    ext_down_in.move(ext_down_in.ports["o1"].dcenter, (0, -spacing / 2))
    
    b_down1 = c << bend
    b_down1.dmirror_y()
    b_down1.connect("o1", ext_down_in.ports["o2"])
    
    v_down1 = c << gf.components.straight(length=H_base + 0.5 * L_extra, cross_section=optical_xs)
    v_down1.connect("o1", b_down1.ports["o2"])
    
    b_down2 = c << bend
    b_down2.connect("o1", v_down1.ports["o2"])
    
    wg_down_active = c << gf.components.straight(length=length, cross_section=optical_xs)
    wg_down_active.connect("o1", b_down2.ports["o2"])
    
    ht_down = c << heater_cell
    ht_down.dmirror_y()
    ht_down.dmove(ht_down.ports["ht_start"].dcenter, b_down2.ports["o2"].dcenter)
    
    b_down3 = c << bend
    b_down3.connect("o1", wg_down_active.ports["o2"])
    
    v_down2 = c << gf.components.straight(length=H_base + 0.5 * L_extra, cross_section=optical_xs)
    v_down2.connect("o1", b_down3.ports["o2"])
    
    b_down4 = c << bend
    b_down4.dmirror_y()
    b_down4.connect("o1", v_down2.ports["o2"])
    
    ext_down_out = c << gf.components.straight(length=10.0, cross_section=optical_xs)
    ext_down_out.connect("o1", b_down4.ports["o2"])
    
    # Expose ports
    c.add_port(name="o1", port=ext_up_in.ports["o1"])
    c.add_port(name="o2", port=ext_up_out.ports["o2"])
    c.add_port(name="o3", port=ext_down_out.ports["o2"])
    c.add_port(name="o4", port=ext_down_in.ports["o1"])
    
    c.add_port(name="e1", port=ht_up.ports["e1"])
    c.add_port(name="e2", port=ht_up.ports["e2"])
    c.add_port(name="e3", port=ht_down.ports["e1"])
    c.add_port(name="e4", port=ht_down.ports["e2"])
    
    c.flatten()
    return c

# Aliases
TO_phase_shifter = heater
TO__phase_shifter = heater

if __name__ == "__main__":
    try:
        import ltoi300
        ltoi300.activate_pdk()
    except Exception:
        pass
    c_eo = EO_Phase_shifter()
    c_eo.write_gds("EO_Phase_shifter.gds")
    print("EO Phase Shifter GDS written.")
    
    c_to = heater()
    c_to.write_gds("TO_phase_shifter.gds")
    print("TO Phase Shifter (heater) GDS written.")
