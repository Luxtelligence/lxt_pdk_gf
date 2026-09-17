import gdsfactory as gf
import numpy as np
from gdsfactory.components import grating_coupler_elliptical_arbitrary

CrossSectionSpec = gf.typings.CrossSectionSpec


@gf.cell
def gc_focusing_arbitrary(
    pitch: tuple,
    fill_factor: tuple,
    alpha_t: float,
    tap_len: float,
    N: int,
    wg_l: float,
    wl_0: float,
    theta: float,
    bias_gap: float,
    sleeve_width: float,
    ridge_thickness: float,
    cross_section: CrossSectionSpec,
    sidewall_angle: float | None = None,
    **kwargs,
):
    """Focusing grating coupler with arbitrary pitch and filling factor, defined in a partial etch technology."""

    kwargs.update(
        {
            "wl_0": wl_0,
            "theta": theta,
            "N": N,
        }
    )

    if sidewall_angle is None:
        sidewall_angle = 0.0

    if not isinstance(pitch, np.ndarray):
        pitch = np.array(pitch, dtype=np.float64)

    if not isinstance(fill_factor, np.ndarray):
        fill_factor = np.array(fill_factor, dtype=np.float64)

    widths = np.round(
        pitch * fill_factor - ridge_thickness * np.tan(np.pi * sidewall_angle / 180.0),
        3,
    )
    gaps = np.round(pitch - widths, 3)
    pitch = np.round(pitch, 3)

    xs = gf.get_cross_section(cross_section)
    layer_ridge = xs.layer
    sleeve_layers = [s.layer for s in xs.sections if s.layer != layer_ridge]
    layer_slab = sleeve_layers[0] if sleeve_layers else None

    gc = grating_coupler_elliptical_arbitrary(
        gaps=tuple(gaps),
        widths=tuple(widths),
        taper_length=tap_len,
        taper_angle=alpha_t,
        wavelength=wl_0,
        fiber_angle=theta,
        nclad=1.0,
        layer_grating=layer_ridge,
        layer_slab=layer_slab,
        taper_to_slab_offset=-tap_len,
        spiked=False,
        bias_gap=bias_gap,
        cross_section=cross_section,
    )

    # Compute sleeve region around the grating coupler teeth

    gc_with_sleeve = gf.Component()
    smooth_param = 0.1

    ridge_region = gc.get_region(layer=layer_ridge)
    sized_region = ridge_region.sized(sleeve_width * 1e3)
    sized_region.smooth(smooth_param * 1e3)

    gc_with_sleeve.add_polygon(ridge_region, layer=layer_ridge)
    if layer_slab:
        gc_with_sleeve.add_polygon(sized_region, layer=layer_slab)

    [
        gc_with_sleeve.add_port(
            port=p,
        )
        for p in gc.ports
    ]

    c = gf.Component()

    if wg_l < 0:
        raise ValueError(
            f"The wg_l parameter must be a non-negative quantity, received {wg_l} instead."
        )
    elif wg_l > 0:
        c_extend = gf.components.extend_ports(
            component=gc_with_sleeve,
            port_names=("o1",),
            length=wg_l,
            cross_section=cross_section,
            allow_width_mismatch=True,
        )
        c << c_extend
        [
            c.add_port(
                port=p,
            )
            for p in c_extend.ports
        ]

    else:
        c = gc_with_sleeve

    c.flatten()

    if layer_slab:
        c.over_under(layer=layer_slab, distance=kwargs.get("over_under_distance", 0.5))

    c.info["gaps"] = tuple(gaps)
    c.info["widths"] = tuple(widths)
    c.info["pitch"] = tuple(pitch)

    return c



def gc_lin_ff(
    no: float = 1.864,
    ne: float = 1.659,
    nm: float = 1.0,
    theta: float = 14.5,
    N: int = 18,
    wl_0: float = 1.55,
    R: float = 0.075,
    wmin: float = 0.45,
    wmin_bias: float = 0.046,
    gmin: float = 0.25,
):
    """
    Compute the pitches, teeth widths and teeth gaps for an apodized grating coupler
    designed to work at the wavelength wl_0.
    """
    Lx = np.zeros(N)
    fx = np.zeros_like(Lx)
    theta_rad = np.pi * theta / 180
    L0 = wl_0 / (no - nm * np.sin(theta_rad))
    f0 = (L0 - gmin) / L0
    x = 0
    for i in range(N):
        fx[i] = f0 - R * x
        nx = fx[i] * no + (1 - fx[i]) * ne
        Lx[i] = wl_0 / (nx - nm * np.sin(theta_rad))
        if (fx[i] - R * Lx[i]) * Lx[i] >= wmin + wmin_bias:
            x += Lx[i]
    return (Lx, fx)


@gf.cell
def gc_lin_chirp(
    Lx: tuple = None,
    fx: tuple = None,
    alpha_t: float = 50.0,
    tap_len: float = 12.5,
    N: int = 18,
    wg_l: float = 0,
    wl_0: float = 1.55,
    theta: float = 14.5,
    bias_gap: float = 0.0,
    sleeve_width: float = 3.5,
    ridge_thickness: float = 0.2,
    sidewall_angle: float = 13.0,
    cross_section: str = "xs_rwg1000",
    **kwargs,
):
    """Focusing grating coupler with a linear filling factor evolution."""

    kwargs.update(
        {
            "wl_0": wl_0,
            "theta": theta,
            "N": N,
        }
    )
    Ldef, fdef = gc_lin_ff(**kwargs)
    if Lx is None:
        Lx = Ldef
    if fx is None:
        fx = fdef

    if not isinstance(Lx, np.ndarray):
        Lx = np.array(Lx, dtype=np.float64)

    if not isinstance(fx, np.ndarray):
        fx = np.array(fx, dtype=np.float64)

    widths = np.round(
        Lx * fx - ridge_thickness * np.tan(np.pi * sidewall_angle / 180.0), 3
    )
    gaps = np.round(Lx - widths, 3)
    Lx = np.round(Lx, 3)

    # Identify "slab" and "ridge" layers from the provided gdsfactory cross-section
    slab_layer = None
    ridge_layer = None
    try:
        xs = gf.get_cross_section(cross_section)
        if hasattr(xs, "sections"):
            for section in xs.sections:
                # Check for "slab" in the section name
                if hasattr(section, "name") and section.name:
                    name_lower = section.name.lower()
                    if "slab" in name_lower and hasattr(section, "layer"):
                        slab_layer = section.layer
                    if "ridge" in name_lower and hasattr(section, "layer"):
                        ridge_layer = section.layer
                # If both found, break
                if slab_layer is not None and ridge_layer is not None:
                    break
        # fallback: also check 'layer' attributes directly if sections is None/empty
        if slab_layer is None and hasattr(xs, "layer") and hasattr(xs, "name"):
            if xs.name and "slab" in xs.name.lower():
                slab_layer = xs.layer
        if ridge_layer is None and hasattr(xs, "layer") and hasattr(xs, "name"):
            if xs.name and "ridge" in xs.name.lower():
                ridge_layer = xs.layer
    except Exception:
        slab_layer = None
        ridge_layer = None


    gc = grating_coupler_elliptical_arbitrary(
        gaps=tuple(gaps),
        widths=tuple(widths),
        taper_length=tap_len,
        taper_angle=alpha_t,
        wavelength=wl_0,
        fiber_angle=theta,
        nclad=1.0,
        layer_grating=ridge_layer,
        layer_slab=None,
        taper_to_slab_offset=-tap_len,
        spiked=False,
        bias_gap=bias_gap,
        cross_section=cross_section,
    )

    # Compute sleeve region around the grating coupler teeth

    gc_with_sleeve = gf.Component()
    smooth_param = 0.1

    ridge_region = gc.get_region(layer=ridge_layer)
    sized_region = ridge_region.sized(sleeve_width * 1e3)
    sized_region.smooth(smooth_param * 1e3)

    gc_with_sleeve.add_polygon(ridge_region, layer=ridge_layer)
    gc_with_sleeve.add_polygon(sized_region, layer=slab_layer)

    [
        gc_with_sleeve.add_port(
            port=p,
        )
        for p in gc.ports
    ]

    c = gf.Component()

    if wg_l < 0:
        raise ValueError(
            f"The wg_l parameter must be a non-negative quantity, received {wg_l} instead."
        )
    elif wg_l > 0:
        c_extend = gf.components.extend_ports(
            component=gc_with_sleeve,
            port_names=("o1",),
            length=wg_l,
            cross_section=cross_section,
            allow_width_mismatch=True,
        )
        c << c_extend
        [
            c.add_port(
                port=p,
            )
            for p in c_extend.ports
        ]

    else:
        c = gc_with_sleeve

    c.flatten()
    c.info["gaps"] = tuple(gaps)
    c.info["widths"] = tuple(widths)
    c.info["Lx"] = tuple(Lx)

    return c


@gf.cell
def gc_uniform_focusing(
    pitch: float = 1.1,
    filling_factor: float = 0.5,
    alpha_t: float = 50.0,
    tap_len: float = 12.5,
    N: int = 30,
    wg_l: float = 0,
    bias_gap: float = 0.0,
    sleeve_width: float = 3.5,
    ridge_thickness: float = 0.2,
    sidewall_angle: float = 13.0,
    cross_section: str = "xs_rwg1000",
    **kwargs,
):
    """Focusing grating coupler with a constant filling factor."""

    Lx = tuple(pitch * np.ones(N))
    fx = tuple(filling_factor * np.ones(N))

    c = gc_lin_chirp(
        Lx=Lx,
        fx=fx,
        alpha_t=alpha_t,
        tap_len=tap_len,
        N=N,
        wg_l=wg_l,
        bias_gap=bias_gap,
        sleeve_width=sleeve_width,
        ridge_thickness=ridge_thickness,
        sidewall_angle=sidewall_angle,
        cross_section=cross_section,
        **kwargs,
    )

    return c
