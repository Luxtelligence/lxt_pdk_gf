import gdsfactory as gf
import numpy as np
from gdsfactory.components import grating_coupler_elliptical_arbitrary

CrossSectionSpec = gf.typings.CrossSectionSpec


@gf.cell
def gc_lin_chirp(
    Lx: tuple,
    fx: tuple,
    alpha_t: float,
    tap_len: float,
    N: int,
    wg_l: float,
    wl_0: float,
    theta: float,
    bias_gap: float,
    sleeve_width: float,
    ridge_thickness: float,
    sidewall_angle: float,
    cross_section: CrossSectionSpec,
    **kwargs,
):
    """Focusing grating coupler with a linear filling factor evolution, defined in a partial etch technology."""

    kwargs.update(
        {
            "wl_0": wl_0,
            "theta": theta,
            "N": N,
        }
    )

    if not isinstance(Lx, np.ndarray):
        Lx = np.array(Lx, dtype=np.float64)

    if not isinstance(fx, np.ndarray):
        fx = np.array(fx, dtype=np.float64)

    widths = np.round(
        Lx * fx - ridge_thickness * np.tan(np.pi * sidewall_angle / 180.0), 3
    )
    gaps = np.round(Lx - widths, 3)
    Lx = np.round(Lx, 3)

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
    c.info["gaps"] = tuple(gaps)
    c.info["widths"] = tuple(widths)
    c.info["Lx"] = tuple(Lx)

    return c
