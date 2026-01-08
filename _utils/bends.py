import gdsfactory as gf
import numpy as np
from gdsfactory.typings import CrossSectionSpec

from _utils.spline import (
    bend_S_spline,
    spline_clamped_path,
)


def L_turn_bend(
    radius: float = 80.0,
    p: float = 1.0,
    with_arc_floorplan: bool = True,
    cross_section: CrossSectionSpec = "xs_rwg700",
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


def S_bend_vert(
    v_offset: float = 25.0,
    h_extent: float = 100.0,
    dx_straight: float = 5.0,
    cross_section: CrossSectionSpec = "xs_rwg700",
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

def bend_euler_tapered(
    radius: float = 45.0,
    w0: float = 3.0,
    wc: float = 1.0,
    cross_section: CrossSectionSpec = "xs_rwg700",
) -> gf.Component:
    euler_path = gf.path.euler(
        radius=radius,
        angle=180.0,
        p=1,
        use_eff=True,
    )
    euler_path.start_angle = 0.0
    euler_path.end_angle = 180.0

    def width_fun(t):
        if any(t > 1) or any(t < 0):
            raise ValueError()
        y = np.zeros_like(t)
        y[t <= 0.5] = w0 + (wc - w0) * 2 * t[t <= 0.5]
        y[t > 0.5] = -(wc - w0) * 2 * t[t > 0.5] + 2 * wc - w0
        return y

    sec = gf.Section(
        layer=cross_section().layer, width=0, width_function=width_fun, port_names=("o1", "o2")
    )

    xs_tapered = gf.CrossSection(sections=(sec,))

    return euler_path.extrude(cross_section=xs_tapered)