import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
from gdsfactory.typings import Coordinate, CrossSectionSpec


def spline_clamped_path(
    t: np.ndarray, start: Coordinate = (0.0, 0.0), end: Coordinate = (120.0, 25.0)
):
    """Returns a spline path with a null first derivative at the extrema."""

    xs = t
    ys = (t**2) * (3 - 2 * t)

    # Rescale to the start and end coordinates

    xs = start[0] + (end[0] - start[0]) * xs
    ys = start[1] + (end[1] - start[1]) * ys

    path = gf.Path(np.column_stack([xs, ys]))
    path.start_angle = path.end_angle = 0.0

    return path


def spline_null_curvature(
    t: np.ndarray, start: Coordinate = (0.0, 0.0), end: Coordinate = (120.0, 25.0)
):
    """Returns a spline path with zero first and second derivatives at the extrema."""

    xs = t
    ys = (t**3) * (6 * t**2 - 15.0 * t + 10.0)

    xs = start[0] + (end[0] - start[0]) * xs
    ys = start[1] + (end[1] - start[1]) * ys

    path = gf.Path(np.column_stack([xs, ys]))
    path.start_angle = path.end_angle = 0.0

    return path


@gf.cell
def bend_S_spline(
    size: tuple[float, float] = (100.0, 30.0),
    cross_section: CrossSectionSpec = "xs_rwg1000",
    npoints: int = 201,
    path_method=spline_clamped_path,
) -> gf.Component:
    """A spline bend merging a vertical offset."""

    t = np.linspace(0, 1, npoints)
    xs = gf.get_cross_section(cross_section)
    path = path_method(t, start=(0.0, 0.0), end=size)

    c = path.extrude(xs)

    return c


@gf.cell
def bend_S_spline_varying_width(
    size: tuple[float, float] = (58, 14.5),
    cross_section1: CrossSectionSpec = "xs_rwg200",
    cross_section2: CrossSectionSpec = "xs_rwg300",
    npoints: int = 201,
    path_method=spline_null_curvature,
) -> gf.Component:
    """
    A spline bend merging a vertical offset with zero
    curvature at both ends. Can accept random cross sections.
    Not tested as a standalone PDK element. Used as a building
    block for cells with knows behaviour.
    """
    # creating new cross sections to have liberty in width selection
    cross_section1_name = str(cross_section1)
    cross_section1_width = float(cross_section1_name[6:]) * 1e-3
    s0 = gf.Section(
        width=cross_section1_width,
        offset=0,
        layer="LN_RIDGE",
        name="_default",
        port_names=("o1", "o2"),
    )
    s1 = gf.Section(width=10.0, offset=0, layer="LN_SLAB", name="slab", simplify=0.03)
    cross_section1 = gf.CrossSection(sections=[s0, s1])

    cross_section2_name = str(cross_section2)
    cross_section2_width = float(cross_section2_name[6:]) * 1e-3
    s0 = gf.Section(
        width=cross_section2_width,
        offset=0,
        layer="LN_RIDGE",
        name="_default",
        port_names=("o1", "o2"),
    )
    s1 = gf.Section(width=10.0, offset=0, layer="LN_SLAB", name="slab", simplify=0.03)
    cross_section2 = gf.CrossSection(sections=[s0, s1])

    t = np.linspace(0, 1, npoints)
    path = path_method(t, start=(0.0, 0.0), end=size)

    xtrans = gf.path.transition(
        cross_section1=cross_section1,
        cross_section2=cross_section2,
        width_type="linear",
    )
    return gf.path.extrude_transition(path, xtrans)


if __name__ == "__main__":
    # Visualize differences between spline and bezier path

    t = np.linspace(0, 1, 600)

    apath = spline_null_curvature(t, end=(50.0, 15.0))
    bpath = spline_clamped_path(t, end=(50.0, 15.0))
    _, ka = apath.curvature()
    _, kb = bpath.curvature()

    plot_args_a = {
        "linewidth": 2.1,
        "label": "Zero curvature",
    }

    plot_args_b = {
        "linewidth": plot_args_a["linewidth"],
        "label": "Zero derivative",
    }

    ap = apath.points
    bp = bpath.points
    # ka = np.column_stack((ap[:-1, 0], curv_apath))
    # kb = np.column_stack((bp[:-1, 0], curv_bpath))

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), tight_layout=True)
    axs[0].plot(ap[:, 0], ap[:, 1], **plot_args_a)
    axs[0].plot(bp[:, 0], bp[:, 1], **plot_args_b)

    axs[0].set_xlabel("x (um)")
    axs[0].set_ylabel("y (um)")

    axs[1].plot(t[0:-1], ka, **plot_args_a)
    axs[1].plot(t[0:-1], kb, **plot_args_b)

    axs[1].set_xlabel("x (um)")
    axs[1].set_ylabel("Curvature (arb.)")

    [axs[k].legend(loc="best") for k in range(2)]
    plt.show()
