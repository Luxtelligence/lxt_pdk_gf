import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
from gdsfactory.typings import Coordinate, CrossSectionSpec
from numpy import float64, ndarray
from scipy.interpolate import make_interp_spline


def centered_diff(a: ndarray) -> ndarray:
    d = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / 2
    return d[1:-1]


def centered_diff2(a: ndarray) -> ndarray:
    d = (np.roll(a, -1, axis=0) - a) - (a - np.roll(a, 1, axis=0))
    return d[1:-1]


def curvature(points: ndarray, t: ndarray) -> ndarray:
    """Args are the points and the tangents at each point.

        points : numpy.array shape (n, 2)
        t: numpy.array of size n

    Return:
        The curvature at each point.

    Computes the curvature at every point excluding the first and last point.

    For a planar curve parametrized as P(t) = (x(t), y(t)), the curvature is given
    by (x' y'' - x'' y' ) / (x' **2 + y' **2)**(3/2)

    """
    # Use centered difference for derivative
    dt = centered_diff(t)
    dp = centered_diff(points)
    dp2 = centered_diff2(points)

    dx = dp[:, 0] / dt
    dy = dp[:, 1] / dt

    dx2 = dp2[:, 0] / dt**2
    dy2 = dp2[:, 1] / dt**2

    return (dx * dy2 - dx2 * dy) / (dx**2 + dy**2) ** (3 / 2)


def path_length(points: ndarray) -> float64:
    """Returns: The path length.

    Args:
        points: With shape (N, 2) representing N points with coordinates x, y.
    """
    dpts = points[1:, :] - points[:-1, :]
    _d = dpts**2
    return np.sum(np.sqrt(_d[:, 0] + _d[:, 1]))


def spline_clamped_path(
    t: np.ndarray, start: Coordinate = (0.0, 0.0), end: Coordinate = (120.0, 25.0)
):
    """Returns a spline path with a null first derivative at the extrema."""

    xs = t
    ys = (t**2) * (3 - 2 * t)

    # Rescale to the start and end coordinates

    xs = start[0] + (end[0] - start[0]) * xs
    ys = start[1] + (end[1] - start[1]) * ys

    return np.column_stack([xs, ys])


def spline_null_curvature(
    t: np.ndarray, start: Coordinate = (0.0, 0.0), end: Coordinate = (120.0, 25.0)
):
    """Returns a spline path with zero first and second derivatives at the extrema."""

    spline = make_interp_spline(
        x=(start[0], end[0]),
        y=(start[1], end[1]),
        k=5,
        bc_type=([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)]),
    )

    xs = np.linspace(start[0], end[0], len(t))

    return np.column_stack([xs, spline(xs)])


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
    path_points = path_method(t, start=(0.0, 0.0), end=size)
    path = gf.Path(path_points)

    # path.start_angle = snap_angle(path.start_angle)
    # path.end_angle = snap_angle(path.end_angle)

    c = gf.Component()
    bend = path.extrude(xs)
    bend_ref = c << bend
    c.add_ports(bend_ref.ports)
    c.absorb(bend_ref)
    curv = curvature(path_points, t)
    length = gf.snap.snap_to_grid(path_length(path_points))
    if max(np.abs(curv)) == 0:
        min_bend_radius = np.inf
    else:
        min_bend_radius = gf.snap.snap_to_grid(1 / max(np.abs(curv)))

    c.info["length"] = float(length)
    c.info["min_bend_radius"] = min_bend_radius
    c.info["start_angle"] = path.start_angle
    c.info["end_angle"] = path.end_angle
    return c


if __name__ == "__main__":
    # Visualize differences between spline and bezier path

    t = np.linspace(0, 1, 600)

    apath = spline_null_curvature(t, end=(50.0, 15.0))
    bpath = spline_clamped_path(t, end=(50.0, 15.0))
    ka = curvature(apath, t)
    kb = curvature(bpath, t)

    plot_args_a = {
        "linewidth": 2.1,
        "label": "Zero curvature",
    }

    plot_args_b = {
        "linewidth": plot_args_a["linewidth"],
        "label": "Zero derivative",
    }

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), tight_layout=True)
    axs[0].plot(apath[:, 0], apath[:, 1], **plot_args_a)
    axs[0].plot(bpath[:, 0], bpath[:, 1], **plot_args_b)

    axs[0].set_xlabel("x (um)")
    axs[0].set_ylabel("y (um)")

    axs[1].plot(t[1:-1], ka, **plot_args_a)
    axs[1].plot(t[1:-1], kb, **plot_args_b)

    axs[1].set_xlabel("x (um)")
    axs[1].set_ylabel("Curvature (arb.)")

    [axs[k].legend(loc="best") for k in range(2)]
    plt.show()
