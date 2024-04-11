import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
from gdsfactory.geometry.functions import curvature, path_length, snap_angle
from gdsfactory.typings import Coordinate, CrossSectionSpec
from scipy.interpolate import make_interp_spline


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
    size: tuple[float, float],
    cross_section: CrossSectionSpec = "xs_sc",
    npoints: int = 201,
    path_method=spline_clamped_path,
) -> gf.Component:
    """A spline bend merging a vertical offset."""

    t = np.linspace(0, 1, npoints)
    xs = gf.get_cross_section(cross_section)
    path_points = path_method(t, start=(0.0, 0.0), end=size)
    path = gf.Path(path_points)

    path.start_angle = snap_angle(path.start_angle)
    path.end_angle = snap_angle(path.end_angle)

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

    apath = spline_null_curvature(t, end=(50.0, 50.0))
    bpath = spline_clamped_path(t, end=(50.0, 50.0))
    ka = curvature(apath, t)
    kb = curvature(bpath, t)

    plt.figure()
    # plt.plot(apath[:, 0], apath[:, 1])
    # plt.plot(bpath[:, 0], bpath[:, 1])

    plt.plot(t[1:-1], ka)
    plt.plot(t[1:-1], kb)
    plt.show()

    # bend = bend_S_spline(size = (100., 30.),
    #                      path_method = spline_clamped_path)
    # bend.show()
    # print(bend.info)
