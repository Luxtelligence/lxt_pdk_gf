import numpy as np
import matplotlib.pyplot as plt
import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Coordinate
from gdsfactory.geometry.functions import curvature, path_length, snap_angle


def spline_clamped_path(t: np.ndarray,
                        start: Coordinate = (0., 0.),
                        end: Coordinate = (120., 25.)):
    """Return a spline path connecting a start point with an end point."""

    xs = t
    ys = (t**2)*(3-2*t)

    # Rescale to the start and end coordinates

    xs = start[0] + (end[0] - start[0])*xs
    ys = start[1] + (end[1] - start[1])*ys

    return np.column_stack([xs, ys])

@gf.cell()
def spline_clamped(
    size: tuple[float, float],
    cross_section: CrossSectionSpec = "xs_sc",
    npoints: int = 201) ->  gf.Component:
    """A spline bend clamped at the start and end points."""

    t = np.linspace(0, 1, npoints)
    xs = gf.get_cross_section(cross_section)
    path_points = spline_clamped_path(t, start = (0., 0.), end = size)
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

    from gdsfactory.components.bezier import bezier_curve

    t = np.linspace(0, 1, 600)
    bpath = bezier_curve(t, control_points = ((0., 0.), (25., 0.), (25., 50.), (50., 50.)))

    spath = spline_clamped_path(t, end = (50., 50.))
    kb = curvature(bpath, t)
    ks = curvature(spath, t)
    
    plt.figure()
    plt.plot(bpath[:, 0], bpath[:, 1])
    plt.plot(spath[:, 0], spath[:, 1])
    
    # plt.plot(t[1:-1], kb)
    # plt.plot(t[1:-1], ks)
    plt.show()

