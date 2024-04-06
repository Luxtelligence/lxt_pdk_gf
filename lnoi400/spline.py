import numpy as np
import matplotlib.pyplot as plt
import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Coordinate
from gdsfactory.geometry.functions import curvature, path_length, snap_angle


def spline_clamped_path(start: Coordinate = (0., 0.),
                        end: Coordinate = (120., 25.),
                        npoints: int = 150):
    """Return a spline path connecting a start point with an end point."""

    xs = np.linspace(0, 1, npoints)
    ys = (xs**2)*(3-2*xs)

    # Rescale to the start and end coordinates

    xs = start[0] + (end[0] - start[0])*xs
    ys = start[1] + (end[1] - start[1])*ys

    return np.column_stack([xs, ys])

@gf.cell()
def spline_clamped(
    size: tuple[float, float] = (120., 25.),
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs) ->  gf.Component:
    """"""

    xs = gf.get_cross_section(cross_section)
    path_points = spline_clamped_path(start = (0., 0.),
                                      end = size,
                                      **kwargs)
    path = gf.Path(path_points)

    path.start_angle = snap_angle(path.start_angle)
    path.end_angle = snap_angle(path.end_angle)

    c = gf.Component()
    bend = path.extrude(xs)
    bend_ref = c << bend
    c.add_ports(bend_ref.ports)
    c.absorb(bend_ref)
    length = gf.snap.snap_to_grid(path_length(path_points))

    c.info["length"] = float(length)
    c.info["start_angle"] = path.start_angle
    c.info["end_angle"] = path.end_angle
    return c

if __name__ == "__main__":    
    
    bend = spline_clamped(size = (100., 10.))
    bend.show()
    print(bend.info)

