import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec


@gf.cell
def ring_single(
    gap: float = 0.6,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    cross_section: CrossSectionSpec = "xs_rwg1000",
    length_extension: float = 10.0,
) -> gf.Component:
    """Returns a single ring.

    ring coupler (cb: bottom) connects to two vertical straights (sl: left, sr: right),
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        cross_section: cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.


    .. code::

                    xxxxxxxxxxxxx
                xxxxx           xxxx
              xxx                   xxx
            xxx                       xxx
           xx                           xxx
           x                             xxx
          xx                              xx▲
          xx                              xx│length_y
          xx                              xx▼
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────o2
    """
    return gf.c.ring_single(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend="bend_euler",
        straight="straight",
        coupler_ring="coupler_ring",
        cross_section=cross_section,
        length_extension=length_extension,
    )
