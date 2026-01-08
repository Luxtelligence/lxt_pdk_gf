import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Layer

from _utils.bends import bend_euler_tapered
from ltoi300.tech import LAYER, xs_rwg700


@gf.cell
def ring_resonator(
    gap: float = 0.6,
    ring_radius: float = 200.0,
    ring_width: float = 1.5,
    bus_width: float = 0.7,
    slab_width: float = 10.0,
    bus_length: float | None = None,
    cross_section: CrossSectionSpec = xs_rwg700,
    layer_ridge: Layer = LAYER.LT_RIDGE,
    layer_slab: Layer = LAYER.LT_SLAB,
) -> gf.Component:
    """A ring resonator with an evanescent coupler."""

    xs_bus = cross_section()

    ring = gf.components.ring(
        layer=layer_ridge,
        radius=ring_radius,
        width=ring_width,
        angle_resolution=0.15,
    )

    slab_outline = gf.components.ring(
        layer=layer_slab,
        radius=ring_radius,
        width=slab_width,
        angle_resolution=1.0,
    )

    bus = gf.components.straight(
        length=(ring_radius * 2 if not bus_length else bus_length),
        cross_section=xs_bus,
    )

    c = gf.Component()
    ring_ref = c << ring
    c << slab_outline
    coupler_ref = c << bus
    coupler_ref.drotate(90)
    coupler_ref.dcenter = [
        ring_radius + gap + 0.5 * bus_width + 0.5 * ring_width,
        ring_ref.dcenter[1],
    ]
    c.add_ports(coupler_ref.ports)
    c.flatten()
    return c


@gf.cell
def racetrack_resonator(
    gap: float = 0.6,
    rt_straight_length: float = 50.0,
    rt_height: float = 80.0,
    rt_width: float = 1.5,
    bus_length: float = 60.0,
    slab_width: float = 10.0,
    cross_section: CrossSectionSpec = xs_rwg700,
    layer_ridge: Layer = LAYER.LT_RIDGE,
    layer_slab: Layer = LAYER.LT_SLAB,
) -> gf.Component:
    """A racetrack resonator with an evanescent coupler. The width of the racetrack is tapered close to the coupling section."""

    if not bus_length:
        bus_length = rt_height

    c = gf.Component()

    xs_racetrack = gf.cross_section.strip(
        layer=layer_ridge,
        width=rt_width,
    )

    xs_slab = gf.cross_section.strip(
        layer=layer_slab,
        width=slab_width,
    )

    xs_bus = gf.get_cross_section(
        cross_section,
    )

    bend_tapered = bend_euler_tapered(
        radius=rt_height / 2.0,
        w0=rt_width,
        wc=cross_section().width,
        cross_section=xs_racetrack,
    )

    bend = gf.components.bend_euler(
        radius=rt_height / 2.0,
        angle=180.0,
        cross_section=xs_racetrack,
        with_arc_floorplan=True,
    )

    bend_slab = gf.components.bend_euler(
        radius=rt_height / 2.0,
        angle=180.0,
        cross_section=xs_slab,
        with_arc_floorplan=True,
    )

    straight_section = gf.components.straight(
        length=rt_straight_length,
        cross_section=xs_racetrack,
    )

    straight_slab = gf.components.straight(
        length=rt_straight_length,
        cross_section=xs_slab,
    )

    # Racetrack placement specifications

    st = c << straight_section
    sb = c << straight_section
    bl = c << bend
    br = c << bend_tapered
    bl.dmirror_x()
    br.connect("o1", sb.ports["o2"])
    st.connect("o2", br.ports["o2"])
    bl.connect("o2", st.ports["o1"])

    sst = c << straight_slab
    sst.dmove(
        origin=sst.ports["o1"].dcenter,
        destination=st.ports["o1"].dcenter,
    )
    ssb = c << straight_slab
    sbl = c << bend_slab
    sbl.dmirror_x()
    sbl.connect("o2", sst.ports["o1"])
    ssb.connect("o1", sbl.ports["o1"])

    # Compute sleeve mask in the tapered coupling region

    sizing_width = 0.5 * (slab_width - rt_width)

    bend_tapered_slab = gf.Component()
    polys = bend_tapered.get_polygons()[layer_ridge]
    bend_region = gf.kdb.Region(polys)
    slab_region = bend_region.sized(sizing_width * 1e3)  # Convert µm to nm (DB units)
    bend_tapered_slab.add_polygon(slab_region, layer=layer_slab)

    sbr = c << bend_tapered_slab
    sbr.dymin = ssb.dymin
    sbr.dxmin = ssb.dxmax - sizing_width

    # Bus waveguide

    bus = c << gf.components.straight(
        length=bus_length,
        cross_section=xs_bus,
    )
    bus.drotate(90.0)
    bus.dcenter = [br.dxmax + 0.5 * cross_section().width + gap, br.dcenter[1]]

    c.add_ports(bus.ports)
    c.add_port(name="st", port=st.ports["o1"])
    c.add_port(name="sb", port=sb.ports["o2"])
    c.flatten()

    return c


if __name__ == "__main__":
    c = ring_resonator()
    c.show()
