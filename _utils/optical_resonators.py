import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from _utils.bends import bend_euler_tapered
from _utils.cross_section import get_slab_extension


@gf.cell(tags=["optical_resonators"])
def ring(
    radius: float,
    cross_section: CrossSectionSpec,
    angle_resolution: float = 0.15,
) -> gf.Component:
    """A ring with a given radius and cross-section."""

    c = gf.Component()

    for section in cross_section.sections:
        if "ridge" in section.name.lower():
            ridge_width = section.width
            ridge_layer = section.layer
        elif "slab" in section.name.lower():
            slab_width = section.width
            slab_layer = section.layer

    c << gf.components.ring(
        radius=radius,
        width=ridge_width,
        layer=ridge_layer,
        angle_resolution=angle_resolution,
    )

    c << gf.components.ring(
        radius=radius,
        width=slab_width,
        layer=slab_layer,
        angle_resolution=angle_resolution,
    )

    c.flatten()
    return c


@gf.cell(tags=["optical_resonators"])
def ring_resonator(
    gap: float,
    ring_radius: float,
    bus_xs: CrossSectionSpec,
    ring_xs: CrossSectionSpec,
    bus_length: float | None = None,
) -> gf.Component:
    """A ring resonator with an evanescent coupler."""

    ring_resonator = ring(
        cross_section=ring_xs,
        radius=ring_radius,
        angle_resolution=0.15,
    )

    bus = gf.components.straight(
        length=(ring_radius * 2 if not bus_length else bus_length),
        cross_section=bus_xs,
    )

    c = gf.Component()
    ring_ref = c << ring_resonator
    coupler_ref = c << bus
    coupler_ref.drotate(90)
    coupler_ref.dcenter = [
        ring_radius + gap + 0.5 * (bus_xs.width + ring_xs.width),
        ring_ref.dcenter[1],
    ]
    c.add_ports(coupler_ref.ports)
    c.flatten()
    return c


@gf.cell(tags=["optical_resonators"])
def racetrack_resonator(
    gap: float,
    rt_straight_length: float,
    rt_height: float,
    rt_width: float,
    bus_length: float,
    xs_bus: CrossSectionSpec,
    xs_racetrack: CrossSectionSpec = "xs_rwg_tapered",
) -> gf.Component:
    """A racetrack resonator with an evanescent coupler. The width of the racetrack is tapered close to the coupling section to match the bus waveguide width."""

    if not bus_length:
        bus_length = rt_height

    xs_racetrack_fixed = xs_racetrack(width=rt_width)
    print(xs_racetrack_fixed.width)
    print(xs_bus.width)

    c = gf.Component()

    bend_tapered = bend_euler_tapered(
        radius=rt_height / 2.0,
        w0=rt_width,
        wc=xs_bus.width,
        cross_section=xs_racetrack,
    )

    bend = gf.components.bend_euler(
        radius=rt_height / 2.0,
        angle=180.0,
        cross_section=xs_racetrack_fixed,
        with_arc_floorplan=True,
    )

    straight_section = gf.components.straight(
        length=rt_straight_length,
        cross_section=xs_racetrack_fixed,
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

    # Bus waveguide
    # br.dxmax includes the slab layer; subtract the slab extension so that
    # `gap` is measured ridge-edge to ridge-edge.
    rt_slab_ext = get_slab_extension(xs_racetrack_fixed)

    bus = c << gf.components.straight(
        length=bus_length,
        cross_section=xs_bus,
    )
    bus.drotate(90.0)
    bus.dcenter = [
        br.dxmax - rt_slab_ext + 0.5 * xs_bus.width + gap,
        br.dcenter[1],
    ]

    c.add_ports(bus.ports)
    c.add_port(name="st", port=st.ports["o1"])
    c.add_port(name="sb", port=sb.ports["o2"])
    c.flatten()

    return c


if __name__ == "__main__":
    from ltoi300.tech import xs_rwg, xs_rwg900

    xs_bus = xs_rwg900()
    xs_racetrack = xs_rwg
    c = racetrack_resonator(
        gap=0.6,
        rt_straight_length=200.0,
        rt_height=100.0,
        rt_width=2.0,
        bus_length=100.0,
        xs_bus=xs_bus,
        xs_racetrack=xs_racetrack,
    )
    c.show()
