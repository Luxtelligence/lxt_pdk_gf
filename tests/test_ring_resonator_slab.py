import gdsfactory as gf
import pytest

import ltoi300
from _utils.optical_resonators import ring
from ltoi300.tech import LAYER, xs_rwg700, xs_rwg900


@pytest.mark.parametrize(
    "cell_name,cross_section,ring_width,gap",
    [
        ("ring_resonator_single_mode_point_coupler_oband", xs_rwg700, 0.7, 1.05),
        ("ring_resonator_multimode_point_coupler_oband", xs_rwg700, 1.5, 0.75),
        ("ring_resonator_single_mode_point_coupler_cband", xs_rwg900, 0.9, 1.5),
        ("ring_resonator_multimode_point_coupler_cband", xs_rwg900, 1.5, 1.2),
    ],
)
def test_ring_resonator_preserves_shared_slab(
    cell_name, cross_section, ring_width, gap
):
    """Building a resonator preserves reusable slab geometry and bus ports.

    @tags ltoi300-ring-slab
    """
    gf.clear_cache()
    ltoi300.PDK.activate()
    try:
        bus_xs = cross_section()
        shared_ring = ring(radius=200.0, cross_section=cross_section(width=ring_width))
        shared_bus = gf.components.straight(length=400.0, cross_section=bus_xs)
        ring_slab = shared_ring.get_region(LAYER.LT_SLAB).merged()
        bus_slab = shared_bus.get_region(LAYER.LT_SLAB).merged()
        assert not ring_slab.is_empty()
        assert not bus_slab.is_empty()

        factory = ltoi300.PDK.cells[cell_name]
        component = factory()
        assert (shared_ring.get_region(LAYER.LT_SLAB) ^ ring_slab).is_empty()
        assert (shared_bus.get_region(LAYER.LT_SLAB) ^ bus_slab).is_empty()

        # A different gap reuses the same cached ring and bus. Both pieces
        # must still contribute slab material, regardless of construction order.
        second = factory(gap=gap + 0.2)
        assert not second.get_region(LAYER.LT_SLAB).is_empty()
        assert second.get_region(LAYER.LT_SLAB).bbox().left < -second.kcl.to_dbu(200.0)
        assert second.get_region(LAYER.LT_SLAB).bbox().right > second.kcl.to_dbu(200.0)
        assert {port.name for port in component.ports} == {"o1", "o2"}
        expected_x = 200.0 + gap + (bus_xs.width + ring_width) / 2
        for name, y, orientation in [("o1", -200.0, 270.0), ("o2", 200.0, 90.0)]:
            port = component.ports[name]
            assert port.center == pytest.approx((expected_x, y), abs=0.001)
            assert port.width == pytest.approx(bus_xs.width)
            assert port.orientation == orientation
            assert port.layer == shared_bus.ports[name].layer
    finally:
        gf.clear_cache()
