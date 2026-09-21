import gdsfactory as gf
import klayout.db as kdb

from lnoi400 import PDK


def test_mzm_gsg_ports_preserve_legacy_electrical_ports():
    """@tags lnoi400-gsg

    The MZM keeps legacy aggregate electrical ports alongside GSG ports.
    """
    PDK.activate()

    component = gf.get_component("mzm_unbalanced_high_speed")

    assert {"e1", "e2", "G1_top", "S1", "G1_bot", "G2_top", "S2", "G2_bot"} <= set(
        component.ports.get_all_named().keys()
    )
    for name in ("G1_top", "S1", "G1_bot", "G2_top", "S2", "G2_bot"):
        orientation = component.ports[name].orientation
        assert min(abs(orientation - 0), abs(orientation - 180)) <= 1e-3


def test_mzm_with_pads_is_registered_and_builds():
    """@tags lnoi400-gsg

    The wrapper is registered in the lnoi400 PDK and exposes its public ports.
    """
    PDK.activate()

    assert "mzm_with_pads" in PDK.cells
    component = gf.get_component("mzm_with_pads")

    assert {"o1", "o2", "e1", "e2"} <= set(component.ports.get_all_named().keys())
    assert component.ports["e1"].port_type == "electrical"
    assert component.ports["e2"].port_type == "electrical"


def test_mzm_with_pads_ports_touch_three_tl_conductors():
    """@tags lnoi400-gsg

    The public pad ports remain on the actual metal after the GSG taper.
    """
    PDK.activate()
    component = gf.get_component("mzm_with_pads")
    tl = kdb.Region(component.begin_shapes_rec(gf.get_layer("TL"))).merged()

    assert tl.size() == 3
    for name in ("e1", "e2"):
        port = component.ports[name]
        x = round(port.dcenter[0] / component.kcl.dbu)
        slice_region = kdb.Region(kdb.Box(x - 1, -10_000_000, x + 1, 10_000_000))
        assert (tl & slice_region).size() == 3
