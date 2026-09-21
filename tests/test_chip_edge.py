"""Verify physical chip-edge placement and constant-width singulation tips."""

import gdsfactory as gf
import klayout.db as kdb
import pytest

import lnoi400
from lnoi400 import cells


def test_mirrored_coupler_defaults_to_singulation_length():
    """Use ten micrometers of constant-width tip without polishing.

    @tags lnoi400-chip-edge
    """
    lnoi400.PDK.activate()
    coupler = cells.double_linear_inverse_taper_mirror()
    assert coupler.settings.input_ext == 10
    assert abs(coupler.ports["o2"].dx - coupler.ports["o1"].dx) == 370


@pytest.mark.parametrize("input_ext", [None, 10.0, 30.0])
@pytest.mark.parametrize("exclusion_zone_width", [50.0, 100.0])
def test_facet_overhang_and_straight_on_both_chip_edges(
    input_ext, exclusion_zone_width
):
    """Keep each facet five micrometers beyond 6/1 as tip length changes.

    @tags lnoi400-chip-edge
    """
    lnoi400.PDK.activate()
    coupler_kwargs = (
        {}
        if input_ext is None
        else {
            "edge_coupler": {
                "component": "double_linear_inverse_taper_mirror",
                "settings": {"input_ext": input_ext},
            }
        }
    )
    input_ext = 30.0 if input_ext is None else input_ext
    die = cells.die_phix_rf(
        nfibers=6,
        npads=2,
        npads_rf=2,
        text=None,
        with_left_fiber_coupler=True,
        **coupler_kwargs,
        exclusion_zone_width=exclusion_zone_width,
    )
    dbu = die.kcl.dbu
    physical_region = kdb.Region(
        die.begin_shapes_rec(gf.get_layer("CHIP_EXCLUSION_ZONE"))
    )
    assert not physical_region.is_empty()
    physical = physical_region.bbox()
    contour = kdb.Region(die.begin_shapes_rec(gf.get_layer("CHIP_CONTOUR"))).bbox()
    assert (physical.right - contour.right) * dbu == exclusion_zone_width
    assert contour.height() * dbu == 4950
    slab = kdb.Region(die.begin_shapes_rec(gf.get_layer("LN_SLAB")))
    assert slab.bbox().right * dbu == physical.right * dbu + 5
    assert slab.bbox().left * dbu == physical.left * dbu - 5

    optical_ports = [port for port in die.ports if port.port_type == "optical"]
    assert {port.dx > 0 for port in optical_ports} == {False, True}
    for port in optical_ports:
        side = 1 if port.dx > 0 else -1
        edge = (physical.right if side == 1 else physical.left) * dbu
        tip = edge + side * 5
        straight_start = tip - side * input_ext
        xmin, xmax = sorted((tip, straight_start))
        clip = kdb.DBox(xmin, port.dy - 3, xmax, port.dy + 3).to_itype(dbu)
        expected = kdb.DBox(xmin, port.dy - 0.125, xmax, port.dy + 0.125).to_itype(dbu)
        assert ((slab & kdb.Region(clip)) ^ kdb.Region(expected)).is_empty()
        assert side * (edge - straight_start) == input_ext - 5


@pytest.mark.parametrize(
    "kwargs, expected", [({}, 57), ({"npads": None}, 49), ({"npads": 12}, 12)]
)
def test_phix_dc_pads_per_side(kwargs, expected):
    """Default to the PHIX pad count while retaining explicit and automatic counts."""
    lnoi400.PDK.activate()
    die = cells.die_phix_rf(**kwargs)
    for orientation in (90, 270):
        pads = [
            p
            for p in die.ports
            if p.port_type == "electrical" and p.orientation == orientation
        ]
        assert len(pads) == expected
        assert len({p.dy for p in pads}) == 1
        positions = sorted(p.dx for p in pads)
        assert all(b - a == 150 for a, b in zip(positions, positions[1:]))
