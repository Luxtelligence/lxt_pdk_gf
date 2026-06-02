import gdsfactory as gf


@gf.cell(tags=["chip_floorplan"])
def chip_frame(
    size: tuple[float, float] = (10_000, 5000),
    exclusion_zone_width: float = 50,
    center: tuple[float, float] = None,
    chip_contour_layer: tuple[int, int] = (6, 0),
    chip_exclusion_zone_layer: tuple[int, int] = (6, 1),
) -> gf.Component:
    """Provide the chip extent and the exclusion zone around the chip frame.
    In the exclusion zone, only the edge couplers routing to the chip facet should be placed.
    Allowed chip dimensions (in either direction): 5000 um, 10000 um, 20000 um."""

    # Check that the chip dimensions have the admissible values.

    snapped_size = []

    if size[0] <= 5050 and size[1] <= 5050:
        raise (ValueError(f"The chip frame size {size} is not supported."))

    if size[0] > 20200 or size[1] > 20200:
        raise (ValueError(f"The chip frame size {size} is not supported."))

    else:
        for s in size:
            if abs(s - 5000.0) <= 50.0:
                snapped_size.append(4950.0)
            elif abs(s - 10000.0) <= 100.0:
                snapped_size.append(10000)
            elif abs(s - 20000.0) <= 200:
                snapped_size.append(20100)
            else:
                raise (ValueError(f"The chip frame size {size} is not supported."))

    # Chip frame elements

    inner_box = gf.components.rectangle(
        size=tuple(snapped_size),
        layer=chip_contour_layer,
        centered=True,
    )

    outer_box = gf.components.rectangle(
        size=tuple(s + 2 * exclusion_zone_width for s in snapped_size),
        layer=chip_exclusion_zone_layer,
        centered=True,
    )

    c = gf.Component()
    ib = c << inner_box
    ob = c << outer_box

    if center:
        ib.dmove(origin=(0.0, 0.0), destination=center)
        ob.dmove(origin=(0.0, 0.0), destination=center)

    c.flatten()

    return c
