from typing import Any

from gdsfactory import typings
from gdsfactory.cross_section import CrossSection, Section, xsection

nm = 1e-3
Sections = tuple[Section, ...]


def get_slab_extension(xs: CrossSection) -> float:
    """Return how far the slab section extends beyond the ridge on each side.

    Because the slab width is defined as ``ridge_width + 2 * slab_offset``,
    the extension equals ``slab_offset`` at every point along the path —
    even when a width_function is used — so it can be read directly from the
    nominal section widths.  Returns 0.0 for strip (ridge-only) cross-sections.
    """
    ridge_section = next(
        (s for s in xs.sections if "ridge" in s.name.lower()), xs.sections[0]
    )
    slab_section = next((s for s in xs.sections if "slab" in s.name.lower()), None)
    if slab_section is None:
        return 0.0
    return (slab_section.width - ridge_section.width) / 2


def slab_etch_cross_section(
    width: float = 0.5,
    offset: float = 0,
    layer: typings.LayerSpec = "SLAB",
    width_function: typings.WidthFunction | None = None,
    sections: Sections | None = None,
    port_names: typings.IOPorts = ("o1", "o2"),
    port_types: typings.IOPorts = ("optical", "optical"),
    bbox_layers: typings.LayerSpecs | None = None,
    bbox_offsets: typings.Floats | None = None,
    radius: float | None = 10.0,
    radius_min: float | None = None,
) -> CrossSection:
    """Return CrossSection.

    Args:
        width: main Section width (um).
        offset: main Section center offset (um).
        layer: main section layer.
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        bbox_layers: list of layers bounding boxes to extrude.
        bbox_offsets: list of offset from bounding box edge.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        cladding_simplify: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting. \
                polygon by more than the value listed here will be removed.
        cladding_centers: center offset for each cladding layer. Defaults to 0.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        main_section_name: name of the main section. Defaults to _default

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.cross_section(width=0.5, offset=0, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()

    """
    section_list: list[Section] = list(sections or [])

    s = [
        Section(
            width=width,
            width_function=width_function,
            offset=offset,
            layer=layer,
            port_names=port_names,
            port_types=port_types,
            name="SLAB",
        )
    ] + section_list

    return CrossSection(
        sections=tuple(s),
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
    )


def partial_etch_cross_section(
    width: float = 0.5,
    width_function: typings.WidthFunction | None = None,
    offset: float = 0,
    layer: typings.LayerSpec = "RIDGE",
    sections: Sections | None = None,
    port_names: typings.IOPorts = ("o1", "o2"),
    port_types: typings.IOPorts = ("optical", "optical"),
    bbox_layers: typings.LayerSpecs | None = None,
    bbox_offsets: typings.Floats | None = None,
    slab_layer: typings.LayerSpec = "SLAB",
    slab_offset: float = 6.0,
    slab_simplify: float = 50 * nm,
    radius: float | None = 10.0,
    radius_min: float | None = None,
) -> CrossSection:
    """Return CrossSection.

    Args:
        width: main Section width (um).
        offset: main Section center offset (um).
        layer: main section layer.
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        bbox_layers: list of layers bounding boxes to extrude.
        bbox_offsets: list of offset from bounding box edge.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        cladding_simplify: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting. \
                polygon by more than the value listed here will be removed.
        cladding_centers: center offset for each cladding layer. Defaults to 0.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        main_section_name: name of the main section. Defaults to _default

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.cross_section(width=0.5, offset=0, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()

    .. code::


           ┌────────────────────────────────────────────────────────────┐
           │                                                            │
           │                                                            │
           │                   boox_layer                               │
           │                                                            │
           │         ┌──────────────────────────────────────┐           │
           │         │                            ▲         │bbox_offset│
           │         │                            │         ├──────────►│
           │         │           slab_offset      │         │           │
           │         │                            │         │           │
           │         ├─────────────────────────▲──┴─────────┤           │
           │         │                         │            │           │
        ─ ─┤         │           core   width  │            │           ├─ ─ center
           │         │                         │            │           │
           │         ├─────────────────────────▼────────────┤           │
           │         │                                      │           │
           │         │                                      │           │
           │         │                                      │           │
           │         │                                      │           │
           │         └──────────────────────────────────────┘           │
           │                                                            │
           │                                                            │
           │                                                            │
           └────────────────────────────────────────────────────────────┘
    """
    section_list: list[Section] = list(sections or [])

    s = [
        Section(
            width=width,
            width_function=width_function,
            offset=offset,
            layer=layer,
            port_names=port_names,
            port_types=port_types,
            name="RIDGE",
        )
    ] + section_list

    #### Add slab section

    if width_function is not None:

        def width_function_slab(t):
            return width_function(t) + 2 * slab_offset

    else:
        width_function_slab = None
    s += [
        Section(
            width=width + 2 * slab_offset,
            width_function=width_function_slab,
            layer=slab_layer,
            simplify=slab_simplify,
            offset=0,
            name="SLAB",
        )
    ]
    return CrossSection(
        sections=tuple(s),
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
    )


@xsection
def ridge_wg(
    width: float = 0.5,
    layer: typings.LayerSpec = "RIDGE",
    radius: float = 50.0,
    radius_min: float | None = None,
    slab_layer: typings.LayerSpec = "SLAB",
    slab_offset: float = 3.0,
    slab_simplify: float = 50 * nm,
    **kwargs: Any,
) -> CrossSection:
    """Return Rib cross_section."""
    return partial_etch_cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        slab_layer=slab_layer,
        slab_offset=slab_offset,
        slab_simplify=slab_simplify,
        **kwargs,
    )
