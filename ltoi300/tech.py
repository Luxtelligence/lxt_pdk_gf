import gdsfactory as gf
from gdsfactory.cross_section import (
    CrossSection,
)
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LogicalLayer,
)
from gdsfactory.typings import Layer, LayerSpec

from ltoi300.config import PATH

nm = 1e-3
ridge_thickness = 180 * nm
slab_thickness = 120 * nm
box_thickness = 7000 * nm
thickness_clad = 2000 * nm
substrate_thickness = 10_000 * nm


class LayerMapLTOI300(LayerMap):
    """Layer map for ltoi300 technology."""

    LT_RIDGE: Layer = (2, 10)
    LT_SLAB: Layer = (3, 10)
    SLAB_NEGATIVE: Layer = (3, 11)
    LABELS: Layer = (4, 0)
    M1: Layer = (20, 0)
    V2: Layer = (40, 0)
    M2: Layer = (22, 0)
    HRM: Layer = (23, 0)
    WAFER: Layer = (990, 0)

    # AUX

    CHIP_CONTOUR: Layer = (6, 0)
    CHIP_EXCLUSION_ZONE: Layer = (6, 1)
    DOC: Layer = (201, 0)
    ERROR: Layer = (50, 1)

    # common gdsfactory layers

    LABEL_INSTANCE: Layer = (66, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (99, 10)
    PORTE: Layer = (99, 11)
    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)


LAYER = LayerMapLTOI300


def get_layer_stack() -> LayerStack:
    """Return ltoi300 LayerStack."""

    lstack = LayerStack(
        layers=dict(
            substrate=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WAFER),
                thickness=substrate_thickness,
                zmin=-substrate_thickness - box_thickness,
                material="si",
                orientation="100",
                mesh_order=101,
            ),
            box=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WAFER),
                thickness=box_thickness,
                zmin=-box_thickness,
                material="sio2",
                mesh_order=100,
            ),
            slab=LayerLevel(
                layer=LogicalLayer(layer=LAYER.LT_SLAB),
                thickness=slab_thickness,
                zmin=0.0,
                sidewall_angle=20.0,
                material="lt",
                mesh_order=1,
            ),
            ridge=LayerLevel(
                layer=LogicalLayer(layer=LAYER.LT_RIDGE),
                thickness=ridge_thickness,
                zmin=slab_thickness,
                sidewall_angle=20.0,
                width_to_z=1,
                material="lt",
                mesh_order=2,
            ),
            clad=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WAFER),
                zmin=0.0,
                material="sio2",
                thickness=thickness_clad,
                mesh_order=99,
            ),
        )
    )

    return lstack


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(filepath=PATH.lyp_yaml)

############################
# Cross-section functions
############################

xsection = gf.xsection


@xsection
def xs_rwg700(
    layer: LayerSpec = (2, 0),
    width: float = 0.7,
    radius: float = 50.0,
    cladding_offset: float = 6.0,
) -> CrossSection:
    """Default routing rib waveguide cross section for O-band"""
    return gf.cross_section.rib(
        width=width,
        layer=layer,
        radius=radius,
        cladding_offsets=(cladding_offset,),
        cladding_layers=((3, 0),),
    )


@xsection
def xs_rwg900(
    layer: LayerSpec = (2, 0),
    width: float = 0.9,
    radius: float = 50.0,
    cladding_offset: float = 6.0,
) -> CrossSection:
    """Default routing rib waveguide cross section for C-band"""
    return gf.cross_section.rib(
        width=width,
        layer=layer,
        radius=radius,
        cladding_offsets=(cladding_offset,),
        cladding_layers=((3, 0),),
    )


@xsection
def xs_swg350(
    layer: LayerSpec = (3, 0),
    width: float = 0.35,
) -> CrossSection:
    """Narrow strip waveguide cross section"""
    return gf.cross_section.strip(
        width=width,
        layer=layer,
    )


@xsection
def xs_swg450(
    layer: LayerSpec = (3, 0),
    width: float = 0.35,
) -> CrossSection:
    """Narrow strip waveguide cross section"""
    return gf.cross_section.strip(
        width=width,
        layer=layer,
    )


@xsection
def xs_uni_cpw(
    central_conductor_width: float = 15.0,
    ground_planes_width: float = 250.0,
    gap: float = 5.0,
) -> CrossSection:
    """Generate cross-section of a uniform coplanar waveguide."""

    offset = 0.5 * (central_conductor_width + ground_planes_width) + gap

    g1 = gf.Section(
        width=ground_planes_width,
        offset=-offset,
        layer=LAYER.M1,
        simplify=50 * nm,
        name="ground_bottom",
    )

    g2 = gf.Section(
        width=ground_planes_width,
        offset=offset,
        layer=LAYER.M1,
        simplify=50 * nm,
        name="ground_top",
    )

    s = gf.Section(
        width=central_conductor_width,
        offset=0.0,
        layer=LAYER.M1,
        simplify=50 * nm,
        name="signal",
    )

    xs_cpw = gf.cross_section.cross_section(
        width=central_conductor_width,
        offset=0.0,
        layer=LAYER.M1,
        sections=(g1, s, g2),
        port_names=gf.cross_section.port_names_electrical,
        port_types=gf.cross_section.port_types_electrical,
    )

    return xs_cpw


@xsection
def xs_ht_wire(
    width: float = 0.9,
    offset: float = 0.0,
    ht_layer: Layer = LAYER.HRM,
) -> CrossSection:
    """Generate cross-section of a heater wire."""

    return gf.cross_section.cross_section(
        width=width,
        offset=offset,
        layer=ht_layer,
        port_names=gf.cross_section.port_names_electrical,
        port_types=gf.cross_section.port_types_electrical,
    )


if __name__ == "__main__":
    c = gf.components.straight(length=100, cross_section=xs_rwg700())
    c.show()
