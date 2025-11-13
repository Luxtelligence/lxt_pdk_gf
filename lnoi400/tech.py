from functools import partial

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

from lnoi400.config import PATH

nm = 1e-3
ridge_thickness = 200 * nm
slab_thickness = 200 * nm
box_thickness = 4700 * nm
thickness_clad = 2000 * nm
tl_separation = 1000 * nm
tl_thickness = 900 * nm
substrate_thickness = 10_000 * nm


class LayerMapLNOI400(LayerMap):
    """Layer map for LXT lnoi400 technology."""

    LN_RIDGE: Layer = (2, 0)
    LN_SLAB: Layer = (3, 0)
    SLAB_NEGATIVE: Layer = (3, 1)
    LABELS: Layer = (4, 0)
    TL: Layer = (21, 0)
    HT: Layer = (21, 1)
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


LAYER = LayerMapLNOI400


def get_layer_stack() -> LayerStack:
    """Return lnoi400 LayerStack."""

    zmin_electrodes = slab_thickness + ridge_thickness + tl_separation

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
                layer=LogicalLayer(layer=LAYER.LN_SLAB),
                thickness=slab_thickness,
                zmin=0.0,
                sidewall_angle=13.0,
                material="ln",
                mesh_order=1,
            ),
            ridge=LayerLevel(
                layer=LogicalLayer(layer=LAYER.LN_RIDGE),
                thickness=ridge_thickness,
                zmin=slab_thickness,
                sidewall_angle=13.0,
                width_to_z=1,
                material="ln",
                mesh_order=2,
            ),
            clad=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WAFER),
                zmin=0.0,
                material="sio2",
                thickness=thickness_clad,
                mesh_order=99,
            ),
            tl=LayerLevel(
                layer=LogicalLayer(layer=LAYER.TL),
                thickness=tl_thickness,
                zmin=zmin_electrodes,
                material="tl_metal",
                mesh_order=6,
            ),
            ht=LayerLevel(
                layer=LogicalLayer(layer=LAYER.HT),
                thickness=tl_thickness,
                zmin=zmin_electrodes,
                material="tl_metal",
                mesh_order=7,
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
def xs_rwg1000(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 1.0,
    radius: float = 60.0,
) -> CrossSection:
    """Routing rib waveguide cross section"""
    sections = (
        gf.Section(
            width=10.0,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
        radius=radius,
    )


@xsection
def xs_rwg2500(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 2.5,
) -> CrossSection:
    sections = (
        gf.Section(
            width=11.5,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
    )


@xsection
def xs_rwg3000(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 3.0,
) -> CrossSection:
    """Multimode rib waveguide cross section"""
    sections = (
        gf.Section(
            width=12.0,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
    )


@xsection
def xs_swg250(
    layer: LayerSpec = "LN_SLAB",
    width: float = 0.25,
) -> CrossSection:
    return gf.cross_section.strip(
        width=width,
        layer=layer,
    )


@xsection
def xs_ht_wire(
    width: float = 0.9,
    offset: float = 0.0,
) -> CrossSection:
    """Generate cross-section of a heater wire."""

    return gf.cross_section.cross_section(
        width=width,
        offset=offset,
        layer=LAYER.HT,
        port_names=gf.cross_section.port_names_electrical,
        port_types=gf.cross_section.port_types_electrical,
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
        layer=LAYER.TL,
        simplify=50 * nm,
        name="ground_bottom",
    )

    g2 = gf.Section(
        width=ground_planes_width,
        offset=offset,
        layer=LAYER.TL,
        simplify=50 * nm,
        name="ground_top",
    )

    s = gf.Section(
        width=central_conductor_width,
        offset=0.0,
        layer=LAYER.TL,
        simplify=50 * nm,
        name="signal",
    )

    xs_cpw = gf.cross_section.cross_section(
        width=central_conductor_width,
        offset=0.0,
        layer=LAYER.TL,
        sections=(g1, s, g2),
        port_names=gf.cross_section.port_names_electrical,
        port_types=gf.cross_section.port_types_electrical,
    )

    return xs_cpw


############################
# Routing functions
############################


route_bundle_rwg1000 = partial(
    gf.routing.route_bundle,
    cross_section="xs_rwg1000",
    straight="straight_rwg1000",
    bend=gf.components.bend_euler,
    radius=60.0,
    separation=7.5,
    start_straight_length=5.0,
    end_straight_length=5.0,
    min_straight_taper=100.0,
    on_collision="show_error",
)

if __name__ == "__main__":
    pass
