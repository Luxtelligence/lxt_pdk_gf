import sys
from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection, get_cross_sections
from gdsfactory.technology import LayerLevel, LayerMap, LayerStack, LogicalLayer

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
    LN_RIDGE = (2, 0)
    LN_SLAB = (3, 0)
    SLAB_NEGATIVE = (3, 1)
    LABELS = (4, 0)
    TL = (21, 0)
    HT = (21, 1)
    WAFER = (99999, 0)

    # AUX

    CHIP_CONTOUR = (6, 0)
    CHIP_EXCLUSION_ZONE = (6, 1)
    DOC = (201, 0)
    ERROR = (50, 1)

    # common gdsfactory layers

    LABEL_INSTANCE = (66, 0)
    DEVREC = (68, 0)
    PORT = (99, 10)
    PORTE = (99, 11)
    TE = (203, 0)
    TM = (204, 0)
    TEXT = (66, 0)


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
LAYER_VIEWS = gf.technology.LayerViews(filepath=PATH.lyp)

############################
# Cross-sections
############################

xs_rwg1000 = partial(
    gf.cross_section.strip,
    layer=LAYER.LN_RIDGE,
    width=1.0,
    sections=(
        gf.Section(
            width=10.0,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    ),
    radius=75.0,
)

xs_rwg2500 = partial(
    xs_rwg1000,
    width=2.5,
    sections=(
        gf.Section(
            width=11.5,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    ),
)

xs_rwg3000 = partial(
    xs_rwg1000,
    width=3.0,
    sections=(
        gf.Section(
            width=12.0,
            layer="LN_SLAB",
            name="slab",
            simplify=50 * nm,
        ),
    ),
)

xs_swg250 = partial(
    gf.cross_section.strip,
    width=0.25,
    layer="LN_SLAB",
)


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


cross_sections = get_cross_sections(sys.modules[__name__])
