import sys
from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection, get_cross_sections
from gdsfactory.technology import LayerMap, LayerViews
from gdsfactory.typings import Layer, LayerLevel, LayerStack

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
    LN_STRIP: Layer = (2, 0)
    LN_RIB: Layer = (3, 0)
    RIB_NEGATIVE: Layer = (3, 1)
    LABELS: Layer = (4, 0)
    TL: Layer = (21, 0)
    HT: Layer = (21, 1)
    WAFER: Layer = (99999, 0)

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


LAYER = LayerMapLNOI400()


def get_layer_stack() -> LayerStack:
    """Return lnoi400 LayerStack."""

    zmin_electrodes = slab_thickness + ridge_thickness + tl_separation

    lstack = LayerStack(
        layers=dict(
            substrate=LayerLevel(
                layer=LAYER.WAFER,
                thickness=substrate_thickness,
                zmin=-substrate_thickness - box_thickness,
                material="si",
                orientation="100",
            ),
            box=LayerLevel(
                layer=LAYER.WAFER,
                thickness=box_thickness,
                zmin=-box_thickness,
                material="sio2",
            ),
            slab=LayerLevel(
                layer=LAYER.LN_RIB, thickness=slab_thickness, zmin=0.0, material="ln"
            ),
            ridge=LayerLevel(
                layer=LAYER.LN_STRIP,
                thickness=ridge_thickness,
                zmin=slab_thickness,
                sidewall_angle=15.0,
                width_to_z=1,
                material="ln",
            ),
            clad=LayerLevel(
                layer=LAYER.WAFER,
                zmin=0.0,
                material="sio2",
                thickness=thickness_clad,
            ),
            tl=LayerLevel(
                layer=LAYER.TL,
                thickness=tl_thickness,
                zmin=zmin_electrodes,
                material="tl_metal",
            ),
            ht=LayerLevel(
                layer=LAYER.TL,
                thickness=tl_thickness,
                zmin=zmin_electrodes,
                material="tl_metal",
            ),
        )
    )

    return lstack


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = LayerViews(filepath=PATH.lyp)

############################
# Cross-sections functions
############################

xf_rwg1000 = partial(
    gf.cross_section.strip,
    layer=LAYER.LN_STRIP,
    width=1.0,
    sections=(gf.Section(width=10.0, layer="LN_RIB", name="slab", simplify=30 * nm),),
    radius=75.0,
    radius_min=60.0,
)

xf_rwg2500 = partial(
    xf_rwg1000,
    width=2.5,
    sections=(gf.Section(width=11.5, layer="LN_RIB", name="slab", simplify=30 * nm),),
)

xf_rwg3000 = partial(
    xf_rwg1000,
    width=3.0,
    sections=(gf.Section(width=12.0, layer="LN_RIB", name="slab", simplify=50 * nm),),
)

xf_swg250 = partial(
    gf.cross_section.strip, width=0.25, layer=LAYER.LN_RIB, simplify=30 * nm
)


def uni_cpw(
    central_conductor_width: float = 15.0,
    ground_planes_width: float = 250.0,
    gap: float = 5.0,
) -> CrossSection:
    """Generate cross-section of a uniform coplanar waveguide."""

    offset = 0.5 * (central_conductor_width + ground_planes_width) + gap

    g1 = gf.Section(
        width=ground_planes_width,
        offset=-offset,
        layer="TL",
        simplify=50 * nm,
        name="ground_bottom",
    )

    g2 = gf.Section(
        width=ground_planes_width,
        offset=offset,
        layer="TL",
        simplify=50 * nm,
        name="ground_top",
    )

    s = gf.Section(
        width=central_conductor_width,
        offset=0.0,
        layer="TL",
        simplify=50 * nm,
        name="signal",
    )

    xs_cpw = gf.cross_section.cross_section(
        width=central_conductor_width,
        offset=0.0,
        layer="TL",
        sections=(g1, s, g2),
        port_names=gf.cross_section.port_names_electrical,
        port_types=gf.cross_section.port_types_electrical,
    )

    return xs_cpw


############################
# Cross-sections
############################

xs_rwg1000 = xf_rwg1000()
xs_rwg2500 = xf_rwg2500()
xs_rwg3000 = xf_rwg3000()
xs_swg250 = xf_swg250()

xs_uni_cpw = uni_cpw()

cross_sections = get_cross_sections(sys.modules[__name__])

if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS.to_yaml(PATH.lyp_yaml)
    t = KLayoutTechnology(
        name="LNOI400",
        layer_map=dict(LAYER),
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
    )
    t.write_tech(tech_dir=PATH.tech_dir)

    path = gf.path.straight(length=1000.0)
    c = path.extrude(xs_uni_cpw)
    c.show()
