import sys
from functools import partial

import gdsfactory as gf
from gdsfactory.technology import LayerMap, LayerViews
from gdsfactory.typings import Layer, LayerLevel, LayerStack
from gdsfactory.cross_section import get_cross_sections

import lnoi400
from lnoi400.config import PATH

nm = 1e-3
ridge_thickness = 200 * nm
slab_thickness = 200 * nm
box_thickness = 4700 * nm
tl_separation = 1000 * nm
tl_thickness = 900 * nm

class LayerMapLNOI400(LayerMap):

    LN_STRIP: Layer = (2, 0)
    LN_RIB: Layer = (3, 0)
    RIB_NEGATIVE: Layer = (3, 1)
    LABELS: Layer = (4, 0)
    TL: Layer = (21, 0)
    HT: Layer = (21, 1)

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
        layers = dict(

            slab = LayerLevel(
                layer = LAYER.LN_RIB,
                thickness = slab_thickness,
                zmin = 0.,
                material = "ln"),

            ridge = LayerLevel(
                layer = LAYER.LN_STRIP,
                thickness = ridge_thickness,
                zmin = slab_thickness,
                sidewall_angle = 15.,
                width_to_z = 1,
                material = "ln"),

            tl = LayerLevel(
                layer = LAYER.TL,
                thickness = tl_thickness,
                zmin = zmin_electrodes,
                material = "tl_metal"),

            ht = LayerLevel(
                layer = LAYER.TL,
                thickness = tl_thickness,
                zmin = zmin_electrodes,
                material = "tl_metal")
            ))

    return lstack

LAYER_STACK = get_layer_stack()
LAYER_VIEWS = LayerViews(filepath = PATH.lyp)

############################
# Cross-sections functions
############################

xf_rwg1000 = partial(
    gf.cross_section.strip,
    layer = LAYER.LN_STRIP,
    width = 1.0,
    sections = (gf.Section(width = 10., layer = "LN_RIB", name = "slab", simplify = 30 * nm),),
    radius = 75.,
    radius_min = 60.,
)

xf_rwg3000 = partial(xf_rwg1000,
                     width = 3.0,
                     sections = (gf.Section(width = 12., layer = "LN_RIB", name = "slab", simplify = 50 * nm),)
                     )

xf_swg250 = partial(gf.cross_section.strip, width = 0.25, layer = LAYER.LN_RIB, simplify = 30*nm)

############################
# Cross-sections
############################

xs_rwg1000 = xf_rwg1000()
xs_rwg3000 = xf_rwg3000()
xs_swg250 = xf_swg250()

cross_sections = get_cross_sections(sys.modules[__name__])

if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS.to_yaml(PATH.lyp_yaml)
    t = KLayoutTechnology(
        name="LNOI400",
        layer_map = dict(LAYER),
        layer_views = LAYER_VIEWS,
        layer_stack = LAYER_STACK,
    )
    t.write_tech(tech_dir = PATH.tech_dir)