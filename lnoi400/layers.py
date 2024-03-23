from gdsfactory.technology import LayerMap
from gdsfactory.typings import Layer


class LayerMap(LayerMap):

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

LAYER = LayerMap()


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    from config import PATH

    LAYER_VIEWS = gf.technology.LayerViews(filepath = PATH.lyp)
    LAYER_VIEWS.to_yaml(PATH.lyp_yaml)
    t = KLayoutTechnology(
        name="LNOI400",
        layer_map = dict(LAYER),
        layer_views = LAYER_VIEWS,
        # layer_stack=LAYER_STACK,
        # connectivity=connectivity,
    )
    t.write_tech(tech_dir = PATH.tech_dir)