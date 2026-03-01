from functools import lru_cache

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from _utils.models import get_models
from ltoi300 import cells, config, models, tech
from ltoi300.config import PATH
from ltoi300.tech import LAYER, LAYER_STACK, LAYER_VIEWS

_models = get_models(models)
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "ltoi300"

# _routing_strategies = dict(
#     route_bundle_rwg1000=tech.route_bundle_rwg1000,
# )


@lru_cache
def get_pdk() -> Pdk:
    """Return LXT ltoi300 PDK."""
    return Pdk(
        name="ltoi300",
        cells=_cells,
        cross_sections=_cross_sections,
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        # routing_strategies=_routing_strategies,
    )


def activate_pdk() -> Pdk:
    pdk = get_pdk()
    pdk.activate()
    return pdk


PDK = activate_pdk()

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
    "PDK",
]
__version__ = "2.0"
