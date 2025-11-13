from functools import lru_cache

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from lnoi400 import cells, config, tech
from lnoi400.config import PATH
from lnoi400.models import get_models
from lnoi400.tech import LAYER, LAYER_STACK, LAYER_VIEWS

_models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "lnoi400"

_routing_strategies = dict(
    route_bundle_rwg1000=tech.route_bundle_rwg1000,
)


@lru_cache
def get_pdk() -> Pdk:
    """Return LXT lnoi400 PDK."""
    return Pdk(
        name="lnoi400",
        cells=_cells,
        cross_sections=_cross_sections,
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        routing_strategies=_routing_strategies,
    )


def activate_pdk() -> None:
    pdk = get_pdk()
    pdk.activate()


PDK = get_pdk()

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
]
__version__ = "1.2.0"
