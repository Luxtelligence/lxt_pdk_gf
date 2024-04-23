import inspect
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
from gdsfactory.pdk import get_cross_section_name
from gdsfactory.typings import CrossSectionSpec
from numpy.typing import NDArray

from gplugins.sax.models import straight as __straight
from gplugins.sax.models import bend as __bend


import matplotlib.pyplot as plt

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Straights
################

def _straight(
    *,
    wl: Float = 1.55,
    length: Float = 10.0,
    cross_section: CrossSectionSpec = "xs_rwg1000") -> sax.SDict:

    cross_section_name = get_cross_section_name(cross_section)

    if cross_section_name == "xs_rwg1000":
        return __straight(
                wl = wl,
                length = length,
                loss = 2e-5,
                wl0 = 1.55,
                neff = 1.8,
                ng = 2.22,
        )

    if cross_section_name == "xs_swg250":
        return __straight(
            wl = wl,
            length = length,
            loss = 1e-4,
            wl0 = 1.55,
            neff = 1.44,
            ng = 1.7,
        )

    else:
        raise ValueError(f"A model for the specified waveguide cross section {cross_section} is not defined.")
    
straight_rwg1000 = partial(_straight, cross_section="xs_rwg1000")
straight_swg250 = partial(_straight, cross_section="xs_swg250")

    
################
# Bends
################

################
# Models Dict
################

def get_models() -> dict[str, Callable[..., sax.SDict]]:
    models = {}
    for name, func in list(globals().items()):
        if not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except ValueError:
            continue
        if str(sig.return_annotation) == "sax.SDict":
            models[name] = func
    return models


if __name__ == "__main__":
    
    import lnoi400.tech

    wl = jnp.linspace(1.5, 1.6, 1000)
    model = straight_swg250
    S = model(wl = wl, length = 2e3)

    plt.figure()
    plt.plot(wl, abs(S["o1", "o2"])**2)
    plt.show()