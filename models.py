import jax
import jax.numpy as jnp
import sax
from gdsfactory.pdk import get_cross_section_name
from gdsfactory.typings import CrossSectionSpec
from numpy.typing import NDArray
import lnoi400.tech

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
                ng = 2.22)    
    else:
        raise ValueError(f"A model for the specified waveguide cross section {cross_section} is not defined.")
    
################
# Bends
################


if __name__ == "__main__":
    wl = jnp.linspace(1.5, 1.6, 1000)
    S = _straight(wl = wl, length = 2e5)

    plt.figure()
    plt.plot(wl, abs(S["o1", "o2"])**2)
    plt.show()