import jax.numpy as jnp
import sax

from .waveguides import straight_rwg1000

sax.set_port_naming_strategy("optical")


def coupler_ring(wl=1.5) -> sax.SDict:
    return sax.models.coupler_ideal(wl=wl)


_ring_single, _ = sax.circuit(
    netlist={
        "instances": {"wg": "straight_rwg1000", "dc": "directional_coupler_balanced"},
        "connections": {
            "dc,o3": "wg,o1",
            "wg,o2": "dc,o2",
        },
        "ports": {
            "o1": "dc,o1",
            "o2": "dc,o4",
        },
    },
    models={
        "straight_rwg1000": straight_rwg1000,
        "directional_coupler_balanced": coupler_ring,
    },
)


def ring_single(wl=1.5, radius: float = 10.0, loss_dB_cm=1.0) -> sax.SType:
    """Custom model for ring_single.

    Args:
        wl: Wavelength in micrometers (default: 1.5)

    Returns:
        sax.SType: S-parameters dictionary
    """
    length = 2 * jnp.pi * radius
    return _ring_single(wl=wl, wg={"length": length, "loss_dB_cm": loss_dB_cm})
