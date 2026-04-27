"""Grating coupler builders for lnoi400."""

import gdsfactory as gf
import numpy as np

from _utils.grating_couplers import gc_focusing_arbitrary
from lnoi400.tech import xs_rwg1000


def build_gc_focusing_1550(
    sleeve_width: float = 4.5,
    cross_section: gf.typings.CrossSectionSpec = xs_rwg1000,
    waveguide_length: float = 10.0,
) -> gf.Component:
    """Returns a focusing grating coupler for 1550 nm (C-band).

    The device geometry (pitch and fill_factor) is pre-computed and stored compactly.
    """
    N = 60

    # Compactly define the changing start and the constant tail of the arrays
    pitch_start = [0.99342, 1.00324, 1.01336]
    pitch_end = 1.02379
    pitch = np.array(pitch_start + [pitch_end] * (N - len(pitch_start)))

    fill_factor_start = [0.73974, 0.66523, 0.58999]
    fill_factor_end = 0.51399
    fill_factor = np.array(
        fill_factor_start + [fill_factor_end] * (N - len(fill_factor_start))
    )

    ridge_thickness = 0.2
    sidewall_angle = 13.0

    return gc_focusing_arbitrary(
        pitch=tuple(pitch.tolist()),
        fill_factor=tuple(fill_factor.tolist()),
        alpha_t=50.0,
        tap_len=12.5,
        N=N,
        wg_l=waveguide_length,
        wl_0=1.55,
        theta=14.5,
        bias_gap=0.075,
        sleeve_width=sleeve_width,
        ridge_thickness=ridge_thickness,
        sidewall_angle=sidewall_angle,
        cross_section=cross_section,
    )
