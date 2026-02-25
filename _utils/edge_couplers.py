from collections.abc import Callable
from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory import Component, CrossSection
from gdsfactory.typings import Layer


def _lin_lin_exp(
    x: list | np.ndarray,
    xp_0: float = 0,
    yp_0: float = 0.35,
    xp_1: float = 0,
    yp_1: float = 0.35,
    xp_2: float = 0.25,
    yp_2: float = 0.5,
    yoffs_exp: float = 0.418,
    yp_max: float = 5.6,
    exp_rate: float = 2.5,
) -> np.ndarray:
    """
    Custom function for linear-linear-exponential profile.
    """

    if not all(0 <= i < 1 for i in (xp_0, xp_1, xp_2)):
        raise ValueError("All the provided x-coordinates must be between 0 and 1.")

    # Piecewise profile definition

    y = np.zeros_like(x)
    if xp_1 > 0:
        y = np.where(x < xp_1, (yp_0 + (yp_1 - yp_0) * (x - xp_0) / (xp_1 - xp_0)), 0)
    if xp_2 > 0:
        y = np.where(
            (x >= xp_1) & (x < xp_2),
            (yp_1 + (yp_2 - yp_1) * (x - xp_1) / (xp_2 - xp_1)),
            y,
        )

    # Exponential expansion part
    b = 1 / (1 - np.exp(-exp_rate))
    a = 1 - b

    y = np.where(
        x >= xp_2,
        yoffs_exp
        + (yp_2 - yoffs_exp) * (a + b * np.exp(exp_rate * x / xp_2 - exp_rate)),
        y,
    )

    # Limit the maximum profile width

    y = np.where(y >= yp_max, yp_max, y)

    return y


def _exp_growth(
    x: list | np.ndarray,
    yp_0: float = 0.25,
    yp_1: float = 0.7,
    exp_rate: float = 2.5,
):
    b = 1 / (1 - np.exp(-exp_rate))
    a = 1 - b
    y = yp_0 + (yp_1 - yp_0) * (a + b * np.exp(exp_rate * (x - 1)))
    return y


@gf.cell
def double_layer_ec_custom(
    lower_taper_xs: CrossSection,
    upper_taper_xs: CrossSection,
    slab_negative_layer: Layer,
    lower_profile: Callable = _lin_lin_exp,
    lower_profile_args: dict | None = None,
    upper_profile: Callable = _exp_growth,
    upper_profile_args: dict | None = None,
    total_taper_length: float = 160.0,
    upper_taper_length: float = 80.0,
    npoints_lower: float = 240,
    npoints_upper: float = 120,
    slab_removal_width: float = 20.0,
    input_ext: float = 0.0,
) -> Component:
    """
    Double layer edge coupler based on inverse tapers with arbitrary profiles.
    Start from a wire cross-section, end with a rib cross-section.
    The default values are optimised for the O-band lensed fiber with the
    MFD=2.15 um. Avoid renaming double_layer_ec_custom to enable backward compatibility.
    """

    double_taper = gf.Component()

    # Pass the kwargs to the profile definitions

    lower_profile = partial(lower_profile, **lower_profile_args)
    upper_profile = partial(upper_profile, **upper_profile_args)

    # Lower taper

    s = gf.Section(
        width=0,
        width_function=lower_profile,
        layer=lower_taper_xs.layer,
        port_names=("o1", "o2"),
    )
    xsl = gf.CrossSection(sections=(s,))

    pl = gf.path.straight(length=total_taper_length, npoints=npoints_lower)

    lower_taper = double_taper << gf.path.extrude(pl, cross_section=xsl)

    # Upper taper
    s = gf.Section(
        width=0,
        width_function=upper_profile,
        layer=upper_taper_xs.layer,
        port_names=("o1", "o2"),
    )
    xsu = gf.CrossSection(sections=(s,))

    pu = gf.path.straight(length=upper_taper_length, npoints=npoints_upper)

    upper_taper = double_taper << gf.path.extrude(pu, cross_section=xsu)
    upper_taper.dmovex(total_taper_length - upper_taper_length)

    s = gf.Section(
        width=lower_profile(0),
        layer=lower_taper_xs.layer,
        port_names=("o1", "o2"),
    )
    xs_ext = gf.CrossSection(sections=(s,))

    if input_ext:
        straight_ext = gf.components.straight(
            cross_section=xs_ext,
            length=input_ext,
        )
        sref = double_taper << straight_ext
        sref.dmovex(-input_ext)

    double_taper.add_port(
        port=sref.ports["o1"]
    ) if input_ext else double_taper.add_port(port=lower_taper.ports["o1"])
    double_taper.add_port(port=upper_taper.ports["o2"])

    # Place the tone inversion box for the slab etch

    if slab_removal_width:
        bn = gf.components.rectangle(
            size=(
                double_taper.ports["o2"].dcenter[0]
                - double_taper.ports["o1"].dcenter[0],
                slab_removal_width,
            ),
            centered=True,
            layer=slab_negative_layer,
        )
        bnref = double_taper << bn
        bnref.dmovex(
            origin=bnref.dxmin,
            destination=-input_ext,
        )

    double_taper.flatten()

    return double_taper
