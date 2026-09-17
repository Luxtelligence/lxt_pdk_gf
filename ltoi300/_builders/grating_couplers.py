from numpy import round, sum, ones
from _utils.grating_couplers import gc_lin_chirp
import gdsfactory as gf

#######################
## Grating couplers cells
#######################

# C-band grating coupler settings

Lx_cband = tuple([1.074] + list(1.093 * ones(59)))
fx_cband = tuple([0.674] + list(0.567 * ones(59)))


@gf.cell
def gc_lt300_cband(
    Lx=Lx_cband,
    fx=fx_cband,
    bias_gap=0.0,
    tap_len=12.0,
    wg_l=20.0,
    alpha_t=50.0,
    theta=14.5,
    ridge_thickness=0.18,
    sidewall_angle=20.0,
    cross_section="xs_rwg900",
):
    gc = gc_lin_chirp(
        Lx=Lx,
        fx=fx,
        tap_len=tap_len,
        bias_gap=bias_gap,
        wg_l=wg_l,
        alpha_t=alpha_t,
        theta=theta,
        ridge_thickness=ridge_thickness,
        sidewall_angle=sidewall_angle,
        cross_section=cross_section,
    )
    return gc


# O-band grating coupler settings

Lx_oband = tuple([0.889] + list(0.911 * ones(59)))
fx_oband = tuple([0.606] + list(0.464 * ones(59)))


@gf.cell
def gc_lt300_oband(
    Lx=Lx_oband,
    fx=fx_oband,
    bias_gap=0.0,
    tap_len=12.0,
    wg_l=20.0,
    alpha_t=50.0,
    theta=14.5,
    ridge_thickness=0.18,
    sidewall_angle=20.0,
    cross_section="xs_rwg700",
):
    gc = gc_lin_chirp(
        Lx=Lx,
        fx=fx,
        tap_len=tap_len,
        bias_gap=bias_gap,
        wg_l=wg_l,
        alpha_t=alpha_t,
        theta=theta,
        ridge_thickness=ridge_thickness,
        sidewall_angle=sidewall_angle,
        cross_section=cross_section,
    )
    return gc

