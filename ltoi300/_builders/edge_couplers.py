from _utils.edge_couplers import double_layer_ec_custom
import gdsfactory as gf
from ltoi300.tech import xs_rwg700, xs_swg350, xs_rwg900, LAYER


def build_oband_ltoi300_edge_coupler(
    input_ext: float = 10.0,
    total_taper_length: float = 160.0,
    upper_taper_length: float = 80.0,
) -> gf.Component:
    return double_layer_ec_custom(
        lower_taper_xs=xs_swg350(),
        upper_taper_xs=xs_rwg700(),
        slab_negative_layer=LAYER.SLAB_NEGATIVE,
        total_taper_length=total_taper_length,
        upper_taper_length=upper_taper_length,
        input_ext=input_ext,
    )

def build_cband_ltoi300_edge_coupler(
    input_ext: float = 10.0,
    total_taper_length: float = 160.0,
    upper_taper_length: float = 80.0,
) -> gf.Component:
    lower_profile_args = {
        "xp_0": 0.0,
        "yp_0": 0.5,
        "xp_1": 0.0,
        "yp_1": 0.5,
        "xp_2": 0.5,
        "yp_2": 1.5,
    }
    upper_profile_args = {
        "yp_0": 0.25,
        "yp_1": 0.9,
    }
    ec = double_layer_ec_custom(
        lower_taper_xs=xs_swg350(),
        upper_taper_xs=xs_rwg900(),
        slab_negative_layer=LAYER.SLAB_NEGATIVE,
        total_taper_length=total_taper_length,
        upper_taper_length=upper_taper_length,
        lower_profile_args=lower_profile_args,
        upper_profile_args=upper_profile_args,
        input_ext=input_ext,
    )
    return ec