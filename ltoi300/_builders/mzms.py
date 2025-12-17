import gdsfactory as gf

from ltoi300._impl.mzm import mzm_unbalanced_LT
from ltoi300.tech import LAYER


@gf.cell
def build_terminated_mzm_cband(
    terminated: bool = True,
    **kwargs,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi2x2_cband")
    kwargs.setdefault("length_imbalance", 100)
    kwargs.setdefault("heater_on_both_branches", False)
    kwargs.setdefault("with_heater", False)
    kwargs.setdefault("heater_offset", 0.0)
    kwargs.setdefault("rf_central_conductor_width", 10.0)
    kwargs.setdefault("rf_gap", 5.5)
    kwargs.setdefault("tl", 53.0)
    kwargs.setdefault("tw", 53.0)
    kwargs.setdefault("th", 1.5)
    kwargs.setdefault("tt", 1.5)
    kwargs.setdefault("communication_band", "C-band")
    kwargs.setdefault("ht_layer", LAYER.HRM)
    kwargs.setdefault("termination_layer", LAYER.HRM)
    kwargs.setdefault("heater_width", 2.5)
    kwargs.setdefault("resistor_length", 190.0 / 2)
    c << mzm_unbalanced_LT(
        **kwargs,
        terminated=terminated,
        bias_tuning_section_length=0.0,
        add_M2andOpenings=True,
    )
    return c
