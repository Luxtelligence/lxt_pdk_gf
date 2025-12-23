import gdsfactory as gf

from _utils.mzm import mzm_unbalanced_LT
from ltoi300.tech import LAYER

############################################
########### O-band builders ################
############################################


def template_build_mzm_oband(
    **kwargs,
):
    """Returns a template for MZM with effective index matching
    for O-band operation."""
    c = gf.Component()
    # Create the MZM component
    # kwargs.setdefault("splitter", "mmi2x2_oband")
    kwargs.setdefault("heater_offset", 0.0)
    kwargs.setdefault("heater_on_both_branches", True)
    kwargs.setdefault("rf_central_conductor_width", 18.0)
    kwargs.setdefault("rf_gap", 6.0)
    kwargs.setdefault("tl", 53.0)
    kwargs.setdefault("tw", 39.0)
    kwargs.setdefault("th", 1.5)
    kwargs.setdefault("tt", 1.5)
    kwargs.setdefault("communication_band", "O-band")
    kwargs.setdefault("ht_layer", LAYER.HRM)
    kwargs.setdefault("termination_layer", LAYER.HRM)
    kwargs.setdefault("heater_width", 2.5)
    kwargs.setdefault("resistor_length", 190.0 / 2)
    # kwargs.setdefault("terminated", True)
    mzm_ref = c << mzm_unbalanced_LT(
        **kwargs,
        add_M2andOpenings=True,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_terminated_mzm_1x2mmi_oband(
    **kwargs,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi1x2_oband")
    kwargs.setdefault("terminated", True)
    mzm_ref = c << template_build_mzm_oband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_unterminated_mzm_1x2mmi_oband(
    **kwargs,
):
    """Create a routed unterminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi1x2_oband")
    kwargs.setdefault("terminated", False)
    mzm_ref = c << template_build_mzm_oband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_terminated_mzm_2x2mmi_oband(
    **kwargs,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi2x2_oband")
    kwargs.setdefault("terminated", True)
    mzm_ref = c << template_build_mzm_oband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_unterminated_mzm_2x2mmi_oband(
    **kwargs,
):
    """Create a routed unterminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi2x2_oband")
    kwargs.setdefault("terminated", False)
    mzm_ref = c << template_build_mzm_oband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c


############################################
########### C-band builders ################
############################################


def template_build_mzm_cband(
    **kwargs,
):
    """Create a template for MZM for C-band operation."""
    c = gf.Component()
    # Create the MZM component
    # kwargs.setdefault("splitter", "mmi2x2_cband")
    kwargs.setdefault("heater_on_both_branches", True)
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
    mzm_ref = c << mzm_unbalanced_LT(
        **kwargs,
        add_M2andOpenings=True,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_terminated_mzm_1x2mmi_cband(
    **kwargs,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi1x2_cband")
    kwargs.setdefault("terminated", True)
    mzm_ref = c << template_build_mzm_cband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_unterminated_mzm_1x2mmi_cband(
    **kwargs,
):
    """Create a routed unterminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi1x2_cband")
    kwargs.setdefault("terminated", False)
    mzm_ref = c << template_build_mzm_cband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_terminated_mzm_2x2mmi_cband(
    **kwargs,
):
    """Create a routed terminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi2x2_cband")
    kwargs.setdefault("terminated", True)
    mzm_ref = c << template_build_mzm_cband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c


def build_unterminated_mzm_2x2mmi_cband(
    **kwargs,
):
    """Create a routed unterminated MZM for wafer-scale testing with edge couplers."""
    c = gf.Component()
    # Create the MZM component
    kwargs.setdefault("splitter", "mmi2x2_cband")
    kwargs.setdefault("terminated", False)
    mzm_ref = c << template_build_mzm_cband(
        **kwargs,
    )
    c.add_ports(mzm_ref.ports)
    return c
