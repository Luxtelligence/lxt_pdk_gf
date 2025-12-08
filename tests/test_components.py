import pathlib

import numpy as np
import pytest
from gdsfactory.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import lnoi400
import ltoi300

pdks = {
    "lnoi400": lnoi400.PDK,
    "ltoi300": ltoi300.PDK,
}

gds_ref_dir = pathlib.Path(__file__).absolute().parent / "gds_ref"

skip_test = {"import_gds"}


def pytest_generate_tests(metafunc):
    if (
        "component_name" in metafunc.fixturenames
        and "pdk_name" in metafunc.fixturenames
    ):
        argvalues = []
        for pdk_name, pdk in pdks.items():
            for component_name in pdk.cells.keys():
                if component_name not in skip_test:
                    argvalues.append((pdk_name, component_name))
        metafunc.parametrize("pdk_name,component_name", argvalues)

    if "model_name" in metafunc.fixturenames and "pdk_name" in metafunc.fixturenames:
        argvalues = []
        for pdk_name, pdk in pdks.items():
            for model_name in pdk.models.keys():
                argvalues.append((pdk_name, model_name))
        metafunc.parametrize("pdk_name,model_name", argvalues)


def test_gds(
    pdk_name: str,
    component_name: str,
) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    pdk = pdks[pdk_name]
    pdk.activate()
    component = pdk.cells[component_name]()

    dirpath = gds_ref_dir / pdk_name
    dirpath.mkdir(exist_ok=True, parents=True)

    difftest(
        component,
        test_name=component_name,
        dirpath=dirpath,
        ignore_sliver_differences=True,
    )


def test_settings(
    pdk_name: str,
    component_name: str,
    data_regression: DataRegressionFixture,
) -> None:
    """Avoid regressions when exporting settings."""
    pdk = pdks[pdk_name]
    pdk.activate()
    component = pdk.cells[component_name]()
    data_regression.check(component.to_dict())


def test_models_with_wavelength_sweep(
    pdk_name: str,
    model_name: str,
    ndarrays_regression: NDArraysRegressionFixture,
) -> None:
    """Test models with different wavelengths to avoid regressions in frequency response."""
    pdk = pdks[pdk_name]
    pdk.activate()
    models = pdk.models

    wl = 1.55

    try:
        model = models[model_name]
        s_params = model(wl=wl)
    except TypeError:
        pytest.skip(f"{model_name} does not accept a wl argument")

    # Convert s_params dictionary to arrays for regression testing
    # s_params is a dict with tuple keys (port pairs) and JAX array values
    arrays_to_check = {}
    for key, value in sorted(s_params.items()):
        # Convert tuple key to string for regression test compatibility
        key_str = f"s_{key[0]}_{key[1]}"
        # Convert JAX arrays to numpy and separate real/imag parts

        value_np = np.array(value)
        arrays_to_check[f"{key_str}_real"] = np.real(value_np)
        arrays_to_check[f"{key_str}_imag"] = np.imag(value_np)

    ndarrays_regression.check(
        arrays_to_check,
        default_tolerance={"atol": 1e-2, "rtol": 1e-2},
    )


if __name__ == "__main__":
    test_models_with_wavelength_sweep("lnoi400", "directional_coupler_balanced", 0)
