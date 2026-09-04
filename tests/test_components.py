import filecmp
import pathlib
import shutil
import tempfile

import klayout.db as kdb
import numpy as np
import pytest
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

_SLIVER_TOLERANCE = 1  # database units


def _gds_difftest(
    component,
    ref_file: pathlib.Path,
    sliver_tolerance: int = _SLIVER_TOLERANCE,
) -> None:
    """Flat-geometry GDS comparator using raw KLayout XOR.

    Writes the run GDS without kfactory metadata so that intermediate
    unnamed cell numbering (which shifts when new cells are registered in
    the global kfactory registry) does not cause spurious failures.
    Uses a recursive-flatten XOR on every layer so the comparison is
    insensitive to cell hierarchy and naming while still catching real
    geometry regressions.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_file = pathlib.Path(tmp) / f"{ref_file.stem}.gds"
        component.write_gds(gdspath=str(run_file), with_metadata=False)

        if not ref_file.exists():
            shutil.copy(run_file, ref_file)
            raise AssertionError(
                f"Reference GDS file for {ref_file.stem!r} not found. "
                f"Wrote new ref to {ref_file}"
            )

        if filecmp.cmp(ref_file, run_file, shallow=False):
            return

        ref_layout = kdb.Layout()
        ref_layout.read(str(ref_file))
        run_layout = kdb.Layout()
        run_layout.read(str(run_file))

        ref_top = ref_layout.top_cell()
        run_top = run_layout.top_cell()

        layer_ids = {(li.layer, li.datatype) for li in ref_layout.layer_infos()} | {
            (li.layer, li.datatype) for li in run_layout.layer_infos()
        }

        failures = []
        for layer, datatype in sorted(layer_ids):
            ref_region = kdb.Region(
                ref_top.begin_shapes_rec(ref_layout.layer(layer, datatype))
            )
            run_region = kdb.Region(
                run_top.begin_shapes_rec(run_layout.layer(layer, datatype))
            )
            xor = ref_region ^ run_region
            if not xor.is_empty() and not xor.sized(-sliver_tolerance).is_empty():
                failures.append(f"({layer}/{datatype})")

        if failures:
            raise AssertionError(
                f"GDS geometry differs on layer(s) {failures}: {ref_file.name}"
            )


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

    _gds_difftest(component, ref_file=dirpath / f"{component_name}.gds")


def test_settings(
    pdk_name: str,
    component_name: str,
    data_regression: DataRegressionFixture,
) -> None:
    """Avoid regressions when exporting settings."""
    pdk = pdks[pdk_name]
    pdk.activate()
    component = pdk.cells[component_name]()
    data_regression.check(component.to_dict(with_ports=True))


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
