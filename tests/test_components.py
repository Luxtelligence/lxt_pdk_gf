import pathlib

import pytest
from gdsfactory.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from lnoi400 import PDK

cells = PDK.cells

skip_test = {"import_gds"}

component_names = set(cells.keys()) - set(skip_test)
dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds").parent / "gds_ref"
dirpath.mkdir(exist_ok=True, parents=True)


@pytest.fixture(params=component_names, scope="function")
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = cells[component_name]()
    difftest(component, test_name=component_name, dirpath=dirpath)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = cells[component_name]()
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component_name: str) -> None:
    component = cells[component_name]()
    component.assert_ports_on_grid()


if __name__ == "__main__":
    import gdsfactory as gf

    double_taper = gf.get_component("double_linear_inverse_taper")
    difftest(double_taper, None)
