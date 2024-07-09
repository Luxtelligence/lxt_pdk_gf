import pathlib

import pytest
from gdsfactory.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from lnoi400 import PDK

cells = PDK.cells

skip_test = {"import_gds"}

component_names = set(cells.keys()) - set(skip_test)
dirpath = pathlib.Path(__file__).absolute().parent / "gds_ref"
dirpath.mkdir(exist_ok=True, parents=True)

pcell_mapping = [
    ("cell", "cell"),
]


@pytest.fixture(params=component_names, scope="function")
def component_name(request) -> str:
    return request.param


# @pytest.fixture(params=pcell_mapping, scope="function")
# def name_mapping(request) -> str:
#     return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = cells[component_name]()
    difftest(
        component,
        test_name=component_name,
        dirpath=dirpath,
        ignore_sliver_differences=True,
    )


# def test_alternative_implementation(
#     name_mapping: tuple,
# ) -> None:
#     """Test against the cells distributed with a different PDK implementation."""

#     # TODO: Implement difftest with layers selection.

#     assert name_mapping[0] == name_mapping[1]


def test_settings(
    component_name: str,
    data_regression: DataRegressionFixture,
) -> None:
    """Avoid regressions when exporting settings."""
    component = cells[component_name]()
    data_regression.check(component.to_dict())


if __name__ == "__main__":
    print(component_names)
