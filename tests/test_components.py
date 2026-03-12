import pathlib

import gdsfactory as gf
import kfactory as kf
import numpy as np
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
    data_regression.check(component.to_dict(with_ports=True))


@pytest.mark.parametrize("component_name", component_names)
def test_optical_port_positions(component_name: str) -> None:
    """Ensure that optical ports are positioned correctly."""
    component = cells[component_name]()
    if isinstance(component, gf.ComponentAllAngle):
        new_component = gf.Component()
        kf.VInstance(component).insert_into_flat(new_component, levels=0)
        new_component.add_ports(component.ports)
        component = new_component
    for port in component.ports:
        if port.port_type == "optical":
            port_layer = port.layer
            port_width = port.width
            port_position = port.center
            port_angle = port.orientation
            cs_region = kf.kdb.Region(component.begin_shapes_rec(port_layer))
            optical_edges = cs_region.edges()

            tolerance = 0.001
            poly = kf.kdb.DBox(-tolerance, -tolerance, tolerance, tolerance)
            dbu_in_um = port.kcl.to_um(1)
            port_marker = (
                kf.kdb.DPolygon(poly).transformed(port.dcplx_trans).to_itype(dbu_in_um)
            )
            port_marker_region = kf.kdb.Region(port_marker)

            interacting_edges = optical_edges.interacting(port_marker_region)
            if interacting_edges.is_empty():
                raise AssertionError(
                    f"No optical edge found for port {port.name} at position {port_position} with width {port_width} and angle {port_angle}."
                )
            port_edge = next(iter(interacting_edges.each()))
            edge_length = port_edge.length() * 0.001
            if not np.isclose(edge_length, port_width, atol=1e-3):
                raise AssertionError(
                    f"Port {port.name} has width {port_width}, but the optical edge length is {edge_length}."
                )


if __name__ == "__main__":
    print(component_names)
