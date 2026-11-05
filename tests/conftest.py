import os
import sys

import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def puck_stl(tmp_path_factory):
    """56 mm x 11 mm cylinder stand-in for the watch charger."""
    m = trimesh.creation.cylinder(radius=28.0, height=11.0, sections=64)
    m.apply_translation([0, 0, 5.5])
    p = tmp_path_factory.mktemp("stl") / "puck.stl"
    m.export(p)
    return str(p)


@pytest.fixture
def bar_stl(tmp_path):
    m = trimesh.creation.box((60.0, 20.0, 10.0))
    p = tmp_path / "bar.stl"
    m.export(p)
    return str(p)
