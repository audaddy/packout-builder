"""Packing several copies of an object (and the watch on it) onto one shared base."""
import math

import cadquery as cq
import numpy as np
import pytest
import trimesh

from cad_engine import PackoutOrganizer, PACKOUT_MODELS
from src import cluster as cl
from src import geometry as geo
from src.geometry import GRID


def _centered(mesh):
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


@pytest.fixture
def stand_stl(tmp_path):
    """Charging-stand-like part: a post with a tilted top face (normal (0.82, 0, 0.57)) holding a
    29.8 mm round puck recess, on a base slab."""
    n = np.array([0.82, 0.0, 0.57])
    n /= np.linalg.norm(n)
    body = cq.Workplane("XY").box(90, 60, 6, centered=(True, True, False))
    body = body.union(cq.Workplane("XY").box(50, 60, 80, centered=(True, True, False)).translate((-10, 0, 0)))
    plane_pt = np.array([15.0, 0.0, 40.0])                               # slant: remove everything beyond this plane
    body = body.cut(cq.Workplane(cq.Plane(origin=tuple(plane_pt), xDir=(0, 1, 0), normal=tuple(n))).rect(400, 400).extrude(200))
    centre = np.array([1.8, 0.0, 59.0])                                  # on that plane
    assert abs(n @ (centre - plane_pt)) < 0.05
    body = body.cut(cq.Workplane(cq.Plane(origin=tuple(centre), xDir=(0, 1, 0), normal=tuple(n))).circle(14.9).extrude(-8))
    path = tmp_path / "stand.stl"
    PackoutOrganizer.to_mesh(body, 0.02, 0.1).export(path)
    return str(path), n, centre


def _simple_unit(box=(40.0, 30.0, 20.0)):
    return _centered(trimesh.creation.box(box))


def test_pack_boxes_without_watch_is_compact_and_collision_free():
    m = _simple_unit()
    pl = cl.ClusterPlanner(m, None, gap=3.0, rotations=(0,))
    plan = pl.plan(4)
    assert len(plan.offsets) == 4
    w, h = plan.size
    assert w * h < 2 * 40 * 2 * 30 * 1.4                 # near 2 x 2 boxes plus gaps
    pts = np.array(plan.offsets)
    for i in range(4):
        for j in range(i + 1, 4):                        # no two boxes overlap (with the gap)
            d = np.abs(pts[i] - pts[j])
            assert d[0] >= 40 - 1e-6 or d[1] >= 30 - 1e-6


def test_pack_respects_watch_and_its_install_path():
    m = _simple_unit((40.0, 40.0, 40.0))
    watch = cl.WatchSpec(np.array([1.0, 0.0, 0.0]), np.array([20.0, 0.0, 25.0]), diameter=40.0, thickness=12.0)
    pl = cl.ClusterPlanner(m, watch, gap=2.0, sweep=30.0, rotations=(0, 90, 180, 270))
    plan = pl.plan(3)
    um, uw = cl._rotate_centered(m, watch, plan.rz)
    V = um.vertices
    for i, (ox, oy) in enumerate(plan.offsets):
        C = uw.center + np.array([ox, oy, 0])
        for j, (px, py) in enumerate(plan.offsets):
            if i == j:
                continue
            P = V + np.array([px, py, 0]) - C
            a = P @ uw.axis
            r = np.linalg.norm(P - a[:, None] * uw.axis, axis=1)
            inside = (a >= -uw.thickness / 2) & (a <= uw.thickness / 2 + 30.0) & (r <= uw.diameter / 2)
            assert not inside.any()                      # no other stand blocks this watch or its way on


def test_pack_constraint_can_make_it_impossible():
    pl = cl.ClusterPlanner(_simple_unit(), None, rotations=(0,))
    assert pl.plan(4, max_wh=(30, 30)) is None
    assert pl.plan(1, max_wh=(60, 60)) is not None


def test_detect_watch_cradle_finds_recess(stand_stl):
    path, n, centre = stand_stl
    w = cl.detect_watch_cradle(cl.load_unit_mesh(path))
    assert w is not None
    assert np.degrees(np.arccos(np.clip(w.axis @ n, -1, 1))) < 3.0          # axis along the face normal
    # the face centre may be anywhere on the recess's axis line... compare in the plane of the face
    assert np.linalg.norm((w.face_center - centre) - n * ((w.face_center - centre) @ n)) < 2.5
    assert w.center == pytest.approx(w.face_center + w.axis * w.thickness / 2)
    # the stand is much bigger than the watch: a plain bar has no cradle at all
    assert w.top_z() > w.center[2]


def test_detect_watch_cradle_refuses_plain_shapes(bar_stl):
    assert cl.detect_watch_cradle(cl.load_unit_mesh(bar_stl)) is None


def test_cluster_block_is_one_valid_solid_with_all_copies(bar_stl):
    mesh, watch, planner = cl.get_planner(bar_stl, use_watch=False)
    p1, p3 = planner.plan(1), planner.plan(3)
    b1, nx1, ny1, i1 = geo.create_stl_cluster_block(bar_stl, p1, 3.0, return_info=True)
    b3, nx3, ny3, i3 = geo.create_stl_cluster_block(bar_stl, p3, 3.0, return_info=True)
    for b in (b1, b3):
        r = PackoutOrganizer.check_part(b)
        assert r["valid"] and r["watertight"]
    assert i3["copies"] == 3 and len(i3["offsets"]) == 3
    assert b3.val().Volume() > 2.5 * b1.val().Volume() * 0.6          # three parts on the plate
    assert i3["size"][0] * i3["size"][1] < 3 * i1["size"][0] * i1["size"][1] * 1.6   # packed tighter than 3 separate bases
    bb = geo.exact_bbox(b3)
    assert bb[0] == pytest.approx(0.0, abs=1.0) and bb[1] == pytest.approx(0.0, abs=1.0)   # corner at the origin


def test_engine_cluster_registers_one_block_without_sockets(bar_stl):
    o = PackoutOrganizer(next(iter(PACKOUT_MODELS)), bed=(256, 256, 256), interlock=True)
    items = o.add_stl_cluster(bar_stl, 4, name="bars", use_watch=False)
    assert len(items) == 1 and items[0].name == "bars" and not items[0].sockets
    assert o.check_part(items[0].solid)["valid"]
    other = o.add_stl_on_base(bar_stl, name="plain")[0]
    items[0].manual, items[0].pos = True, (0, 0)
    other.manual, other.pos = True, (items[0].nx, 0)
    o.layout()
    assert o.seam_key_count() == 0                       # no sockets on the custom base, so no clips between them


def test_engine_cluster_splits_when_the_bed_is_small(bar_stl):
    o = PackoutOrganizer("Organizer #48-22-8430", bed=(110, 110, 200))
    items = o.add_stl_cluster(bar_stl, 12, name="bars", use_watch=False)
    assert len(items) >= 2 and [i.name for i in items] == [f"bars_{k}" for k in range(1, len(items) + 1)]
    assert all(o.fits_bed(i.solid) for i in items)
    with pytest.raises(ValueError):
        PackoutOrganizer(bed=(20, 20, 200)).add_stl_cluster(bar_stl, 1, use_watch=False)


def test_engine_cluster_without_cradle_adds_a_note(bar_stl):
    o = PackoutOrganizer(next(iter(PACKOUT_MODELS)), bed=(256, 256, 256))
    items = o.add_stl_cluster(bar_stl, 2, name="bars", use_watch=True)
    assert any("no charger recess" in n for n in items[0].notes)
