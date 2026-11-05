import cadquery as cq
import pytest

from cad_engine import PackoutOrganizer
from src import geometry as geo
from src.geometry import GRID, FOOT_H

check = PackoutOrganizer.check_part


def vol(wp):
    return wp.val().Volume()


def inside(wp, x, y, z):
    return wp.val().isInside(cq.Vector(x, y, z))


def assert_printable(wp):
    r = check(wp)
    assert r["valid"] and r["watertight"], r


# ---- base block ---------------------------------------------------------
def test_base_block_footprint_and_height():
    b = geo.create_base_block(2, 3, 30)
    x0, y0, z0, x1, y1, z1 = geo.exact_bbox(b)
    assert (x1 - x0) == pytest.approx(2 * GRID - 0.5, abs=1e-3)
    assert (y1 - y0) == pytest.approx(3 * GRID - 0.5, abs=1e-3)
    assert (z1 - z0) == pytest.approx(30)
    assert_printable(b)


def test_base_block_rejects_tiny_height():
    with pytest.raises(ValueError):
        geo.create_base_block(1, 1, FOOT_H)


# ---- phone slot -----------------------------------------------------------
def test_phone_slot_is_printable_and_hollow():
    blk, nx, ny = geo.create_phone_slot_block()
    assert_printable(blk)
    assert vol(blk) < vol(geo.create_base_block(nx, ny, 70))
    cx = nx * GRID / 2

    def empty_ys(z):
        return [y / 2 for y in range(1, int(ny * GRID * 2) - 1) if not inside(blk, cx, y / 2, z)]  # block spans y 0.25..ny*42-0.25

    top, bottom = empty_ys(69.5), empty_ys(FOOT_H + 1.5 + 1.0)
    assert top and bottom                       # slot open at the top and reaches the floor
    assert sum(top) / len(top) > sum(bottom) / len(bottom)    # leans towards +Y
    assert inside(blk, 2.0, 2.0, 30)            # corner wall is solid
    assert inside(blk, 21.0, 21.0, 3.0)         # foot under the slot is solid


def test_phone_slot_clearance_removes_more_material():
    tight, *_ = geo.create_phone_slot_block(clearance=0.2, finger_scoops=False)
    loose, *_ = geo.create_phone_slot_block(clearance=1.0, finger_scoops=False)
    assert vol(loose) < vol(tight)


def test_phone_slot_angle_changes_footprint_need():
    _, _, ny_up = geo.create_phone_slot_block(angle=0)
    _, _, ny_flat = geo.create_phone_slot_block(angle=60, height_mm=60)
    assert ny_flat > ny_up


def test_phone_slot_scoops_remove_material():
    a, *_ = geo.create_phone_slot_block(finger_scoops=True)
    b, *_ = geo.create_phone_slot_block(finger_scoops=False)
    assert vol(a) < vol(b)


def test_phone_slot_too_small_grid_raises():
    with pytest.raises(ValueError):
        geo.create_phone_slot_block(grid_size=(1, 1))


def test_phone_slot_bad_angle():
    with pytest.raises(ValueError):
        geo.create_phone_slot_block(angle=95)


# ---- STL insert -------------------------------------------------------------
def test_stl_insert_pocket(puck_stl):
    blk, nx, ny = geo.create_stl_insert_block(puck_stl, offset=0.4)
    assert_printable(blk)
    cx, cy = nx * GRID / 2, ny * GRID / 2
    floor_z = FOOT_H + 1.5
    assert not inside(blk, cx, cy, floor_z + 5)            # puck volume is empty
    assert not inside(blk, cx + 28.2, cy, floor_z + 5)     # inside the +0.4 offset
    assert inside(blk, cx + 29.5, cy, floor_z + 5)         # wall beyond the offset


def test_stl_offset_grows_pocket(puck_stl):
    small, *_ = geo.create_stl_insert_block(puck_stl, offset=0.0, block_grid_size=(2, 2),
                                            cable_channel=False, finger_scoop=False, height_mm=20)
    big, *_ = geo.create_stl_insert_block(puck_stl, offset=1.0, block_grid_size=(2, 2),
                                          cable_channel=False, finger_scoop=False, height_mm=20)
    assert vol(big) < vol(small)


def test_stl_load_reports_size_and_mode(puck_stl):
    solid, info = geo.load_stl_as_solid(puck_stl, 0.4)
    assert info["mode"] == "mesh"
    assert info["size_z"] == pytest.approx(11.0, abs=0.01)
    assert info["size_x"] == pytest.approx(56.0, abs=0.1)
    assert solid.isValid()


def test_stl_too_big_for_grid(puck_stl):
    with pytest.raises(ValueError):
        geo.create_stl_insert_block(puck_stl, block_grid_size=(1, 1))


def test_offset_mesh_moves_every_face_outward():
    import trimesh
    box = trimesh.creation.box((10, 10, 10))
    grown = geo.offset_mesh(box, 0.5)
    assert grown.bounds[1].tolist() == pytest.approx([5.5] * 3, abs=1e-6)   # miter-compensated


# ---- hub ---------------------------------------------------------------------
def test_hub_pocket_and_channels():
    with_c, nx, ny = geo.create_hub_cutout_block(cable_pass_through=True)
    without, *_ = geo.create_hub_cutout_block(cable_pass_through=False)
    assert_printable(with_c)
    assert_printable(without)
    assert not inside(with_c, nx * GRID / 2, ny * GRID / 2, 20)
    # side tunnel leaves the block: point in the tunnel on the front wall is empty
    assert not inside(with_c, nx * GRID / 2, 1.0, FOOT_H + 4 + 3 + 2)


def test_hub_grid_too_small():
    with pytest.raises(ValueError):
        geo.create_hub_cutout_block(grid_size=(1, 1))


# ---- baseplate + keys -------------------------------------------------------
def test_baseplate_tile_and_key_fit():
    plate = geo.create_baseplate_tile(2, 2, keys=[(2 * GRID, 21, False), (21, 0, True)])
    assert_printable(plate)
    plain = geo.create_baseplate_tile(2, 2)
    assert vol(plate) < vol(plain)
    key = geo.create_dovetail_key()
    assert_printable(key)
    x0, y0, z0, x1, y1, z1 = geo.exact_bbox(key)
    assert (z1 - z0) < geo.KEY_DEPTH            # key is thinner than the socket
    assert (x1 - x0) == pytest.approx(16.0)


def test_block_foot_sits_in_baseplate_pocket():
    plate = geo.create_baseplate_tile(1, 1)
    foot = geo.create_base_block(1, 1, 20).translate((0, 0, geo.SLAB_H))
    # block foot + pocket must not collide: interference volume ~ 0
    assert vol(plate.intersect(foot)) < 1.0


# ---- phone presets ----------------------------------------------------------
def test_phone_preset_uses_db_dims_and_clearance():
    from src.phone_db import get_phone
    ultra = get_phone("Samsung Galaxy S23 Ultra")
    blk, nx, ny = geo.create_phone_slot_block(phone="Samsung Galaxy S23 Ultra", angle=0, height_mm=60)
    assert_printable(blk)
    explicit, nx2, ny2 = geo.create_phone_slot_block(phone_dims=ultra.dims, clearance=ultra.clearance, angle=0, height_mm=60)
    assert (nx, ny) == (nx2, ny2) and vol(blk) == pytest.approx(vol(explicit))
    base, *_ = geo.create_phone_slot_block(phone="Samsung Galaxy S23", angle=0, height_mm=60)
    assert vol(base) != pytest.approx(vol(blk))        # different presets, different slots


def test_phone_preset_clearance_override():
    a, *_ = geo.create_phone_slot_block(phone="Samsung Galaxy S23", clearance=1.2, finger_scoops=False)
    b, *_ = geo.create_phone_slot_block(phone="Samsung Galaxy S23", finger_scoops=False)   # DB default 0.6
    assert vol(a) < vol(b)


# ---- universal STL ------------------------------------------------------------
def test_analyze_stl_and_rotation(bar_stl):
    a = geo.analyze_stl(bar_stl)
    assert a["watertight"] and a["size"] == pytest.approx((60, 20, 10), abs=1e-6)
    assert a["volume_cm3"] == pytest.approx(12.0)
    r = geo.analyze_stl(bar_stl, (0, 0, 90))
    assert r["size"] == pytest.approx((20, 60, 10), abs=1e-6)
    up = geo.analyze_stl(bar_stl, (90, 0, 0))
    assert up["size"] == pytest.approx((60, 10, 20), abs=1e-6)
    with pytest.raises(ValueError):
        geo.analyze_stl(bar_stl, (0, 0, 45))


def test_analyze_stl_warns_on_open_and_tiny(tmp_path):
    import trimesh
    m = trimesh.creation.box((1.0, 1.0, 1.0))
    m.faces = m.faces[:-2]            # punch a hole
    p = tmp_path / "open.stl"
    m.export(p)
    a = geo.analyze_stl(str(p))
    assert not a["watertight"]
    assert any("inches" in w for w in a["warnings"])


def test_suggest_grid_units():
    assert geo.suggest_grid_units((30, 30), 0.4) == (1, 1)
    assert geo.suggest_grid_units((60, 20), 0.4) == (2, 1)
    assert geo.suggest_grid_units((60, 20), 0.4)[0] >= geo.suggest_grid_units((30, 20), 0.4)[0]


def test_custom_stl_block_clearance_and_grid(bar_stl):
    blk, nx, ny = geo.create_custom_stl_subtraction_block(bar_stl, 2, 1, 0.4)
    assert (nx, ny) == (2, 1)
    assert_printable(blk)
    cx, cy, fz = nx * GRID / 2, ny * GRID / 2, FOOT_H + 1.5
    assert not inside(blk, cx + 30.3, cy, fz + 5)       # inside the +0.4 clearance
    assert inside(blk, cx + 30.9, cy, fz + 5)           # beyond it: wall
    loose, *_ = geo.create_custom_stl_subtraction_block(bar_stl, 2, 1, 0.6)
    assert not inside(loose, cx + 30.5, cy, fz + 5)
    assert vol(loose) < vol(blk)


def test_custom_stl_block_rotation_changes_footprint(bar_stl):
    _, nx, ny = geo.create_custom_stl_subtraction_block(bar_stl)
    _, rx, ry = geo.create_custom_stl_subtraction_block(bar_stl, rotation_deg=(0, 0, 90))
    assert (nx, ny) == (2, 1) and (rx, ry) == (1, 2)
    blk, *_ = geo.create_custom_stl_subtraction_block(bar_stl, rotation_deg=(90, 0, 0), grid_units_x=2, grid_units_y=1)
    assert_printable(blk)


def test_custom_stl_block_cable_features(bar_stl):
    plain, *_ = geo.create_custom_stl_subtraction_block(bar_stl, 2, 2)
    side, *_ = geo.create_custom_stl_subtraction_block(bar_stl, 2, 2, cable_channel_bool=True)
    groove, *_ = geo.create_custom_stl_subtraction_block(bar_stl, 2, 2, cable_floor_groove=True)
    assert vol(side) < vol(plain)
    assert vol(groove) != pytest.approx(vol(plain))
    for b in (side, groove):
        assert_printable(b)
    # side tunnel is open to the front face at floor level
    cx = 2 * GRID / 2
    assert not inside(side, cx, 1.0, FOOT_H + 1.5 + 2)


def test_custom_stl_block_validation(bar_stl):
    with pytest.raises(ValueError):
        geo.create_custom_stl_subtraction_block(bar_stl, 1, 1)         # 60 mm bar cannot fit 1x1
    with pytest.raises(ValueError):
        geo.create_custom_stl_subtraction_block(bar_stl, 2, None)
    with pytest.raises(ValueError):
        geo.create_custom_stl_subtraction_block(bar_stl, 0, 1)
    with pytest.raises(ValueError):
        geo.create_custom_stl_subtraction_block(bar_stl, 2, 1, clearance_mm=-1)


def test_custom_stl_return_info(bar_stl):
    *_, info = geo.create_custom_stl_subtraction_block(bar_stl, return_info=True)
    assert info["mode"] == "silhouette"
    *_, info = geo.create_custom_stl_subtraction_block(bar_stl, pocket="exact", return_info=True)
    assert info["mode"] == "mesh"


# ---- scanned contour block ------------------------------------------------------
def _shapely():
    import shapely.geometry as sg
    return sg


def test_scanned_block_removes_area_times_depth():
    sg = _shapely()
    poly = sg.box(0, 0, 70, 30)
    blk, nx, ny, info = geo.create_scanned_contour_block(poly, depth_mm=25, return_info=True)
    assert (nx, ny) == (2, 1)
    assert_printable(blk)
    base = geo.create_base_block(nx, ny, info["height_mm"])
    assert vol(base) - vol(blk) == pytest.approx(70 * 30 * 25, rel=1e-3)
    assert info["height_mm"] == pytest.approx(FOOT_H + 1.5 + 25)
    assert info["area_mm2"] == pytest.approx(2100)


def test_scanned_block_is_centred_and_open_at_top():
    sg = _shapely()
    blk, nx, ny = geo.create_scanned_contour_block(sg.box(500, -80, 560, -50), 2, 1, depth_mm=20)   # far from origin
    cx, cy = nx * GRID / 2, ny * GRID / 2
    top = FOOT_H + 1.5 + 20
    assert not inside(blk, cx, cy, top - 0.5) and not inside(blk, cx, cy, FOOT_H + 1.5 + 0.5)
    assert not inside(blk, cx + 29, cy + 14, top - 1)
    assert inside(blk, cx + 33, cy, top - 1)                   # wall beyond the 60 mm outline
    assert inside(blk, cx, cy, FOOT_H + 0.5)                   # solid floor under the pocket


def test_scanned_block_follows_outline_shape():
    sg = _shapely()
    ell = sg.Polygon([(0, 0), (60, 0), (60, 15), (20, 15), (20, 40), (0, 40)])          # L shape
    blk, nx, ny = geo.create_scanned_contour_block(ell, 2, 2, depth_mm=15)
    ox, oy = nx * GRID / 2 - 30, ny * GRID / 2 - 20            # polygon bbox centre is (30, 20)
    z = FOOT_H + 1.5 + 10
    assert not inside(blk, ox + 50, oy + 5, z)                 # inside the foot of the L
    assert inside(blk, ox + 50, oy + 30, z)                    # the notch of the L stays solid
    assert not inside(blk, ox + 10, oy + 35, z)                # inside the upright of the L


def test_scanned_block_keeps_islands():
    sg = _shapely()
    ring = sg.box(0, 0, 70, 40).difference(sg.box(25, 12, 45, 28))
    blk, nx, ny = geo.create_scanned_contour_block(ring, depth_mm=15)
    assert_printable(blk)
    cx, cy, z = nx * GRID / 2, ny * GRID / 2, FOOT_H + 1.5 + 10
    assert inside(blk, cx, cy, z)                              # island post is solid
    assert not inside(blk, cx - 30, cy, z)                     # ring body is cut


def test_scanned_block_complex_outline_is_printable():
    from src import scanner
    poly = scanner.prepare_cutout(scanner.demo_item_polygon(), 0.6, scoops=True)
    assert len(poly.exterior.coords) > 100
    blk, nx, ny = geo.create_scanned_contour_block(poly, depth_mm=20)
    assert_printable(blk)


def test_scanned_block_validation():
    sg = _shapely()
    p = sg.box(0, 0, 60, 30)
    with pytest.raises(ValueError, match="walls"):
        geo.create_scanned_contour_block(p, 1, 1, depth_mm=10)
    with pytest.raises(ValueError, match="both block_grid"):
        geo.create_scanned_contour_block(p, 2, None, depth_mm=10)
    with pytest.raises(ValueError, match="depth_mm"):
        geo.create_scanned_contour_block(p, depth_mm=0)
    with pytest.raises(ValueError, match="above the feet"):
        geo.create_scanned_contour_block(p, depth_mm=30, height_mm=30)
    with pytest.raises(ValueError, match="MultiPolygon"):
        geo.create_scanned_contour_block(sg.MultiPolygon([sg.box(0, 0, 5, 5), sg.box(10, 10, 15, 15)]))
    with pytest.raises(ValueError, match="shapely Polygon"):
        geo.create_scanned_contour_block([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="empty or smaller"):
        geo.create_scanned_contour_block(sg.box(0, 0, 0.5, 0.5))
    with pytest.raises(ValueError, match="grid size"):
        geo.create_scanned_contour_block(p, 0, 1, depth_mm=10)


def test_scanned_block_explicit_height_and_bowtie_repair():
    sg = _shapely()
    bow = sg.Polygon([(0, 0), (40, 30), (40, 0), (0, 30)])       # self-intersecting 'bow tie'
    blk, *_ = geo.create_scanned_contour_block(bow, depth_mm=10)  # repaired by buffer(0)
    assert_printable(blk)
    tall, nx, ny, info = geo.create_scanned_contour_block(sg.box(0, 0, 40, 20), depth_mm=10, height_mm=40, return_info=True)
    assert info["height_mm"] == 40 and geo.exact_bbox(tall)[5] == pytest.approx(40)


# ---- silhouette (drop-in) pocket ------------------------------------------
@pytest.fixture
def hook_stl(tmp_path):
    """Charging-stand-like part: a base slab with an arm overhanging it, so its
    exact 3D negative could not be dropped in from above."""
    import trimesh
    base = trimesh.creation.box((40, 30, 6))
    post = trimesh.creation.box((8, 30, 40)); post.apply_translation((-16, 0, 20))
    arm = trimesh.creation.box((30, 30, 6)); arm.apply_translation((-3, 0, 40))
    m = trimesh.util.concatenate([base, post, arm])
    path = tmp_path / "hook.stl"
    m.export(path)
    return str(path)


def test_stl_silhouette_pocket_lets_overhanging_part_drop_in(hook_stl):
    blk, nx, ny, info = geo.create_custom_stl_subtraction_block(hook_stl, return_info=True)
    assert info["mode"] == "silhouette"
    assert_printable(blk)
    # the full top-down footprint is open at the top, even under the arm's overhang
    cx, cy, top = nx * GRID / 2, ny * GRID / 2, blk.val().BoundingBox().zmax
    for dx in (-18, 0, 15):
        assert not inside(blk, cx + dx, cy, top - 1)
        assert not inside(blk, cx + dx, cy, FOOT_H + 1.5 + 3)


def test_stl_silhouette_pocket_is_deeper_open_than_exact(hook_stl):
    sil, *_ = geo.create_custom_stl_subtraction_block(hook_stl, pocket="silhouette")
    exact, *_ = geo.create_custom_stl_subtraction_block(hook_stl, pocket="exact")
    assert vol(sil) < vol(exact)          # exact negative leaves the space under the arm filled


def test_stl_pocket_mode_validated(hook_stl):
    with pytest.raises(ValueError):
        geo.create_custom_stl_subtraction_block(hook_stl, pocket="bogus")


# ---- STL on an integrated base --------------------------------------------
def test_stl_on_base_fuses_part_to_grid_base(bar_stl):
    blk, nx, ny, info = geo.create_stl_on_base(bar_stl, base_thickness=3.0, return_info=True)
    assert_printable(blk)
    assert info["mode"] == "mesh" and (nx, ny) == (2, 1)
    x0, y0, z0, x1, y1, z1 = geo.exact_bbox(blk)
    assert (x1 - x0) == pytest.approx(nx * GRID - 0.5, abs=1e-3)       # base footprint = grid cells
    assert z1 == pytest.approx(info["height_mm"], abs=1e-3)
    cx, cy = nx * GRID / 2, ny * GRID / 2
    assert inside(blk, cx, cy, FOOT_H + 3.0 + 2.0)                     # the part stands on the base (solid, no pocket)
    assert inside(blk, cx, cy, FOOT_H + 1.0)                           # slab


def test_stl_on_base_rejects_part_larger_than_grid(bar_stl):
    with pytest.raises(ValueError):
        geo.create_stl_on_base(bar_stl, (1, 1))
    with pytest.raises(ValueError):
        geo.create_stl_on_base(bar_stl, base_thickness=0.5)


# ---- baseplate-free interlock ------------------------------------------------
def test_interlock_base_flat_underside_with_edge_sockets():
    blk = geo.create_base_block(2, 2, 20)
    il = geo.add_interlock_base(blk, 2, 2)
    assert_printable(il)
    assert vol(il) > vol(blk)                                          # feet gaps filled by a flat prism
    # a socket sits at the middle of each perimeter cell edge: open at the bottom, solid above it
    for (x, y, rot) in geo.interlock_socket_positions(2, 2):
        dx, dy = (0.0, 4.0) if not rot else (4.0, 0.0)                 # a few mm into the block from the edge
        px = min(max(x + (0.5 if x == 0 else -0.5 if x == 2 * GRID else dx), 0.4), 2 * GRID - 0.4)
        py = min(max(y + (0.5 if y == 0 else -0.5 if y == 2 * GRID else dy), 0.4), 2 * GRID - 0.4)
        assert not inside(il, px, py, 1.0)
        assert inside(il, px, py, geo.KEY_DEPTH + 1.0)
    assert inside(il, GRID, GRID, 1.0)                                 # middle of the underside is solid


def test_interlock_socket_count():
    assert len(geo.interlock_socket_positions(1, 1)) == 4
    assert len(geo.interlock_socket_positions(3, 2)) == 2 * 3 + 2 * 2


# ---- phone rack (N slots in one block) --------------------------------------
S23 = "Samsung Galaxy S23"


def _phone_overlap_cm3(blk, info, stance, count, angle):
    """Volume of material that a real phone-sized box would collide with in every slot."""
    import math
    length, width, thick = geo.S23_DIMS
    vert, across = (length, width) if stance == "portrait" else (width, length)
    a = math.radians(angle)
    ph = cq.Workplane("XY").box(across, thick, vert, centered=(True, True, False)).translate((0, 0, 0.05))
    ph = ph.rotate((0, 0, 0), (1, 0, 0), -angle)
    lift = (info["T"] / 2) * math.sin(a)          # the slot's lowest corner sits at floor_z (slot is T thick, phone is thinner)
    total = 0.0
    nx, ny = info["nx"], info["ny"]
    cx, cy = nx * GRID / 2, ny * GRID / 2
    for i in range(count):
        y = cy + i * info["pitch"] - (info["ymid0"] * 2 + (count - 1) * info["pitch"]) / 2
        placed = ph.translate((cx, y, info["floor_z"] + lift))
        total += blk.val().intersect(placed.val()).Volume()
    return total / 1000


@pytest.mark.parametrize("stance,angle", [("portrait", 0), ("portrait", 20), ("landscape", 0), ("landscape", 15)])
def test_phone_rack_holds_phones_without_collision(stance, angle):
    blk, nx, ny, info = geo.create_phone_rack_block(S23, 5, clearance=0.6, stance=stance, angle=angle, return_info=True)
    assert_printable(blk)
    assert info["count"] == 5
    assert _phone_overlap_cm3(blk, info, stance, 5, angle) < 0.05      # < 50 mm3 of grazing contact


def test_phone_rack_any_number_of_phones():
    sizes = {}
    for n in (1, 2, 3, 5, 8, 10, 13):
        blk, nx, ny, info = geo.create_phone_rack_block(S23, n, return_info=True)
        assert_printable(blk)
        sizes[n] = ny
        # one slot per phone: the slot axis of every phone is open at mid height
        for i in range(n):
            y = ny * GRID / 2 + i * info["pitch"] - (info["ymid0"] * 2 + (n - 1) * info["pitch"]) / 2
            assert not inside(blk, nx * GRID / 2, y, info["floor_z"] + 40)
    assert list(sizes.values()) == sorted(sizes.values()) and sizes[13] > sizes[1]


def test_phone_rack_features_change_the_solid():
    base, *_ = geo.create_phone_rack_block(S23, 5, vent=False, finger_scoops=False, cable_groove=False)
    vented, *_ = geo.create_phone_rack_block(S23, 5, vent=True, finger_scoops=False, cable_groove=False)
    scooped, *_ = geo.create_phone_rack_block(S23, 5, vent=False, finger_scoops=True, cable_groove=False)
    assert vol(vented) < vol(base) and vol(scooped) < vol(base)
    taller, *_ = geo.create_phone_rack_block(S23, 5, hold_frac=0.9)
    shorter, *_ = geo.create_phone_rack_block(S23, 5, hold_frac=0.5)
    assert geo.exact_bbox(taller)[5] > geo.exact_bbox(shorter)[5]


def test_phone_rack_validation_and_pose_choice():
    for kw in (dict(count=0), dict(count=61), dict(count=3, stance="x"), dict(count=3, angle=70),
               dict(count=3, separator=1.0), dict(count=3, hold_frac=0.1)):
        with pytest.raises(ValueError):
            geo.create_phone_rack_block(S23, **kw)
    dims = geo.get_phone(S23).dims
    assert geo.choose_rack_pose(dims, 0.6, 98)[0] == "landscape"         # 99 mm case: stands on its long edge
    assert geo.choose_rack_pose(dims, 0.6, 400) == ("portrait", 0)       # deep enough: upright like the reference
    assert geo.choose_rack_pose(dims, 0.6, 45) is None                   # low-profile case: flat single slots
