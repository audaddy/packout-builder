import io
import math
import zipfile

import pytest

from cad_engine import PackoutOrganizer, split_cells, PACKOUT_MODELS


def test_split_cells_balanced():
    assert split_cells(7, 4) == [4, 3]
    assert split_cells(4, 4) == [4]
    assert split_cells(10, 4) == [4, 3, 3]
    with pytest.raises(ValueError):
        split_cells(3, 0)


@pytest.mark.parametrize("model", list(PACKOUT_MODELS))
@pytest.mark.parametrize("bed", [(180, 180, 180), (200, 200, 200), (256, 256, 256), (120, 300, 100)])
def test_tiles_cover_grid_and_fit_bed(model, bed):
    o = PackoutOrganizer(model, bed=bed)
    tiles = o.plan_tiles()
    covered = set()
    for t in tiles:
        assert t.nx * 42 <= bed[0] and t.ny * 42 <= bed[1]
        for i in range(t.nx):
            for j in range(t.ny):
                cell = (t.cx + i, t.cy + j)
                assert cell not in covered
                covered.add(cell)
    assert len(covered) == o.cells_x * o.cells_y


def test_custom_interior_and_leftover():
    o = PackoutOrganizer(interior=(250, 100, 90), bed=(220, 220, 250))
    assert (o.cells_x, o.cells_y) == (5, 2)
    assert o.leftover == pytest.approx((40.0, 16.0))
    assert o.model == "Custom"


def test_invalid_inputs():
    with pytest.raises(ValueError):
        PackoutOrganizer("nope")
    with pytest.raises(ValueError):
        PackoutOrganizer(interior=(30, 100, 90))
    with pytest.raises(ValueError):
        PackoutOrganizer(interior=(100, 100, 90), bed=(0, 200, 200))


def test_seam_keys_match_sockets():
    o = PackoutOrganizer("Compact Organizer #48-22-8435", bed=(200, 200, 200))   # 4x7 -> two tiles stacked
    tiles = o.plan_tiles()
    assert len(tiles) == 2
    assert o.seam_key_count() == 4
    assert len(o.tile_keys(tiles[0])) == 4 and len(o.tile_keys(tiles[1])) == 4
    big = PackoutOrganizer("Organizer #48-22-8430", bed=(200, 200, 200))   # 10x7 -> 3x2 tiles
    assert len(big.plan_tiles()) == 6
    assert big.seam_key_count() == 2 * 7 + 1 * 10


def test_item_larger_than_bed_rejected():
    o = PackoutOrganizer(interior=(300, 300, 100), bed=(100, 100, 100))
    with pytest.raises(ValueError):
        o.add_hub(hub_dims=(150, 60, 20))


def test_item_taller_than_bed_rejected():
    o = PackoutOrganizer(interior=(300, 300, 100), bed=(200, 200, 50))
    with pytest.raises(ValueError):
        o.add_phone_slots(1, height_mm=70)


def test_layout_no_overlap_and_warnings(puck_stl):
    o = PackoutOrganizer("Compact Organizer #48-22-8435")
    o.add_phone_slots(2, angle=60)
    o.add_stl_insert(puck_stl)
    o.add_hub()
    assert o.layout() == []
    seen = set()
    for it in o.items:
        for i in range(it.nx):
            for j in range(it.ny):
                c = (it.pos[0] + i, it.pos[1] + j)
                assert c not in seen
                seen.add(c)
                assert c[0] < o.cells_x and c[1] < o.cells_y


def test_layout_manual_position_and_overflow():
    o = PackoutOrganizer(interior=(130, 90, 100))        # 3 x 2 cells
    o.add_phone_slots(3)                                 # 2x1 each -> only 3 cells' worth... 2 fit, 1 won't
    items = o.items
    items[0].pos, items[0].manual = (1, 1), True
    warnings = o.layout()
    assert items[0].pos == (1, 1)
    assert any("does not fit" in w for w in warnings)


def test_layout_bad_manual_position_falls_back():
    o = PackoutOrganizer(interior=(130, 90, 100))
    o.add_phone_slots(1)
    o.items[0].pos, o.items[0].manual = (2, 0), True      # 2x1 at x=2 would overflow 3 columns
    w = o.layout()
    assert any("manual position" in x for x in w)
    assert o.items[0].pos is not None


def test_height_warning_vs_interior():
    o = PackoutOrganizer(interior=(130, 90, 60), bed=(200, 200, 200))
    o.add_phone_slots(1, height_mm=70)
    assert any("interior height" in w for w in o.layout())


def test_export_zip_contents(puck_stl):
    o = PackoutOrganizer("Compact Organizer #48-22-8435")
    o.add_phone_slots(1)
    o.add_hub()
    o.layout()
    parts = o.build_parts()
    data = o.export_zip(parts, "both")
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "stl/phone_slot.stl" in names and "step/usbc_hub.step" in names and "README.txt" in names
    assert any(n.startswith("stl/baseplate_") for n in names)
    assert all(o.check_part(w)["watertight"] for w in parts.values())
    with pytest.raises(ValueError):
        o.export_zip(parts, "obj")


def test_phone_lid_clearance_warning_and_tilt_helper():
    from src import geometry as geo
    o = PackoutOrganizer("Compact Organizer #48-22-8435")
    o.add_phone_slots(1, phone="Samsung Galaxy S23", angle=15)          # 141 mm tall phone, 99 mm interior
    assert any("lid will not close" in w for w in o.layout())
    ang = geo.min_tilt_for_height((146.3, 70.9, 7.6), 0.6, o.interior[2] - 1.0)
    assert 45 < ang < 70
    o2 = PackoutOrganizer("Compact Organizer #48-22-8435")
    o2.add_phone_slots(1, phone="Samsung Galaxy S23", angle=ang)
    assert o2.layout() == []
    assert geo.min_tilt_for_height((300, 70, 8), 0.6, 20) is None


def test_phone_preset_names_item_and_low_profile_case():
    from src.packout_db import get_case
    o = PackoutOrganizer(get_case("8431").label, bed=(300, 300, 200))    # near-flat Ultra needs a 252 mm block
    o.add_phone_slots(1, phone="Samsung Galaxy S23 Ultra", angle=82, height_mm=43)
    assert o.items[0].name == "s23_ultra_slot"
    assert o.layout() == []


def test_flat_phone_block_exceeds_small_bed():
    from src.packout_db import get_case
    o = PackoutOrganizer(get_case("8431").label, bed=(200, 200, 200))
    with pytest.raises(ValueError, match="exceeds the printer bed"):
        o.add_phone_slots(1, phone="Samsung Galaxy S23 Ultra", angle=82, height_mm=43)


def test_scanned_block_in_engine_and_profile_export():
    import ezdxf
    from shapely.geometry import box
    o = PackoutOrganizer("Organizer #48-22-8430", bed=(220, 220, 250))
    o.add_scanned_block(box(0, 0, 80, 30), name="pliers", depth_mm=22)
    o.add_hub()
    assert o.items[0].kind == "scan" and o.items[0].profile is not None
    assert o.layout() == []
    parts = o.build_parts()
    assert o.check_part(parts["pliers"])["watertight"]
    names = zipfile.ZipFile(io.BytesIO(o.export_zip(parts, "stl"))).namelist()
    assert "stl/pliers.stl" in names and "profiles/pliers.svg" in names and "profiles/pliers.dxf" in names
    assert not any(n.startswith("profiles/usbc") for n in names)
    z = zipfile.ZipFile(io.BytesIO(o.export_zip(parts, "stl")))
    doc = ezdxf.read(io.StringIO(z.read("profiles/pliers.dxf").decode()))
    assert len(list(doc.modelspace().query("LWPOLYLINE"))) == 1


def test_scanned_block_too_big_for_bed_or_depth():
    from shapely.geometry import box
    o = PackoutOrganizer("Organizer #48-22-8430", bed=(100, 100, 250))
    with pytest.raises(ValueError, match="exceeds the printer bed"):
        o.add_scanned_block(box(0, 0, 150, 30), depth_mm=20)
    o2 = PackoutOrganizer("Organizer #48-22-8430", bed=(220, 220, 40))
    with pytest.raises(ValueError, match="exceeds bed Z"):
        o2.add_scanned_block(box(0, 0, 60, 30), depth_mm=45)


def test_scanned_block_height_warning_vs_low_profile_case():
    from shapely.geometry import box
    from src.packout_db import get_case
    o = PackoutOrganizer(get_case("8431").label)          # 45.7 mm interior
    o.add_scanned_block(box(0, 0, 60, 30), depth_mm=40)   # block = 4.75 + 1.5 + 40 = 46.25 mm
    assert any("interior height" in w for w in o.layout())


# ---- baseplate-free interlock mode ---------------------------------------
def test_interlock_mode_has_no_baseplates_and_counts_clips(puck_stl):
    o = PackoutOrganizer(next(iter(PACKOUT_MODELS)), interlock=True)
    o.add_stl_insert(puck_stl, name="a")
    o.add_stl_insert(puck_stl, name="b")
    o.items[0].manual, o.items[0].pos = True, (0, 0)
    o.items[1].manual, o.items[1].pos = True, (o.items[0].nx, 0)       # touching along x
    assert o.layout() == [] or all("height" in w for w in o.layout())
    assert o.build_baseplates() == {}
    assert o.seam_key_count() == min(o.items[0].ny, o.items[1].ny)
    parts = o.build_parts()
    assert not any(k.startswith("baseplate") for k in parts)
    assert [k for k in parts if k.startswith("dovetail_")] == ["dovetail_clip"]       # one model, not one per clip
    uniq = {n: q for n, _, q in o.unique_parts(parts)}
    assert uniq["dovetail_clip"] == o.seam_key_count()
    assert "no baseplates" in o.summary()
    for it in o.items:
        assert o.check_part(it.solid)["valid"]
    # separated blocks need no clips
    o.items[1].pos = (o.items[0].nx + 1, 0)
    assert o.seam_key_count() == 0


def test_default_mode_unchanged_still_has_baseplates():
    o = PackoutOrganizer(next(iter(PACKOUT_MODELS)))
    assert not o.interlock and o.build_baseplates()


def test_stl_on_base_in_engine(puck_stl):
    o = PackoutOrganizer(next(iter(PACKOUT_MODELS)), interlock=True)
    items = o.add_stl_on_base(puck_stl, name="obj")
    assert items[0].nx >= 1 and items[0].height > 0
    assert o.check_part(items[0].solid)["valid"]


# ---- phone racks in the engine -------------------------------------------
def test_phone_rack_is_one_block_for_all_phones():
    o = PackoutOrganizer(next(iter(PACKOUT_MODELS)), bed=(256, 256, 256))
    items = o.add_phone_rack(5, phone="Samsung Galaxy S23", stance="landscape")
    assert len(items) == 1 and items[0].name == "s23_rack"
    assert o.check_part(items[0].solid)["valid"]


def test_phone_rack_splits_to_fit_small_bed():
    o = PackoutOrganizer("Organizer #48-22-8430", bed=(100, 300, 200))
    items = o.add_phone_rack(30, phone="Samsung Galaxy S23", stance="portrait")      # too long for a 300 mm bed in one piece
    assert len(items) >= 2 and all(i.nx * 42 <= 100 or i.ny * 42 <= 100 for i in items)
    assert [i.name for i in items] == [f"s23_rack_{k}" for k in range(1, len(items) + 1)]
    assert all(o.footprint_fits(i.nx * 42, i.ny * 42) for i in items)
    with pytest.raises(ValueError):
        PackoutOrganizer(bed=(60, 60, 200)).add_phone_rack(1, phone="Samsung Galaxy S23")


def test_phone_rack_lid_warning():
    o = PackoutOrganizer("Compact Organizer #48-22-8435")          # 99 mm interior; upright S23 is far taller
    o.add_phone_rack(2, phone="Samsung Galaxy S23", stance="portrait", angle=0)
    assert any("lid will not close" in w for w in o.layout())


# ---- half-cell floor use (baseplate-free) ----------------------------------------
def test_half_cell_grid_only_without_baseplates():
    on = PackoutOrganizer("Compact Organizer #48-22-8435", interlock=True)         # 203 x 305 mm interior
    off = PackoutOrganizer("Compact Organizer #48-22-8435")
    assert on.grid_cells == (4.5, 7.0) and off.grid_cells == (4.0, 7.0)
    assert on.leftover == pytest.approx((203.2 - 189.0, 304.8 - 294.0))
    assert (off.cells_x, off.cells_y) == (4, 7)                                    # baseplate tiles stay whole cells


def test_custom_shaped_block_reserves_half_cells_and_can_sit_on_a_half_cell(bar_stl):
    o = PackoutOrganizer("Compact Organizer #48-22-8435", bed=(256, 256, 256), interlock=True)
    cl_item = o.add_stl_cluster(bar_stl, 2, name="bars", use_watch=False)[0]
    w, h = o.check_part(cl_item.solid)["size"][:2]
    assert cl_item.ux * 21 >= w - 1e-6 and cl_item.uy * 21 >= h - 1e-6          # reserves at least its real footprint
    assert cl_item.ux <= 2 * cl_item.nx and cl_item.uy <= 2 * cl_item.ny        # in half cells: never more than whole cells
    grid_item = o.add_stl_on_base(bar_stl, name="plain")[0]
    assert grid_item.ux == 2 * grid_item.nx and grid_item.sockets
    cl_item.manual, cl_item.pos = True, (0.5, 0)
    o.layout()
    assert cl_item.pos == (0.5, 0) and grid_item.pos is not None
    assert all(float(c).is_integer() for c in grid_item.pos)                       # socketed blocks stay on whole cells
    assert "cells" in o.summary()


def test_clips_only_where_sockets_line_up():
    o = PackoutOrganizer("Compact Organizer #48-22-8435", interlock=True)
    import trimesh, tempfile, os
    d = tempfile.mkdtemp()
    p = os.path.join(d, "b.stl")
    trimesh.creation.box((30.0, 30.0, 10.0)).export(p)
    a = o.add_stl_on_base(p, name="a")[0]
    b = o.add_stl_on_base(p, name="b")[0]
    a.manual, a.pos = True, (0, 0)
    b.manual, b.pos = True, (a.nx, 0)
    o.layout()
    assert o.seam_key_count() == 1                      # aligned neighbours: one clip per shared cell edge
    b.pos = (a.nx, 0.5)                                 # half a cell off: sockets do not line up
    o.layout()
    assert o.seam_key_count() == 0


def test_unique_parts_merges_identical_and_exports_single_file(puck_stl):
    o = PackoutOrganizer(next(iter(PACKOUT_MODELS)))
    o.add_stl_insert(puck_stl, name="a")
    o.add_stl_insert(puck_stl, name="a_2")
    parts = o.build_parts()
    assert sum(k.startswith("dovetail_") for k in parts) <= 1
    uniq = o.unique_parts(parts)
    assert len({id(w) for _, w, _ in uniq}) == len(uniq)
    names = {n: q for n, _, q in uniq}
    assert max(names.values()) >= 1
    first = next(w for n, w, _ in uniq if not n.startswith(("baseplate", "dovetail")))
    stl = o.export_part(first, "stl")
    assert stl[:5] == b"solid" or len(stl) > 100
    assert o.export_part(first, "step")[:10].startswith(b"ISO-10303")
    with pytest.raises(ValueError):
        o.export_part(first, "obj")
