"""Shapeoko 4 XL G-code: parse the program back and check what it would actually cut."""
import re

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from src import cnc
from src.cnc import CncError, CncJob, generate_gcode


def parse(text):
    """-> list of (cmd, x, y, z, f) with modal XYZ/F carried forward; comments split the program."""
    pos = [None, None, None]
    f = None
    out = []
    for line in text.splitlines():
        if line.startswith("("):
            out.append(("comment", line, None, None, None))
            continue
        words = dict(re.findall(r"([A-Z])(-?\d+\.?\d*)", line))
        cmd = line.split()[0]
        for i, a in enumerate("XYZ"):
            if a in words:
                pos[i] = float(words[a])
        if "F" in words:
            f = float(words["F"])
        out.append((cmd, *pos, f))
    return out


def sections(text):
    """Split the move list at the 'Cut-out' comment: (pocket_moves, cutout_moves)."""
    moves = parse(text)
    idx = next((i for i, m in enumerate(moves) if m[0] == "comment" and m[1].startswith("(Cut-out:")), len(moves))
    return moves[:idx], moves[idx:]


def cutting(moves):
    return [m for m in moves if m[0] == "G1"]


@pytest.fixture(scope="module")
def shape():
    # 60 x 40 mm pocket with a 12 x 12 mm island, in a 3 x 2 cell (126 x 84 mm) block
    p = box(30, 22, 90, 62).difference(box(54, 36, 66, 48))
    return p, (126.0, 84.0)


@pytest.fixture(scope="module")
def result(shape):
    prof, size = shape
    return generate_gcode(prof, size, CncJob(name="t", pocket_depth=12.0, stock_thickness=25.4))


# --------------------------------------------------------------------------- machine
def test_shapeoko_4_xl_cutting_area():
    m = cnc.MACHINES["shapeoko_4_xl"]
    assert m.name == "Shapeoko 4 XL"
    assert (m.x, m.y, m.z) == pytest.approx((838.2, 444.5, 101.6))        # 33 x 17.5 x 4 in


# --------------------------------------------------------------------------- program structure
def test_program_header_and_footer(result):
    lines = result.gcode.splitlines()
    cmds = [ln for ln in lines if not ln.startswith("(")]
    assert cmds[:4] == ["G21", "G90", "G17", "G94"]
    assert "M3 S18000" in cmds and "M6 T1" in cmds
    assert cmds[-3:] == ["G0 Z5", "G0 X0 Y0", "M30"] and "M5" in cmds
    assert not any(c.split()[0] in ("G2", "G3", "G81", "G82") for c in cmds)    # plain G0/G1 only


def test_tool_change_can_be_omitted(shape):
    prof, size = shape
    r = generate_gcode(prof, size, CncJob(pocket_depth=10, tool_change=False, cut_out=False))
    assert "M6" not in r.gcode


def test_all_moves_stay_on_the_stock_and_machine(result):
    W, H, T = result.stock
    m = cnc.MACHINES["shapeoko_4_xl"]
    for cmd, x, y, z, f in parse(result.gcode):
        if cmd not in ("G0", "G1"):
            continue
        if x is not None:
            assert -1e-6 <= x <= W + 1e-6 and x <= m.x
        if y is not None:
            assert -1e-6 <= y <= H + 1e-6 and y <= m.y
        assert z is None or (-(T + 0.31) <= z <= 5.0 + 1e-6)


def test_cutting_moves_always_have_a_feed_rate(result):
    assert all(m[4] and m[4] > 0 for m in cutting(parse(result.gcode)))
    assert max(m[4] for m in cutting(parse(result.gcode))) <= 1500.0


# --------------------------------------------------------------------------- pocket
def test_pocket_reaches_depth_and_never_goes_deeper(result):
    pocket, _ = sections(result.gcode)
    zs = [m[3] for m in cutting(pocket)]
    assert min(zs) == pytest.approx(-12.0, abs=1e-3)
    assert result.layers == 4                                              # 12 mm / 3 mm


def test_pocket_clears_the_outline_and_stays_inside_it(shape, result):
    prof, _ = shape
    r = 6.35 / 2
    pocket, _ = sections(result.gcode)
    final = [m for m in cutting(pocket) if m[3] == pytest.approx(-12.0, abs=1e-3)]
    pts = np.array([[m[1], m[2]] for m in final])
    # the program's coordinates have the block at (margin, margin)
    placed = Polygon(prof.exterior.coords, [i.coords for i in prof.interiors])
    placed = __import__("shapely.affinity", fromlist=["x"]).translate(placed, 15, 15)
    swept = unary_union([LineString(pts[i:i + 2]).buffer(r) for i in range(len(pts) - 1)
                         if np.hypot(*(pts[i + 1] - pts[i])) < 5])       # skip the jumps between loops
    gouge = swept.difference(placed.buffer(0.02)).area
    uncut = placed.difference(swept).area
    assert gouge < 0.5                                                     # never cuts the walls or the island
    assert uncut / placed.area < 0.02                                      # only the inside-corner radii remain
    assert result.uncut_pct == pytest.approx(100 * uncut / placed.area, abs=0.5)


def test_island_is_left_standing(shape, result):
    prof, _ = shape
    pocket, _ = sections(result.gcode)
    pts = [(m[1], m[2]) for m in cutting(pocket) if m[3] == pytest.approx(-12.0, abs=1e-3)]
    island = box(54 + 15, 36 + 15, 66 + 15, 48 + 15)                       # in program coordinates
    from shapely.geometry import Point
    assert not any(island.contains(Point(p)) for p in pts)


def test_each_level_starts_with_a_ramp_not_a_straight_plunge(result):
    pocket, _ = sections(result.gcode)
    cuts = cutting(pocket)
    # the first moves after the entry are XY+Z moves that go down slowly
    first = [m for m in cuts[:40] if m[1] is not None and m[3] < -0.01]
    assert first and first[0][4] == 500.0
    slopes = []
    for a, b in zip(cuts[1:30], cuts[2:31]):
        d = np.hypot(b[1] - a[1], b[2] - a[2])
        if d > 0.1 and b[3] < a[3]:
            slopes.append((a[3] - b[3]) / d)
    assert slopes and max(slopes) < np.tan(np.radians(6))                 # ~3 deg ramp, never steeper than 6


def test_climb_milling_direction(result):
    outer = [p for p in result.pocket_paths if len(p) > 5]
    def signed_area(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    areas = sorted(signed_area(p) for p in outer)
    assert areas[0] < 0 < areas[-1]                                        # island loops clockwise, outer loops counter-clockwise


# --------------------------------------------------------------------------- cut-out and tabs
def test_cutout_goes_through_and_leaves_tabs(result):
    _, cut = sections(result.gcode)
    zs = [m[3] for m in cutting(cut)]
    assert min(zs) == pytest.approx(-(25.4 + 0.3), abs=1e-3)
    tab_top = -(25.4 + 0.3) + 2.5
    last_level = [m for m in cut if m[0] == "comment" and "level" in m[1]][-1]
    seq = cut[cut.index(last_level):]
    levels = [round(m[3], 3) for m in cutting(seq)]
    raised = [z == pytest.approx(tab_top, abs=1e-3) for z in levels]
    transitions = sum(1 for a, b in zip(raised, raised[1:]) if a != b)
    assert transitions == 8                                                # 4 tabs = 4 up + 4 down
    assert "4 tabs" in " ".join(result.warnings)


def test_cutout_path_is_the_block_outline_offset_by_the_bit(result):
    r = 6.35 / 2
    loop = result.cutout_paths[0]
    xs, ys = loop[:, 0], loop[:, 1]
    assert xs.min() == pytest.approx(15 - r, abs=0.02) and xs.max() == pytest.approx(15 + 126 + r, abs=0.02)
    assert ys.min() == pytest.approx(15 - r, abs=0.02) and ys.max() == pytest.approx(15 + 84 + r, abs=0.02)
    area = 0.5 * np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1])
    assert area < 0                                                        # clockwise = climb on an outside cut


def test_no_cutout_option(shape):
    prof, size = shape
    r = generate_gcode(prof, size, CncJob(pocket_depth=10, cut_out=False))
    assert "Cut-out" not in r.gcode and r.cutout_paths == []


# --------------------------------------------------------------------------- settings and estimates
def test_material_defaults_scale_with_bit_size():
    big = cnc.material_defaults("MDF / plywood", 6.35)
    small = cnc.material_defaults("MDF / plywood", 3.175)
    assert small["feed"] < big["feed"] and small["stepdown"] < big["stepdown"]
    assert big["feed"] == 1500.0 and big["stepdown"] == 3.0
    with pytest.raises(CncError):
        cnc.material_defaults("unobtainium", 6.35)


def test_estimates_and_stock_size(result):
    assert result.stock == pytest.approx((156.0, 114.0, 25.4))
    assert result.block_origin == (15.0, 15.0)
    assert 1 < result.minutes < 120 and result.cut_length_mm > 1000


def test_deeper_pocket_than_flute_warns(shape):
    prof, size = shape
    r = generate_gcode(prof, size, CncJob(pocket_depth=22.0, stock_thickness=30.0))
    assert any("flute" in w for w in r.warnings)


def test_smaller_bit_reaches_more(shape):
    prof, size = shape
    thin = box(30, 22, 90, 62).difference(box(30, 40, 80, 44))            # forms a 4 mm slot gap
    a = generate_gcode(thin, size, CncJob(tool_diameter=6.35, pocket_depth=6, cut_out=False))
    b = generate_gcode(thin, size, CncJob(tool_diameter=3.175, pocket_depth=6, cut_out=False))
    assert b.uncut_area_mm2 <= a.uncut_area_mm2


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("kw,msg", [
    (dict(stock_thickness=20.0, pocket_depth=19.5), "floor"),
    (dict(stock_thickness=120.0, pocket_depth=20.0), "Z travel"),
    (dict(tool_diameter=0), "tool diameter"),
    (dict(tool_diameter=70.0, cut_out=False), "wider"),
    (dict(margin=5.0), "margin"),
    (dict(stepover_pct=0), "stepover"),
    (dict(pocket_depth=0), "depth"),
])
def test_bad_jobs_are_rejected(shape, kw, msg):
    prof, size = shape
    with pytest.raises(CncError, match=msg):
        generate_gcode(prof, size, CncJob(**kw))


def test_stock_larger_than_machine_is_rejected(shape):
    prof, _ = shape
    with pytest.raises(CncError, match="cutting area"):
        generate_gcode(prof, (900.0, 900.0), CncJob(cut_out=False))


def test_stock_that_only_fits_turned_warns(shape):
    prof, _ = shape
    r = generate_gcode(prof, (400.0, 800.0), CncJob(pocket_depth=6, cut_out=False))
    assert any("90 degrees" in w for w in r.warnings)


def test_real_block_end_to_end():
    """The scanned-tool block from the geometry module drives the CNC job unchanged."""
    from src.geometry import GRID, create_scanned_contour_block
    from shapely import affinity
    base = Polygon([(0, 0), (70, 0), (70, 25), (30, 25), (30, 60), (0, 60)])   # L-shaped tool outline
    blk, nx, ny, info = create_scanned_contour_block(affinity.translate(base, 20, 20), None, None, 15.0, return_info=True)
    r = generate_gcode(info["profile"], (nx * GRID, ny * GRID), CncJob(name="l_tool", pocket_depth=15.0, stock_thickness=25.4))
    assert r.stock[0] == nx * GRID + 30 and "M30" in r.gcode and r.uncut_pct < 3
