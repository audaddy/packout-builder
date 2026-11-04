"""Printer presets and round-bed tiling."""
import math

import pytest

from cad_engine import PackoutOrganizer
from src import printers as pr
from src.geometry import GRID


def test_exact_preset_list():
    assert list(pr.PRINTERS) == ["bambu_p2s", "prusa_xl", "prusa_core_one_l", "prusa_ht90"]
    assert pr.DEFAULT_PRINTER == "bambu_p2s"


@pytest.mark.parametrize("key,bed,rnd", [
    ("bambu_p2s", (256, 256, 256), False),
    ("prusa_xl", (360, 360, 360), False),
    ("prusa_core_one_l", (300, 300, 330), False),
    ("prusa_ht90", (300, 300, 400), True),          # round plate, diameter 300
])
def test_published_build_volumes(key, bed, rnd):
    p = pr.get_printer(key)
    assert p.bed == bed and p.round_bed is rnd


def test_labels_are_unique_and_resolve_back():
    labels = pr.printer_labels()
    assert len(set(labels)) == 4
    for p in pr.PRINTERS.values():
        assert pr.printer_by_label(p.label) is p
    assert "Ø300" in pr.get_printer("prusa_ht90").label and "round" in pr.get_printer("prusa_ht90").label
    assert pr.printer_by_label("Custom") is None


def test_unknown_printer():
    with pytest.raises(KeyError):
        pr.get_printer("ender3")


@pytest.mark.parametrize("key", list(pr.PRINTERS))
@pytest.mark.parametrize("model", ["Compact Organizer #48-22-8435", "Organizer #48-22-8430"])
def test_every_tile_fits_every_printer(key, model):
    p = pr.get_printer(key)
    org = PackoutOrganizer(model=model, bed=p.bed, bed_round=p.round_bed)
    tiles = org.plan_tiles()
    assert sum(t.nx * t.ny for t in tiles) == org.cells_x * org.cells_y
    for t in tiles:
        assert org.footprint_fits(t.nx * GRID, t.ny * GRID)
        if p.round_bed:
            assert math.hypot(t.nx * GRID, t.ny * GRID) <= p.x
        else:
            assert t.nx * GRID <= p.x and t.ny * GRID <= p.y


def test_round_bed_needs_smaller_tiles_than_a_square_bed_of_the_same_size():
    sq = PackoutOrganizer(model="Organizer #48-22-8430", bed=(300, 300, 400))
    rd = PackoutOrganizer(model="Organizer #48-22-8430", bed=(300, 300, 400), bed_round=True)
    assert len(rd.plan_tiles()) > len(sq.plan_tiles())
    assert max(t.nx * t.ny for t in rd.plan_tiles()) < max(t.nx * t.ny for t in sq.plan_tiles())


def test_round_bed_footprint_rule():
    org = PackoutOrganizer(bed=(300, 300, 400), bed_round=True)
    assert org.footprint_fits(210, 210)            # diagonal 297
    assert not org.footprint_fits(252, 168)        # diagonal 303 even though it fits a 300 square in one axis...
    assert not org.footprint_fits(300, 100)
    sq = PackoutOrganizer(bed=(300, 300, 400))
    assert sq.footprint_fits(300, 100)


def test_round_bed_rejects_a_block_that_overhangs_the_circle():
    org = PackoutOrganizer(bed=(300, 300, 400), bed_round=True)
    with pytest.raises(ValueError, match="exceeds the printer bed"):
        org._register("big", "stl", (None, 7, 7), 20.0, 1)       # 294 x 294 mm: diagonal 416 > 300


def test_summary_mentions_round_bed():
    org = PackoutOrganizer(bed=(300, 300, 400), bed_round=True)
    org.add_phone_slots(1, phone="Samsung Galaxy S23", height_mm=60.0, angle=60.0)
    org.build_parts()
    text = "\n".join(org.summary()) if not isinstance(org.summary(), str) else org.summary()
    assert "Ø300" in text and "round" in text


def test_round_bed_keeps_square_dimensions():
    org = PackoutOrganizer(bed=(300, 123, 400), bed_round=True)     # a stray Y is ignored for a round plate
    assert org.bed == (300.0, 300.0, 400.0)
