import json

import pytest

from src import packout_db as pk
from src import phone_db as ph
from src.assets import packout_images as img


def test_phone_presets_match_spec():
    assert ph.get_phone("Samsung Galaxy S23").dims == (146.3, 70.9, 7.6)
    assert ph.get_phone("samsung galaxy s23+").dims == (157.8, 76.2, 7.6)
    assert ph.get_phone("s23_ultra").dims == (163.4, 78.1, 8.9)
    assert all(ph.get_phone(n).clearance == 0.6 for n in ph.list_phones())


def test_phone_lookup_error_suggests():
    with pytest.raises(KeyError, match="did you mean"):
        ph.get_phone("Samsung Galaxy S23 Ultr")


def test_register_and_json_roundtrip(tmp_path):
    ph.register_phone("Test Phone", 150, 70, 8, clearance=0.8)
    assert ph.get_phone("test phone").clearance == 0.8
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"name": "JSON Phone", "length": 160, "width": 75, "thickness": 9}]))
    assert ph.load_phones_json(f) == 1
    assert ph.get_phone("JSON Phone").dims == (160, 75, 9)
    assert ph.get_phone("JSON Phone").clearance == ph.DEFAULT_CLEARANCE
    for k in ("test_phone", "json_phone"):
        ph.PHONES.pop(k)


def test_phone_validation():
    with pytest.raises(ValueError):
        ph.register_phone("Bad", 0, 70, 8)
    with pytest.raises(ValueError):
        ph.register_phone("Bad", 150, 70, 8, clearance=9)


def test_case_db():
    labels = pk.case_labels()
    assert len(labels) == 5 and all("#48-22-" in l for l in labels)
    c = pk.get_case("48-22-8435")
    assert c is pk.get_case("8435") is pk.get_case(c.label)
    assert c.interior_mm == pytest.approx((203.2, 304.8, 99.06))
    L, W, H = c.dims_ldh_in
    assert (L, W) == (12.0, 8.0)
    assert pk.get_case("8431").interior_in[2] < pk.get_case("8430").interior_in[2] < pk.get_case("8432").interior_in[2]
    with pytest.raises(KeyError):
        pk.get_case("9999")


def test_footprint():
    f = pk.footprint(pk.get_case("8430").interior_mm)
    assert (f["cells_x"], f["cells_y"]) == (10, 7)
    assert f["leftover_x"] == pytest.approx(457.2 - 420)
    assert 0 < f["usable_pct"] <= 100


@pytest.mark.parametrize("case", list(pk.CASES.values()) + [None])
def test_thumbnails_are_valid_svg(case):
    import xml.etree.ElementTree as ET
    svg = pk.thumbnail_svg(case, selected=True)
    assert ET.fromstring(svg).tag.endswith("svg")
    assert img.svg_data_uri(svg).startswith("data:image/svg+xml;base64,")
