"""Headless UI smoke test (Streamlit AppTest) for the photo scanner flow."""
import os

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def button(at, label):
    return [b for b in at.button if b.label == label][0]


def test_scanner_tab_demo_photo_to_generated_project():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    [c for c in at.checkbox if "demo photo" in c.label][0].check()
    at.run()
    assert not at.exception and not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Markers"] == "4 / 4"
    assert metrics["Item size"].startswith("153")                     # demo pliers are ~153 x 66 mm

    button(at, "Preview block").click().run()
    assert not at.exception and not at.error
    assert at.session_state["scan_preview"] is not None
    assert {"stl", "svg", "dxf"} <= set(at.session_state["scan_preview"])

    # CNC panel for the previewed block
    cm = {m.label: m.value for m in at.metric}
    assert cm["Stock to cut"].endswith("mm") and cm["Est. run time"].endswith("min")
    assert not at.warning or all("flute" in w.value or "tabs" in w.value for w in at.warning)

    button(at, "Add to project").click().run()
    assert len(at.session_state["scan_items"]) == 1

    button(at, "Generate project").click().run()
    assert not at.exception and not at.error
    parts = [s for s in at.selectbox if s.label == "Part"][0].options
    assert "scanned_tool" in parts


def test_scanner_tab_without_photo_shows_prompt():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert any("Upload a photo" in i.value for i in at.info)


def test_cnc_panel_reacts_to_bit_and_reports_errors():
    at = AppTest.from_file(APP, default_timeout=240).run()
    [c for c in at.checkbox if "demo photo" in c.label][0].check()
    at.run()
    button(at, "Preview block").click().run()
    at.number_input(key="cnc_depth").set_value(25.0).run()                 # leaves a 0.4 mm floor in 25.4 mm stock
    assert not at.exception
    assert any("floor" in e.value for e in at.error)
    at.number_input(key="cnc_depth").set_value(12.0).run()
    assert not at.error
    [s for s in at.selectbox if s.label == "Bit (flat endmill)"][0].select("1/8 in (3.175 mm)").run()
    assert not at.exception


def test_printer_presets_in_ui():
    at = AppTest.from_file(APP, default_timeout=120).run()
    presets = [s for s in at.selectbox if s.label == "Preset"][0]
    assert len(presets.options) == 5 and presets.options[-1] == "Custom"
    assert [o.split(" (")[0] for o in presets.options[:4]] == ["Bambu Lab P2S", "Prusa XL", "Prusa CORE One L", "Prusa HT90"]
    assert (at.session_state["bed_x"], at.session_state["bed_z"]) == (256.0, 256.0)         # P2S default
    presets.select(presets.options[3]).run()
    assert at.session_state["bed_round"] is True and at.session_state["bed_x"] == 300.0 and at.session_state["bed_z"] == 400.0
    assert any(n.label == "Bed diameter (mm)" for n in at.number_input)
    button(at, "Generate project").click().run()
    assert not at.exception and not at.error
    presets = [s for s in at.selectbox if s.label == "Preset"][0]          # fresh handle after each run
    presets.select(presets.options[1]).run()
    assert at.session_state["bed_round"] is False and at.session_state["bed_y"] == 360.0


def test_layout_flags_stale_project_and_unplaced_blocks():
    at = AppTest.from_file(APP, default_timeout=240).run()
    button(at, "Generate project").click().run()
    assert not at.exception
    assert not any("changed since" in w.value for w in at.warning)
    # a block added after generating must not leave the old layout on screen
    [c for c in at.checkbox if "demo photo" in c.label][0].check()
    at.run()
    button(at, "Preview block").click().run()
    button(at, "Add to project").click().run()
    at.run()
    assert any("changed since" in w.value for w in at.warning)
    assert "Layout canvas" not in [s.value for s in at.subheader]
    # regenerating shows it; the default 2 phones + hub + a 4x2 scanned tool leave nothing free for a 2x2 block
    button(at, "Generate project").click().run()
    assert not any("changed since" in w.value for w in at.warning)
    assert "scanned_tool" in [s for s in at.selectbox if s.label == "Part"][0].options
