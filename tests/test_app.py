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

    button(at, "Add to project").click().run()
    assert len(at.session_state["scan_items"]) == 1

    button(at, "Generate project").click().run()
    assert not at.exception and not at.error
    parts = [s for s in at.selectbox if s.label == "Part"][0].options
    assert "scanned_tool" in parts


def test_scanner_tab_without_photo_shows_prompt():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert any("Upload a photo" in i.value for i in at.info)
