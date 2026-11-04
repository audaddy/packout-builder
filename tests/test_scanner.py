"""Tests for the photo scanner using synthetic photos with known ground truth."""
import io
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box

from src import scanner as sc

RECT = affinity.rotate(box(-30, -15, 30, 15), 30)      # 60 x 30 mm rectangle turned 30 degrees
CENTRE = 100.0                                          # simulate_photo puts the item at the sheet centre


def truth(item):
    return affinity.translate(item, CENTRE, CENTRE)


@pytest.fixture(scope="module")
def rect_photo():
    return sc.simulate_photo(RECT, return_transform=True)


# --------------------------------------------------------------------------- sheet
def test_marker_layout_is_inside_workspace_and_disjoint():
    layout = sc.marker_layout()
    assert sorted(layout) == list(sc.MARKER_IDS)
    boxes = [box(x, y, x + sc.MARKER_MM, y + sc.MARKER_MM) for x, y in layout.values()]
    border = box(0, 0, sc.WORKSPACE_MM, sc.WORKSPACE_MM)
    assert all(border.contains(b) for b in boxes)
    assert all(not a.intersects(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    for mid in sc.MARKER_IDS:
        c = sc.marker_corners_mm(mid)
        assert c.shape == (4, 2)
        assert np.linalg.norm(c[1] - c[0]) == pytest.approx(sc.MARKER_MM)


@pytest.mark.parametrize("paper,w,h", [("A4", 595.28, 841.89), ("Letter", 612.0, 792.0)])
def test_calibration_pdf(paper, w, h):
    data = sc.calibration_sheet_pdf(paper)
    assert data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF")
    m = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", data)
    assert m and float(m.group(1)) == pytest.approx(w, abs=0.1) and float(m.group(2)) == pytest.approx(h, abs=0.1)
    assert data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= 1


def test_calibration_pdf_rejects_unknown_paper():
    with pytest.raises(ValueError):
        sc.calibration_sheet_pdf("A0")


def test_pdf_workspace_fits_on_both_papers():
    for w, h in sc.PAPER_SIZES.values():
        assert w >= sc.WORKSPACE_MM and h >= sc.WORKSPACE_MM + 60     # room for instructions/scale bar


# --------------------------------------------------------------------------- markers & scale
def test_detect_markers_corner_positions():
    ppm, margin = 4.0, 15.0
    found = sc.detect_markers(sc.render_calibration_image(ppm, margin))
    assert sorted(found) == [0, 1, 2, 3]
    for mid, corners in found.items():
        expected = (sc.marker_corners_mm(mid) + margin) * ppm
        assert np.abs(corners - expected).max() < 1.0            # sub-pixel/pixel accuracy


def test_detect_markers_in_very_large_image():
    """Markers of ~300 px (a 12 MP photo of a sheet) need the multi-scale search."""
    found = sc.detect_markers(sc.render_calibration_image(12.0, 10.0))
    assert sorted(found) == [0, 1, 2, 3]


@pytest.mark.parametrize("ppm", [3.0, 4.0, 6.0, 8.0])
def test_pixel_to_mm_ratio_of_flat_sheet(ppm):
    """1 source pixel = 1/ppm mm on an un-tilted render."""
    markers = sc.detect_markers(sc.render_calibration_image(ppm, 15.0))
    H, rms, n = sc.compute_homography(markers, 5.0)
    assert n == 16 and rms < 0.1
    assert 1.0 / sc._source_px_per_mm(H, 5.0) == pytest.approx(1.0 / ppm, rel=0.01)


def test_homography_matches_ground_truth_under_perspective(rect_photo):
    photo, M = rect_photo
    r = sc.scan_photo(photo)
    assert r.marker_ids == [0, 1, 2, 3]
    pts = np.array([(30, 30), (170, 40), (160, 170), (100, 100)], np.float64)
    photo_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M)
    back = cv2.perspectiveTransform(photo_pts, r.homography).reshape(-1, 2) / r.px_per_mm
    assert np.abs(back - pts).max() < 0.3                         # mm
    assert r.reproj_rms_mm < 0.3
    assert abs(r.scale_error_pct) < 1.0
    assert r.mm_per_px == pytest.approx(1 / r.px_per_mm)
    assert r.warped.shape[:2] == (1000, 1000)


@pytest.mark.parametrize("ppm", [4.0, 8.0])
def test_warped_resolution_follows_px_per_mm(rect_photo, ppm):
    r = sc.scan_photo(rect_photo[0], px_per_mm=ppm)
    assert r.warped.shape[0] == int(200 * ppm) and r.mm_per_px == pytest.approx(1 / ppm)
    assert r.polygon.area == pytest.approx(RECT.area, rel=0.03)


# --------------------------------------------------------------------------- contour extraction
@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_contour_matches_ground_truth(rect_photo, method):
    r = sc.scan_photo(rect_photo[0], method=method)
    t = truth(RECT)
    assert r.polygon.area == pytest.approx(t.area, rel=0.03)
    assert r.polygon.symmetric_difference(t).area / t.area < 0.04
    for a, b in zip(r.polygon.bounds, t.bounds):
        assert a == pytest.approx(b, abs=1.0)                     # within 1 mm everywhere
    assert not r.polygon.interiors
    assert r.warnings == []


@pytest.mark.parametrize("colour", [(70, 70, 70), (20, 200, 240), (180, 90, 30)])
def test_contour_for_different_item_colours(colour):
    r = sc.scan_photo(sc.simulate_photo(RECT, item_bgr=colour))
    assert r.polygon.area == pytest.approx(RECT.area, rel=0.04)


def test_contour_with_strong_lighting_gradient_and_noise():
    r = sc.scan_photo(sc.simulate_photo(RECT, gradient=0.35, noise_sigma=6, rotation_deg=-25, perspective=0.2))
    assert r.polygon.area == pytest.approx(RECT.area, rel=0.04)


def test_nonconvex_item_keeps_concavity():
    item = sc.demo_item_polygon()
    r = sc.scan_photo(sc.simulate_photo(item))
    t = truth(item)
    assert r.polygon.symmetric_difference(t).area / t.area < 0.05
    assert r.polygon.area < r.polygon.convex_hull.area * 0.95        # jaw/handle gap survived


def test_item_with_hole_is_kept_as_island():
    ring = box(-35, -20, 35, 20).difference(box(-10, -8, 10, 8))
    r = sc.scan_photo(sc.simulate_photo(ring))
    assert len(r.polygon.interiors) == 1
    assert r.polygon.area == pytest.approx(ring.area, rel=0.04)


# --------------------------------------------------------------------------- failure handling
def test_empty_sheet_has_no_object():
    with pytest.raises(sc.ScanError, match="No object"):
        sc.scan_photo(sc.simulate_photo(None))


def test_photo_without_markers():
    with pytest.raises(sc.ScanError, match="0 of 4"):
        sc.scan_photo(np.full((600, 800, 3), 180, np.uint8))


def _cover_markers(photo, M, ids):
    out = photo.copy()
    for i in ids:
        x0, y0 = sc.marker_layout()[i]
        quad = cv2.perspectiveTransform(np.array([[[x0 - 1, y0 - 1], [x0 + 26, y0 - 1], [x0 + 26, y0 + 26], [x0 - 1, y0 + 26]]], np.float64), M)
        cv2.fillPoly(out, [quad.astype(np.int32)], (255, 255, 255))
    return out


def test_one_marker_hidden_warns_but_works(rect_photo):
    r = sc.scan_photo(_cover_markers(*rect_photo, [3]))
    assert r.marker_ids == [0, 1, 2]
    assert any("Marker 3" in w for w in r.warnings)
    assert r.polygon.area == pytest.approx(RECT.area, rel=0.04)


def test_two_markers_hidden_fails(rect_photo):
    with pytest.raises(sc.ScanError, match="2 of 4"):
        sc.scan_photo(_cover_markers(*rect_photo, [0, 2]))


def test_item_touching_markers_warns():
    big = affinity.rotate(box(-92, -40, 92, 40), 20)
    r = sc.scan_photo(sc.simulate_photo(big))
    assert any("touches the markers" in w for w in r.warnings)


def test_bad_inputs():
    with pytest.raises(sc.ScanError):
        sc.scan_photo(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        sc.scan_photo(sc.simulate_photo(RECT), px_per_mm=0.5)
    with pytest.raises(ValueError):
        sc.segment_item(np.zeros((1000, 1000, 3), np.uint8), 5.0, method="magic")


def test_decode_image_roundtrip_and_garbage():
    img = sc.simulate_photo(RECT)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    dec = sc.decode_image(buf.tobytes())
    assert dec.shape == img.shape
    assert sc.scan_photo(dec).polygon.area == pytest.approx(RECT.area, rel=0.04)    # survives JPEG compression
    with pytest.raises(sc.ScanError):
        sc.decode_image(b"not an image")


# --------------------------------------------------------------------------- polygon processing
def test_smooth_polygon_simplifies_noisy_outline():
    rng = np.random.default_rng(1)
    ang = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    r = 30 + rng.normal(0, 0.12, ang.size)
    noisy = Polygon(np.c_[r * np.cos(ang), r * np.sin(ang)])
    smooth = sc.smooth_polygon(noisy)
    assert len(smooth.exterior.coords) < len(noisy.exterior.coords) / 2
    assert smooth.area == pytest.approx(np.pi * 30 ** 2, rel=0.01)
    assert smooth.is_valid
    # smoother than the input: much less boundary length wasted on jitter
    assert smooth.length < noisy.length * 0.9 and smooth.length == pytest.approx(2 * np.pi * 30, rel=0.01)


def test_smoothing_rounds_corners_by_sigma_not_by_edge_length():
    """Regression: corner-cutting schemes shaved ~16% off a 60x30 rectangle."""
    r = box(0, 0, 60, 30)
    s = sc.smooth_polygon(r, sigma_mm=0.5)
    assert s.area == pytest.approx(r.area, rel=0.005)
    assert s.bounds == pytest.approx(r.bounds, abs=0.2)
    longer = sc.smooth_polygon(box(0, 0, 160, 30), sigma_mm=0.5)
    assert longer.area == pytest.approx(160 * 30, rel=0.005)


def test_smooth_polygon_drops_tiny_holes_keeps_big_ones():
    p = box(0, 0, 80, 40).difference(box(10, 10, 12, 12)).difference(box(30, 10, 60, 30))
    out = sc.smooth_polygon(p, min_hole_area_mm2=40)
    assert len(out.interiors) == 1


def test_expand_polygon_adds_clearance_on_every_side():
    r = box(0, 0, 60, 30)
    e = sc.expand_polygon(r, 0.6)
    assert e.bounds == pytest.approx((-0.6, -0.6, 60.6, 30.6), abs=1e-6)
    assert sc.expand_polygon(r, 0) is r
    with pytest.raises(ValueError):
        sc.expand_polygon(r, -1)


def test_expand_polygon_shrinks_holes():
    p = box(0, 0, 60, 40).difference(box(20, 10, 40, 30))
    e = sc.expand_polygon(p, 1.0)
    assert Polygon(e.interiors[0]).area < Polygon(p.interiors[0]).area


def test_finger_scoops_bite_out_of_sides():
    r = box(0, 0, 60, 30)
    s = sc.add_finger_scoops(r, 10, ("left", "right"))
    assert s.bounds == pytest.approx((-10, 0, 70, 30), abs=0.05)
    assert s.area == pytest.approx(r.area + math_pi_half(10) * 2, rel=0.02)
    t = sc.add_finger_scoops(r, 8, ("top", "bottom"))
    assert t.bounds == pytest.approx((0, -8, 60, 38), abs=0.05)
    with pytest.raises(ValueError):
        sc.add_finger_scoops(r, 10, ("diagonal",))
    with pytest.raises(ValueError):
        sc.add_finger_scoops(r, 0)


def math_pi_half(r):
    return np.pi * r * r / 2


def test_finger_scoop_follows_tapered_outline():
    tri = Polygon([(0, 0), (60, 10), (60, 20), (0, 30)])            # left side at x=0, right narrower
    s = sc.add_finger_scoops(tri, 6, ("right",))
    assert s.bounds[2] == pytest.approx(66, abs=0.1)


def test_prepare_cutout_pipeline_and_print_scale():
    base = sc.prepare_cutout(RECT, clearance_mm=0.6)
    assert base.area == pytest.approx(RECT.buffer(0.6).area, rel=0.01)
    scaled = sc.prepare_cutout(RECT, clearance_mm=0.0, print_scale=0.98)
    assert scaled.area == pytest.approx(RECT.area * 0.98 ** 2, rel=0.02)
    with_scoops = sc.prepare_cutout(RECT, 0.6, scoops=True, scoop_radius_mm=10)
    assert with_scoops.area > base.area
    with pytest.raises(ValueError):
        sc.prepare_cutout(RECT, print_scale=0)


def test_center_polygon():
    c = sc.center_polygon(box(10, 20, 30, 60))
    assert c.bounds == pytest.approx((-10, -20, 10, 20))


# --------------------------------------------------------------------------- 2D export
def test_svg_export_is_real_world_millimetres():
    poly = box(0, 0, 60, 30).difference(box(20, 10, 30, 20))
    svg = sc.polygon_to_svg(poly, margin_mm=2.0)
    root = ET.fromstring(svg)
    assert root.attrib["width"] == "64.000mm" and root.attrib["height"] == "34.000mm"
    assert root.attrib["viewBox"] == "0 0 64.000 34.000"
    d = root[0].attrib["d"]
    assert d.count("M") == 2 and d.count("Z") == 2               # outline + island
    ys = [float(m) for m in re.findall(r",(-?[\d.]+)", d)]
    assert min(ys) == pytest.approx(2.0) and max(ys) == pytest.approx(32.0)


def test_svg_export_flips_y_to_match_top_down_view():
    tri = Polygon([(0, 0), (20, 0), (10, 30)])                    # apex up in CAD
    d = ET.fromstring(sc.polygon_to_svg(tri, margin_mm=0))[0].attrib["d"]
    pts = [tuple(map(float, p.split(","))) for p in re.findall(r"(-?[\d.]+,-?[\d.]+)", d)]
    apex = [p for p in pts if abs(p[0] - 10) < 1e-6][0]
    assert apex[1] == pytest.approx(0.0)                          # apex at the TOP of the SVG (y=0)


def test_dxf_export_roundtrip():
    import ezdxf
    poly = box(5, 5, 65, 35).difference(box(20, 15, 30, 25))
    doc = ezdxf.read(io.StringIO(sc.polygon_to_dxf(poly).decode()))
    lines = list(doc.modelspace().query("LWPOLYLINE"))
    assert len(lines) == 2 and all(e.closed for e in lines)
    assert {e.dxf.layer for e in lines} == {"CUT", "HOLE"}
    assert doc.units == ezdxf.units.MM
    pts = np.array([p[:2] for p in lines[0].get_points()] if lines[0].dxf.layer == "CUT" else [p[:2] for p in lines[1].get_points()])
    assert pts.min(axis=0) == pytest.approx((0, 0)) and pts.max(axis=0) == pytest.approx((60, 30))
