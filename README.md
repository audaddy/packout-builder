# Packout insert builder

Parametric, 3D-printable Gridfinity-style inserts for Milwaukee Packout organizers (CadQuery).

    python -m pip install -r requirements.txt
    streamlit run app.py          # web UI
    python -m pytest              # unit tests

* `src/phone_db.py` – phone presets (Galaxy S23 / S23+ / S23 Ultra) + `register_phone()` / JSON loading (`assets/phones_custom.json`).
* `src/packout_db.py`, `src/assets/packout_images.py` – case specs and offline SVG thumbnails.
* `src/theme.py`, `.streamlit/config.toml` – Milwaukee red/black dark theme (restart Streamlit after editing config.toml).
* `src/scanner.py` – photo scanner: calibration-sheet PDF, ArUco homography, silhouette extraction, shapely smoothing / clearance / finger scoops, SVG and DXF profile export.
* `src/geometry.py` – CadQuery builders: phone slot, STL insert, scanned-contour block, USB-C hub, baseplate tile, dovetail key.
* `cad_engine.py` – `PackoutOrganizer`: tiling to the printer bed, layout, validation, STL/STEP zip export.
* `app.py` – Streamlit UI (layout view, 3D preview, print checks, export).
* `assets/sample_watch_charger.stl` is a **placeholder puck**; upload the real Galaxy Watch Ultra charger STL (Z-up) in the UI.

Notes: blocks sit on baseplate tiles via Gridfinity feet; tiles join with printed bow-tie dovetail keys.
Case interiors come from Milwaukee's published specs and exclude dividers – use Custom to override.
STL tolerance offset is a miter-compensated vertex offset (approximate on very sharp or thin features).

## Photo tool scanner

1. In the **📷 Photo Tool Scanner** tab, download the calibration sheet (A4 or Letter) and print it at 100%. Check the 200 mm border and the 100 mm scale bar with a ruler; if they are off, enter the measured border size under *Scan settings*.
2. Lay the item flat on the sheet and photograph it from above (tilt is fine). All four corner markers must be visible, the lighting should be even, and the item must contrast with the white paper.
3. Check the red edge overlay, set depth / clearance / grid, preview the block, then **Add to project**. The flat profile can be downloaded as SVG or DXF, and is also written to `profiles/` in the export zip.

Accuracy on synthetic photos with known ground truth: outline area within about 1%, marker fit error about 0.1 mm. Hard shadows are detected as part of the item, so use diffuse light. Needs OpenCV 4.x (`opencv-python>=4.8,<5`).
