# Packout insert builder

Parametric, 3D-printable Gridfinity-style inserts for Milwaukee Packout organizers (CadQuery).

    python -m pip install -r requirements.txt
    streamlit run app.py          # web UI
    python -m pytest              # unit tests

* `src/printers.py` – printer presets: Bambu Lab P2S, Prusa XL, Prusa CORE One L, Prusa HT90 (round Ø300 bed; tiles must fit the circle diagonally) or Custom.
* `src/cnc.py` – Shapeoko 4 XL G-code for scanned-tool pockets (see below).
* Phone racks (tab 2): all phones go into **one rack block** after the Maker Club phone rack - slots side by side with 5 mm separators, a holder low enough that ~40% of each phone sticks out, a cable groove under each phone, arch-topped airflow windows and finger scoops. Any number of phones: the grid grows with the count, and a rack longer than the printer bed is split into the fewest equal racks that fit. Style *Auto* picks upright, on its long edge (low), or flat single slots so the phones clear the lid.
* `src/phone_db.py` – phone presets (Galaxy S23 / S23+ / S23 Ultra) + `register_phone()` / JSON loading (`assets/phones_custom.json`).
* `src/packout_db.py`, `src/assets/packout_images.py` – case specs and offline SVG thumbnails.
* `src/theme.py`, `.streamlit/config.toml` – Milwaukee red/black dark theme (restart Streamlit after editing config.toml). Font is Helvetica Neue (Milwaukee's site font; system font on macOS/iOS, Helvetica/Arial elsewhere).
* `src/scanner.py` – photo scanner: calibration-sheet PDF, ArUco homography, silhouette extraction, shapely smoothing / clearance / finger scoops, SVG and DXF profile export.
* `src/geometry.py` – CadQuery builders (custom STL blocks default to a *drop-in outline* pocket – the part's top-down silhouette cut straight down, so stands/hooks/overhangs don't fill the block; `pocket="exact"` gives the 3D negative): phone slot, STL insert, scanned-contour block, USB-C hub, baseplate tile, dovetail key.
* `cad_engine.py` – `PackoutOrganizer`: tiling to the printer bed, layout, validation, STL/STEP zip export.
* `app.py` – Streamlit UI (layout view, 3D preview, print checks, export).
* `assets/sample_watch_charger.stl` is a **placeholder puck**; upload the real Galaxy Watch Ultra charger STL (Z-up) in the UI.

Notes: by default (**Baseplate-free** option in tab 1) there are no baseplates: every block has a flat underside with a half dovetail
socket at the middle of each cell edge, and blocks that touch are locked together with small loose bow-tie clips (the app counts
them from the layout). Untick the option for Gridfinity feet on printed baseplate tiles joined by dovetail keys.

Custom STL tab: **Object on a base** fuses the model itself to a grid base (print it as one piece); **Holder block** cuts the model
out of a block (drop-in outline pocket by default, or the exact 3D negative).
Case interiors come from Milwaukee's published specs and exclude dividers – use Custom to override.
STL tolerance offset is a miter-compensated vertex offset (approximate on very sharp or thin features).

## Photo tool scanner

1. In the **📷 Photo Tool Scanner** tab, download the calibration sheet (A4 or Letter, PDF or PNG) and print it at 100%. Check the 200 mm border and the 100 mm scale bar with a ruler; if they are off, enter the measured border size under *Scan settings*.
2. Lay the item flat on the sheet and photograph it from above (tilt is fine). All four corner markers must be visible, the lighting should be even, and the item must contrast with the white paper.
3. Check the red edge overlay, set depth / clearance / grid, preview the block, then **Add to project**. The flat profile can be downloaded as SVG or DXF, and is also written to `profiles/` in the export zip.

Accuracy on synthetic photos with known ground truth: outline area within about 1%, marker fit error about 0.1 mm. Hard shadows are detected as part of the item, so use diffuse light. Needs OpenCV 4.x (`opencv-python>=4.8,<5`).

## CNC (Shapeoko 4 XL)

After **Preview block** in the scanner tab, open *CNC program (Shapeoko 4 XL)*: pick material, bit and stock thickness,
check the toolpath preview and download the `.nc` file.

* Mills the scanned outline as a flat-bottom pocket (islands stay standing) in Z levels with concentric passes, a 3 degree ramped
  entry and climb milling; optionally through-cuts the block out of the stock with holding tabs.
* Work zero in Carbide Motion: X0 Y0 = lower-left corner of the stock, Z0 = top of the stock.
* Plain G0/G1, mm, absolute. `M6 T1` can be switched off for senders that reject it.
* Feeds, speeds and stepdown are conservative starting points for a 1/4 in bit, scaled for smaller bits. Test on scrap first.
* The bit cannot reach narrow gaps and sharp inside corners; the panel reports how much of the outline that is.
* Machine limits come from Carbide 3D: 33 x 17.5 x 4 in cutting area.
