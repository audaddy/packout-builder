# Packout insert builder

Parametric, 3D-printable Gridfinity-style inserts for Milwaukee Packout organizers (CadQuery).

    python -m pip install -r requirements.txt
    streamlit run app.py          # web UI
    python -m pytest              # unit tests

* `src/phone_db.py` – phone presets (Galaxy S23 / S23+ / S23 Ultra) + `register_phone()` / JSON loading (`assets/phones_custom.json`).
* `src/packout_db.py`, `src/assets/packout_images.py` – case specs and offline SVG thumbnails.
* `src/theme.py`, `.streamlit/config.toml` – Milwaukee red/black dark theme (restart Streamlit after editing config.toml).
* `src/geometry.py` – CadQuery builders: phone slot, STL insert, USB-C hub, baseplate tile, dovetail key.
* `cad_engine.py` – `PackoutOrganizer`: tiling to the printer bed, layout, validation, STL/STEP zip export.
* `app.py` – Streamlit UI (layout view, 3D preview, print checks, export).
* `assets/sample_watch_charger.stl` is a **placeholder puck**; upload the real Galaxy Watch Ultra charger STL (Z-up) in the UI.

Notes: blocks sit on baseplate tiles via Gridfinity feet; tiles join with printed bow-tie dovetail keys.
Case interiors come from Milwaukee's published specs and exclude dividers – use Custom to override.
STL tolerance offset is a miter-compensated vertex offset (approximate on very sharp or thin features).
