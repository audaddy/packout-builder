"""Generate a PLACEHOLDER watch-charger puck (not the real Galaxy Watch Ultra
charger).  Replace assets/sample_watch_charger.stl with the real STL.
Usage: python assets/make_sample_charger.py
"""
import pathlib
import trimesh

out = pathlib.Path(__file__).with_name("sample_watch_charger.stl")
disc = trimesh.creation.cylinder(radius=28.0, height=9.0, sections=96)
disc.apply_translation([0, 0, 4.5])
rim = trimesh.creation.cylinder(radius=23.0, height=2.0, sections=96)
rim.apply_translation([0, 0, 10.0])
mesh = trimesh.util.concatenate([disc, rim])
mesh = trimesh.boolean.union([disc, rim], engine=None) if False else mesh.convex_hull
mesh.export(out)
print("wrote", out, mesh.bounds.tolist(), "watertight:", mesh.is_watertight)
