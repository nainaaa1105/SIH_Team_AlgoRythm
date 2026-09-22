"""Test-wide fixtures.

The geo cache is disabled for the whole suite. It is an on-disk SQLite
store keyed by rounded coordinates, so without this a test that patches
`search_least_cloudy_scene` (or the WorldCover reader, or Open-Meteo) can
be handed a value another test recorded minutes earlier and never call
its own mock at all. That made `test_cloud_gate_cdse` fail in ways that
depended on test execution order, which is exactly the kind of flake a
cache should never introduce.
"""
import os

os.environ.setdefault("FIRESIGHT_GEO_CACHE_DISABLED", "1")
