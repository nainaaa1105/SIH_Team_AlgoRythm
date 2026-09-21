# Archive

## `dashboard-cesium-legacy.html`

The dashboard that `static/index.html` replaced: a CesiumJS 3D globe with
its own operator UI.

It is kept for reference only, and deliberately **outside `static/`**, because
the gateway mounts `static/` at `/` with `StaticFiles(html=True)` — anything
left in that directory is publicly served. This page carries a bundled
`HOTSPOTS` sample array and falls back to it whenever the backend is
unreachable, so serving it would have put fabricated detections back on a
reachable URL.

The current dashboard has no bundled dataset and no client-side classifier.
When the gateway is unreachable or the database is empty it says so and
shows nothing.
