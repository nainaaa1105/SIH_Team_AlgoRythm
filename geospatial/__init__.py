"""Member 3 — facility attribution, land-cover context, plume dispersion,
threat corridors and evacuation routing.

Package is named `geospatial`, not `app`: it runs alongside Member 1's
`app` package and Member 2's `classifier` package, importing M1's DB
models, config and Celery app rather than duplicating them. (Two
packages named `app` on one sys.path shadow each other — that cost real
debugging time during M2's build.)
"""
