"""Member 6 — API gateway and dashboard aggregation layer.

Architecture Layer 3 is M6's: the FastAPI gateway that the client talks
to. M1 built the app and its own three routers but never mounted M2-M5's,
and the WebSocket channel in the plan was never implemented. This package
does both, plus the aggregation endpoints the dashboard needs so the
event card is one request instead of five.
"""
