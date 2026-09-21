"""Whether a cluster currently reads as "Active" — one definition,
shared by `gateway.routes_dashboard._status_for` (the dashboard's
Active/Under Control/Extinguished/Verified label) and
`geospatial.wui_analysis` (which downgrades a WUI-critical read to
WATCH for a fire that isn't Active, so it doesn't fire the same
air-tanker-dispatch urgency as a genuinely escalating one). Active means
escalating per M5's Kalman-filter forecast, or PTSI flagging the fire's
thermal behaviour as abnormal for its own source class — not that the
fire is safe, just that it isn't currently getting worse by either
signal this pipeline actually measures.

Two entry points because the two callers fetch differently: the
dashboard list endpoint already batches KalmanStateRow/PTSIRegistry
across every cluster on the page (see `_forecast_for`/`_ptsi_for`'s
batch siblings), so `is_active_from_signals` takes the already-fetched
values; the WUI evaluation is single-cluster, so `is_fire_active` does
its own two-row read. Both funnel into the same boolean expression.
"""
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session


def is_active_from_signals(escalating: bool, ptsi: Optional[Dict[str, Any]]) -> bool:
    if escalating:
        return True
    if ptsi and ptsi.get("is_behaving_normally") is False:
        return True
    return False


def is_fire_active(session: Session, cluster_id: int) -> bool:
    from temporal.db.models import KalmanStateRow, PTSIRegistry

    kalman = session.get(KalmanStateRow, cluster_id)
    escalating = bool(kalman.escalating) if kalman is not None else False

    ptsi_row = session.get(PTSIRegistry, cluster_id)
    ptsi = {"is_behaving_normally": ptsi_row.is_behaving_normally} if ptsi_row is not None else None

    return is_active_from_signals(escalating, ptsi)
