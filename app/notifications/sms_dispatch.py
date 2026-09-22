"""Emergency SMS dispatch — TEST MODE ONLY.

Builds the SMS body entirely from real, already-computed incident data
(geospatial.suppression.estimate_suppression, the same source
GET /geospatial/suppression/{id} and the detail panel's Suppression
Estimate card already use — see geospatial/api/routes_geo.py), sends it
through HttpSMS (app.notifications.sms_client), and records the attempt
on the existing (previously unused) `alerts` table. Nothing here invents
a number: a field genuinely not computable for this cluster's class
prints as "Not applicable", never a fabricated value.

Recipient is get_sms_recipient() — test-mode only, see
app.notifications.sms_recipient for what changes when nearest-fire-
station routing is built later.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# A second click within this window re-reports the earlier attempt
# instead of sending again — the real protection against a double
# "Escalate and Dispatch" click while the first request is still (or
# just finished) in flight. Not a permanent per-cluster lock: a fire
# that flares up again later can still be re-dispatched.
DUPLICATE_WINDOW = timedelta(minutes=2)

GOOGLE_MAPS_NAV_URL = "https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"


class ClusterNotFound(RuntimeError):
    pass


def _gather_incident(db: Session, cluster_id: int) -> Dict[str, Any]:
    """Everything the SMS body needs, pulled from the same real sources
    the dashboard's own event/suppression endpoints already use — no
    second, divergent calculation."""
    from geoalchemy2.shape import to_shape

    from app.db.models import Cluster
    from classifier.db.models import Classification
    from geospatial.decision_engine import evaluate_decision_support
    from geospatial.features_io import load_hotspot_rows
    from geospatial.suppression import estimate_suppression

    cluster = db.get(Cluster, cluster_id)
    if cluster is None:
        raise ClusterNotFound(f"cluster {cluster_id} not found")

    point = to_shape(cluster.centroid)
    lon, lat = point.x, point.y

    classification = db.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
    predicted_class = classification.predicted_class if classification else None

    hotspot_rows = load_hotspot_rows(db, cluster_id)
    sensor = None
    for row in hotspot_rows:
        source = (row.get("source") or "").upper()
        if "VIIRS" in source:
            sensor = "VIIRS"
            break
        if "MODIS" in source:
            sensor = "MODIS"
            break

    decision = evaluate_decision_support(cluster_id, db)
    suppression = estimate_suppression(
        predicted_class=predicted_class,
        frp_max_mw=decision.get("frp_max_mw"),
        area_ha=decision.get("area_ha"),
        coa_type=decision.get("coa_type"),
        sensor=sensor,
    )

    state = district = None
    try:
        from geospatial.admin_boundaries import resolve as resolve_admin
        state, district = resolve_admin(lon, lat)
    except Exception:  # noqa: BLE001
        pass

    location_name = None
    try:
        from gateway.routes_dashboard import _top_facility_for
        attribution = _top_facility_for(db, cluster_id, lon, lat)
        if attribution:
            location_name = attribution.get("name") or attribution.get("nearby_named_feature")
    except Exception:  # noqa: BLE001
        logger.info("Facility name lookup failed for cluster %s", cluster_id, exc_info=True)

    if not location_name:
        location_name = ", ".join(p for p in (district, state) if p) or None

    return {
        "cluster_id": cluster_id,
        "lat": lat,
        "lon": lon,
        "location_name": location_name,
        "predicted_class": predicted_class,
        "suppression": suppression,
    }


def _fmt_water_required(suppression: Dict[str, Any]) -> str:
    volume_l = suppression.get("primary_volume_l")
    if volume_l is not None:
        return f"{volume_l:,.0f} L"
    volume_m3 = suppression.get("primary_volume_m3")
    if volume_m3 is not None:
        return f"Not applicable ({suppression.get('resource_label', 'fill material')}: {volume_m3:,.0f} m3)"
    return "Not available"


def _fmt_truck_trips(suppression: Dict[str, Any]) -> Optional[int]:
    trips = suppression.get("tanker_trips")
    if trips:
        return int(trips)
    loads = suppression.get("truck_loads")
    if loads:
        return int(loads)
    return None


def build_sms_body(incident: Dict[str, Any], map_url: str) -> str:
    suppression = incident["suppression"]
    lat, lon = incident["lat"], incident["lon"]

    truck_trips = _fmt_truck_trips(suppression)
    ground_required = bool(truck_trips)
    air_required = bool(suppression.get("air_support_recommended"))
    air_trips = suppression.get("air_drop_trips") or 0

    nav_url = GOOGLE_MAPS_NAV_URL.format(lat=lat, lon=lon)

    lines = [
        "FIREOPS EMERGENCY ALERT",
        "",
        f"Location: {incident['location_name'] or 'Location pending confirmation'}",
        "",
        f"Coordinates: {lat:.5f}, {lon:.5f}",
        "",
        f"Water Required: {_fmt_water_required(suppression)}",
        f"Fire Truck Trips: {truck_trips if truck_trips is not None else 'Not available'}",
        f"Ground Support: {'Required' if ground_required else 'Not Required'}",
        f"Air Support: {'Required' if air_required else 'Not Required'}",
    ]
    if air_required and air_trips:
        lines.append(f"Aerial Tanks: {air_trips}")
    lines += [
        "",
        "Fire Location:",
        map_url,
        "",
        "NAVIGATE:",
        nav_url,
        "",
        "Respond immediately.",
    ]
    return "\n".join(lines)


def _recent_dispatch(db: Session, cluster_id: int) -> Optional[Any]:
    from app.db.models import Alert

    cutoff = datetime.now(timezone.utc) - DUPLICATE_WINDOW
    return (
        db.query(Alert)
        .filter(
            Alert.cluster_id == cluster_id,
            Alert.channel == "sms",
            Alert.dispatched_at >= cutoff,
        )
        .order_by(Alert.dispatched_at.desc())
        .first()
    )


def dispatch_emergency_sms(
    db: Session, cluster_id: int, settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Build and send the emergency SMS for one cluster, recording the
    attempt. Returns a dict the API layer serialises directly — never
    raises for an ordinary send failure (bad number, HttpSMS down,
    no recipient configured); those come back as a normal
    ok=False result so the caller can show an honest error instead of a
    500. Only a genuinely missing cluster raises (ClusterNotFound)."""
    from app.db.models import Alert
    from app.notifications.sms_client import send_sms
    from app.notifications.sms_recipient import get_sms_recipient

    settings = settings or get_settings()

    existing = _recent_dispatch(db, cluster_id)
    if existing is not None:
        return {
            "ok": existing.status == "sent",
            "duplicate": True,
            "recipient": existing.recipient,
            "dispatched_at": existing.dispatched_at,
            "error": existing.error,
        }

    incident = _gather_incident(db, cluster_id)

    recipient = get_sms_recipient(cluster_id, settings)
    if not recipient:
        return {"ok": False, "duplicate": False, "error": "No SMS recipient configured (TEST_SMS_RECIPIENT is empty)"}

    map_url = f"{settings.frontend_base_url.rstrip('/')}/?cluster={cluster_id}"
    body = build_sms_body(incident, map_url)

    result = send_sms(recipient, body, request_id=f"cluster-{cluster_id}-{int(datetime.now(timezone.utc).timestamp())}")

    alert = Alert(
        cluster_id=cluster_id,
        severity="emergency",
        message=body,
        dispatched_at=datetime.now(timezone.utc),
        channel="sms",
        recipient=recipient,
        status="sent" if result.ok else "failed",
        provider="httpsms",
        provider_message_id=result.provider_message_id,
        error=result.error,
    )
    db.add(alert)
    db.flush()

    return {
        "ok": result.ok,
        "duplicate": False,
        "recipient": recipient,
        "message": body,
        "provider_message_id": result.provider_message_id,
        "error": result.error,
    }
