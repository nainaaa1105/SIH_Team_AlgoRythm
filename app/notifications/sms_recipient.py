"""Recipient selection for emergency SMS dispatch.

TEST MODE ONLY right now: get_sms_recipient() always returns the single
configured TEST_SMS_RECIPIENT, never a real fire station. This is the
one place a later "nearest fire station" phase needs to change — every
caller already asks "who should this go to for cluster_id" without
knowing or caring how the answer was decided, so that phase is a change
to this function's body, not to any of its callers.
"""
from typing import Optional

from app.config import Settings, get_settings


class SmsModeNotImplemented(RuntimeError):
    """Raised when SMS_MODE names a routing mode that doesn't exist yet
    (e.g. a future "station" mode before that phase is built)."""


def get_sms_recipient(cluster_id: int, settings: Optional[Settings] = None) -> Optional[str]:
    """Who the emergency SMS for this cluster should go to.

    `cluster_id` is accepted (not just ignored) so this signature already
    matches what nearest-fire-station routing will need later — that
    phase looks up the cluster's real location and returns the nearest
    station's configured number instead of the fixed test number.

    Returns None if no recipient is configured (caller must not send).
    """
    settings = settings or get_settings()

    if settings.sms_mode != "test":
        raise SmsModeNotImplemented(
            f"SMS_MODE={settings.sms_mode!r} has no routing implementation yet — "
            "only 'test' mode (fixed TEST_SMS_RECIPIENT) exists so far. "
            "Nearest-fire-station routing is a later phase."
        )

    recipient = (settings.test_sms_recipient or "").strip()
    return recipient or None
