"""HttpSMS (https://httpsms.com) client — sends a real SMS through a
registered Android phone acting as the gateway.

This makes an actual authenticated POST to HttpSMS's API and reports
back exactly what it said. It never fabricates a "sent" result: with no
API key configured, or on any request failure, it returns ok=False with
the real reason, the same "real data or an honest failure, never a
guess" rule the rest of this codebase's external-API clients follow
(see app/enrichment/osm_facilities.py).

API shape (https://docs.httpsms.com/): POST {base}/messages/send,
header x-api-key, JSON body {from, to, content, request_id}, success
response {"data": {"id": ...}}, error response {"message": ...}.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class SmsSendResult:
    ok: bool
    provider_message_id: Optional[str] = None
    error: Optional[str] = None


def send_sms(
    to: str, body: str, request_id: Optional[str] = None, settings: Optional[Settings] = None,
) -> SmsSendResult:
    settings = settings or get_settings()

    if not settings.httpsms_api_key or not settings.httpsms_from_number:
        return SmsSendResult(
            ok=False,
            error="HTTPSMS_API_KEY / HTTPSMS_FROM_NUMBER not configured — see .env.example",
        )

    url = f"{settings.httpsms_base_url.rstrip('/')}/messages/send"
    payload = {"from": settings.httpsms_from_number, "to": to, "content": body}
    if request_id:
        payload["request_id"] = request_id

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"x-api-key": settings.httpsms_api_key, "Content-Type": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        message_id = (data.get("data") or {}).get("id")
        return SmsSendResult(ok=True, provider_message_id=message_id)
    except requests.RequestException as exc:
        logger.warning("HttpSMS send to %s failed", to, exc_info=True)
        detail = str(exc)
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                detail = (response.json() or {}).get("message") or response.text[:500]
            except ValueError:
                detail = response.text[:500]
        return SmsSendResult(ok=False, error=detail)
