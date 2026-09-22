"""INSAT-3DS/MOSDAC is pending manual approval; EUMETSAT IODC is the real
(not sample) temporary stand-in. These tests cover the source-selection
logic and the CAP XML parser without hitting either real network API.
"""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.config import Settings
from app.ingestion import eumetsat_iodc, geostationary_supplementary

CAP_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>IODC-FIRE-20260913-0001</identifier>
  <sender>eumetsat.int</sender>
  <sent>2026-09-13T06:00:00+00:00</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <category>Fire</category>
    <event>Active Fire Detection</event>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2026-09-13T06:00:00+00:00</effective>
    <area>
      <areaDesc>Detected hotspot</areaDesc>
      <circle>23.8,86.43 1.5</circle>
    </area>
  </info>
  <info>
    <category>Fire</category>
    <event>Active Fire Detection</event>
    <severity>Extreme</severity>
    <certainty>Observed</certainty>
    <effective>2026-09-13T06:15:00+00:00</effective>
    <area>
      <areaDesc>Detected hotspot</areaDesc>
      <circle>21.2,81.6 1.5</circle>
    </area>
  </info>
</alert>
"""


@pytest.fixture
def cap_file(tmp_path):
    path = tmp_path / "sample.cap"
    path.write_text(CAP_SAMPLE, encoding="utf-8")
    return path


# --- CAP parsing ---------------------------------------------------------

def test_parses_one_record_per_circle(cap_file):
    records = eumetsat_iodc.parse_cap_fire_product(cap_file)
    assert len(records) == 2
    lats = {round(r["lat"], 2) for r in records}
    assert lats == {23.8, 21.2}


def test_severity_maps_to_a_confidence_ordering(cap_file):
    records = sorted(eumetsat_iodc.parse_cap_fire_product(cap_file), key=lambda r: r["lat"])
    moderate, extreme = records[1], records[0]  # 21.2 (extreme) < 23.8 (moderate)
    assert extreme["confidence"] > moderate["confidence"]


def test_raises_a_clear_error_on_a_non_cap_file(tmp_path):
    bogus = tmp_path / "bogus.cap"
    bogus.write_text("<root><nothing/></root>", encoding="utf-8")
    with pytest.raises(eumetsat_iodc.EumetsatClientError):
        eumetsat_iodc.parse_cap_fire_product(bogus)


def test_unparseable_circle_is_skipped_not_fatal(tmp_path):
    bad = tmp_path / "bad.cap"
    bad.write_text(
        CAP_SAMPLE.replace("23.8,86.43 1.5", "not-a-circle"), encoding="utf-8"
    )
    records = eumetsat_iodc.parse_cap_fire_product(bad)
    assert len(records) == 1  # the one good circle survives


# --- token / download plumbing -------------------------------------------

def test_no_token_without_credentials():
    settings = Settings(eumetsat_consumer_key="", eumetsat_consumer_secret="")
    assert eumetsat_iodc._get_access_token(settings) is None


def test_download_returns_none_without_credentials():
    settings = Settings(eumetsat_consumer_key="", eumetsat_consumer_secret="")
    assert eumetsat_iodc.download_latest_product(settings) is None


def test_fetch_all_returns_empty_list_on_client_error(cap_file):
    settings = Settings(eumetsat_consumer_key="k", eumetsat_consumer_secret="s")
    with patch.object(eumetsat_iodc, "download_latest_product", return_value=cap_file), \
         patch.object(eumetsat_iodc, "parse_cap_fire_product", side_effect=eumetsat_iodc.EumetsatClientError("x")):
        assert eumetsat_iodc.fetch_all(settings) == []


# --- source selection ------------------------------------------------------

def test_prefers_mosdac_when_both_are_configured():
    settings = Settings(
        mosdac_username="u", mosdac_password="p",
        eumetsat_consumer_key="k", eumetsat_consumer_secret="s",
    )
    assert geostationary_supplementary.active_source_name(settings) == "MOSDAC_INSAT3DS"


def test_falls_back_to_eumetsat_when_mosdac_is_not_configured():
    settings = Settings(
        mosdac_username="", mosdac_password="",
        eumetsat_consumer_key="k", eumetsat_consumer_secret="s",
    )
    assert geostationary_supplementary.active_source_name(settings) == "EUMETSAT_IODC_BACKUP"


def test_reports_none_configured_when_neither_is_set():
    settings = Settings(mosdac_username="", mosdac_password="",
                         eumetsat_consumer_key="", eumetsat_consumer_secret="")
    assert geostationary_supplementary.active_source_name(settings) == "none_configured"
    assert geostationary_supplementary.fetch_all(settings) == []


def test_switches_back_to_mosdac_the_moment_credentials_are_filled_in():
    """The user's explicit requirement: entering the MOSDAC key later must
    switch back with no code change."""
    before = Settings(mosdac_username="", mosdac_password="",
                       eumetsat_consumer_key="k", eumetsat_consumer_secret="s")
    after = Settings(mosdac_username="u", mosdac_password="p",
                      eumetsat_consumer_key="k", eumetsat_consumer_secret="s")

    assert geostationary_supplementary.active_source_name(before) == "EUMETSAT_IODC_BACKUP"
    assert geostationary_supplementary.active_source_name(after) == "MOSDAC_INSAT3DS"


def test_dispatch_calls_the_selected_module_only():
    settings = Settings(mosdac_username="", mosdac_password="",
                         eumetsat_consumer_key="k", eumetsat_consumer_secret="s")
    with patch.object(geostationary_supplementary.insat3ds, "fetch_all") as mosdac_fetch, \
         patch.object(geostationary_supplementary.eumetsat_iodc, "fetch_all", return_value=[{"source": "EUMETSAT_IODC"}]) as eumetsat_fetch:
        records = geostationary_supplementary.fetch_all(settings)

    eumetsat_fetch.assert_called_once()
    mosdac_fetch.assert_not_called()
    assert records == [{"source": "EUMETSAT_IODC"}]
