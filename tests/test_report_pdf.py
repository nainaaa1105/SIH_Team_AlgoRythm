"""Fire detail report download — now PDF instead of a plain-text file.

No browser, just asserting the hooks a live page needs are present in
static/index.html — same pattern as test_wui.py / test_gateway.py.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def _report_fn():
    src = html()
    return src.split("async function downloadReport()")[1].split("\n    }\n")[0]


def test_jspdf_library_is_loaded():
    assert "jspdf" in html().lower()
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/' in html()


def test_report_is_saved_as_pdf_not_txt():
    fn = _report_fn()
    assert "doc.save('agni-pehchan-cluster-' + fire.clusterId + '.pdf')" in fn
    assert ".txt'" not in fn
    assert "Blob(" not in fn  # the old text-file path is gone, not just unused


def test_report_generation_guards_a_missing_pdf_library():
    fn = _report_fn()
    assert "typeof window.jspdf === 'undefined'" in fn


def test_report_button_disabled_for_the_whole_generation():
    fn = _report_fn()
    assert "btn.disabled = true;" in fn
    assert "btn.disabled = false;" in fn
    assert "finally" in fn


def test_report_fetches_a_live_suppression_estimate_for_the_pdf():
    """Regression: the plain-text report never included water/material
    quantity, tanker/truck trips, or ground/air support at all — the
    explicit ask was for exactly this. Must reuse the same live source
    (API.suppression) the Suppression Estimate card and the emergency
    SMS already use, not a second, divergent calculation."""
    fn = _report_fn()
    assert "await API.suppression(fire.clusterId)" in fn


def test_pdf_includes_the_resource_and_suppression_section():
    fn = _report_fn()
    assert "RESOURCE & SUPPRESSION REQUIREMENT" in fn
    assert "Quantity required" in fn
    assert "Vehicle trips required" in fn
    assert "Ground support" in fn
    assert "Air support" in fn


def test_pdf_includes_name_coordinates_sensor_and_location():
    """Explicit fields the request called out by name."""
    fn = _report_fn()
    assert "w.row('Name', fire.name);" in fn
    assert "w.row('Latitude'" in fn
    assert "w.row('Longitude'" in fn
    assert "w.row('Sensor', fire.sensorProduct || fire.sensor);" in fn
    assert "w.row('Location'" in fn


def test_pdf_never_fabricates_a_missing_field():
    fn = _report_fn()
    assert "not available" in fn


def test_pdf_writer_paginates_long_reports():
    src = html()
    fn = src.split("function makePdfWriter(doc)")[1].split("\n    }\n")[0]
    assert "doc.addPage()" in fn


def test_pdf_no_longer_shows_the_live_pipeline_output_disclaimer():
    """Removed by request: the trailing 'All values are live pipeline
    output...' line at the very bottom of the PDF. The 'Generated ...'
    timestamp line right above it stays."""
    fn = _report_fn()
    assert "All values are live pipeline output" not in fn
    assert "were not produced for this cluster and are not substituted" not in fn
    assert "w.para('Generated '" in fn


def test_pdf_no_longer_includes_escalation_forecast_or_persistence():
    """Removed by request: both sections dropped from the bottom of the
    PDF entirely — not just hidden when their data is missing."""
    fn = _report_fn()
    assert "ESCALATION FORECAST" not in fn
    assert "PERSISTENCE (PTSI)" not in fn
    assert "ev.forecast" not in fn
    assert "ev.ptsi" not in fn


def test_pdf_no_longer_shows_a_stray_method_row():
    """Removed by request: the bare 'Method: shap' row sat oddly right
    after the numbered SHAP reasoning list — the reasons themselves
    stay, just not this row."""
    fn = _report_fn()
    assert "explanation_method" not in fn
    assert "(i + 1) + '. ' + r" in fn  # the numbered reasoning list itself is untouched
