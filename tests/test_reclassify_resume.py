"""--reclassify's cluster ordering (scripts/run_live_cycle.py) -- the
part that makes an interrupted pass resume on the next identical
invocation instead of restarting from scratch.

Real incident: a Railway auto-deploy (triggered by an unrelated git
push) kills the shell session running `--reclassify` partway through.
Re-pasting the same command used to redo all ~7,800 clusters from the
top, because the old ordering (Cluster.last_seen, descending) carries
no memory of which clusters this pass already reached. The fix orders
by Classification.updated_at ascending with nulls first instead:
already-reclassified clusters have a freshly bumped updated_at and
sort to the back, so a resumed run naturally picks up the untouched
(oldest-updated-or-never-classified) ones first. That state lives in
Postgres, which survives a container restart, unlike anything the
process could keep in memory or on local disk.

This script needs a live DB session to run end to end (no fixtures in
this suite -- see test_gateway.py's module docstring for why), so this
is a source-level regression guard on the exact ordering, the same
style used for the other DB-dependent gateway logic in
test_generic_facility_name.py.
"""
import ast
from pathlib import Path


def _reclassify_source() -> str:
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_live_cycle.py"
    return path.read_text(encoding="utf-8")


def test_reclassify_orders_by_classification_updated_at_not_cluster_recency():
    source = _reclassify_source()
    assert "Classification.updated_at.asc().nullsfirst()" in source
    assert "Cluster.last_seen.desc()" not in source


def test_reclassify_joins_classification_so_never_classified_clusters_are_included():
    """An outer join, not an inner one -- a cluster with no Classification
    row yet (predicted_class is null on the left side) must still be
    selected, and sort first via nullsfirst()."""
    source = _reclassify_source()
    assert ".outerjoin(Classification, Classification.cluster_id == Cluster.id)" in source


def test_run_live_cycle_module_parses_and_documents_the_resume_behaviour():
    """The docstring is the only place a human running this from a
    Railway shell sees an explanation -- make sure the module still
    imports cleanly (syntax-valid) and keeps that explanation."""
    source = _reclassify_source()
    ast.parse(source)  # would raise SyntaxError on a bad edit
    module_doc = ast.get_docstring(ast.parse(source))
    assert "resume" in module_doc.lower()
