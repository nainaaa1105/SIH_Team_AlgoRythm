"""Detecting an administrative/generic facility name so LOCATION can try
a real, specific replacement instead.

Real report: the LOCATION field showed "Cluster 8 and Cluster 9 Coal
Mines" -- confirmed genuine OSM data (BCCL/ECL's real administrative
naming convention for the Jharia/Raniganj coalfields, source=OSM), not a
bug or fabricated text. But an administrative zone label is a much less
useful LOCATION than an actual site name, so it should trigger the same
already-existing, already-honest "look for a real nearby named feature"
fallback that a completely unnamed facility already gets
(nearest_named_industrial_feature -- "never invents a name", only used
when a real neighbour is found close enough to plausibly be the same
site). No fabrication added: a generic name with no better real
alternative nearby still shows the real (if generic) name, never blank
and never invented.
"""
from gateway.routes_dashboard import _GENERIC_FACILITY_NAME


def test_matches_the_actually_observed_cluster_names():
    """These are real, live facility names found in the deployed
    database (source=OSM) -- not hypothetical."""
    assert _GENERIC_FACILITY_NAME.match("Cluster 8 and Cluster 9 Coal Mines")
    assert _GENERIC_FACILITY_NAME.match("Cluster 11 and Cluster 7 (BCCL) Coal Mines")
    assert _GENERIC_FACILITY_NAME.match("Cluster 6 (BCCL) Coal Mines")
    assert _GENERIC_FACILITY_NAME.match("Cluster 2 (BCCL) Coal Mines")
    assert _GENERIC_FACILITY_NAME.match("Cluster 11 (ECL) Coal Mines")


def test_does_not_match_a_real_specific_company_name():
    """Regression guard: must not become so broad it starts treating
    genuinely specific names as generic and searching for a replacement
    that was never needed."""
    assert not _GENERIC_FACILITY_NAME.match("Sunflag Steel")
    assert not _GENERIC_FACILITY_NAME.match("SPS Steel and Power Limited")
    assert not _GENERIC_FACILITY_NAME.match("Children community sentre")
    assert not _GENERIC_FACILITY_NAME.match("Jharia Colliery")


def test_does_not_match_cluster_appearing_mid_name():
    """Anchored to the start -- a real company whose name merely
    contains the word "Cluster" somewhere later must not be treated as
    the administrative-zone pattern."""
    assert not _GENERIC_FACILITY_NAME.match("Bharat Industrial Cluster 8 Pvt Ltd")


def test_is_case_insensitive():
    assert _GENERIC_FACILITY_NAME.match("cluster 8 coal mines")
