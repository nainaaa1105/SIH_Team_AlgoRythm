"""Evacuation routing out of a hazard polygon.

Lowest-priority M3 deliverable and deliberately opt-in
(`routing_enabled`, default False): building an OSMnx road graph is slow
and network-bound, which is the wrong thing to do inside a per-cluster
task that fires every 30 minutes. Enable it for a specific incident, or
run it from the script rather than the live pipeline.

osmnx/networkx are lazy imports so the rest of M3 doesn't depend on a
heavy geospatial stack being installed.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from geospatial.geometry import destination_point, haversine_m

logger = logging.getLogger(__name__)


@dataclass
class EvacuationPath:
    coordinates: List[Tuple[float, float]]
    length_km: float
    origin_name: Optional[str]
    destination_name: Optional[str]


def point_in_ring(lon: float, lat: float, ring: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon on a (lon, lat) ring.

    Used to decide which candidate destinations are still inside the
    hazard area. Kept dependency-free and pure so it is unit-testable
    without shapely or a database.
    """
    if len(ring) < 3:
        return False

    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        # Does the horizontal ray at `lat` cross this edge?
        if (y1 > lat) != (y2 > lat):
            if y2 != y1:
                x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
                if lon < x_cross:
                    inside = not inside
    return inside


def safe_destinations(
    lon: float,
    lat: float,
    hazard_ring: Sequence[Tuple[float, float]],
    distance_m: float = 8000.0,
    bearings: Optional[Sequence[float]] = None,
) -> List[Tuple[float, float]]:
    """Candidate refuge points outside the hazard polygon.

    Casts points on a ring around the source and keeps the ones the
    hazard doesn't cover — the crosswind and upwind directions survive,
    which is exactly the advice you'd give someone on the ground.
    """
    if bearings is None:
        bearings = [b * 30.0 for b in range(12)]

    candidates = []
    for bearing in bearings:
        point = destination_point(lon, lat, bearing, distance_m)
        if not point_in_ring(point[0], point[1], hazard_ring):
            candidates.append(point)
    return candidates


def shortest_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    search_radius_m: float = 10000.0,
) -> Optional[EvacuationPath]:
    """Road-network shortest path between two (lon, lat) points.

    Returns None if osmnx isn't installed or the network can't be built,
    so a missing optional dependency degrades routing rather than
    failing the incident's whole geospatial enrichment.
    """
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError:
        logger.info("osmnx/networkx not installed — evacuation routing unavailable")
        return None

    try:
        graph = ox.graph_from_point(
            (origin[1], origin[0]), dist=search_radius_m, network_type="drive"
        )
        # osmnx takes (X=lon, Y=lat); passing them the other way round
        # silently returns a wrong-but-plausible node.
        origin_node = ox.nearest_nodes(graph, X=origin[0], Y=origin[1])
        destination_node = ox.nearest_nodes(graph, X=destination[0], Y=destination[1])

        node_path = nx.shortest_path(graph, origin_node, destination_node, weight="length")
        coordinates = [(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in node_path]
    except Exception:
        logger.warning("Routing failed from %s to %s", origin, destination, exc_info=True)
        return None

    length_km = sum(
        haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(coordinates, coordinates[1:])
    ) / 1000.0

    return EvacuationPath(
        coordinates=coordinates,
        length_km=length_km,
        origin_name=None,
        destination_name=None,
    )
