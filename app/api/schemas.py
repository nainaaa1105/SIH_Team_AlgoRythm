"""Pydantic response models for the read API — kept separate from the ORM
models so the wire format (GeoJSON-flavoured) can evolve independently of
the DB schema, per M6's contract needs.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class HotspotOut(BaseModel):
    id: int
    source: str
    lon: float
    lat: float
    acq_datetime: datetime
    frp: Optional[float]
    brightness: Optional[float]
    confidence: Optional[float]
    daynight: Optional[str]
    cluster_id: Optional[int]

    class Config:
        from_attributes = True


class ClusterOut(BaseModel):
    id: int
    lon: float
    lat: float
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    n_detections: int
    cloud_fraction: Optional[float]
    optical_available: bool
    status: str
    wui_threat: bool = False
    wui_distance_m: Optional[float] = None
    wui_eta_hours: Optional[float] = None
    wui_bearing_deg: Optional[float] = None
    wui_threatened_asset: Optional[str] = None
    is_crown_fire: bool = False
    rdi_score: Optional[float] = None
    coa_type: Optional[str] = None

    class Config:
        from_attributes = True


class FacilityOut(BaseModel):
    id: int
    name: Optional[str]
    facility_type: Optional[str]
    source: Optional[str]
    lon: float
    lat: float
    prior_weight: float

    class Config:
        from_attributes = True


class HotspotList(BaseModel):
    count: int
    items: List[HotspotOut]


class ClusterList(BaseModel):
    count: int
    items: List[ClusterOut]


class FacilityList(BaseModel):
    count: int
    items: List[FacilityOut]
