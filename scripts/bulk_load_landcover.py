"""One-time (Day 1) / annual-refresh: fetch a MODIS MCD12Q1 land-cover
GeoTIFF for India from Google Earth Engine and save it locally so
app/enrichment/landcover.py can sample it per-cluster without a live GEE
call on every request.

OPTIONAL — not required to run the pipeline. This project has no GCP
service account, so the primary land-cover path is now
`geospatial.attribution.landcover_worldcover_cog` (ESA WorldCover 10m,
read live from a public, keyless AWS S3 bucket — see that module's
docstring). This script only matters if you later get a GEE account and
want the MODIS 500m raster as an extra fallback tier underneath it.

Usage: python -m scripts.bulk_load_landcover --year 2023 --out data/landcover/mcd12q1_india_2023.tif
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_mcd12q1(year: int, out_path: str, bbox: tuple) -> None:
    try:
        import ee
    except ImportError:
        raise SystemExit("earthengine-api is required: pip install earthengine-api, then `earthengine authenticate`")

    from app.config import get_settings

    settings = get_settings()
    if settings.gee_service_account:
        credentials = ee.ServiceAccountCredentials(settings.gee_service_account, settings.gee_private_key_file)
        ee.Initialize(credentials)
    else:
        ee.Initialize()  # falls back to `earthengine authenticate`-cached user credentials

    west, south, east, north = bbox
    region = ee.Geometry.Rectangle([west, south, east, north])
    image = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .first()
        .select("LC_Type1")
        .clip(region)
    )

    # ee.batch export is async and lands in Google Drive/Cloud Storage, not
    # a local file directly — this kicks off the export task; download the
    # result manually (or via gsutil/gdown) once it completes.
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"mcd12q1_india_{year}",
        folder="sih162_landcover",
        fileNamePrefix=f"mcd12q1_india_{year}",
        region=region,
        scale=500,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    logger.info(
        "Export task started (id=%s). Check status at "
        "https://code.earthengine.google.com/tasks, then download the "
        "GeoTIFF from Drive folder 'sih162_landcover' to %s",
        task.id, out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--out", default="data/landcover/mcd12q1_india.tif")
    parser.add_argument("--bbox", default="68.0,6.5,97.5,37.5", help="west,south,east,north")
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    export_mcd12q1(args.year, args.out, bbox)


if __name__ == "__main__":
    main()
