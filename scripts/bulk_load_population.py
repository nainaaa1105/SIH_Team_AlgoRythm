"""One-time load: fetch the WorldPop population raster from a GitHub
Release asset into data/population/, since the file (~1.8GB) is far too
large to ship in git or the Docker image (GitHub blocks anything over
100MB in a normal commit; even Git LFS would bloat every future deploy).

The release is private (same repo as the code), so this uses the GitHub
API's asset-download endpoint with a token, not the plain browser URL --
an anonymous request to a private repo's release asset gets a 404.

Usage:
    GITHUB_TOKEN=... python -m scripts.bulk_load_population \\
        --repo nainaaa1105/SIH_Team_AlgoRythm --asset-id 584273578

Safe to re-run: verifies the existing file's size before re-downloading,
so a second run after a successful one is a no-op.
"""
import argparse
import logging
import os
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB — large enough to be efficient over a slow link, small enough to log progress


def _asset_metadata(repo: str, asset_id: int, token: str) -> dict:
    resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/releases/assets/{asset_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def download_asset(repo: str, asset_id: int, dest: Path, token: str) -> None:
    meta = _asset_metadata(repo, asset_id, token)
    expected_size = meta["size"]
    name = meta["name"]

    if dest.exists() and dest.stat().st_size == expected_size:
        logger.info("%s already present at %s (%d bytes) -- skipping download", name, dest, expected_size)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    logger.info("Downloading %s (%.2f GB) -> %s", name, expected_size / 1e9, dest)

    # The asset-download endpoint (not the metadata one above) needs
    # Accept: application/octet-stream to get the actual bytes -- with
    # the vnd.github+json Accept header it returns JSON metadata instead,
    # same URL either way.
    with requests.get(
        f"{GITHUB_API}/repos/{repo}/releases/assets/{asset_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        written = 0
        last_logged_pct = -1
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                pct = int(written * 100 / expected_size)
                if pct != last_logged_pct:
                    logger.info("  %d%% (%.2f / %.2f GB)", pct, written / 1e9, expected_size / 1e9)
                    last_logged_pct = pct

    actual_size = tmp_path.stat().st_size
    if actual_size != expected_size:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download incomplete: got {actual_size} bytes, expected {expected_size}. "
            "Removed the partial file -- re-run to try again."
        )

    tmp_path.rename(dest)
    logger.info("Done: %s (%d bytes, verified)", dest, actual_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="nainaaa1105/SIH_Team_AlgoRythm")
    parser.add_argument("--asset-id", type=int, default=584273578,
                         help="GitHub release asset ID for worldpop_india.tif")
    parser.add_argument("--dest", default="data/population/worldpop_india.tif")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set -- needed to read this private repo's release asset.")

    download_asset(args.repo, args.asset_id, Path(args.dest), token)


if __name__ == "__main__":
    main()
