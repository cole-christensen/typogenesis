"""Google Fonts downloader for Typogenesis training data.

Downloads font families from Google Fonts, filtering for Latin-script fonts
with adequate character coverage (A-Z, a-z, 0-9).

Usage:
    python -m data.download_fonts --output-dir data/fonts --limit 1500
    python -m data.download_fonts --output-dir data/fonts --api-key YOUR_KEY

The downloader is incremental: it skips fonts already present in the output directory.
"""

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimum characters a font must support to be useful for training
REQUIRED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
GOOGLE_FONTS_API_URL = "https://www.googleapis.com/webfonts/v1/webfonts"
# Public GitHub raw URL for the Google Fonts collection (no API key needed)
GOOGLE_FONTS_OFL_BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"
GOOGLE_FONTS_APACHE_BASE = "https://raw.githubusercontent.com/google/fonts/main/apache"


def fetch_font_list_from_api(api_key: str) -> list[dict]:
    """Fetch the full font list from the Google Fonts API.

    Args:
        api_key: Google Fonts API key.

    Returns:
        List of font family metadata dicts.
    """
    url = f"{GOOGLE_FONTS_API_URL}?key={api_key}&sort=popularity"
    logger.info("Fetching font list from Google Fonts API...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    families = data.get("items", [])
    logger.info(f"Found {len(families)} font families from API")
    return families


def fetch_font_list_from_csv() -> list[dict]:
    """Fetch font metadata from the Google Fonts metadata CSV (no API key needed).

    Returns:
        List of font family metadata dicts with keys: family, category, subsets, files.
    """
    logger.info("Fetching font family list from Google Fonts repo...")

    # Use the Google Fonts API without key for basic listing (limited but works)
    # Fall back to fetching the METADATA.pb index
    api_url = f"{GOOGLE_FONTS_API_URL}?sort=popularity"
    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        families = data.get("items", [])
        if families:
            logger.info(f"Found {len(families)} font families from public API")
            return families
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.warning(f"Public API failed: {e}")

    # If API fails, we'll need an API key
    raise RuntimeError(
        "Could not fetch font list. The Google Fonts API requires an API key. "
        "Get one at https://developers.google.com/fonts/docs/developer_api "
        "and pass it via --api-key or GOOGLE_FONTS_API_KEY env var."
    )


def has_required_coverage(font_path: Path) -> bool:
    """Check if a font file has the required character coverage.

    Args:
        font_path: Path to a TTF/OTF file.

    Returns:
        True if the font contains glyphs for all required characters.
    """
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        if cmap is None:
            return False
        font_chars = {chr(cp) for cp in cmap}
        return REQUIRED_CHARS.issubset(font_chars)
    except Exception:
        return False


def sanitize_family_name(name: str) -> str:
    """Convert a font family name to a safe directory name.

    Args:
        name: Font family name (e.g. "Noto Sans").

    Returns:
        Sanitized name (e.g. "noto-sans").
    """
    return name.lower().replace(" ", "-").replace("'", "").replace('"', "")


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a file from a URL to a local path.

    Args:
        url: URL to download from.
        dest: Local file path to save to.
        timeout: Request timeout in seconds.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Typogenesis-ML/0.1")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.debug(f"Failed to download {url}: {e}")
        return False


def download_font_family(
    family_meta: dict,
    output_dir: Path,
) -> dict | None:
    """Download all font files for a single family.

    Args:
        family_meta: Font family metadata from Google Fonts API.
        output_dir: Root output directory.

    Returns:
        Dict of downloaded file info, or None if family was skipped/failed.
    """
    family_name = family_meta.get("family", "")
    safe_name = sanitize_family_name(family_name)
    family_dir = output_dir / safe_name
    metadata_path = family_dir / "metadata.json"

    # Skip if already downloaded and has metadata
    if metadata_path.exists():
        return None

    files = family_meta.get("files", {})
    if not files:
        return None

    # Filter to Latin subset if available
    subsets = family_meta.get("subsets", [])
    if subsets and "latin" not in subsets:
        return None

    downloaded_styles = {}
    has_valid_font = False

    for style, url in files.items():
        # Google Fonts API returns http URLs, upgrade to https
        url = url.replace("http://", "https://")
        file_ext = ".ttf"
        if url.endswith(".otf"):
            file_ext = ".otf"

        dest = family_dir / f"{style}{file_ext}"
        if dest.exists():
            # Already have this file, check coverage
            if has_required_coverage(dest):
                has_valid_font = True
                downloaded_styles[style] = str(dest.relative_to(output_dir))
            continue

        if download_file(url, dest):
            if has_required_coverage(dest):
                has_valid_font = True
                downloaded_styles[style] = str(dest.relative_to(output_dir))
            else:
                # Font lacks required characters, remove it
                dest.unlink(missing_ok=True)

    if not has_valid_font:
        # Clean up empty directory
        if family_dir.exists() and not any(family_dir.iterdir()):
            family_dir.rmdir()
        return None

    # Save metadata
    meta = {
        "family": family_name,
        "category": family_meta.get("category", "unknown"),
        "subsets": subsets,
        "styles": downloaded_styles,
        "license": family_meta.get("license", "unknown"),
    }
    family_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2))

    return meta


def download_fonts(
    output_dir: Path,
    api_key: str | None = None,
    limit: int = 1500,
) -> dict:
    """Download Google Fonts for training.

    Args:
        output_dir: Directory to save fonts to.
        api_key: Optional Google Fonts API key. If not provided,
                 attempts to use the public API endpoint.
        limit: Maximum number of font families to download.

    Returns:
        Summary dict with counts and paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch font list
    families = fetch_font_list_from_api(api_key) if api_key else fetch_font_list_from_csv()

    # Limit to requested count
    families = families[:limit]
    logger.info(f"Processing {len(families)} font families (limit={limit})")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, family_meta in enumerate(families):
        family_name = family_meta.get("family", "unknown")

        result = download_font_family(family_meta, output_dir)
        if result is None:
            # Check if it was skipped (already exists) vs failed
            safe_name = sanitize_family_name(family_name)
            if (output_dir / safe_name / "metadata.json").exists():
                skipped += 1
            else:
                failed += 1
        else:
            downloaded += 1
            styles = list(result.get("styles", {}).keys())
            logger.info(
                f"[{i + 1}/{len(families)}] Downloaded {family_name} "
                f"({len(styles)} styles)"
            )

        # Progress logging every 100 families
        if (i + 1) % 100 == 0:
            logger.info(
                f"Progress: {i + 1}/{len(families)} "
                f"(downloaded={downloaded}, skipped={skipped}, failed={failed})"
            )

    summary = {
        "total_families_processed": len(families),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "output_dir": str(output_dir),
    }
    logger.info(f"Download complete: {summary}")

    # Save summary
    summary_path = output_dir / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def main():
    import os

    parser = argparse.ArgumentParser(description="Download Google Fonts for ML training")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/fonts"),
        help="Directory to save downloaded fonts",
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("GOOGLE_FONTS_API_KEY"),
        help="Google Fonts API key (or set GOOGLE_FONTS_API_KEY env var)",
    )
    parser.add_argument(
        "--limit", type=int, default=1500,
        help="Maximum number of font families to download",
    )
    args = parser.parse_args()

    summary = download_fonts(args.output_dir, args.api_key, args.limit)
    print(json.dumps(summary, indent=2))

    if summary["downloaded"] == 0 and summary["skipped"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
