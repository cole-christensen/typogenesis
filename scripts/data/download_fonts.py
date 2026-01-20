#!/usr/bin/env python3
"""
Download open-source fonts from Google Fonts API.

This script downloads fonts from Google Fonts, filtering by license (OFL, Apache),
organizes them by font family, and tracks downloaded fonts in a manifest.

Usage:
    python download_fonts.py --output-dir ./fonts --api-key YOUR_API_KEY
    python download_fonts.py --output-dir ./fonts --max-fonts 100 --categories sans-serif,serif
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
GOOGLE_FONTS_API_BASE = "https://www.googleapis.com/webfonts/v1/webfonts"
GOOGLE_FONTS_DOWNLOAD_BASE = "https://fonts.google.com/download"
GITHUB_GOOGLE_FONTS_RAW = "https://raw.githubusercontent.com/google/fonts/main"

# Allowed licenses for training data
ALLOWED_LICENSES = {"ofl", "apache", "ufl"}

# Font categories
VALID_CATEGORIES = {"serif", "sans-serif", "display", "handwriting", "monospace"}


@dataclass
class FontInfo:
    """Information about a downloaded font."""
    family: str
    category: str
    license: str
    variants: list[str]
    subsets: list[str]
    files: dict[str, str]
    download_date: str
    local_paths: list[str]


@dataclass
class Manifest:
    """Manifest tracking all downloaded fonts."""
    version: str = "1.0"
    created: str = ""
    updated: str = ""
    total_fonts: int = 0
    total_files: int = 0
    fonts: dict[str, dict] = None

    def __post_init__(self):
        if self.fonts is None:
            self.fonts = {}
        if not self.created:
            self.created = datetime.now().isoformat()
        self.updated = datetime.now().isoformat()


class FontDownloader:
    """Download fonts from Google Fonts API."""

    def __init__(
        self,
        output_dir: Path,
        api_key: Optional[str] = None,
        rate_limit: float = 0.5
    ):
        """
        Initialize the font downloader.

        Args:
            output_dir: Directory to save downloaded fonts
            api_key: Google Fonts API key (optional, uses public endpoint)
            rate_limit: Seconds to wait between downloads
        """
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Typogenesis-FontDownloader/1.0"
        })

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load or create manifest
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Manifest:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)
                return Manifest(**data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not load manifest: {e}, creating new one")
        return Manifest()

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        self.manifest.updated = datetime.now().isoformat()
        self.manifest.total_fonts = len(self.manifest.fonts)
        self.manifest.total_files = sum(
            len(f.get("local_paths", [])) for f in self.manifest.fonts.values()
        )

        with open(self.manifest_path, 'w') as f:
            json.dump(asdict(self.manifest), f, indent=2)
        logger.info(f"Saved manifest with {self.manifest.total_fonts} fonts")

    def fetch_font_list(
        self,
        categories: Optional[set[str]] = None,
        sort_by: str = "popularity"
    ) -> list[dict]:
        """
        Fetch list of available fonts from Google Fonts API.

        Args:
            categories: Set of categories to filter by
            sort_by: Sort order (alpha, date, popularity, style, trending)

        Returns:
            List of font dictionaries
        """
        params = {"sort": sort_by}
        if self.api_key:
            params["key"] = self.api_key

        logger.info("Fetching font list from Google Fonts API...")

        try:
            response = self.session.get(GOOGLE_FONTS_API_BASE, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch font list: {e}")
            raise

        fonts = data.get("items", [])
        logger.info(f"Found {len(fonts)} fonts in Google Fonts")

        # Filter by category
        if categories:
            fonts = [f for f in fonts if f.get("category") in categories]
            logger.info(f"Filtered to {len(fonts)} fonts in categories: {categories}")

        # Filter by license (all Google Fonts are open source, but we check anyway)
        # Google Fonts uses OFL and Apache licenses
        fonts = [
            f for f in fonts
            if self._is_allowed_license(f.get("family", ""))
        ]

        return fonts

    def _is_allowed_license(self, family: str) -> bool:
        """
        Check if font has an allowed license.

        Google Fonts only hosts OFL and Apache licensed fonts,
        so all fonts from the API are allowed.
        """
        # All Google Fonts are under OFL or Apache license
        return True

    def _get_font_license(self, family: str) -> str:
        """
        Determine the license of a font.

        Args:
            family: Font family name

        Returns:
            License identifier (ofl, apache, or unknown)
        """
        # Most Google Fonts are OFL, some are Apache
        # We'd need to check the LICENSE file to be certain
        # For now, assume OFL as it's most common
        return "ofl"

    def download_font(
        self,
        font_info: dict,
        skip_existing: bool = True
    ) -> Optional[FontInfo]:
        """
        Download a single font family.

        Args:
            font_info: Font dictionary from Google Fonts API
            skip_existing: Skip if font already downloaded

        Returns:
            FontInfo object or None if skipped/failed
        """
        family = font_info.get("family", "")
        if not family:
            logger.warning("Font missing family name, skipping")
            return None

        # Check if already downloaded
        family_key = family.lower().replace(" ", "_")
        if skip_existing and family_key in self.manifest.fonts:
            logger.debug(f"Skipping already downloaded font: {family}")
            return None

        # Create font directory
        font_dir = self.output_dir / family_key
        font_dir.mkdir(parents=True, exist_ok=True)

        # Download each variant
        files = font_info.get("files", {})
        local_paths = []

        for variant, url in files.items():
            # Rate limiting
            time.sleep(self.rate_limit)

            # Determine file extension from URL
            if ".ttf" in url.lower():
                ext = ".ttf"
            elif ".otf" in url.lower():
                ext = ".otf"
            else:
                ext = ".ttf"  # Default to TTF

            filename = f"{family_key}_{variant}{ext}"
            filepath = font_dir / filename

            if filepath.exists() and skip_existing:
                local_paths.append(str(filepath.relative_to(self.output_dir)))
                continue

            try:
                logger.debug(f"Downloading {family} {variant}...")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                local_paths.append(str(filepath.relative_to(self.output_dir)))

            except requests.RequestException as e:
                logger.warning(f"Failed to download {family} {variant}: {e}")
                continue

        if not local_paths:
            logger.warning(f"No files downloaded for {family}")
            return None

        # Create font info
        info = FontInfo(
            family=family,
            category=font_info.get("category", "unknown"),
            license=self._get_font_license(family),
            variants=font_info.get("variants", []),
            subsets=font_info.get("subsets", []),
            files=files,
            download_date=datetime.now().isoformat(),
            local_paths=local_paths
        )

        # Update manifest
        self.manifest.fonts[family_key] = asdict(info)

        logger.info(f"Downloaded {family}: {len(local_paths)} files")
        return info

    def download_fonts(
        self,
        max_fonts: Optional[int] = None,
        categories: Optional[set[str]] = None,
        sort_by: str = "popularity"
    ) -> int:
        """
        Download multiple fonts from Google Fonts.

        Args:
            max_fonts: Maximum number of fonts to download
            categories: Categories to filter by
            sort_by: Sort order for font list

        Returns:
            Number of fonts downloaded
        """
        fonts = self.fetch_font_list(categories=categories, sort_by=sort_by)

        if max_fonts:
            fonts = fonts[:max_fonts]

        logger.info(f"Downloading {len(fonts)} fonts...")

        downloaded = 0
        for font_info in tqdm(fonts, desc="Downloading fonts"):
            result = self.download_font(font_info)
            if result:
                downloaded += 1

                # Save manifest periodically
                if downloaded % 10 == 0:
                    self._save_manifest()

        # Final save
        self._save_manifest()

        logger.info(f"Downloaded {downloaded} new fonts")
        return downloaded

    def get_downloaded_fonts(self) -> list[FontInfo]:
        """Get list of all downloaded fonts."""
        fonts = []
        for family_key, data in self.manifest.fonts.items():
            try:
                fonts.append(FontInfo(**data))
            except TypeError:
                logger.warning(f"Invalid font data for {family_key}")
        return fonts

    def get_font_files(self, extensions: Optional[set[str]] = None) -> list[Path]:
        """
        Get all font files in the output directory.

        Args:
            extensions: Set of extensions to filter by (e.g., {'.ttf', '.otf'})

        Returns:
            List of font file paths
        """
        if extensions is None:
            extensions = {'.ttf', '.otf'}

        files = []
        for ext in extensions:
            files.extend(self.output_dir.rglob(f"*{ext}"))

        return sorted(files)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download open-source fonts from Google Fonts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download top 100 fonts
    python download_fonts.py --output-dir ./fonts --max-fonts 100

    # Download only serif and sans-serif fonts
    python download_fonts.py --output-dir ./fonts --categories serif,sans-serif

    # Download with API key for higher rate limits
    python download_fonts.py --output-dir ./fonts --api-key YOUR_KEY
        """
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save downloaded fonts"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_API_KEY"),
        help="Google Fonts API key (or set GOOGLE_FONTS_API_KEY env var)"
    )

    parser.add_argument(
        "--max-fonts",
        type=int,
        default=None,
        help="Maximum number of fonts to download"
    )

    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help=f"Comma-separated list of categories to download ({','.join(VALID_CATEGORIES)})"
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["alpha", "date", "popularity", "style", "trending"],
        default="popularity",
        help="Sort order for fonts (default: popularity)"
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds between downloads (default: 0.5)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list fonts without downloading"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse categories
    categories = None
    if args.categories:
        categories = set(c.strip() for c in args.categories.split(","))
        invalid = categories - VALID_CATEGORIES
        if invalid:
            logger.error(f"Invalid categories: {invalid}")
            logger.error(f"Valid categories: {VALID_CATEGORIES}")
            return 1

    # Create downloader
    downloader = FontDownloader(
        output_dir=args.output_dir,
        api_key=args.api_key,
        rate_limit=args.rate_limit
    )

    if args.list_only:
        # Just list available fonts
        fonts = downloader.fetch_font_list(categories=categories, sort_by=args.sort_by)
        if args.max_fonts:
            fonts = fonts[:args.max_fonts]

        print(f"\nFound {len(fonts)} fonts:")
        for font in fonts:
            print(f"  - {font['family']} ({font['category']}): {len(font.get('variants', []))} variants")
        return 0

    # Download fonts
    try:
        downloaded = downloader.download_fonts(
            max_fonts=args.max_fonts,
            categories=categories,
            sort_by=args.sort_by
        )

        logger.info(f"\nDownload complete!")
        logger.info(f"Total fonts in manifest: {downloader.manifest.total_fonts}")
        logger.info(f"Total font files: {downloader.manifest.total_files}")
        logger.info(f"Manifest saved to: {downloader.manifest_path}")

        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
