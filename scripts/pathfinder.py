"""
Pathfinder AI Explorer — discovers official government visa URLs for each country.

Input:  config/countries.json
Output: config/sources.json (updated with validated URLs)

Uses Gemini 1.5 Flash to find official government pages for student, work, and
spouse visa categories.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
COUNTRIES_PATH = ROOT_DIR / "config" / "countries.json"
SOURCES_PATH = ROOT_DIR / "config" / "sources.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pathfinder")

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------
VISA_CATEGORIES = ["student", "work", "spouse"]

SYSTEM_PROMPT = (
    "You are an immigration research assistant. "
    "For the given country, find the OFFICIAL government website URLs for the "
    "following visa categories: student visa, work visa, and spouse/partner visa.\n\n"
    "Rules:\n"
    "1. Only return URLs from official government domains (e.g. .gov, .gov.uk, "
    ".gc.ca, .gob, .go.jp, etc.).\n"
    "2. If you cannot find an official URL for a category, return null for that entry.\n"
    "3. Return ONLY valid JSON — no markdown fences, no commentary.\n\n"
    "Return format (strict JSON):\n"
    "{\n"
    '  "student": [{"url": "<URL>", "title": "<page title>"}],\n'
    '  "work":    [{"url": "<URL>", "title": "<page title>"}],\n'
    '  "spouse":  [{"url": "<URL>", "title": "<page title>"}]\n'
    "}\n"
)


def configure_gemini() -> genai.Client:
    """Initialise and return the Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def load_countries() -> list[str]:
    """Load country slugs from config."""
    with open(COUNTRIES_PATH, "r", encoding="utf-8") as fh:
        countries = json.load(fh)
    logger.info("Loaded %d countries from %s", len(countries), COUNTRIES_PATH)
    return countries


def load_existing_sources() -> dict:
    """Load existing sources.json or return empty dict."""
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            if content:
                return json.loads(content)
    return {}


def save_sources(sources: dict) -> None:
    """Persist sources dict to config/sources.json."""
    with open(SOURCES_PATH, "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=2, ensure_ascii=False)
    logger.info("Saved sources to %s", SOURCES_PATH)


def discover_urls(client: genai.Client, country_slug: str) -> dict | None:
    """Ask Gemini for official visa URLs for a single country."""
    country_display = country_slug.replace("-", " ").title()
    prompt = (
        f"Find the official government visa/immigration URLs for: {country_display}.\n"
        f"Country slug: {country_slug}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        data = json.loads(raw)
        return data
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON from Gemini for %s: %s", country_slug, exc)
        return None
    except Exception as exc:
        logger.warning("Gemini API error for %s: %s", country_slug, exc)
        return None


def run() -> None:
    """Main pipeline: iterate countries, discover URLs, update sources."""
    client = configure_gemini()
    countries = load_countries()
    sources = load_existing_sources()

    for idx, slug in enumerate(countries, start=1):
        logger.info("[%d/%d] Discovering URLs for: %s", idx, len(countries), slug)

        result = discover_urls(client, slug)
        if result is None:
            logger.warning("Skipping %s — no valid response.", slug)
            continue

        # Merge into sources (preserve existing entries, update with new)
        if slug not in sources:
            sources[slug] = {}

        for category in VISA_CATEGORIES:
            urls = result.get(category)
            if urls is not None:
                sources[slug][category] = urls

        # Respect API rate limits
        time.sleep(1.5)

    save_sources(sources)
    logger.info("Pathfinder complete — %d countries processed.", len(countries))


if __name__ == "__main__":
    run()
