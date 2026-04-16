"""
Modular Scraper Engine — scrapes official visa pages and structures data via AI.

Input:  config/sources.json, config/apps.json
Output: data/{country}_{category}.json

1. Loads URLs from sources.json (populated by pathfinder.py).
2. Scrapes raw text with requests + BeautifulSoup.
3. Sends text to Gemini 1.5 Flash for structured extraction (original language).
4. Appends country-specific app download links from apps.json.
5. Writes one JSON file per country-category pair into data/.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCES_PATH = ROOT_DIR / "config" / "sources.json"
APPS_PATH = ROOT_DIR / "config" / "apps.json"
DATA_DIR = ROOT_DIR / "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("engine_scraper")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = (
    "Mozilla/5.0 (compatible; LandoBot/1.0; +https://github.com/lando-data)"
)

STRUCTURING_SYSTEM_PROMPT = (
    "Extract visa details from the provided text in its original language. "
    "Output a valid JSON with fields: visa_category, visa_type_detail, "
    "processing_time, fee_amount, currency, last_updated_official, "
    "reference_urls. If data is missing, return null for that field.\n\n"
    "Rules:\n"
    "1. DO NOT translate any content. Keep all text in the original language.\n"
    "2. Return ONLY valid JSON — no markdown fences, no commentary.\n"
    "3. If multiple visa sub-types exist, return a JSON array of objects.\n"
    "4. Dates should be in ISO 8601 format where possible.\n"
)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def configure_gemini() -> genai.Client:
    """Initialise and return the Gemini client for structuring."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------
def load_sources() -> dict:
    """Load sources.json."""
    with open(SOURCES_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_apps() -> dict:
    """Load apps.json (may not exist yet)."""
    if APPS_PATH.exists():
        with open(APPS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
def scrape_page(url: str) -> str | None:
    """Fetch a URL and return cleaned body text."""
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive whitespace
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def scrape_sources(source_entries: list[dict]) -> str:
    """Scrape all URLs for a single category and merge text."""
    all_text: list[str] = []
    for entry in source_entries:
        url = entry.get("url")
        if not url:
            continue
        logger.info("  Scraping: %s", url)
        text = scrape_page(url)
        if text:
            all_text.append(f"--- Source: {url} ---\n{text}")
    return "\n\n".join(all_text)


# ---------------------------------------------------------------------------
# AI Structuring
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
def _call_gemini(client: genai.Client, prompt: str, system_instruction: str) -> str:
    """Call Gemini with retry on transient errors (503, 429, etc.)."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    return response.text.strip()


def _parse_json(raw: str) -> dict | list:
    """Strip markdown fences and parse JSON."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


def structure_with_ai(
    client: genai.Client, raw_text: str, country: str, category: str
) -> dict | list | None:
    """Send raw text to Gemini and parse the structured response."""
    if not raw_text.strip():
        return None

    # Truncate very long pages to stay within token limits
    max_chars = 60_000
    if len(raw_text) > max_chars:
        raw_text = raw_text[:max_chars]

    prompt = (
        f"Country: {country.replace('-', ' ').title()}\n"
        f"Visa category: {category}\n\n"
        f"--- BEGIN PAGE TEXT ---\n{raw_text}\n--- END PAGE TEXT ---"
    )

    try:
        raw = _call_gemini(client, prompt, STRUCTURING_SYSTEM_PROMPT)
        return _parse_json(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON from Gemini for %s/%s: %s", country, category, exc)
        return None
    except Exception as exc:
        logger.warning("Gemini error for %s/%s: %s", country, category, exc)
        return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_output(country: str, category: str, data: dict | list, apps: list) -> None:
    """Write structured data + app links to data/{country}_{category}.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{country}_{category}.json"
    filepath = DATA_DIR / filename

    output = {
        "country": country,
        "visa_category": category,
        "data": data,
        "essential_apps": apps,
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    logger.info("  Saved %s", filepath)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run() -> None:
    """Iterate sources, scrape, structure, and save."""
    client = configure_gemini()
    sources = load_sources()
    apps = load_apps()

    if not sources:
        logger.warning(
            "sources.json is empty. Run pathfinder.py first to discover URLs."
        )
        return

    # Remove stale data files for countries no longer in sources
    if DATA_DIR.exists():
        valid_prefixes = set(sources.keys())
        for filepath in DATA_DIR.glob("*.json"):
            # Filename format: {country}_{category}.json
            prefix = filepath.stem.rsplit("_", 1)[0]
            if prefix not in valid_prefixes:
                filepath.unlink()
                logger.info("Removed stale data file: %s", filepath.name)

    total_countries = len(sources)
    for idx, (country, categories) in enumerate(sources.items(), start=1):
        logger.info("[%d/%d] Processing country: %s", idx, total_countries, country)

        country_apps = apps.get(country, [])

        for category, entries in categories.items():
            logger.info("  Category: %s (%d source(s))", category, len(entries))

            raw_text = scrape_sources(entries)
            if not raw_text:
                logger.warning("  No text scraped for %s/%s — skipping.", country, category)
                continue

            structured = structure_with_ai(client, raw_text, country, category)
            if structured is None:
                logger.warning("  AI structuring failed for %s/%s — skipping.", country, category)
                continue

            save_output(country, category, structured, country_apps)

            # Respect rate limits
            time.sleep(1.5)

    logger.info("Engine scraper complete.")


if __name__ == "__main__":
    run()
