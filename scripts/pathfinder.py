"""
Pathfinder AI Explorer — discovers official government visa URLs for each country.

Input:  config/countries.json
Output: config/sources.json (updated with validated URLs)
        config/apps.json   (essential relocation apps per country)

Uses Gemini 2.5 Flash to discover visa categories dynamically per country.

sources.json schema (per category):
  { "urls": [...], "source": "ai" | "manual" }

Entries tagged "source": "manual" (added via GitHub Issues) are never
overwritten or removed by the pathfinder.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
COUNTRIES_PATH = ROOT_DIR / "config" / "countries.json"
SOURCES_PATH = ROOT_DIR / "config" / "sources.json"
APPS_PATH = ROOT_DIR / "config" / "apps.json"

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
SYSTEM_PROMPT = (
    "You are an immigration research assistant. "
    "For the given country, identify the most relevant visa/permit categories "
    "based on that country's actual immigration system, then find the OFFICIAL "
    "government website URLs for each category.\n\n"
    "Consider categories such as (but not limited to):\n"
    "- Student visa\n"
    "- Work visa / work permit\n"
    "- Spouse / partner / family reunification visa\n"
    "- Skilled worker / talent visa\n"
    "- Digital nomad visa\n"
    "- Investor / entrepreneur visa\n"
    "- Working holiday visa\n"
    "- Permanent residency\n"
    "- Any other visa type unique to this country (e.g. Gold Card, Top Talent Pass)\n\n"
    "Rules:\n"
    "1. Only return URLs from official government domains (e.g. .gov, .gov.uk, "
    ".gc.ca, .gob, .go.jp, etc.).\n"
    "2. Use short, lowercase, hyphenated keys for each category (e.g. 'student', "
    "'skilled-worker', 'digital-nomad', 'working-holiday', 'permanent-residency').\n"
    "3. Include 5-10 of the most relevant categories for the country.\n"
    "4. If you cannot find an official URL for a category, omit it entirely.\n"
    "5. Return ONLY valid JSON — no markdown fences, no commentary.\n\n"
    "Return format (strict JSON):\n"
    "{\n"
    '  "<category-slug>": [{"url": "<URL>", "title": "<page title>"}],\n'
    '  "<category-slug>": [{"url": "<URL>", "title": "<page title>"}]\n'
    "}\n"
)

APPS_SYSTEM_PROMPT = (
    "You are a relocation assistant. For the given country, find the most essential "
    "mobile apps that a new immigrant would need. Focus on:\n"
    "- Government identity/authentication apps\n"
    "- Healthcare apps\n"
    "- Public transport apps\n"
    "- Banking/payment apps widely used in the country\n\n"
    "Rules:\n"
    "1. Only include apps that are real and currently available.\n"
    "2. Return 3-5 of the MOST essential apps.\n"
    "3. Return ONLY valid JSON — no markdown fences, no commentary.\n\n"
    "Return format (strict JSON array):\n"
    "[\n"
    '  {"app_name": "<name>", "description": "<one-line description>", '
    '"platform": "iOS/Android", '
    '"download_url_ios": "<App Store URL or null>", '
    '"download_url_android": "<Play Store URL or null>"}\n'
    "]\n"
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


def _diff_json(old: dict, new: dict) -> bool:
    """Return True if the two dicts serialise to different JSON."""
    return json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True)


def save_sources(sources: dict, original: dict) -> None:
    """Persist sources dict to config/sources.json only if changed."""
    if not _diff_json(original, sources):
        logger.info("sources.json unchanged — skipping write.")
        return
    with open(SOURCES_PATH, "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=2, ensure_ascii=False)
    logger.info("Saved sources to %s", SOURCES_PATH)


def load_existing_apps() -> dict:
    """Load existing apps.json or return empty dict."""
    if APPS_PATH.exists():
        with open(APPS_PATH, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            if content:
                return json.loads(content)
    return {}


def save_apps(apps: dict, original: dict) -> None:
    """Persist apps dict to config/apps.json only if changed."""
    if not _diff_json(original, apps):
        logger.info("apps.json unchanged — skipping write.")
        return
    with open(APPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(apps, fh, indent=2, ensure_ascii=False)
    logger.info("Saved apps to %s", APPS_PATH)


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


def discover_urls(client: genai.Client, country_slug: str) -> dict | None:
    """Ask Gemini for official visa URLs for a single country."""
    country_display = country_slug.replace("-", " ").title()
    prompt = (
        f"Find the official government visa/immigration URLs for: {country_display}.\n"
        f"Country slug: {country_slug}"
    )

    try:
        raw = _call_gemini(client, prompt, SYSTEM_PROMPT)
        return _parse_json(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON from Gemini for %s: %s", country_slug, exc)
        return None
    except Exception as exc:
        logger.warning("Gemini API error for %s: %s", country_slug, exc)
        return None


def discover_apps(client: genai.Client, country_slug: str) -> list | None:
    """Ask Gemini for essential relocation apps for a single country."""
    country_display = country_slug.replace("-", " ").title()
    prompt = (
        f"Find the most essential mobile apps for someone relocating to: {country_display}."
    )

    try:
        raw = _call_gemini(client, prompt, APPS_SYSTEM_PROMPT)
        return _parse_json(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON (apps) from Gemini for %s: %s", country_slug, exc)
        return None
    except Exception as exc:
        logger.warning("Gemini API error (apps) for %s: %s", country_slug, exc)
        return None


def _migrate_sources(sources: dict) -> dict:
    """Migrate legacy flat format → wrapped {urls, source} format."""
    migrated = {}
    for country, categories in sources.items():
        migrated[country] = {}
        for cat, value in categories.items():
            if isinstance(value, dict) and "urls" in value:
                migrated[country][cat] = value          # already new format
            elif isinstance(value, list):
                migrated[country][cat] = {"urls": value, "source": "ai"}
            else:
                migrated[country][cat] = {"urls": [], "source": "ai"}
    return migrated


def run() -> None:
    """Main pipeline: iterate countries, discover URLs, update sources."""
    client = configure_gemini()
    countries = load_countries()
    sources_raw = load_existing_sources()
    sources = _migrate_sources(sources_raw)
    original_sources = json.loads(json.dumps(sources))  # deep copy for diff

    apps = load_existing_apps()
    original_apps = json.loads(json.dumps(apps))        # deep copy for diff

    for idx, slug in enumerate(countries, start=1):
        logger.info("[%d/%d] Discovering URLs for: %s", idx, len(countries), slug)

        result = discover_urls(client, slug)
        if result is None:
            logger.warning("Skipping %s — no valid response.", slug)
        else:
            # Merge: preserve manual entries, update/add AI entries
            existing = sources.get(slug, {})
            for cat, urls in result.items():
                if cat in existing and existing[cat].get("source") == "manual":
                    logger.info("  Skipping manual category: %s/%s", slug, cat)
                    continue
                existing[cat] = {"urls": urls, "source": "ai"}
            sources[slug] = existing
            logger.info("  Found %d AI categories: %s", len(result), list(result.keys()))

        # Discover essential apps
        logger.info("[%d/%d] Discovering apps for: %s", idx, len(countries), slug)
        app_result = discover_apps(client, slug)
        if app_result is not None:
            apps[slug] = app_result

        # Respect API rate limits
        time.sleep(1.5)

    # Remove countries no longer in the list
    removed = [k for k in sources if k not in countries]
    for slug in removed:
        del sources[slug]
        logger.info("Removed stale country from sources: %s", slug)

    removed_apps = [k for k in apps if k not in countries]
    for slug in removed_apps:
        del apps[slug]
        logger.info("Removed stale country from apps: %s", slug)

    save_sources(sources, original_sources)
    save_apps(apps, original_apps)
    logger.info("Pathfinder complete — %d countries processed.", len(countries))


if __name__ == "__main__":
    run()
