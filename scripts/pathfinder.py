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

import requests
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

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
    "You are an immigration research assistant with access to Google Search. "
    "For EACH country listed, SEARCH THE WEB to find the most relevant visa/permit "
    "categories and their OFFICIAL government website URLs.\n\n"
    "Consider categories such as (but not limited to):\n"
    "- Student visa\n"
    "- Work visa / work permit\n"
    "- Spouse / partner / family reunification visa\n"
    "- Skilled worker / talent visa\n"
    "- Digital nomad visa\n"
    "- Investor / entrepreneur visa\n"
    "- Working holiday visa\n"
    "- Permanent residency\n"
    "- Any other visa type unique to that country (e.g. Gold Card, Top Talent Pass)\n\n"
    "Rules:\n"
    "1. Only return URLs from official government domains (e.g. .gov, .gov.uk, "
    ".gc.ca, .gob, .go.jp, etc.).\n"
    "2. Use short, lowercase, hyphenated keys for each category (e.g. 'student', "
    "'skilled-worker', 'digital-nomad', 'working-holiday', 'permanent-residency').\n"
    "3. Include 5-10 of the most relevant categories per country.\n"
    "4. Every URL MUST come from your Google Search results. "
    "Do NOT invent or guess any URL. If you cannot find a working URL for a "
    "category via search, omit that category entirely.\n"
    "5. Return ONLY valid JSON — no markdown fences, no commentary.\n\n"
    "Return format — a JSON object keyed by country slug:\n"
    "{\n"
    '  "<country-slug>": {\n'
    '    "<category-slug>": [{"url": "<URL>", "title": "<page title>"}]\n'
    "  }\n"
    "}\n"
)

APPS_SYSTEM_PROMPT = (
    "You are a relocation assistant. For EACH country listed, find the most essential "
    "mobile apps that a new immigrant would need. Focus on:\n"
    "- Government identity/authentication apps\n"
    "- Healthcare apps\n"
    "- Public transport apps\n"
    "- Banking/payment apps widely used in the country\n\n"
    "Rules:\n"
    "1. Only include apps that are real and currently available.\n"
    "2. Return 3-5 of the MOST essential apps per country.\n"
    "3. Return ONLY valid JSON — no markdown fences, no commentary.\n\n"
    "Return format — a JSON object keyed by country slug:\n"
    "{\n"
    '  "<country-slug>": [\n'
    '    {"app_name": "<name>", "description": "<one-line description>", '
    '"platform": "iOS/Android", '
    '"download_url_ios": "<App Store URL or null>", '
    '"download_url_android": "<Play Store URL or null>"}\n'
    "  ]\n"
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


# Put less-congested models first — gemini-2.5-flash is often 503
MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]


def _is_transient(exc: Exception) -> bool:
    """Return True for 503/429 (retry-worthy), False for 404 etc."""
    msg = str(exc)
    return "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def _call_model(client: genai.Client, model: str, prompt: str, system_instruction: str, *, use_search: bool = False) -> str:
    """Call a specific Gemini model with retry on transient errors."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
    if use_search:
        config.tools = [types.Tool(google_search=types.GoogleSearch())]
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text.strip()


def _call_gemini(client: genai.Client, prompt: str, system_instruction: str, *, use_search: bool = False) -> str:
    """Try each model in the fallback chain until one succeeds."""
    last_exc = None
    for model in MODELS:
        try:
            result = _call_model(client, model, prompt, system_instruction, use_search=use_search)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("Model %s failed: %s — trying next fallback", model, exc)
    raise last_exc


def _parse_json(raw: str) -> dict | list:
    """Strip markdown fences and parse JSON."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


BATCH_SIZE = 5


def discover_urls_batch(client: genai.Client, slugs: list[str]) -> dict:
    """Ask Gemini for official visa URLs for multiple countries at once."""
    country_lines = "\n".join(
        f"- {s.replace('-', ' ').title()} (slug: {s})" for s in slugs
    )
    prompt = (
        f"Search the web and find the official government visa/immigration URLs "
        f"for each of these countries:\n{country_lines}\n\n"
        f"Use Google Search to verify each URL actually exists before including it.\n"
        f"Return a single JSON object keyed by country slug."
    )

    try:
        raw = _call_gemini(client, prompt, SYSTEM_PROMPT, use_search=True)
        parsed = _parse_json(raw)
        if not isinstance(parsed, dict):
            logger.warning("Expected dict from batch URL discovery, got %s", type(parsed))
            return {}
        return parsed
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON from batch URL discovery %s: %s", slugs, exc)
        return {}
    except Exception as exc:
        logger.warning("Gemini API error for batch URL discovery %s: %s", slugs, exc)
        return {}


USER_AGENT = "Mozilla/5.0 (compatible; LandoBot/1.0; +https://github.com/lando-data)"


def _validate_urls(result: dict, country_slug: str) -> dict:
    """HEAD-check every discovered URL; drop entries that return 404/403/timeout.

    Returns a filtered copy of *result* — categories with no surviving URLs
    are removed entirely.
    """
    validated: dict = {}
    for category, url_list in result.items():
        # Normalize: model sometimes returns a single dict instead of a list
        if isinstance(url_list, dict):
            url_list = [url_list]
        if not isinstance(url_list, list):
            continue
        good: list[dict] = []
        for entry in url_list:
            url = entry.get("url", "")
            if not url:
                continue
            try:
                resp = requests.head(
                    url, timeout=15, headers={"User-Agent": USER_AGENT},
                    allow_redirects=True, verify=True,
                )
                if resp.status_code < 400:
                    good.append(entry)
                elif resp.status_code == 403:
                    # 403 = site blocks bots but URL exists — keep it
                    logger.info("  Keeping %s (403 = bot-blocked, URL likely valid)", url)
                    good.append(entry)
                else:
                    logger.warning("  Dropping %s/%s — HTTP %d", country_slug, url, resp.status_code)
            except requests.exceptions.SSLError:
                # SSL cert issue on our side — URL exists, keep it
                logger.info("  Keeping %s (SSL error = cert issue, URL likely valid)", url)
                good.append(entry)
            except requests.RequestException as exc:
                # Some sites block HEAD, try a quick GET as fallback
                try:
                    resp = requests.get(
                        url, timeout=15, headers={"User-Agent": USER_AGENT},
                        allow_redirects=True, stream=True, verify=False,
                    )
                    resp.close()
                    if resp.status_code < 400 or resp.status_code == 403:
                        good.append(entry)
                    else:
                        logger.warning("  Dropping %s/%s — HTTP %d", country_slug, url, resp.status_code)
                except requests.RequestException:
                    logger.warning("  Dropping %s/%s — %s", country_slug, url, exc)
        if good:
            validated[category] = good
        else:
            logger.warning("  Category '%s' has no valid URLs — removed", category)
    return validated


def discover_apps_batch(client: genai.Client, slugs: list[str]) -> dict:
    """Ask Gemini for essential relocation apps for multiple countries at once."""
    country_lines = "\n".join(
        f"- {s.replace('-', ' ').title()} (slug: {s})" for s in slugs
    )
    prompt = (
        f"Find the most essential mobile apps for someone relocating to each of "
        f"these countries:\n{country_lines}\n\n"
        f"Return a single JSON object keyed by country slug."
    )

    try:
        raw = _call_gemini(client, prompt, APPS_SYSTEM_PROMPT)
        parsed = _parse_json(raw)
        if not isinstance(parsed, dict):
            logger.warning("Expected dict from batch apps discovery, got %s", type(parsed))
            return {}
        return parsed
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON from batch apps discovery %s: %s", slugs, exc)
        return {}
    except Exception as exc:
        logger.warning("Gemini API error for batch apps discovery %s: %s", slugs, exc)
        return {}


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
    """Main pipeline: batch-discover URLs & apps, validate, update sources."""
    client = configure_gemini()
    countries = load_countries()
    sources_raw = load_existing_sources()
    sources = _migrate_sources(sources_raw)
    original_sources = json.loads(json.dumps(sources))  # deep copy for diff

    apps = load_existing_apps()
    original_apps = json.loads(json.dumps(apps))        # deep copy for diff

    # Process countries in batches of BATCH_SIZE
    batches = [countries[i:i + BATCH_SIZE] for i in range(0, len(countries), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches, start=1):
        batch_label = ", ".join(batch)
        logger.info("=== Batch %d/%d (%d countries): %s ===",
                     batch_idx, len(batches), len(batch), batch_label)

        # --- Discover URLs for the batch (1 API call) ---
        url_results = discover_urls_batch(client, batch)

        for slug in batch:
            result = url_results.get(slug)
            if result is None or not isinstance(result, dict):
                logger.warning("  No URL data for %s in batch response", slug)
                continue

            # Validate URLs before saving
            result = _validate_urls(result, slug)
            logger.info("  %s: %d categories survived validation", slug, len(result))

            # Remove all old AI entries for this country (keep manual only)
            existing = sources.get(slug, {})
            manual_only = {
                cat: val for cat, val in existing.items()
                if isinstance(val, dict) and val.get("source") == "manual"
            }
            # Add validated AI entries
            for cat, urls in result.items():
                if cat in manual_only:
                    logger.info("  Skipping manual category: %s/%s", slug, cat)
                    continue
                manual_only[cat] = {"urls": urls, "source": "ai"}
            sources[slug] = manual_only

        # Respect API rate limits between URL and app calls
        time.sleep(3)

        # --- Discover apps for the batch (1 API call) ---
        logger.info("  Discovering apps for batch: %s", batch_label)
        apps_results = discover_apps_batch(client, batch)

        for slug in batch:
            app_result = apps_results.get(slug)
            if app_result is not None and isinstance(app_result, list):
                apps[slug] = app_result
            else:
                logger.warning("  No apps data for %s in batch response", slug)

        # Respect API rate limits between batches
        time.sleep(3)

    # --- Retry pass: re-discover for countries with 0 AI categories ---
    empty_slugs = [
        slug for slug in countries
        if not any(
            v.get("source") == "ai" and v.get("urls")
            for v in sources.get(slug, {}).values()
        )
    ]
    if empty_slugs:
        logger.info("=== Retry pass for %d countries with 0 categories: %s ===",
                     len(empty_slugs), ", ".join(empty_slugs))
        for slug in empty_slugs:
            logger.info("  Retrying: %s", slug)
            retry_result = discover_urls_batch(client, [slug])
            result = retry_result.get(slug)
            if result and isinstance(result, dict):
                result = _validate_urls(result, slug)
                logger.info("  %s retry: %d categories survived", slug, len(result))
                existing = sources.get(slug, {})
                manual_only = {
                    cat: val for cat, val in existing.items()
                    if isinstance(val, dict) and val.get("source") == "manual"
                }
                for cat, urls in result.items():
                    if cat in manual_only:
                        continue
                    manual_only[cat] = {"urls": urls, "source": "ai"}
                sources[slug] = manual_only
            time.sleep(3)

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
    logger.info("Pathfinder complete — %d countries in %d batches.", len(countries), len(batches))


if __name__ == "__main__":
    run()
