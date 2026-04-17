"""
Scraper for witzepause.com
Collects all jokes from all categories and saves them to witze.txt
One joke per block separated by blank lines.
"""

import argparse
import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.witzepause.com"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "witze.txt")
DELAY = 0.3  # seconds between requests
TIMEOUT = 10

EXCLUDED_EXACT_PATHS = {
    "/",
    "/zufall",
    "/eintragen",
    "/impressum",
    "/datenschutz",
    "/nutzungsbedingungen",
}

EXCLUDED_PREFIXES = (
    "/suche",
    "/ajax",
    "/img",
)


def get_session():
    """Create a requests session with proper headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session


def normalize_internal_path(href, include_query=False):
    """
    Normalize internal links to '/slug' form.

    If include_query=True, preserve URL query string (e.g. '/slug?page=2').
    Returns None for external/invalid links.
    """
    if not href:
        return None

    href = href.strip()
    parsed_base = urlparse(BASE_URL)

    if href.startswith("http://") or href.startswith("https://"):
        parsed = urlparse(href)
        if parsed.netloc and parsed.netloc != parsed_base.netloc:
            return None
        path = parsed.path or "/"
        query = parsed.query
    else:
        parsed = urlparse(href)
        path = parsed.path or "/"
        query = parsed.query

    if not path.startswith("/"):
        return None

    path = re.sub(r"/+", "/", path)

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    if include_query and query:
        return f"{path}?{query}"

    return path


def is_probable_category_path(path):
    """Heuristic filter for category-like paths on witzepause.com."""
    if not path:
        return False

    path = path.split("?", 1)[0]

    if path in EXCLUDED_EXACT_PATHS:
        return False

    if any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False

    # Only top-level slugs, e.g. /kevin-witze
    return bool(re.fullmatch(r"/[a-z0-9\-]+", path))


def fetch_soup(session, url):
    """Fetch a URL and return a parsed BeautifulSoup object, or None on errors."""
    try:
        resp = session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        resp.encoding = 'utf-8'
        return BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def get_categories(session):
    """Extract all category links from the homepage."""
    print("Fetching homepage...")
    soup = fetch_soup(session, BASE_URL)
    if not soup:
        return []

    categories = []
    seen_paths = set()

    # Extract internal one-segment links (category candidates)
    for link in soup.find_all('a', href=True):
        path = normalize_internal_path(link.get('href'))
        if not is_probable_category_path(path):
            continue

        if path in seen_paths:
            continue

        seen_paths.add(path)
        categories.append(urljoin(BASE_URL, path))

    print(f"Found {len(categories)} categories")
    return categories


def clean_joke_text(text):
    """Clean and normalize joke text."""
    if not text:
        return ""

    # Remove common metadata tail if present
    text = re.split(r"\s+—\s*Autor\s*:\s*", text, maxsplit=1)[0]
    text = re.split(r"\s+Autor\s*:\s*", text, maxsplit=1)[0]

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def extract_jokes_from_soup(soup):
    """Extract all jokes from a parsed page."""
    jokes = []

    # Primary selector used on category pages
    joke_blocks = soup.select('blockquote.well.well-small')

    # Fallback if markup changes
    if not joke_blocks:
        joke_blocks = soup.find_all('blockquote')

    for block in joke_blocks:
        p_tag = block.find('p')
        if p_tag and p_tag.get_text(strip=True):
            raw_text = p_tag.get_text(' ', strip=True)
        else:
            raw_text = block.get_text(' ', strip=True)

        text = clean_joke_text(raw_text)

        # Avoid junk entries
        if not text or len(text) < 5:
            continue

        jokes.append(text)

    return jokes


def find_next_page_url(soup, current_url):
    """Find URL of the next pagination page, if present."""
    # Most reliable marker on this site: text contains "Weiter"
    for link in soup.find_all('a', href=True):
        label = link.get_text(' ', strip=True)
        if "Weiter" in label:
            candidate = urljoin(current_url, link['href'])
            path = normalize_internal_path(candidate, include_query=True)
            if not path:
                continue
            return urljoin(BASE_URL, path)

    # Fallback for rel="next"
    for link in soup.find_all('a', href=True):
        rel_values = [v.lower() for v in (link.get('rel') or [])]
        if 'next' in rel_values:
            candidate = urljoin(current_url, link['href'])
            path = normalize_internal_path(candidate, include_query=True)
            if not path:
                continue
            return urljoin(BASE_URL, path)

    return None


def get_all_pages_in_category(session, category_url, max_pages=None):
    """Scrape all jokes from all pages in a category via pagination links."""
    all_jokes = []
    visited_pages = set()

    current_url = category_url
    page_num = 1

    while current_url:
        if current_url in visited_pages:
            print("loop detected, stopping")
            break

        if max_pages is not None and page_num > max_pages:
            print(f"page limit reached ({max_pages}), stopping")
            break

        visited_pages.add(current_url)

        print(f"  Scraping page {page_num}: {current_url}...", end=" ", flush=True)
        soup = fetch_soup(session, current_url)
        if not soup:
            print("fetch failed, stopping")
            break

        jokes = extract_jokes_from_soup(soup)
        print(f"got {len(jokes)} jokes")

        # Non-category page or unexpected structure
        if not jokes and page_num == 1:
            print("no jokes found on first page, skipping category")
            break

        all_jokes.extend(jokes)

        next_page_url = find_next_page_url(soup, current_url)
        if not next_page_url or next_page_url in visited_pages:
            break

        current_url = next_page_url
        page_num += 1
        time.sleep(DELAY)

    return all_jokes


def scrape_all_categories(session, max_categories=None, max_pages_per_category=None):
    """Scrape jokes from all categories."""
    categories = get_categories(session)
    if max_categories is not None:
        categories = categories[:max_categories]

    all_jokes = []
    seen_jokes = set()

    for i, category_url in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Scraping {category_url}...")

        jokes = get_all_pages_in_category(
            session,
            category_url,
            max_pages=max_pages_per_category,
        )

        before_unique = len(all_jokes)
        for joke in jokes:
            if joke not in seen_jokes:
                all_jokes.append(joke)
                seen_jokes.add(joke)

        unique_added = len(all_jokes) - before_unique
        print(
            f"  Category total: {len(jokes)} jokes "
            f"({unique_added} new unique, unique so far: {len(all_jokes)})"
        )

        time.sleep(DELAY)

    return all_jokes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape all jokes from witzepause.com into a TXT file."
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="Output TXT file path (default: nanoGPT/data/witze.txt)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY,
        help="Delay in seconds between requests (default: 0.3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT,
        help="Request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=None,
        help="Optional limit for category count (debug/testing)",
    )
    parser.add_argument(
        "--max-pages-per-category",
        type=int,
        default=None,
        help="Optional limit for pages per category (debug/testing)",
    )
    return parser.parse_args()


def main():
    global DELAY, TIMEOUT

    args = parse_args()
    DELAY = args.delay
    TIMEOUT = args.timeout

    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(__file__), output_file)

    print("=" * 60)
    print("Witze-Pause.com Scraper")
    print("=" * 60)
    print("Hinweis: Laut Nutzungsbedingungen liegen Urheberrechte bei den jeweiligen Autoren.")
    print("Bitte prüfe die rechtliche Nutzung der Daten für Trainingszwecke eigenverantwortlich.")
    print("=" * 60)

    session = get_session()

    try:
        all_jokes = scrape_all_categories(
            session,
            max_categories=args.max_categories,
            max_pages_per_category=args.max_pages_per_category,
        )

        print("\n" + "=" * 60)
        print(f"Total unique jokes collected: {len(all_jokes)}")
        print(f"Writing to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(all_jokes))

        # Print stats
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"[OK] Done! File size: {file_size_mb:.2f} MB")
        print("=" * 60)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
