import csv
import html
import json
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import urlencode

import requests
from django.core.management.base import BaseCommand, CommandError

GOODREADS_QUOTES_BASE_URL = "https://www.goodreads.com/quotes/tag/{tag}"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_TAG = "relationships"

"""
python3 manage.py scrape_goodreads_quotes --tag relationships --output-csv ./relationships_quotes.csv
"""


@dataclass(frozen=True)
class QuoteItem:
    text: str
    author: str
    page: int
    url: str


def _strip_html_tags(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw)


def _normalize_spaces(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def _decode_text(raw: str) -> str:
    return html.unescape(_normalize_spaces(_strip_html_tags(raw)))


def _extract_quotes_from_page(
    page_html: str, page_number: int, page_url: str
) -> List[QuoteItem]:
    quote_blocks = re.findall(
        r'<div class="quoteText">\s*(.*?)\s*</div>\s*<div class="quoteFooter">',
        page_html,
        flags=re.DOTALL,
    )
    quotes: List[QuoteItem] = []
    for block in quote_blocks:
        author_match = re.search(
            r'<span class="authorOrTitle">\s*(.*?)\s*</span>', block
        )
        if not author_match:
            raise CommandError(
                f"Could not extract author from quote block on page {page_number} ({page_url})."
            )

        author = _decode_text(author_match.group(1)).rstrip(",")
        body_html = block.split("<br/>", 1)[0]
        quote_text = _decode_text(body_html).strip('“”" ')
        if not quote_text:
            raise CommandError(
                f"Could not extract quote text from quote block on page {page_number} ({page_url})."
            )

        quotes.append(
            QuoteItem(
                text=quote_text,
                author=author,
                page=page_number,
                url=page_url,
            )
        )
    return quotes


def _has_next_page(page_html: str) -> bool:
    return bool(re.search(r'rel="next"', page_html))


class Command(BaseCommand):
    help = "Scrape quotes from Goodreads by tag (default: relationships)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--tag",
            default=DEFAULT_TAG,
            help=f"Goodreads quote tag. Default: {DEFAULT_TAG}",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=0,
            help="Maximum number of pages to scrape. 0 means all pages.",
        )
        parser.add_argument(
            "--output-json",
            default="",
            help="Optional output JSON file path.",
        )
        parser.add_argument(
            "--output-csv",
            default="",
            help="Optional output CSV file path.",
        )

    def handle(self, *args, **options):
        tag = str(options["tag"]).strip()
        if not tag:
            raise CommandError("Parameter --tag is required.")

        max_pages = int(options["max_pages"])
        if max_pages < 0:
            raise CommandError("Parameter --max-pages cannot be negative.")

        output_json = str(options["output_json"]).strip()
        output_csv = str(options["output_csv"]).strip()

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            }
        )

        all_quotes: List[QuoteItem] = []
        page = 1

        while True:
            if max_pages and page > max_pages:
                break

            page_url = f"{GOODREADS_QUOTES_BASE_URL.format(tag=tag)}?{urlencode({'page': page})}"
            self.stdout.write(f"Scraping page {page}: {page_url}")

            response = session.get(page_url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()

            quotes = _extract_quotes_from_page(
                page_html=response.text,
                page_number=page,
                page_url=page_url,
            )

            if not quotes:
                if page == 1:
                    raise CommandError(
                        f"No quotes found for tag '{tag}' on the first page."
                    )
                break

            all_quotes.extend(quotes)
            self.stdout.write(f"  Found {len(quotes)} quotes on page {page}.")

            if not _has_next_page(response.text):
                break
            page += 1

        self.stdout.write(
            self.style.SUCCESS(f"Total quotes extracted: {len(all_quotes)}")
        )

        if output_json:
            with open(output_json, "w", encoding="utf-8") as json_file:
                json.dump(
                    [
                        {
                            "text": item.text,
                            "author": item.author,
                            "page": item.page,
                            "url": item.url,
                        }
                        for item in all_quotes
                    ],
                    json_file,
                    ensure_ascii=False,
                    indent=2,
                )
            self.stdout.write(self.style.SUCCESS(f"Saved JSON: {output_json}"))

        if output_csv:
            with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["text", "author", "page", "url"],
                )
                writer.writeheader()
                for item in all_quotes:
                    writer.writerow(
                        {
                            "text": item.text,
                            "author": item.author,
                            "page": item.page,
                            "url": item.url,
                        }
                    )
            self.stdout.write(self.style.SUCCESS(f"Saved CSV: {output_csv}"))

        if not output_json and not output_csv:
            for idx, item in enumerate(all_quotes, start=1):
                self.stdout.write(f'{idx}. "{item.text}" — {item.author}')
