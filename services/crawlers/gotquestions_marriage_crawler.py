import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests

LIST_URL = "https://www.gotquestions.org/content.html"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_DELAY_SECONDS = 1
USER_AGENT = (
    "WaChatBot/1.0 (+https://github.com/avictorino/wachat; "
    "contact=admin@wachat.local)"
)

LOGGER = logging.getLogger(__name__)

VERSE_PATTERN = re.compile(
    r"\b(?:[1-3]\s*)?"
    r"(?:Genesis|Gen|Exodus|Exod|Leviticus|Lev|Numbers|Num|Deuteronomy|Deut|"
    r"Joshua|Josh|Judges|Ruth|1\s*Samuel|2\s*Samuel|1\s*Kings|2\s*Kings|"
    r"1\s*Chronicles|2\s*Chronicles|Ezra|Nehemiah|Esther|Job|Psalms?|Psalm|Ps|"
    r"Proverbs|Prov|Ecclesiastes|Song of Songs|Song|Isaiah|Isa|Jeremiah|Jer|"
    r"Lamentations|Ezekiel|Ezek|Daniel|Dan|Hosea|Joel|Amos|Obadiah|Jonah|Micah|"
    r"Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Matthew|Matt|Mark|Luke|"
    r"John|Acts|Romans|Rom|1\s*Corinthians|2\s*Corinthians|Galatians|Ephesians|"
    r"Philippians|Colossians|1\s*Thessalonians|2\s*Thessalonians|1\s*Timothy|"
    r"2\s*Timothy|Titus|Philemon|Hebrews|James|1\s*Peter|2\s*Peter|"
    r"1\s*John|2\s*John|3\s*John|Jude|Revelation|Rev)\s+"
    r"\d{1,3}:\d{1,3}(?:-\d{1,3})?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionAnswerPair:
    question: str
    answer: str
    verses: List[str]


@dataclass(frozen=True)
class ParsedArticle:
    title: str
    url: str
    question: str
    pairs: List[QuestionAnswerPair]


class GotQuestionsMarriageCrawler:
    def __init__(self, list_url: str = LIST_URL):
        self.list_url = list_url
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def fetch_list_page(self) -> str:
        return self.fetch_page(self.list_url)

    def fetch_page(self, url: str) -> str:
        LOGGER.info("event=fetch_page url=%s", url)
        response = self._session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)
        return response.text

    def extract_links(
        self, list_html: str, base_url: Optional[str] = None
    ) -> List[str]:
        soup = self._build_soup(list_html)
        base = base_url or self.list_url
        parsed_base = urlparse(self.list_url)
        discovered_links = set()

        for anchor in soup.select("a[href]"):
            href = str(anchor.get("href", "")).strip()
            if not href:
                continue
            absolute_url = urljoin(base, href)
            parsed_url = urlparse(absolute_url)
            if parsed_url.netloc != parsed_base.netloc:
                continue
            if not parsed_url.path.endswith(".html"):
                continue
            normalized = self._normalize_url(absolute_url)
            discovered_links.add(normalized)

        return sorted(discovered_links)

    def crawl_article_links(self) -> List[str]:
        to_visit = [self.list_url]
        visited_navigation_pages = set()
        article_links = set()

        while to_visit:
            current_url = to_visit.pop(0)
            if current_url in visited_navigation_pages:
                continue

            visited_navigation_pages.add(current_url)
            html = self.fetch_page(current_url)
            links = self.extract_links(html, base_url=current_url)

            for link in links:
                if self._is_navigation_page(link):
                    if link not in visited_navigation_pages:
                        to_visit.append(link)
                    continue
                if self._is_article_page(link):
                    article_links.add(link)

            LOGGER.info(
                "event=crawl_progress visited_navigation_pages=%s queued=%s articles=%s current=%s",
                len(visited_navigation_pages),
                len(to_visit),
                len(article_links),
                current_url,
            )

        return sorted(article_links)

    def fetch_article(self, url: str) -> str:
        return self.fetch_page(url)

    def parse_article(self, url: str, article_html: str) -> ParsedArticle:
        soup = self._build_soup(article_html)

        # Remove non-content nodes before extracting main text.
        for tag_name in ["script", "style", "noscript", "header", "footer", "nav"]:
            for element in soup.find_all(tag_name):
                element.decompose()

        main_container = soup.find("article")
        if main_container is None:
            main_container = soup.find("main")
        if main_container is None:
            main_container = soup.find(
                "div", class_=re.compile("content", re.IGNORECASE)
            )
        if main_container is None:
            main_container = soup.body
        if main_container is None:
            raise ValueError(
                f"Could not locate main content container for article {url}."
            )

        title_text = ""
        if soup.title and soup.title.text:
            title_text = soup.title.text.strip()
        if not title_text:
            heading = main_container.find(["h1", "h2"])
            title_text = heading.get_text(" ", strip=True) if heading else url

        main_question = self._extract_main_question(main_container, fallback=title_text)
        pairs = self._extract_question_answer_pairs(
            main_container=main_container,
            main_question=main_question,
        )
        if not pairs:
            raise ValueError(f"No question/answer pairs extracted from article {url}.")

        return ParsedArticle(
            title=title_text,
            url=url,
            question=main_question,
            pairs=pairs,
        )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _build_soup(self, html: str):
        try:
            from bs4 import BeautifulSoup
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "beautifulsoup4 is required for GotQuestions crawling. "
                "Install dependency 'beautifulsoup4'."
            ) from exc
        return BeautifulSoup(html, "html.parser")

    def _extract_main_question(self, main_container, fallback: str) -> str:
        heading = main_container.find("h1")
        if heading and heading.get_text(" ", strip=True):
            return heading.get_text(" ", strip=True)
        heading = main_container.find("h2")
        if heading and heading.get_text(" ", strip=True):
            return heading.get_text(" ", strip=True)
        return fallback.strip()

    def _extract_question_answer_pairs(
        self,
        main_container,
        main_question: str,
    ) -> List[QuestionAnswerPair]:
        content_nodes = main_container.find_all(["h2", "h3", "p", "li"])
        has_answer_marker = any(
            node.get_text(" ", strip=True).lower() == "answer" for node in content_nodes
        )
        answer_started = not has_answer_marker
        current_subquestion = ""
        current_answer_parts: List[str] = []
        pairs: List[QuestionAnswerPair] = []

        for node in content_nodes:
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            normalized = text.lower()

            if not answer_started:
                if normalized == "answer":
                    answer_started = True
                continue

            if normalized in {"translate", "audio", "related topics", "return to"}:
                continue

            if node.name in {"h2", "h3"}:
                self._flush_pair(
                    pairs=pairs,
                    main_question=main_question,
                    subquestion=current_subquestion,
                    answer_parts=current_answer_parts,
                )
                current_subquestion = text
                current_answer_parts = []
                continue

            current_answer_parts.append(text)

        self._flush_pair(
            pairs=pairs,
            main_question=main_question,
            subquestion=current_subquestion,
            answer_parts=current_answer_parts,
        )
        return pairs

    def _flush_pair(
        self,
        pairs: List[QuestionAnswerPair],
        main_question: str,
        subquestion: str,
        answer_parts: List[str],
    ) -> None:
        if not answer_parts:
            return

        answer_text = self._clean_text("\n\n".join(answer_parts))
        if subquestion:
            answer_text = self._clean_text(f"{subquestion}\n\n{answer_text}")
        verses = sorted(set(VERSE_PATTERN.findall(answer_text)))
        pairs.append(
            QuestionAnswerPair(
                question=main_question,
                answer=answer_text,
                verses=verses,
            )
        )

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        cleaned_path = parsed.path.rstrip("/")
        if not cleaned_path:
            cleaned_path = "/"
        return f"{parsed.scheme}://{parsed.netloc}{cleaned_path}"

    def _is_navigation_page(self, url: str) -> bool:
        path = urlparse(url).path.rsplit("/", 1)[-1].lower()
        return path.startswith("content") and path.endswith(".html")

    def _is_article_page(self, url: str) -> bool:
        path = urlparse(url).path.rsplit("/", 1)[-1].lower()
        if not path.endswith(".html"):
            return False
        return not path.startswith("content")
