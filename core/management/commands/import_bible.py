import os
import re
from html import unescape
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from django.core.management.base import BaseCommand, CommandError
from openai import OpenAI

from core.models import BibleTextFlat, ThemeV2

BASE_URL = "https://ebible.org/porbr2018/"
TRANSLATION = "porbr2018"
EMBEDDING_MODEL = "text-embedding-3-large"

BOOKS = [
    ("Gênesis", "OT", "GEN"),
    ("Êxodo", "OT", "EXO"),
    ("Levítico", "OT", "LEV"),
    ("Números", "OT", "NUM"),
    ("Deuteronômio", "OT", "DEU"),
    ("Josué", "OT", "JOS"),
    ("Juízes", "OT", "JDG"),
    ("Rute", "OT", "RUT"),
    ("1 Samuel", "OT", "1SA"),
    ("2 Samuel", "OT", "2SA"),
    ("1 Reis", "OT", "1KI"),
    ("2 Reis", "OT", "2KI"),
    ("1 Crônicas", "OT", "1CH"),
    ("2 Crônicas", "OT", "2CH"),
    ("Esdras", "OT", "EZR"),
    ("Neemias", "OT", "NEH"),
    ("Ester", "OT", "EST"),
    ("Jó", "OT", "JOB"),
    ("Salmos", "OT", "PSA"),
    ("Provérbios", "OT", "PRO"),
    ("Eclesiastes", "OT", "ECC"),
    ("Cânticos", "OT", "SNG"),
    ("Isaías", "OT", "ISA"),
    ("Jeremias", "OT", "JER"),
    ("Lamentações", "OT", "LAM"),
    ("Ezequiel", "OT", "EZK"),
    ("Daniel", "OT", "DAN"),
    ("Oséias", "OT", "HOS"),
    ("Joel", "OT", "JOL"),
    ("Amós", "OT", "AMO"),
    ("Obadias", "OT", "OBA"),
    ("Jonas", "OT", "JON"),
    ("Miqueias", "OT", "MIC"),
    ("Naum", "OT", "NAM"),
    ("Habacuque", "OT", "HAB"),
    ("Sofonias", "OT", "ZEP"),
    ("Ageu", "OT", "HAG"),
    ("Zacarias", "OT", "ZEC"),
    ("Malaquias", "OT", "MAL"),
    ("Mateus", "NT", "MAT"),
    ("Marcos", "NT", "MRK"),
    ("Lucas", "NT", "LUK"),
    ("João", "NT", "JHN"),
    ("Atos", "NT", "ACT"),
    ("Romanos", "NT", "ROM"),
    ("1 Coríntios", "NT", "1CO"),
    ("2 Coríntios", "NT", "2CO"),
    ("Gálatas", "NT", "GAL"),
    ("Efésios", "NT", "EPH"),
    ("Filipenses", "NT", "PHP"),
    ("Colossenses", "NT", "COL"),
    ("1 Tessalonicenses", "NT", "1TH"),
    ("2 Tessalonicenses", "NT", "2TH"),
    ("1 Timóteo", "NT", "1TI"),
    ("2 Timóteo", "NT", "2TI"),
    ("Tito", "NT", "TIT"),
    ("Filemom", "NT", "PHM"),
    ("Hebreus", "NT", "HEB"),
    ("Tiago", "NT", "JAS"),
    ("1 Pedro", "NT", "1PE"),
    ("2 Pedro", "NT", "2PE"),
    ("1 João", "NT", "1JN"),
    ("2 João", "NT", "2JN"),
    ("3 João", "NT", "3JN"),
    ("Judas", "NT", "JUD"),
    ("Apocalipse", "NT", "REV"),
]

BOOK_BY_CODE = {}
for order, (book_name, testament, code) in enumerate(BOOKS, start=1):
    BOOK_BY_CODE[code] = (book_name, testament, order)

HREF_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
CHAPTER_LINK_PATTERN = re.compile(
    r"(?P<code>[1-3]?[A-Z]{2,3})(?P<chapter>\d{1,3})\.x?html?$",
    re.IGNORECASE,
)
VERSE_MARKER_PATTERN = re.compile(
    r'<(?:span|sup|a)[^>]*(?:id|class|name)=["\'][^"\']*(?:v|verse)[^"\']*["\'][^>]*>\s*(\d{1,3})\s*</(?:span|sup|a)>',
    re.IGNORECASE,
)
VERSE_REF_ATTR_PATTERN = re.compile(
    r'(?:id|data-usfm|data-ref|name)=["\']'
    r"(?P<code>[1-3]?[A-Z]{2,3})[._:-](?P<chapter>\d{1,3})[._:-](?P<verse>\d{1,3})"
    r'[^"\']*["\']',
    re.IGNORECASE,
)
VERSE_REF_ATTR_COMPACT_PATTERN = re.compile(
    r'(?:id|data-usfm|data-ref|name)=["\']'
    r"(?P<code>[1-3]?[A-Z]{2,3})(?P<chapter>\d{1,3})[._:-](?P<verse>\d{1,3})"
    r'[^"\']*["\']',
    re.IGNORECASE,
)
VERSE_ONLY_ATTR_PATTERN = re.compile(
    r'(?:id|name|data-verse)=["\'](?:v|verse)?[._:-]?(?P<verse>\d{1,3})["\']',
    re.IGNORECASE,
)
TAG_PATTERN_WITH_ATTRS = re.compile(r"<(?P<tag>\w+)(?P<attrs>[^>]*)>", re.IGNORECASE)
SCRIPT_STYLE_PATTERN = re.compile(
    r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
    re.IGNORECASE | re.DOTALL,
)
VERSE_BLOCK_PATTERN = re.compile(
    r"\[\[V:(\d{1,3})\]\]\s*(.*?)(?=\[\[V:\d{1,3}\]\]|$)",
    re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
WS_PATTERN = re.compile(r"\s+")
BLOCK_TAG_PATTERN = re.compile(r"</?(?:p|div|li|tr|h[1-6]|br)\b[^>]*>", re.IGNORECASE)
VERSE_LINE_PATTERN = re.compile(r"^(\d{1,3})\s+(.+)$")
BOOK_CHAPTER_TAIL_PATTERN = re.compile(r"\b.+?\s<\s*\d+\s*>\s*$")
COPYRIGHT_TAIL_PATTERN = re.compile(r"\s*©\s*\d{4}.*$", re.IGNORECASE | re.DOTALL)
VERSE_NUMBER_PREFIX_PATTERN = re.compile(r"^\d{1,3}\s+")
IRRELEVANT_TEXT_PATTERN = re.compile(
    r"^(?:cap[íi]tulo|cap\.|anterior|pr[oó]ximo|índice|sum[áa]rio|in[íi]cio|home)\b",
    re.IGNORECASE,
)


def clean_html_text(value: str) -> str:
    return WS_PATTERN.sub(" ", unescape(TAG_PATTERN.sub(" ", value))).strip()


def normalize_verse_text(text: str, verse: Optional[int] = None) -> str:
    normalized = WS_PATTERN.sub(" ", text).strip()
    normalized = COPYRIGHT_TAIL_PATTERN.sub("", normalized).strip()
    normalized = BOOK_CHAPTER_TAIL_PATTERN.sub("", normalized).strip()
    if verse is not None and normalized.startswith(f"{verse} "):
        normalized = normalized[len(str(verse)) + 1 :].strip()  # noqa
        normalized = VERSE_NUMBER_PREFIX_PATTERN.sub("", normalized).strip()
    return normalized


def is_important_text(text: str) -> bool:
    if not text:
        return False
    if IRRELEVANT_TEXT_PATTERN.match(text):
        return False
    if len(text) < 8 and not re.search(r"[A-Za-zÀ-ÿ]{3,}", text):
        return False
    return True


def discover_chapter_links(index_html: str) -> list[tuple[int, int, str]]:
    parsed_base = urlparse(BASE_URL)
    discovered = {}

    for href in HREF_PATTERN.findall(index_html):
        absolute = urljoin(BASE_URL, href)
        parsed = urlparse(absolute)
        if parsed.netloc != parsed_base.netloc:
            continue
        filename = parsed.path.rsplit("/", 1)[-1]
        match = CHAPTER_LINK_PATTERN.search(filename)
        if not match:
            continue
        code = match.group("code").upper()
        chapter = int(match.group("chapter"))
        if code not in BOOK_BY_CODE:
            continue
        _, _, book_order = BOOK_BY_CODE[code]
        discovered[(book_order, chapter)] = absolute

    return [
        (book_order, chapter, discovered[(book_order, chapter)])
        for book_order, chapter in sorted(discovered.keys())
    ]


def extract_verse_from_attrs(
    attrs: str, expected_code: str, expected_chapter: int
) -> Optional[int]:
    match = VERSE_REF_ATTR_PATTERN.search(attrs)
    if match:
        if (
            match.group("code").upper() == expected_code
            and int(match.group("chapter")) == expected_chapter
        ):
            return int(match.group("verse"))

    match = VERSE_REF_ATTR_COMPACT_PATTERN.search(attrs)
    if match:
        if (
            match.group("code").upper() == expected_code
            and int(match.group("chapter")) == expected_chapter
        ):
            return int(match.group("verse"))

    match = VERSE_ONLY_ATTR_PATTERN.search(attrs)
    if match:
        return int(match.group("verse"))

    return None


def parse_chapter_verses(
    html: str, expected_code: str, expected_chapter: int
) -> list[tuple[int, str]]:
    cleaned_html = SCRIPT_STYLE_PATTERN.sub(" ", html)

    verses_map = {}
    anchors = []
    for match in TAG_PATTERN_WITH_ATTRS.finditer(cleaned_html):
        verse = extract_verse_from_attrs(
            match.group("attrs"), expected_code, expected_chapter
        )
        if verse is None:
            continue
        anchors.append((verse, match.end(), match.start()))

    if anchors:
        anchors.sort(key=lambda item: item[2])
        for idx, (verse, content_start, _) in enumerate(anchors):
            next_start = (
                anchors[idx + 1][2] if idx + 1 < len(anchors) else len(cleaned_html)
            )
            chunk = cleaned_html[content_start:next_start]
            text = normalize_verse_text(clean_html_text(chunk), verse=verse)
            if is_important_text(text):
                previous = verses_map.get(verse, "")
                if len(text) > len(previous):
                    verses_map[verse] = text

    if not verses_map:
        marked = VERSE_MARKER_PATTERN.sub(r" [[V:\1]] ", cleaned_html)
        text = clean_html_text(marked)
        for match in VERSE_BLOCK_PATTERN.finditer(text):
            verse = int(match.group(1))
            body = normalize_verse_text(
                WS_PATTERN.sub(" ", match.group(2)).strip(), verse=verse
            )
            if is_important_text(body):
                previous = verses_map.get(verse, "")
                if len(body) > len(previous):
                    verses_map[verse] = body

    if not verses_map:
        line_ready = BLOCK_TAG_PATTERN.sub("\n", cleaned_html)
        text_lines = unescape(TAG_PATTERN.sub(" ", line_ready))
        current_verse = None
        current_parts = []

        def flush_line_verse():
            nonlocal current_verse, current_parts
            if current_verse is None:
                return
            body = normalize_verse_text(
                WS_PATTERN.sub(" ", " ".join(current_parts)).strip(),
                verse=current_verse,
            )
            if is_important_text(body):
                previous = verses_map.get(current_verse, "")
                if len(body) > len(previous):
                    verses_map[current_verse] = body
            current_verse = None
            current_parts = []

        for raw_line in text_lines.splitlines():
            line = WS_PATTERN.sub(" ", raw_line).strip()
            if not line:
                continue
            line_match = VERSE_LINE_PATTERN.match(line)
            if line_match:
                flush_line_verse()
                current_verse = int(line_match.group(1))
                current_parts = [line_match.group(2)]
                continue
            if current_verse is not None:
                current_parts.append(line)

        flush_line_verse()

    verses = sorted(verses_map.items(), key=lambda item: item[0])

    if len(verses) < 3:
        text = normalize_verse_text(clean_html_text(cleaned_html))
        sequence_verses = parse_sequential_verses(text)
        if len(sequence_verses) > len(verses):
            verses = sequence_verses

    if not verses:
        raise RuntimeError(
            f"Nenhum verso encontrado em {expected_code} {expected_chapter}."
        )
    return verses


def parse_sequential_verses(text: str) -> list[tuple[int, str]]:
    content = text
    content = normalize_verse_text(content)
    has_first_verse_marker = bool(re.search(r"(?:^|\s)1\s+", content))
    if not has_first_verse_marker and re.search(r"(?:^|\s)2\s+", content):
        content = f"1 {content}"

    verses = []
    expected = 1
    search_pos = 0

    while True:
        current_match = re.search(rf"(?:^|\s){expected}\s+", content[search_pos:])
        if not current_match:
            break
        # current_abs_start = search_pos + current_match.start()
        current_abs_end = search_pos + current_match.end()
        next_expected = expected + 1
        next_match = re.search(
            rf"(?:^|\s){next_expected}\s+", content[current_abs_end:]
        )

        if next_match:
            next_abs_start = current_abs_end + next_match.start()
            verse_text = content[current_abs_end:next_abs_start].strip()
            search_pos = next_abs_start
        else:
            verse_text = content[current_abs_end:].strip()
            search_pos = len(content)

        verse_text = normalize_verse_text(verse_text, verse=expected)
        if is_important_text(verse_text):
            verses.append((expected, verse_text))
        expected += 1
        if search_pos >= len(content):
            break

    return verses


def generate_embedding(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding
    if not isinstance(embedding, list):
        raise RuntimeError("Embedding inválido retornado pela OpenAI.")
    return embedding


def classify_theme(
    client: OpenAI,
    model: str,
    text: str,
    allowed_themes: list[str],
) -> str:
    allowed = ", ".join(allowed_themes)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classifique versos bíblicos em uma única chave de tema. "
                    "Retorne somente a chave."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Classifique o verso bíblico em exatamente uma chave.\n"
                    f"Chaves permitidas: {allowed}\n"
                    "Responda apenas com a chave, sem explicação.\n"
                    f"Verso: {text}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Classificação de tema sem conteúdo.")
    theme = content.strip()
    if theme not in allowed_themes:
        raise RuntimeError(f"Tema inválido: {theme}")
    return theme


def get_resume_point() -> Optional[Tuple[int, int, int]]:
    last = (
        BibleTextFlat.objects.filter(translation=TRANSLATION)
        .order_by("-book_order", "-chapter", "-verse")
        .values_list("book_order", "chapter", "verse")
        .first()
    )
    if not last:
        return None
    return last


class Command(BaseCommand):
    help = "Importa a Bíblia porbr2018 por scraping HTML e salva em BibleTextFlat."

    def handle(self, *args, **options):
        default_model = os.environ.get("DEFAULT_MODEL")
        if not default_model:
            raise CommandError("Variável DEFAULT_MODEL é obrigatória.")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise CommandError("Variável OPENAI_API_KEY é obrigatória.")

        allowed_themes = list(
            ThemeV2.objects.values_list("id", flat=True).order_by("id")
        )
        if not allowed_themes:
            raise CommandError("Nenhum tema encontrado em ThemeV2.")

        openai_client = OpenAI(api_key=openai_api_key)
        session = requests.Session()
        index_response = session.get(BASE_URL, timeout=120)
        index_response.raise_for_status()
        index_response.encoding = "utf-8"
        chapter_links = discover_chapter_links(index_response.text)
        if not chapter_links:
            raise CommandError("Nenhum capítulo encontrado no índice HTML.")

        resume_point = get_resume_point()
        total = 0

        for book_order, chapter, chapter_url in chapter_links:
            if resume_point and (book_order, chapter, 0) <= resume_point:
                continue

            chapter_response = session.get(chapter_url, timeout=120)
            chapter_response.raise_for_status()
            chapter_response.encoding = "utf-8"

            book_name, testament, code = BOOKS[book_order - 1]
            verses = parse_chapter_verses(chapter_response.text, code, chapter)
            for verse, text in verses:
                if resume_point and (book_order, chapter, verse) <= resume_point:
                    continue

                embedding = generate_embedding(openai_client, text)
                theme = classify_theme(
                    client=openai_client,
                    model=default_model,
                    text=text,
                    allowed_themes=allowed_themes,
                )

                BibleTextFlat.objects.create(
                    translation=TRANSLATION,
                    testament=testament,
                    book=book_name,
                    book_order=book_order,
                    chapter=chapter,
                    verse=verse,
                    reference=f"{book_name}:{chapter}:{verse}",
                    text=text,
                    embedding=embedding,
                    theme_id=theme,
                )
                total += 1

        self.stdout.write(
            self.style.SUCCESS(f"Importação concluída: {total} versos novos.")
        )
