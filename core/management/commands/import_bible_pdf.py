import os
import re
import unicodedata
from pathlib import Path

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from pdfminer.high_level import extract_text

from core.models import BibleTextFlat, ThemeV2

EMBEDDING_MODEL = "nomic-embed-text"
TRANSLATION = "porbr2018"
VERSE_PATTERN = re.compile(r"^(\d+)\s+(.+)$")

BOOKS = [
    ("Gênesis", "OT"),
    ("Êxodo", "OT"),
    ("Levítico", "OT"),
    ("Números", "OT"),
    ("Deuteronômio", "OT"),
    ("Josué", "OT"),
    ("Juízes", "OT"),
    ("Rute", "OT"),
    ("1 Samuel", "OT"),
    ("2 Samuel", "OT"),
    ("1 Reis", "OT"),
    ("2 Reis", "OT"),
    ("1 Crônicas", "OT"),
    ("2 Crônicas", "OT"),
    ("Esdras", "OT"),
    ("Neemias", "OT"),
    ("Ester", "OT"),
    ("Jó", "OT"),
    ("Salmos", "OT"),
    ("Provérbios", "OT"),
    ("Eclesiastes", "OT"),
    ("Cânticos", "OT"),
    ("Isaías", "OT"),
    ("Jeremias", "OT"),
    ("Lamentações", "OT"),
    ("Ezequiel", "OT"),
    ("Daniel", "OT"),
    ("Oséias", "OT"),
    ("Joel", "OT"),
    ("Amós", "OT"),
    ("Obadias", "OT"),
    ("Jonas", "OT"),
    ("Miqueias", "OT"),
    ("Naum", "OT"),
    ("Habacuque", "OT"),
    ("Sofonias", "OT"),
    ("Ageu", "OT"),
    ("Zacarias", "OT"),
    ("Malaquias", "OT"),
    ("Mateus", "NT"),
    ("Marcos", "NT"),
    ("Lucas", "NT"),
    ("João", "NT"),
    ("Atos", "NT"),
    ("Romanos", "NT"),
    ("1 Coríntios", "NT"),
    ("2 Coríntios", "NT"),
    ("Gálatas", "NT"),
    ("Efésios", "NT"),
    ("Filipenses", "NT"),
    ("Colossenses", "NT"),
    ("1 Tessalonicenses", "NT"),
    ("2 Tessalonicenses", "NT"),
    ("1 Timóteo", "NT"),
    ("2 Timóteo", "NT"),
    ("Tito", "NT"),
    ("Filemom", "NT"),
    ("Hebreus", "NT"),
    ("Tiago", "NT"),
    ("1 Pedro", "NT"),
    ("2 Pedro", "NT"),
    ("1 João", "NT"),
    ("2 João", "NT"),
    ("3 João", "NT"),
    ("Judas", "NT"),
    ("Apocalipse", "NT"),
]

BOOK_ALIASES = {
    "Cantares": "Cânticos",
    "Cântico dos Cânticos": "Cânticos",
    "1o Samuel": "1 Samuel",
    "2o Samuel": "2 Samuel",
    "1o Reis": "1 Reis",
    "2o Reis": "2 Reis",
    "1o Crônicas": "1 Crônicas",
    "2o Crônicas": "2 Crônicas",
    "1o Coríntios": "1 Coríntios",
    "2o Coríntios": "2 Coríntios",
    "1o Tessalonicenses": "1 Tessalonicenses",
    "2o Tessalonicenses": "2 Tessalonicenses",
    "1o Timóteo": "1 Timóteo",
    "2o Timóteo": "2 Timóteo",
    "1o Pedro": "1 Pedro",
    "2o Pedro": "2 Pedro",
    "1o João": "1 João",
    "2o João": "2 João",
    "3o João": "3 João",
}


def normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"\s+", " ", ascii_only).strip().lower()
    return compact


BOOK_INDEX = {}
for idx, (book_name, testament) in enumerate(BOOKS, start=1):
    BOOK_INDEX[normalize(book_name)] = (book_name, testament, idx)

for alias, canonical in BOOK_ALIASES.items():
    canonical_data = BOOK_INDEX[normalize(canonical)]
    BOOK_INDEX[normalize(alias)] = canonical_data


def parse_verses(content: str):
    current_book = None
    current_testament = None
    current_book_order = None
    current_chapter = None
    current_verse = None
    current_parts = []
    verses = []

    def flush_current():
        nonlocal current_verse, current_parts
        if current_book and current_chapter and current_verse and current_parts:
            text = " ".join(current_parts).strip()
            verses.append(
                {
                    "translation": TRANSLATION,
                    "testament": current_testament,
                    "book": current_book,
                    "book_order": current_book_order,
                    "chapter": current_chapter,
                    "verse": current_verse,
                    "reference": f"{current_book}:{current_chapter}:{current_verse}",
                    "text": text,
                }
            )
        current_verse = None
        current_parts = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        book_data = BOOK_INDEX.get(normalize(line))
        if book_data:
            flush_current()
            current_book, current_testament, current_book_order = book_data
            current_chapter = None
            continue

        if current_book is None:
            continue

        if line.isdigit():
            flush_current()
            current_chapter = int(line)
            continue

        verse_match = VERSE_PATTERN.match(line)
        if verse_match:
            if current_chapter is None:
                raise RuntimeError(f"Verso sem capítulo em {current_book}: {line}")
            flush_current()
            current_verse = int(verse_match.group(1))
            current_parts = [verse_match.group(2).strip()]
            continue

        if current_verse is not None:
            current_parts.append(line)

    flush_current()
    return verses


def generate_embedding(session: requests.Session, ollama_url: str, text: str):
    response = session.post(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    embedding = payload.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError(f"Embedding inválido: {payload}")
    return embedding


def classify_theme(
    session: requests.Session,
    ollama_url: str,
    model: str,
    text: str,
    allowed_themes: list[str],
) -> str:
    allowed = ", ".join(allowed_themes)
    prompt = (
        "Classifique o verso bíblico em exatamente uma chave.\n"
        f"Chaves permitidas: {allowed}\n"
        "Responda apenas com a chave, sem explicação.\n"
        f"Verso: {text}"
    )
    response = session.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    theme = payload.get("response", "").strip()
    if theme not in allowed_themes:
        raise RuntimeError(f"Tema inválido: {theme}")
    return theme


class Command(BaseCommand):
    help = "Importa porbr2018_all.pdf e salva versos em BibleTextFlat."

    def handle(self, *args, **options):
        ollama_url = os.environ.get("OLLAMA_BASE_URL")
        if not ollama_url:
            raise CommandError("Variável OLLAMA_BASE_URL é obrigatória.")

        default_model = os.environ.get("DEFAULT_MODEL")
        if not default_model:
            raise CommandError("Variável DEFAULT_MODEL é obrigatória.")

        pdf_path = Path(settings.BASE_DIR) / "model/porbr2018_all.pdf"
        if not pdf_path.exists():
            raise CommandError(f"Arquivo não encontrado: {pdf_path}")

        content = extract_text(str(pdf_path))
        verses = parse_verses(content)
        allowed_themes = list(
            ThemeV2.objects.values_list("id", flat=True).order_by("id")
        )
        if not allowed_themes:
            raise CommandError("Nenhum tema encontrado em ThemeV2.")

        session = requests.Session()
        total = 0
        for verse in verses:
            embedding = generate_embedding(session, ollama_url, verse["text"])
            theme = classify_theme(
                session,
                ollama_url,
                default_model,
                verse["text"],
                allowed_themes,
            )
            BibleTextFlat.objects.create(
                translation=verse["translation"],
                testament=verse["testament"],
                book=verse["book"],
                book_order=verse["book_order"],
                chapter=verse["chapter"],
                verse=verse["verse"],
                reference=verse["reference"],
                text=verse["text"],
                embedding=embedding,
                theme_id=theme,
            )
            total += 1

        self.stdout.write(self.style.SUCCESS(f"Importação concluída: {total} versos."))
