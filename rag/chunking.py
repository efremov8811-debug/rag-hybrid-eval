"""Разбиение markdown на чанки по заголовкам.

Заголовок секции дописывается в начало каждого чанка: без него кусок вроде
"Межповерочный интервал: 5 лет" теряет привязку к прибору и всплывает в поиске
по любому вопросу про интервалы.
"""

from dataclasses import dataclass
import re

from .config import MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS, MIN_CHUNK_CHARS

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    doc: str
    title: str
    section: str
    index: int


def _split_sections(markdown: str):
    """Возвращает список (h1, h2, тело). H1 считается названием документа."""
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        return [("", "", markdown.strip())]

    sections = []
    doc_title = ""
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()

        if level == 1:
            doc_title = heading
            if body:
                sections.append((doc_title, "", body))
            continue

        sections.append((doc_title, heading, body))
    return sections


def _window(text: str):
    """Режет длинный текст по абзацам, добирая до лимита, с перекрытием."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces, current = [], ""

    for p in paragraphs:
        if not current:
            current = p
        elif len(current) + len(p) + 2 <= MAX_CHUNK_CHARS:
            current = f"{current}\n\n{p}"
        else:
            pieces.append(current)
            tail = current[-CHUNK_OVERLAP_CHARS:]
            current = f"{tail}\n\n{p}" if len(p) < MAX_CHUNK_CHARS else p
    if current:
        pieces.append(current)

    # Абзац длиннее лимита режем жёстко, иначе один кусок перевесит остальные.
    out = []
    for piece in pieces:
        while len(piece) > MAX_CHUNK_CHARS * 1.6:
            out.append(piece[:MAX_CHUNK_CHARS])
            piece = piece[MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS:]
        out.append(piece)
    return out


def chunk_markdown(markdown: str, doc: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_title, section, body in _split_sections(markdown):
        if len(body) < MIN_CHUNK_CHARS:
            continue
        for piece in _window(body):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            header = " / ".join(x for x in (doc_title, section) if x)
            text = f"{header}\n\n{piece}" if header else piece
            chunks.append(
                Chunk(
                    text=text,
                    doc=doc,
                    title=doc_title,
                    section=section,
                    index=len(chunks),
                )
            )
    return chunks
