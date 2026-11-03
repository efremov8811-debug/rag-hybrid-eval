"""Индексация корпуса: чанкинг, эмбеддинги, запись в Qdrant и BM25.

    python ingest.py
"""

import shutil
import sys
import time

from qdrant_client import QdrantClient, models

from rag import config
from rag.chunking import chunk_markdown
from rag.lexical import build as build_bm25
from rag.search import embed_passages

# Windows-консоль по умолчанию не UTF-8, иначе кириллица в выводе ломается.
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    files = sorted(config.DOCS_DIR.glob("*.md"))
    if not files:
        print(f"Нет документов в {config.DOCS_DIR}")
        return 1

    t0 = time.time()
    chunks = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, path.name))

    print(f"Документов: {len(files)}, чанков: {len(chunks)}")
    lengths = sorted(len(c.text) for c in chunks)
    print(
        f"Длина чанка: мин {lengths[0]}, медиана {lengths[len(lengths) // 2]}, "
        f"макс {lengths[-1]} символов"
    )

    if config.QDRANT_PATH.exists():
        shutil.rmtree(config.QDRANT_PATH)

    print(f"Считаю эмбеддинги моделью {config.DENSE_MODEL}")
    vectors = embed_passages([c.text for c in chunks])

    client = QdrantClient(path=str(config.QDRANT_PATH))
    client.create_collection(
        config.COLLECTION,
        vectors_config=models.VectorParams(
            size=config.DENSE_DIM, distance=models.Distance.COSINE
        ),
    )
    client.upsert(
        config.COLLECTION,
        points=[
            models.PointStruct(
                id=i,
                vector=vectors[i],
                payload={
                    "doc": c.doc,
                    "title": c.title,
                    "section": c.section,
                    "text": c.text,
                },
            )
            for i, c in enumerate(chunks)
        ],
    )

    build_bm25([c.text for c in chunks]).save(config.BM25_PATH)
    client.close()

    print(f"Индекс собран за {time.time() - t0:.1f} с, каталог {config.QDRANT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
