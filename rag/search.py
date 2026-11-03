"""Поиск: векторная ветка + BM25, слияние RRF, переранжирование кросс-энкодером.

Режимы ("dense", "hybrid", "rerank") заданы явно, чтобы eval мог померить
вклад каждого слоя по отдельности, а не поверить на слово.
"""

from dataclasses import dataclass
from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from . import config
from .lexical import BM25Index


@dataclass
class Hit:
    chunk_id: int
    doc: str
    section: str
    text: str
    score: float


@lru_cache(maxsize=1)
def _encoder() -> TextEmbedding:
    return TextEmbedding(config.DENSE_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(config.RERANK_MODEL)


def embed_query(text: str) -> list[float]:
    return list(next(iter(_encoder().query_embed(text))))


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [list(v) for v in _encoder().passage_embed(texts)]


def _rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: складываем 1/(k+позиция) по всем веткам.

    Сливаем по рангам, а не по сырым очкам: косинус и BM25 живут в разных
    шкалах, и нормировать их друг к другу означало бы подгонять коэффициент
    под конкретный корпус.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Searcher:
    def __init__(self):
        self.client = QdrantClient(path=str(config.QDRANT_PATH))
        self.bm25 = BM25Index.load(config.BM25_PATH)
        self._payloads: dict[int, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                config.COLLECTION, limit=256, offset=offset, with_payload=True
            )
            for p in points:
                self._payloads[int(p.id)] = p.payload
            if offset is None:
                break

    def _dense(self, query: str, top_k: int) -> list[int]:
        result = self.client.query_points(
            config.COLLECTION, query=embed_query(query), limit=top_k
        )
        return [int(p.id) for p in result.points]

    def _hit(self, chunk_id: int, score: float) -> Hit:
        p = self._payloads[chunk_id]
        return Hit(chunk_id, p["doc"], p.get("section", ""), p["text"], score)

    def search(self, query: str, top_k: int = 5, mode: str = "rerank") -> list[Hit]:
        dense_ids = self._dense(query, config.DENSE_TOP_K)
        if mode == "dense":
            return [self._hit(i, 1.0 / (r + 1)) for r, i in enumerate(dense_ids[:top_k])]

        bm25_ids = [i for i, _ in self.bm25.search(query, config.BM25_TOP_K)]
        fused = _rrf([dense_ids, bm25_ids], config.RRF_K)[: config.FUSION_TOP_K]
        if mode == "hybrid":
            return [self._hit(i, s) for i, s in fused[:top_k]]

        candidates = [i for i, _ in fused[: config.RERANK_CANDIDATES]]
        docs = [self._payloads[i]["text"] for i in candidates]
        scores = list(_reranker().rerank(query, docs))
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [self._hit(i, float(s)) for i, s in ranked[:top_k]]
