"""Лексическая ветка поиска: BM25 с русским стеммингом.

Нужна рядом с векторной, потому что эмбеддинги плохо держат точные токены:
артикулы вроде ДД-200, номера регистров 40030 и числа «1200 метров» модель
обобщает, а пользователь ищет именно их.
"""

import pickle
import re

import Stemmer
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-zA-Zа-яёА-ЯЁ0-9]+(?:[-/][a-zA-Zа-яёА-ЯЁ0-9]+)*")
_stemmer = Stemmer.Stemmer("russian")

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "для",
    "это", "эти", "этот", "при", "их", "уже", "может", "можно", "есть",
}


def tokenize(text: str) -> list[str]:
    words = [w.lower() for w in _TOKEN_RE.findall(text)]
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    # Стеммим только кириллицу: артикулы и латиницу портить нельзя.
    return [
        _stemmer.stemWord(w) if re.fullmatch(r"[а-яё]+", w) else w
        for w in words
    ]


class BM25Index:
    def __init__(self, corpus_tokens: list[list[str]]):
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_k] if s > 0]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.bm25, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = cls.__new__(cls)
            obj.bm25 = pickle.load(f)
            return obj


def build(texts: list[str]) -> BM25Index:
    return BM25Index([tokenize(t) for t in texts])
