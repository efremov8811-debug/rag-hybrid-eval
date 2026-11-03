"""Замер качества поиска на размеченном наборе вопросов.

Для каждого вопроса известен документ-источник и строка, которая обязана
попасть в найденный чанк. Проверка по строке строже проверки по документу:
нужный файл можно зацепить соседней секцией, где ответа нет.

    python evaluate.py

Метрики: Recall@1, Recall@3, Recall@5 и MRR по трём режимам, чтобы видеть
вклад каждого слоя отдельно.
"""

import json
import sys
import time

from rag import config
from rag.search import Searcher

# Windows-консоль по умолчанию не UTF-8, иначе кириллица в выводе ломается.
sys.stdout.reconfigure(encoding="utf-8")

MODES = ["dense", "hybrid", "rerank"]
TOP_K = 5


def load_questions():
    path = config.ROOT / "eval" / "questions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(searcher: Searcher, questions, mode: str):
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    misses = []
    t0 = time.time()

    for item in questions:
        # Ответ может честно лежать в нескольких документах, тогда их перечисляют списком.
        expected = item["doc"] if isinstance(item["doc"], list) else [item["doc"]]
        results = searcher.search(item["q"], top_k=TOP_K, mode=mode)
        rank = None
        for position, hit in enumerate(results, start=1):
            correct = hit.doc in expected and item["must_contain"] in hit.text
            if correct:
                rank = position
                break

        if rank is None:
            misses.append((item["q"], results[0].doc if results else "—"))
            continue

        reciprocal += 1.0 / rank
        for k in hits_at:
            if rank <= k:
                hits_at[k] += 1

    n = len(questions)
    elapsed = time.time() - t0
    return {
        "mode": mode,
        "recall@1": hits_at[1] / n,
        "recall@3": hits_at[3] / n,
        "recall@5": hits_at[5] / n,
        "mrr": reciprocal / n,
        "sec_per_query": elapsed / n,
        "misses": misses,
    }


def main() -> int:
    questions = load_questions()
    searcher = Searcher()
    print(f"Вопросов в наборе: {len(questions)}")

    # Прогрев: первый вызов грузит onnx-модели с диска, иначе эта задержка
    # целиком падает на первый замеряемый режим и искажает сравнение.
    print("Прогрев моделей…")
    for mode in MODES:
        searcher.search("прогрев", top_k=TOP_K, mode=mode)
    print()

    rows = []
    for mode in MODES:
        r = evaluate(searcher, questions, mode)
        rows.append(r)
        print(
            f"{mode:<8} "
            f"R@1 {r['recall@1']:.2f}  "
            f"R@3 {r['recall@3']:.2f}  "
            f"R@5 {r['recall@5']:.2f}  "
            f"MRR {r['mrr']:.3f}  "
            f"{r['sec_per_query'] * 1000:.0f} мс/запрос"
        )

    print()
    for r in rows:
        if r["misses"]:
            print(f"Промахи в режиме {r['mode']}:")
            for q, got in r["misses"]:
                print(f"  — {q}  (верх выдачи: {got})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
