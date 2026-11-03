"""Вопрос к корпусу.

    python ask.py "сколько стоит поверка расходомера"
    python ask.py "..." --mode hybrid --top-k 3
    python ask.py "..." --answer        # сгенерировать ответ по найденному

Без ключа модели работает как поиск: выдаёт источники. Это намеренно —
источники и есть то, что проверяется, генерация поверх них вторична.
"""

import argparse
import os
import sys

from rag.search import Searcher

# Windows-консоль по умолчанию не UTF-8, иначе кириллица в выводе ломается.
sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы по документации поставщика измерительных приборов. "
    "Используй только приведённые фрагменты. Если ответа в них нет, скажи, что "
    "в документации это не описано, и не додумывай. В конце ответа перечисли "
    "названия документов, на которые опирался."
)


def generate(question: str, hits) -> str:
    """Генерация ответа. Ключ берётся из окружения, в репозитории его нет."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return "DEEPSEEK_API_KEY не задан, генерация пропущена."

    import httpx

    context = "\n\n---\n\n".join(
        f"[{h.doc} / {h.section}]\n{h.text}" for h in hits
    )
    response = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Фрагменты документации:\n\n{context}\n\nВопрос: {question}",
                },
            ],
            "temperature": 0.1,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", default="rerank", choices=["dense", "hybrid", "rerank"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()

    hits = Searcher().search(args.question, top_k=args.top_k, mode=args.mode)

    if args.answer:
        print(generate(args.question, hits))
        print("\n" + "-" * 60)

    for i, h in enumerate(hits, start=1):
        head = h.text.split("\n\n", 1)[-1].replace("\n", " ")
        print(f"{i}. [{h.score:.3f}] {h.doc} / {h.section}")
        print(f"   {head[:220]}{'…' if len(head) > 220 else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
