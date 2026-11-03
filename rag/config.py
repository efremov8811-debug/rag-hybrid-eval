"""Параметры пайплайна. Всё, что имеет смысл крутить, собрано здесь."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = ROOT / "docs"
QDRANT_PATH = ROOT / ".qdrant"
BM25_PATH = ROOT / ".qdrant" / "bm25.pkl"
COLLECTION = "pribor"

# Мультиязычная модель: 384 измерения, 220 МБ, работает на CPU через onnxruntime.
DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DENSE_DIM = 384

# Кросс-энкодер для переранжирования. Версия int8: 570 МБ вместо 2,3 ГБ у fp32.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"

# Чанкинг. Документы размечены заголовками, поэтому режем по ним,
# а слишком длинные секции добиваем окном с перекрытием.
MAX_CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 80

# Поиск: сколько берём из каждой ветки до слияния и сколько отдаём реранкеру.
DENSE_TOP_K = 50
BM25_TOP_K = 50
FUSION_TOP_K = 20
RRF_K = 60

# Сколько кандидатов уходит в кросс-энкодер. Он на порядок дороже поиска,
# поэтому число подбирается замером: больше кандидатов — выше потолок recall,
# но линейно растёт задержка.
RERANK_CANDIDATES = 12
