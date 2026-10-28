"""
The two pretrained models used by V2 (both run on CPU, downloaded once from Hugging Face):

- Sentence embeddings (all-MiniLM-L6-v2): turns text into vectors, so we can
  find the evidence sentences most similar in meaning to the claim.
- NLI cross-encoder (nli-deberta-v3-small): reads (evidence, claim) together and
  outputs probabilities for entailment / contradiction / neutral.
"""

import re
import unicodedata
from functools import lru_cache

from core.preprocessing import _PUNCT_MAP

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def nli_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(NLI_MODEL, device="cpu")


def rank_by_similarity(query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
    """Return [(index_in_texts, cosine_similarity)] for the top_k most similar texts."""
    if not texts:
        return []
    from sentence_transformers import util
    model = embedder()
    q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    t = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
    scores = util.cos_sim(q, t)[0]
    top = scores.topk(min(top_k, len(texts)))
    return [(int(i), float(s)) for s, i in zip(top.values, top.indices)]


def normalize(text: str) -> str:
    """Small fixes that confuse the NLI model: "2,000" vs "2000", curly quotes."""
    return _THOUSANDS.sub("", unicodedata.normalize("NFKC", text).translate(_PUNCT_MAP))


def nli(premises: list[str], hypothesis: str) -> list[dict]:
    """
    For each premise (evidence sentence): probabilities that it
    entails / contradicts / is neutral to the hypothesis (the claim).
    """
    if not premises:
        return []
    model = nli_model()
    pairs = [(normalize(p), normalize(hypothesis)) for p in premises]
    probs = model.predict(pairs, apply_softmax=True)
    # Read the label order from the model config instead of assuming it.
    id2label = {int(k): v.lower() for k, v in model.model.config.id2label.items()}
    return [{id2label[j]: float(row[j]) for j in range(len(row))} for row in probs]
