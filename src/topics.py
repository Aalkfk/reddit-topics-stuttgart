# /src/topics.py
from typing import List, Tuple, Dict
import numpy as np
import warnings

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- optional: gensim für c_v Coherence ---
try:
    from gensim.corpora import Dictionary  # type: ignore
    from gensim.models.coherencemodel import CoherenceModel  # type: ignore
    _HAS_GENSIM = True
except Exception:
    _HAS_GENSIM = False


# -------- Vektorisierung --------
# N‑Gramme (1–2) wie im Konzept, begrenzte Featurezahl für Stabilität/Performance.
def vectorize_count(texts: List[str], max_features: int = 20000, ngram_range=(1, 2)) -> Tuple[CountVectorizer, np.ndarray]:
    v = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = v.fit_transform(texts)
    return v, X

def vectorize_tfidf(texts: List[str], max_features: int = 20000, ngram_range=(1, 2)) -> Tuple[TfidfVectorizer, np.ndarray]:
    v = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = v.fit_transform(texts)
    return v, X


# -------- k-Auswahl: bevorzugt c_v (gensim), sonst Log-Likelihood --------
# c_v bewertet semantische Kohärenz von Top‑Terms je Topic; Log‑Likelihood als fallbac
def _coherence_c_v(texts_tok: List[List[str]], topics_terms: List[List[str]]) -> float:
    """Berechnet c_v Coherence. Erwartet Tokenlisten und Topic-Term-Listen."""
    if not _HAS_GENSIM:
        return float("nan")
    dct = Dictionary(texts_tok)
    cm = CoherenceModel(topics=topics_terms, texts=texts_tok, dictionary=dct, coherence="c_v")
    return float(cm.get_coherence())

def _topic_term_lists(lda: LatentDirichletAllocation, v_count: CountVectorizer, topn: int = 12) -> List[List[str]]:
    terms = v_count.get_feature_names_out()
    topics: List[List[str]] = []
    for i in range(lda.n_components):
        comp = lda.components_[i]
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([str(terms[j]) for j in idx])
    return topics

def select_k(
    texts_tok: List[List[str]],
    v_count: CountVectorizer,
    X_count,
    k_range: range = range(5, 15),
    topn: int = 12,
) -> Tuple[int, Dict[int, float], str]:
    """
    Wählt k anhand c_v (falls gensim verfügbar & Daten ausreichend), sonst via Log-Likelihood.
    Returns: (best_k, scores_by_k, metric_used['coherence_c_v'|'loglik'])
    """
    scores: Dict[int, float] = {}
    use_coherence = _HAS_GENSIM and len(texts_tok) >= 10 and X_count.shape[0] >= 10
    best_k, best_score = None, -np.inf
    metric = "coherence_c_v" if use_coherence else "loglik"

    for k in k_range:
        if k >= X_count.shape[0]:  # k darf nicht >= #Dokumente sein
            continue
        lda = LatentDirichletAllocation(n_components=k, random_state=42, learning_method="batch")
        lda.fit(X_count)

        if use_coherence:
            topics_terms = _topic_term_lists(lda, v_count, topn=topn)
            score = _coherence_c_v(texts_tok, topics_terms)
            if np.isnan(score):
                # Fallback lokal, falls gensim da ist, aber c_v nicht berechenbar
                score = float(lda.score(X_count))
                metric = "loglik"
        else:
            score = float(lda.score(X_count))

        scores[k] = score
        if score > best_score:
            best_k, best_score = k, score

    if best_k is None:
        best_k = max(2, min(6, X_count.shape[0] - 1))
        scores[best_k] = -1.0
        metric = "loglik"

    return best_k, scores, metric

def fit_lda_with_k(
    texts_tok: List[List[str]],
    texts_joined: List[str],
    k_range: range = range(6, 13),
    topn: int = 12,
):
    """Hilfsfunktion: Vektorisieren, k selektieren, finales LDA mit `best_k` fitten"""
    v_count, X_count = vectorize_count(texts_joined)
    best_k, diag, metric = select_k(texts_tok, v_count, X_count, k_range=k_range, topn=topn)
    lda = LatentDirichletAllocation(n_components=best_k, random_state=42, learning_method="batch")
    lda.fit(X_count)
    return lda, v_count, X_count, best_k, diag, metric

def terms_for_topics(lda: LatentDirichletAllocation, v_count: CountVectorizer, topn: int = 12):
    """Liefert Top‑Terms je Topic inkl. Gewichten für die Darstellung im Report"""
    terms = v_count.get_feature_names_out()
    topics = []
    for i in range(lda.n_components):
        comp = lda.components_[i]
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([(terms[j], float(comp[j])) for j in idx])
    return topics


# -------- LSA stabil ----
# Wird stillschweigend übersprungen, wenn  Matrixzu dünn, zu klein, …
def lsa_top_terms(texts: List[str], requested_topics: int = 8, topn: int = 12):
    """Optionale LSA (Konzept). Läuft nur, wenn Matrix „gesund“ ist; sonst sauber überspringen."""
    if not texts or all(not (t and t.strip()) for t in texts):
        return []

    v_tfidf, X_tfidf = vectorize_tfidf(texts)
    n_samples, n_features = X_tfidf.shape

    # Mindestanforderungen: genug Samples/Features + keine Nullmatrix
    if n_samples < 2 or n_features < 2 or X_tfidf.nnz == 0:
        return []

    density = X_tfidf.nnz / float(n_samples * n_features)
    if density < 1e-6:
        return []

    safe_max = max(0, min(n_samples - 1, n_features - 1))
    n_comp = min(requested_topics, safe_max)
    if n_comp < 2:
        return []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=7)
            svd.fit(X_tfidf)
    except Exception:
        return []

    terms = v_tfidf.get_feature_names_out()
    topics: List[List[Tuple[str, float]]] = []
    for comp in svd.components_:
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([(terms[j], float(comp[j])) for j in idx])
    return topics
