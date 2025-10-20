# /src/utils.py
# Purpose: helpers for stats and sampling
# ==========================================================
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import Counter
from .preprocess import tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

def top_flairs(df_posts: pd.DataFrame, topn: int = 20) -> pd.DataFrame:
    vc = df_posts["flair_text"].fillna("(ohne Flair)").value_counts().head(topn)
    return vc.rename_axis("flair").reset_index(name="count")

def top_users(df_posts: pd.DataFrame, topn: int = 20) -> pd.DataFrame:
    vc = df_posts["author"].fillna("[unknown]").value_counts().head(topn)
    return vc.rename_axis("author").reset_index(name="count")

def tfidf_top_terms_per_flair(df: pd.DataFrame, flair_col: str = "flair_text", text_col: str = "text_all", topn_terms: int = 12, max_features: int = 20000):
    out = {}
    for flair, grp in df.groupby(flair_col):
        if len(grp) < 3:
            continue
        texts = grp[text_col].fillna("").map(lambda s: " ".join(tokenize(s)))
        v = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
        X = v.fit_transform(texts)
        idx = np.argsort(X.sum(axis=0).A1)[::-1][:topn_terms]
        terms = v.get_feature_names_out()[idx]
        out[flair if pd.notna(flair) else "(ohne Flair)"] = list(terms)
    return out

def sample_docs_for_topics(df: pd.DataFrame, doc_topic: np.ndarray, topics: np.ndarray, top_k: int = 5) -> pd.DataFrame:
    rows = []
    for t in range(doc_topic.shape[1]):
        idx = np.where(topics == t)[0]
        if len(idx) == 0:
            continue
        idx_sorted = idx[np.argsort(doc_topic[idx, t])[::-1][:top_k]]
        for j in idx_sorted:
            rows.append({
                "topic": t,
                "title": df.iloc[j]["title"],
                "flair": df.iloc[j]["flair_text"],
                "permalink": df.iloc[j]["permalink"],
            })
    return pd.DataFrame(rows)
