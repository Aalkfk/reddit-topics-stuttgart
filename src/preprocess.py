import re
import hashlib
from typing import Tuple, Set, List

import pandas as pd

# NLTK-Stopwörter (optional), sonst Fallback
def _get_stopwords() -> Set[str]:
    try:
        from nltk.corpus import stopwords  # type: ignore
        sw = set(stopwords.words("german"))
        sw |= {"dass", "daß", "im", "ins", "vom", "zum", "zur", "für", "über", "r", "stuttgart"}
        return sw
    except Exception:
        return set(
            """aber als also am an auch auf aus bei bin bis da dann dass dem den der des die dir doch dort du durch ein eine einem einen einer eines er es fürs gegen habe haben hat hier ich ihr im in ist ja jede jedem jeden jeder jedes kann kein keine kleinen kommt mal mehr mit muss nach nicht nun oder sehr sein seine seiner seit so soll sollen sondern während war waren was wir wird wirds wo zu zum zur über unter zwischen usw usw""".split()
        )

GERMAN_SW: Set[str] = _get_stopwords()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-zA-ZäöüÄÖÜß0-9]+")

def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = NON_WORD_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = clean_text(s)
    toks = [t for t in s.split(" ") if t and t not in GERMAN_SW and len(t) > 2]
    return toks

def attach_norm_hash(df: pd.DataFrame, title_col: str = "title", text_col: str = "selftext") -> pd.DataFrame:
    def _norm(row) -> str:
        a = clean_text(str(row.get(title_col, "")))
        b = clean_text(str(row.get(text_col, "")))
        return f"{a}\n{b}"

    df = df.copy()
    h = df.apply(_norm, axis=1).map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    df["norm_hash"] = h
    return df

def drop_duplicates_by_hash(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df2 = df.drop_duplicates(subset=["norm_hash"]).reset_index(drop=True)
    return df2, before - len(df2)

def heuristic_spam_mask(
    df: pd.DataFrame,
    min_tokens: int = 10,
    max_url_ratio: float = 0.5,
    title_col: str = "title",
    text_col: str = "selftext",
) -> pd.Series:
    """Heuristik gem. Konzept: Mindest-Tokenanzahl & URL-Anteil – ohne Autor-Blacklist."""
    titles = df[title_col].fillna("")
    bodies = df[text_col].fillna("")
    combined = titles + " " + bodies
    cleaned = combined.map(clean_text)
    tokens = cleaned.map(lambda s: [t for t in s.split(" ") if t])
    tok_counts = tokens.map(len)

    url_counts = (titles + " " + bodies).str.count(r"https?://|www\.")
    url_ratio = url_counts / tok_counts.clip(lower=1)

    keep = (tok_counts >= min_tokens) & (url_ratio <= max_url_ratio)
    return keep
