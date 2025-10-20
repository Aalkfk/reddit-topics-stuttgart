# /src/main.py
import os
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict
from src.preprocess import tokenize

import numpy as np
import pandas as pd

# Paket-Import mit Fallback
HERE = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(ROOT, "out")
DATADIR = os.path.join(ROOT, "data")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

try:
    from src.fetch import fetch_subreddit_posts
    from src.preprocess import (
        attach_norm_hash,
        drop_duplicates_by_hash,
        heuristic_spam_mask,
        clean_text,
    )
    from src.topics import fit_lda_with_k, terms_for_topics, lsa_top_terms
    from src.utils import top_flairs, top_users, tfidf_top_terms_per_flair, sample_docs_for_topics
except ImportError:
    import sys
    sys.path.append(os.path.join(ROOT, "src"))
    from fetch import fetch_subreddit_posts
    from preprocess import (
        attach_norm_hash,
        drop_duplicates_by_hash,
        heuristic_spam_mask,
        clean_text,
    )
    from topics import fit_lda_with_k, terms_for_topics, lsa_top_terms
    from utils import top_flairs, top_users, tfidf_top_terms_per_flair, sample_docs_for_topics


# ---------- Rendering ----------
def _html_escape(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def render_html_report(meta: Dict, flair_stats: pd.DataFrame, user_stats: pd.DataFrame,
                       per_flair: Dict, global_block: Dict, method_text: str, ethics_text: str) -> str:
    def df_to_html(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "<i>n/a</i>"
        return df.to_html(index=False, escape=False)
    parts = []
    parts.append(f"<h1>/r/Stuttgart – Themenanalyse</h1>")
    parts.append(f"<p><b>Dokument erstellt:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>")

    parts.append(f"<h2>Meta</h2><ul><li>Dokumente: {meta.get('n_docs')}</li><li>Flairs: {meta.get('n_flairs')}</li></ul>")
    parts.append(f"<h2>Methodik (Kurz)</h2><div style='white-space:pre-wrap'>{_html_escape(method_text)}</div>")
    parts.append(f"<h2>Ethik & Nutzung</h2><div style='white-space:pre-wrap'>{_html_escape(ethics_text)}</div>")

    parts.append("<h2>Top Flairs</h2>" + df_to_html(flair_stats))
    parts.append("<h2>Aktivste Nutzer</h2>" + df_to_html(user_stats))

    # Global Block
    parts.append("<hr><h2>Global – LDA Themen</h2>")
    if global_block.get("lda_terms"):
        parts.append("<ol>" + "".join(["<li>" + ", ".join([_html_escape(w) for w,_ in t]) + "</li>" for t in global_block["lda_terms"]]) + "</ol>")
    if global_block.get("lsa_terms"):
        parts.append("<h3>LSA (Vergleich)</h3><ol>" + "".join(["<li>" + ", ".join([_html_escape(w) for w,_ in t]) + "</li>" for t in global_block["lsa_terms"]]) + "</ol>")

    # Pro Flair
    parts.append("<hr><h2>Pro Flair</h2>")
    for flair, blk in per_flair.items():
        parts.append(f"<h3>{_html_escape(str(flair if flair else '(ohne Flair)'))}</h3>")
        if blk.get("tfidf_terms"):
            parts.append("<b>TF-IDF Top-Begriffe:</b> " + ", ".join([_html_escape(t) for t in blk["tfidf_terms"]]))
        if blk.get("lda_terms"):
            parts.append("<details><summary><b>LDA Themen</b></summary><ol>" + "".join(["<li>" + ", ".join([_html_escape(w) for w,_ in t]) + "</li>" for t in blk["lda_terms"]]) + "</ol></details>")
        if blk.get("lsa_terms"):
            parts.append("<details><summary><b>LSA (Vergleich)</b></summary><ol>" + "".join(["<li>" + ", ".join([_html_escape(w) for w,_ in t]) + "</li>" for t in blk["lsa_terms"]]) + "</ol></details>")
        if blk.get("samples_df") is not None and not blk["samples_df"].empty:
            parts.append("<b>Beispiel-Posts pro LDA-Thema:</b>" + df_to_html(blk["samples_df"][["topic","title","flair","permalink"]]))

    return "\n".join(parts)


# ---------- Pipeline helpers ----------
def prepare_text_columns(df_posts: pd.DataFrame, df_comments: pd.DataFrame, include_comments: bool) -> pd.DataFrame:
    df = df_posts.copy()
    df["comments_text"] = ""
    if include_comments and not df_comments.empty:
        comments_concat = df_comments.groupby("post_id")["body"].apply(lambda s: "\n".join(s.tolist()))
        df = df.merge(comments_concat.rename("comments_text"), left_on="id", right_index=True, how="left")
        df["comments_text"] = df["comments_text"].fillna("")
    df["text_all"] = (df["title"].fillna("") + "\n" + df["selftext"].fillna("") + "\n" + df["comments_text"]).str.strip()
    return df

def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = attach_norm_hash(df, title_col="title", text_col="selftext")
    df, n_dups = drop_duplicates_by_hash(df)
    print(f"[Quality] Removed duplicates: {n_dups}")
    mask = heuristic_spam_mask(df, min_tokens=10, max_url_ratio=0.5)
    removed = int((~mask).sum())
    df = df.loc[mask].reset_index(drop=True)
    print(f"[Quality] Removed low-quality/spam: {removed}")
    return df

def build_method_text(metric_used: str, k_diag_global: Dict[int, float], min_docs_per_flair: int, start: str, end: str) -> str:
    lines = [
        f"Zeitraum: {start} bis {end}",
        "Vorverarbeitung: Bereinigung (Kleinbuchstaben, URLs entfernen), Tokenisierung, Stoppwörter (NLTK/fallback), N-Gramme (1–2).",
        "Merkmale: Bag-of-Words (Count) + TF-IDF.",
        ("Themenmodellierung: LDA (k via Coherence c_v) mit Fallback Log-Likelihood." if metric_used=="coherence_c_v"
         else "Themenmodellierung: LDA (k via Log-Likelihood-Gitterwahl)."),
        "Optional: LSA (TruncatedSVD) zum Vergleich (wird nur bei stabiler Matrix gezeigt).",
        f"Pro-Flair-Modellierung ab mindestens {min_docs_per_flair} Dokumenten, sonst globaler Fallback.",
        "Qualitätssicherung: Duplikate (MD5 auf normalisiertem Text), heuristisches Filtering (Token-Mindestlänge, URL-Anteil).",
    ]
    if k_diag_global:
        best_k = max(k_diag_global, key=lambda k: k_diag_global[k])
        lines.append(f"k-Selektion (global): bestes k = {best_k} in {sorted(list(k_diag_global.keys()))}.")
    return "\n".join(lines)

def build_ethics_text() -> str:
    return (
        "• Ausschließlich öffentliche Inhalte (Reddit-Beiträge/Kommentare) werden verarbeitet.\n"
        "• Keine Profilbildung, keine personenbezogene Auswertung; Ergebnisse sind aggregiert (Flairs, Themen, Top-Begriffe).\n"
        "• Beachtung der Reddit-API-Nutzungsbedingungen und Content Policies; Zugriff über offizielle API.\n"
        "• Kommentare werden nur optional zur Kontextanreicherung einbezogen.\n"
        "• Links im Report verweisen auf Originalbeiträge; keine Speicherung sensibler Personendaten."
    )


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    today = datetime.now(tz=timezone.utc).date()
    default_start = (today - timedelta(days=365)).isoformat()
    default_end = today.isoformat()

    parser.add_argument("--subreddit", default="stuttgart")
    parser.add_argument("--start", default=default_start, help="Startdatum YYYY-MM-DD (inkl., UTC)")
    parser.add_argument("--end", default=default_end, help="Enddatum YYYY-MM-DD (exkl., UTC)")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--include-comments", action="store_true", help="Kommentare einbeziehen (Default: aus)")
    parser.add_argument("--max-comments-per-post", type=int, default=20)
    parser.add_argument("--min-docs-per-flair", type=int, default=40)
    parser.add_argument("--kmin", type=int, default=6)
    parser.add_argument("--kmax", type=int, default=13)
    args = parser.parse_args()

    fetched = fetch_subreddit_posts(
        subreddit=args.subreddit,
        start=args.start,
        end=args.end,
        limit=args.limit,
        include_comments=args.include_comments,
        max_comments_per_post=args.max_comments_per_post if args.include_comments else 0,
        throttle_sec=0.0,
    )
    df_posts = fetched["posts"]
    df_comments = fetched["comments"]

    if df_posts.empty:
        raise SystemExit("Keine Beiträge im angegebenen Zeitraum gefunden.")

    # Rohdaten speichern
    df_posts.to_csv(os.path.join(DATADIR, f"raw_r_{args.subreddit}_posts.csv"), index=False)
    df_comments.to_csv(os.path.join(DATADIR, f"raw_r_{args.subreddit}_comments.csv"), index=False)

    df = prepare_text_columns(df_posts, df_comments, include_comments=args.include_comments)
    df = apply_quality_filters(df)

    # Stats
    flair_stats = top_flairs(df)
    user_stats = top_users(df)

    # GLOBAL LDA (k via c_v wenn möglich, sonst Log-Likelihood)
    texts_tok = df["text_all"].map(lambda s: tokenize(s)).tolist()
    texts_joined = [" ".join(toks) for toks in texts_tok]

    lda, v_count, X_count, best_k, k_diag, metric_used = fit_lda_with_k(
        texts_tok, texts_joined, k_range=range(args.kmin, args.kmax + 1), topn=12
    )
    lda_terms_global = terms_for_topics(lda, v_count, topn=12)

    # LSA (robust; wird ggf. ausgelassen)
    try:
        from src.topics import lsa_top_terms as _lsa_top_terms
    except ImportError:
        from topics import lsa_top_terms as _lsa_top_terms

    lsa_terms_global = _lsa_top_terms(texts_joined, requested_topics=min(8, max(6, best_k)), topn=12)


    # Pro Flair
    per_flair = {}
    tfidf_terms_map = tfidf_top_terms_per_flair(df)
    for flair, grp in df.groupby("flair_text"):
        flair_name = flair if pd.notna(flair) else "(ohne Flair)"
        blk = {}
        blk["tfidf_terms"] = tfidf_terms_map.get(flair_name, [])
        if len(grp) >= args.min_docs_per_flair:
            tok = grp["text_all"].map(lambda s: [t for t in clean_text(s).split() if t]).tolist()
            joined = [" ".join(t) for t in tok]
            lda_f, v_c_f, X_c_f, best_k_f, _, _ = fit_lda_with_k(
                tok, joined, k_range=range(args.kmin, args.kmax + 1), topn=12
            )
            blk["lda_terms"] = terms_for_topics(lda_f, v_c_f, topn=12)
            try:
                blk["lsa_terms"] = lsa_top_terms(joined, requested_topics=min(8, max(6, best_k_f)), topn=12)
            except Exception:
                blk["lsa_terms"] = []
            # Samples
            doc_topic_f = lda_f.transform(X_c_f)
            topics_f = doc_topic_f.argmax(axis=1)
            blk["samples_df"] = sample_docs_for_topics(grp.reset_index(drop=True), doc_topic_f, topics_f, top_k=3)
        else:
            blk["lda_terms"] = []
            blk["lsa_terms"] = []
            blk["samples_df"] = pd.DataFrame()
        per_flair[flair_name] = blk

    global_block = {"lda_terms": lda_terms_global, "lsa_terms": lsa_terms_global}
    meta = {"n_docs": int(len(df)), "n_flairs": int(df["flair_text"].nunique())}

    method_text = build_method_text(metric_used, k_diag, args.min_docs_per_flair, args.start, args.end)
    ethics_text = build_ethics_text()

    html = render_html_report(meta, flair_stats, user_stats, per_flair, global_block, method_text, ethics_text)

    out_html = os.path.join(OUTDIR, f"report_r_{args.subreddit}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    flair_stats.to_csv(os.path.join(OUTDIR, "top_flairs.csv"), index=False)
    user_stats.to_csv(os.path.join(OUTDIR, "top_users.csv"), index=False)
    print(f"[OK] Report written to {out_html}")

if __name__ == "__main__":
    main()
