from __future__ import annotations
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

try:
    import praw  # type: ignore
except Exception as e:
    raise RuntimeError("PRAW is required. pip install praw") from e


@dataclass
class PostRecord:
    id: str
    created_utc: float
    author: str
    title: str
    selftext: str
    permalink: str
    score: int
    num_comments: int
    flair_text: Optional[str]


@dataclass
class CommentRecord:
    post_id: str
    comment_id: str
    author: str
    body: str


def _mk_reddit() -> "praw.Reddit":
    load_dotenv()
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "rStuttgartTopics/1.0"),
        ratelimit_seconds=5,
    )


def fetch_subreddit_posts(
    subreddit: str = "stuttgart",
    start: Optional[str] = None,  # "YYYY-MM-DD" inkl.
    end: Optional[str] = None,    # "YYYY-MM-DD" exkl.; None => jetzt
    limit: int = 1500,
    include_comments: bool = False,
    max_comments_per_post: int = 0,
    throttle_sec: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """Gemäß Konzept: Posts der letzten 12 Monate (Default) via Reddit-API; Kommentare optional."""
    reddit = _mk_reddit()
    sr = reddit.subreddit(subreddit)

    # Zeitraum vorbereiten
    if end:
        end_ts = datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()
    else:
        end_ts = datetime.now(tz=timezone.utc).timestamp()

    if start:
        start_ts = datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()
    else:
        # Default: 12 Monate
        start_ts = (datetime.now(tz=timezone.utc).replace(microsecond=0) - pd.DateOffset(years=1)).timestamp()

    posts: List[PostRecord] = []
    comments: List[CommentRecord] = []

    # Wir nutzen nur new() und filtern per Datum, bis wir unter start_ts fallen
    count = 0
    for s in sr.new(limit=limit):
        created = float(getattr(s, "created_utc", 0.0))
        if created > end_ts:
            continue
        if created < start_ts:
            break

        p = PostRecord(
            id=s.id,
            created_utc=created,
            author=str(getattr(s.author, "name", "[deleted]")),
            title=s.title or "",
            selftext=getattr(s, "selftext", "") or "",
            permalink="https://www.reddit.com" + getattr(s, "permalink", ""),
            score=int(getattr(s, "score", 0)),
            num_comments=int(getattr(s, "num_comments", 0)),
            flair_text=getattr(s, "link_flair_text", None),
        )
        posts.append(p)
        count += 1

        if include_comments and max_comments_per_post > 0 and p.num_comments > 0:
            s.comment_sort = "top"
            s.comments.replace_more(limit=0)
            c_added = 0
            for c in s.comments.list():
                if c_added >= max_comments_per_post:
                    break
                comments.append(
                    CommentRecord(
                        post_id=s.id,
                        comment_id=c.id,
                        author=str(getattr(c.author, "name", "[deleted]")),
                        body=c.body or "",
                    )
                )
                c_added += 1

        if throttle_sec:
            time.sleep(throttle_sec)

    df_posts = pd.DataFrame([asdict(p) for p in posts])
    df_comments = pd.DataFrame([asdict(c) for c in comments]) if comments else pd.DataFrame(columns=["post_id","comment_id","author","body"])

    if not df_posts.empty:
        df_posts = df_posts.sort_values("created_utc", ascending=False).reset_index(drop=True)
    return {"posts": df_posts, "comments": df_comments}
