# 1) GitHub-Repo klonen
git clone https://github.com/Aalkfk/reddit-topics-stuttgart

cd reddit-topics-stuttgart

# 2) Virtuelle Umgebung vorbereiten

python -m venv .venv

# macOS/Linux:

source .venv/bin/activate

pip install -r requirements.txt

# 3) Reddit-Creds in .env anlegen
export REDDIT_CLIENT_ID="..."

export REDDIT_CLIENT_SECRET="..."

export REDDIT_USER_AGENT="stuttgart-topics/1.0"

# 4) Starten
python -m src.main

====
# Parameters
--subreddit (str, Default: stuttgart)

Auswahl des Subs.


--start / --end (YYYY-MM-DD)

Start inkl., End exkl. in UTC; Default = letzte 12 Monate bis heute.


--limit (int)

Max. Anzahl Posts, die aus new() geholt werden.


--include-comments (Flag)

Bezieht Kommentare ein (Kontext), wenn gesetzt.


--max-comments-per-post (int, Default: 20)

Kappung der je Post gelesenen Kommentare (nur wenn --include-comments).


--min-docs-per-flair (int, Default: 40)

Mindestanzahl Dokumente, ab der je Flair ein eigenes LDA gerechnet wird, sonst globaler Fallback.


--kmin / --kmax (int)

Untere/obere Grenze für die k-Suche bei LDA (Coherence c_v mit Fallback Log-Likelihood). Intern als range(kmin, kmax+1) an fit_lda_with_k/select_k weitergegeben.