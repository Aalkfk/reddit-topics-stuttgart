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
