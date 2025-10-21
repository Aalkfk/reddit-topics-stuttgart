<div id="readme-top"></div>

# Stuttgart Reddit Topic Explorer

**Aufgabenstellung**

Extrahiere die häufigsten Themen aus Reddit-Beiträgen des Subreddits **/r/Stuttgart** mit Python. Die Analyse nutzt die Reddit-API (PRAW) bzw. vorhandene Daten und führt Topic Modeling mittels **LDA** durch.

**Zusammenfassung**

Pipeline mit konfigurierbaren Parametern: Abruf aus **r/Stuttgart** (optional inkl. Kommentare), Zeitfenster standardmäßig **letzte 12 Monate**, Topic Modeling via **LDA** mit k-Suche (Coherence **c_v**, Fallback **Log-Likelihood**). Start per `python -m src.main`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#environment-variables">Environment Variables</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#configuration">Configuration</a></li>
  </ol>
</details>

---

## About The Project

Dieses Projekt extrahiert Themen aus **/r/Stuttgart**-Posts per Python und führt **LDA**-Topic-Modeling mit konfigurierbarer k-Suche durch.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

- **Python** (mit `venv` und `pip`)

### Installation

```bash
# 1) GitHub-Repo klonen
git clone https://github.com/Aalkfk/reddit-topics-stuttgart
cd reddit-topics-stuttgart

# 2) Virtuelle Umgebung vorbereiten
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Lege Reddit-Creds in `.env` an:

```bash
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
export REDDIT_USER_AGENT="stuttgart-topics/1.0"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

```bash
# 4) Starten
python -m src.main
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Configuration

**Parameter**

- `--subreddit` *(str, Default: stuttgart)*  
  Auswahl des Subs.

- `--start` / `--end` *(YYYY-MM-DD)*  
  Start inkl., End exkl. in **UTC**; Default = letzte 12 Monate bis heute.

- `--limit` *(int)*  
  Max. Anzahl Posts, die aus `new()` geholt werden.

- `--include-comments` *(Flag)*  
  Bezieht Kommentare ein (Kontext), wenn gesetzt.

- `--max-comments-per-post` *(int, Default: 20)*  
  Kappung der je Post gelesenen Kommentare (nur wenn `--include-comments`).

- `--min-docs-per-flair` *(int, Default: 40)*  
  Mindestanzahl Dokumente, ab der je Flair ein eigenes **LDA** gerechnet wird, sonst globaler Fallback.

- `--kmin` / `--kmax` *(int)*  
  Untere/obere Grenze für die k-Suche bei **LDA** (Coherence **c_v** mit Fallback **Log-Likelihood**).  
  Intern als `range(kmin, kmax+1)` an `fit_lda_with_k` / `select_k` weitergegeben.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
