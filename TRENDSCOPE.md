# TrendScope AI

Trend intelligence on top of the original trending-topics pipeline. Standard library plus the packages already used by the project; heavy AI models are optional.

## Run (Windows PowerShell)

```powershell
cd C:\KIT-GITHUB-Project\Trending-Topic\twitter-trending-topics
.venv\Scripts\python trendscope.py            # http://localhost:8765 (next free port if busy); seeds a synthetic demo on first run
.venv\Scripts\python trendscope.py --port 9000 --no-browser
.venv\Scripts\python -W ignore -m unittest discover -s tests     # 230 unit and API tests (59 verify trend detection and lifecycle states, 55 verify sentiment, NLP, optional-model fallbacks and evaluation metrics)
.venv\Scripts\python tests\run_ui_tests.py                       # 35 browser tests (needs Chrome or Edge)
.venv\Scripts\python tests\capture_ui.py                         # screenshots of every tab (desktop + phone) into .\screenshots, plus console-error and overflow checks
```

Data is stored in `trendscope_data\trendscope.db` (SQLite). Delete it to start fresh.

## Modules

| Area | Where |
|---|---|
| Dashboard, Analysis Studio, Data Centre, Trends, Topic Explorer, Sentiment, Evaluation Lab, History, Alerts | `trendscope\static\` (hash-routed single page app) |
| Uploads (CSV/JSON/JSONL), mapping, validation, duplicates, synthetic generator | `trendscope\datacentre.py` |
| Filters, pipeline, run comparison | `trendscope\engine.py` |
| Burst z-score and Poisson p-value, growth, velocity, lifecycle | `trendscope\trends.py` |
| Topic algorithms: `pivot` (the original `topicDetection.py`), `kmeans` (TF-IDF, restarts, silhouette), optional `embeddings` and `bertopic` | `trendscope\topics.py` |
| Sentiment: `vader`, `lexicon`, `sentistrength` (needs Java), optional `transformer`; experimental emotions | `trendscope\sentiment.py` |
| Accuracy, precision, recall, F1, confusion matrices, purity, ARI, timing | `trendscope\evaluation.py` |
| History, saved configurations, annotations | `trendscope\db.py` |
| CSV, JSON and HTML exports | `trendscope\reports.py` |
| Alert rules | `trendscope\alerts.py` |
| Optional X API v2 recent search | `trendscope\xapi.py` |

## Trend definitions

- **Burst z-score**: (mean of last w bins minus baseline mean) / max(baseline std, sqrt(baseline mean)); **p-value** is the Poisson tail.
- **Growth**: recent bins versus the bins before. **Trend ramp**: smoother growth over a window of max(6, n/5) bins.
- **Lifecycle** (re-evaluated every bin): new, rising, peaking, stable, declining, inactive. A slope counts only if it is more than 2 standard errors from zero.
  - `new`: first seen in the last 2 bins, after silence. `rising` / `declining`: significant slope over the recent window; `declining` also if the level has fallen below half of its peak by more than 2 Poisson standard deviations (so noise on a small topic is not read as a decline).
  - `peaking`: neither rising nor declining, and the recent level is clearly above the topic's own earlier level (more than 2 standard errors and at least +30%). `stable`: neither rising nor declining and not clearly above its earlier level: flat or low growth, or too little history to show a peak.
  - Conservative by design: on noisy counts a ramp of about +2.5% per bin is usually reported as `peaking` (it ends at its highest level), while +8% per bin is reliably `rising`. Steady topics from 1 to 100 tweets per bin are `stable` in over 90% of runs.
  - Runs saved before `stable` existed keep their old labels (flat topics show as `peaking`) and still load, compare and export.
- **Time bins** are anchored to the end of the data (the last bin ends at the last tweet), so the "recent" bins used for growth, z-score and emerging flags are always complete. Only the oldest bin can be partial. Runs saved before this change keep their original bin layout.
- **`bursting`** (KPI) counts topics whose most recent bins are bursting now; the Trends tab lists earlier bursts separately.
- **Emerging**: rising or new, and a burst, growth of +100% or a ramp of +50%.
- **Score**: 0.5 x burst + 0.3 x growth + 0.2 x share.

## Sentiment and NLP

- **Scores** are normalised to [-1, 1]: VADER compound, the opinion lexicon `(pos - neg) / (pos + neg)`, SentiStrength scale / 4, and transformer `P(positive) - P(negative)`. **VADER is the default.**
- **Labels**: positive if score >= the positive threshold, negative if score <= minus the negative threshold, otherwise neutral. A score of exactly 0 is always neutral (also with thresholds of 0). Thresholds must be between 0 and 1; anything else is rejected (HTTP 400).
- **VADER vs the lexicon**: VADER understands negation and intensity ("not good" is negative, "very good!!!" scores above "good"); the lexicon counts polar words, so any single polar word scores +-1 and "not good" is positive. The Evaluation Lab measures the gap on labelled data (it widens sharply on hard cases).
- **Confidence**: there is no separate confidence value. The size of the score is the only strength signal (it grows with evidence for VADER, but is always +-1 for the lexicon).
- **Emotions** (experimental): whole-word matches against a small hand-written lexicon, counted once per tweet. Negation is not understood ("not happy" still counts as joy).
- **Optional models** (`transformer`, `embeddings`, `bertopic`, `sentistrength`) are never downloaded or installed automatically by the tests. When a package is missing they show as unavailable with the install command, and choosing one anyway gives a clear error (HTTP 400), never fabricated results. When a model cannot be loaded (for example no internet for the first download) the error says so; the model is loaded once per process. Embedding clusters use k-means++ with 4 restarts.
- The tests cover this code with small fakes for the optional packages and for the Java process. They check our integration code, not the quality of the real models.

## Data sources

Synthetic data is yellow-bannered, uploads grey, live X data green. Live retrieval is optional: set the `X_BEARER_TOKEN` environment variable before starting. It uses `GET https://api.x.com/2/tweets/search/recent` (last 7 days; plan limits and per-read costs apply). The token is never typed into the page or stored, and tests use a fake HTTP layer only.

## Limitations

- Live X retrieval and the real SentiStrength jar, transformer, sentence-transformers and BERTopic models have never been run on this machine (not installed). Only the integration code around them is tested, with fakes.
- The original document-pivot method is quadratic: slow above about 1,500 tweets.
- Synthetic text is templated, so evaluation scores on it are optimistic.
- VADER and the lexicon are English only; emotion analysis is a small hand-written lexicon.

- CSV exports prefix text cells that start with `=`, `+`, `-`, `@`, tab or CR with an apostrophe (spreadsheet formula protection). Only the exported copy changes; stored data and the JSON export are untouched.
- Growth over a single time bin is noisy for small topics (for example +100% next to a `declining` stage); the stage uses the significance-tested slope, so trust the stage over the single-bin growth figure.
