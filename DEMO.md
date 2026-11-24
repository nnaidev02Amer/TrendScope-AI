# Demo and dashboard

## Run the demo (no Twitter/X credentials needed)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install nltk matplotlib python-twitter pymongo
.venv\Scripts\python -c "import nltk; nltk.download('vader_lexicon')"
.venv\Scripts\python demo.py
```

`demo.py` generates 400 **synthetic** tweets (fixed seed, reproducible), runs the unchanged
analysis (`main.py -a`), builds `demo/out/dashboard.html` and serves it on
http://localhost:8765/dashboard.html (next free port if busy). Options: `--no-serve`, `--no-browser`, `--port N`, `--tweets N`.

## Dashboard for your own analysis

```powershell
.venv\Scripts\python main.py -a tweets.json out
.venv\Scripts\python dashboard.py out --source live      # or: synthetic | unspecified
```

This writes `out/dashboard.html` (standalone, no external libraries). The coloured banner at the top
always states whether the data is synthetic, live Twitter/X, or unspecified.

## Synthetic vs live data

| | Synthetic demo | Live |
|---|---|---|
| Source | `demo.py` templates | `main.py -r ...` (needs API keys in `twitterApiHandle.py`) |
| Banner | yellow, "SYNTHETIC DEMONSTRATION DATA" | green, "LIVE TWITTER/X DATA" |

Retrieval uses the 2018 Twitter streaming API and has not been tested here.

## Notes
- Sentiment: VADER compound score; positive above 0, negative below 0.
- Per-topic counts in `stats.csv` use the same threshold as `sentimentStats.json`.
- Python 3 standard library plus the packages above; the dashboard itself needs nothing extra.


## TrendScope AI

Full application: `.venv\Scripts\python trendscope.py` (see TRENDSCOPE.md). The older one-page demo above still works.
