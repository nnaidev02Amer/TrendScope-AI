"""Sentiment algorithms. Every algorithm maps a list of texts to scores in [-1, 1].

  vader         VADER compound score (NLTK). Baseline; English only.
  lexicon       Hu & Liu opinion lexicon: (pos - neg) / (pos + neg) word counts. Already in the repo.
  sentistrength SentiStrength JAR via Java (existing sentiStrength.py). Needs Java on PATH.
  transformer   Optional Hugging Face model (needs `pip install transformers torch`; downloads a model).
"""
import importlib.util
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vader = None
_lex = None
_WORD = re.compile(r"[a-z']+")


def _get_vader():
    global _vader
    if _vader is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def _get_lexicon():
    global _lex
    if _lex is None:
        sets = []
        for name in ('POSITIVE-WORDS.TXT', 'NEGATIVE-WORDS.TXT'):
            with open(os.path.join(ROOT, 'opinionLexicon', name), encoding='utf-8', errors='replace') as f:
                sets.append({l.strip().lower() for l in f if l.strip() and not l.startswith(';')})
        _lex = tuple(sets)
    return _lex


def availability():
    """Which algorithms can run on this machine, with a reason when they cannot."""
    out = {}
    try:
        _get_vader()
        out['vader'] = {'ok': True, 'note': 'Baseline, English only'}
    except Exception as e:  # missing nltk data
        out['vader'] = {'ok': False, 'note': 'NLTK vader_lexicon missing: python -c "import nltk; nltk.download(\'vader_lexicon\')" (%s)' % type(e).__name__}
    try:
        _get_lexicon()
        out['lexicon'] = {'ok': True, 'note': 'Hu & Liu opinion lexicon, English only'}
    except OSError as e:
        out['lexicon'] = {'ok': False, 'note': 'Opinion lexicon files not found: %s' % e}
    jar = os.path.join(ROOT, 'SentiStrength', 'SentiStrengthCom.jar')
    if shutil.which('java') and os.path.isfile(jar):
        out['sentistrength'] = {'ok': True, 'note': 'Runs SentiStrength through Java'}
    else:
        out['sentistrength'] = {'ok': False, 'note': 'Needs Java on PATH (java not found)' if not shutil.which('java') else 'SentiStrength jar missing'}
    if importlib.util.find_spec('transformers') and importlib.util.find_spec('torch'):
        out['transformer'] = {'ok': True, 'note': 'Hugging Face pipeline (model downloads on first use); not tested in this project'}
    else:
        out['transformer'] = {'ok': False, 'note': 'Optional: pip install transformers torch'}
    return out


def score(algo, texts):
    if not texts:
        return []
    if algo == 'vader':
        sia = _get_vader()
        return [sia.polarity_scores(t)['compound'] for t in texts]
    if algo == 'lexicon':
        pos, neg = _get_lexicon()
        res = []
        for t in texts:
            words = _WORD.findall(t.lower())
            p = sum(1 for w in words if w in pos)
            n = sum(1 for w in words if w in neg)
            res.append((p - n) / (p + n) if p + n else 0.0)
        return res
    if algo == 'sentistrength':
        if not availability()['sentistrength']['ok']:
            raise RuntimeError('SentiStrength unavailable: ' + availability()['sentistrength']['note'])
        import sentiStrength
        cwd = os.getcwd()
        os.chdir(ROOT)  # sentiStrength.py uses relative paths
        try:
            vals = sentiStrength.rate([t for t in texts])
        finally:
            os.chdir(cwd)
        if len(vals) != len(texts):
            raise RuntimeError('SentiStrength returned %d scores for %d texts' % (len(vals), len(texts)))
        return [v / 4.0 for v in vals]
    if algo == 'transformer':
        if not availability()['transformer']['ok']:
            raise RuntimeError('transformers/torch are not installed')
        pipe = _get_transformer()
        res = []
        for out in pipe(list(texts), batch_size=16):
            d = {o['label'].lower(): o['score'] for o in out}
            if 'positive' not in d and 'negative' not in d:
                raise RuntimeError('The transformer model returned unrecognised labels %s (expected positive/negative/neutral).' % sorted(d))
            res.append(d.get('positive', 0.0) - d.get('negative', 0.0))
        return res
    raise ValueError('Unknown sentiment algorithm: %s' % algo)


TRANSFORMER_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
_tf_pipe = None


def _get_transformer():
    """Load the model once. A failed load (no internet for the first download, no disk space...) becomes a clear RuntimeError."""
    global _tf_pipe
    if _tf_pipe is None:
        try:
            from transformers import pipeline
            _tf_pipe = pipeline('sentiment-analysis', model=TRANSFORMER_MODEL, top_k=None, truncation=True)
        except Exception as e:
            raise RuntimeError('The transformer model %s could not be loaded (it is downloaded once on first use, so it needs internet or a local copy): %s: %s'
                               % (TRANSFORMER_MODEL, type(e).__name__, e))
    return _tf_pipe


def check_thresholds(pos_t, neg_t):
    if not (0 <= pos_t <= 1 and 0 <= neg_t <= 1):
        raise ValueError('Sentiment thresholds must be between 0 and 1 (got positive %s, negative %s).' % (pos_t, neg_t))


def to_label(s, pos_t=0.05, neg_t=0.05):
    """Score -> label. A score of exactly 0 (no sentiment found) is always neutral, even with a threshold of 0
    (this matches the original pipeline, which uses a strict > 0)."""
    return 'positive' if s > 0 and s >= pos_t else ('negative' if s < 0 and s <= -neg_t else 'neutral')


# ------------------------------------------------------------------ emotions (experimental)
EMOTIONS = {
    'joy': 'love loved happy thrilled amazing fantastic wonderful brilliant excellent great delighted glad excited celebrate enjoy',
    'anger': 'angry furious hate hated pathetic outrage outraged annoyed mad disgusting livid rage',
    'sadness': 'sad disappointed disappointing sorry unhappy miserable heartbroken upset gloomy tragic grief',
    'fear': 'afraid scared worried fear panic anxious danger dread terrified threat warning',
    'surprise': 'surprised shocked unexpected wow astonished sudden unbelievable stunned',
}
_EMO = {w: e for e, ws in EMOTIONS.items() for w in ws.split()}


def emotions(texts):
    """Experimental: counts of words from a tiny hand-written lexicon. Not a validated model."""
    from collections import Counter
    c = Counter()
    for t in texts:
        found = {_EMO[w] for w in _WORD.findall(t.lower()) if w in _EMO}
        c.update(found)
    return {e: c.get(e, 0) for e in EMOTIONS}
