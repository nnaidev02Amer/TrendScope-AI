"""Model Evaluation Lab. Metrics are computed only from real predictions against real labels.

Sentiment: accuracy, per-class precision / recall / F1, macro-F1, confusion matrix, processing time.
Topics:    purity, adjusted Rand index (ARI), coverage, number of clusters, processing time.
Labels come from synthetic ground truth, an uploaded label column, or tweets you annotated manually.
"""
import time
from collections import Counter, defaultdict

from . import sentiment, topics

CLASSES = ['positive', 'neutral', 'negative']


def confusion(true, pred):
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = [[0] * 3 for _ in range(3)]
    for t, p in zip(true, pred):
        m[idx[t]][idx[p]] += 1
    return m


def prf(m):
    out = {}
    for i, c in enumerate(CLASSES):
        tp = m[i][i]
        fp = sum(m[j][i] for j in range(3)) - tp
        fn = sum(m[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4), 'support': sum(m[i])}
    return out


def sentiment_metrics(true, pred):
    m = confusion(true, pred)
    per = prf(m)
    n = len(true)
    return {'n': n, 'accuracy': round(sum(m[i][i] for i in range(3)) / n, 4) if n else 0.0, 'per_class': per,
            'macro_f1': round(sum(v['f1'] for v in per.values()) / 3, 4),
            'macro_precision': round(sum(v['precision'] for v in per.values()) / 3, 4),
            'macro_recall': round(sum(v['recall'] for v in per.values()) / 3, 4),
            'confusion': m, 'classes': CLASSES}


def evaluate_sentiment(rows, algos, pos_t=0.05, neg_t=0.05):
    """rows must carry label_sentiment. Returns one result per algorithm (errors are reported, never faked)."""
    sentiment.check_thresholds(pos_t, neg_t)
    rows = [r for r in rows if r.get('label_sentiment') in CLASSES]
    if len(rows) < 20:
        raise ValueError('Need at least 20 labelled tweets for a meaningful evaluation (have %d).' % len(rows))
    texts = [r['text'] for r in rows]
    true = [r['label_sentiment'] for r in rows]
    dist = dict(Counter(true))
    res = []
    for a in algos:
        try:
            t0 = time.time()
            scores = sentiment.score(a, texts)
            dt = time.time() - t0
            pred = [sentiment.to_label(s, pos_t, neg_t) for s in scores]
            res.append(dict(sentiment_metrics(true, pred), algo=a, seconds=round(dt, 3), ms_per_1000=round(dt * 1e6 / len(texts), 1)))
        except Exception as e:
            res.append({'algo': a, 'error': '%s: %s' % (type(e).__name__, e)})
    return {'kind': 'sentiment', 'n': len(rows), 'label_distribution': dist, 'results': res}


def purity(true, pred):
    by = defaultdict(Counter)
    for t, p in zip(true, pred):
        by[p][t] += 1
    return sum(max(c.values()) for c in by.values()) / len(true) if true else 0.0


def ari(true, pred):
    def c2(x):
        return x * (x - 1) / 2
    cont = defaultdict(int)
    for t, p in zip(true, pred):
        cont[(t, p)] += 1
    a, b = Counter(true), Counter(pred)
    n = len(true)
    sum_ij = sum(c2(v) for v in cont.values())
    sum_a, sum_b = sum(c2(v) for v in a.values()), sum(c2(v) for v in b.values())
    expected = sum_a * sum_b / c2(n) if n > 1 else 0
    mx = (sum_a + sum_b) / 2
    return (sum_ij - expected) / (mx - expected) if mx != expected else 1.0


def evaluate_topics(rows, algos, params):
    rows = [r for r in rows if r.get('label_topic') and r.get('ts')]
    true_all = [r['label_topic'] for r in rows]
    scored = [i for i, t in enumerate(true_all) if t != 'noise']
    if len(scored) < 30 or len(set(true_all[i] for i in scored)) < 2:
        raise ValueError('Need at least 30 tweets with a topic label covering 2+ topics.')
    res = []
    for a in algos:
        try:
            t0 = time.time()
            labels, info = topics.assign(rows, a, params)
            dt = time.time() - t0
            kept = [i for i in scored if labels[i] >= 0]
            tt, pp = [true_all[i] for i in kept], [labels[i] for i in kept]
            res.append({'algo': a, 'seconds': round(dt, 3), 'clusters': len({l for l in labels if l >= 0}),
                        'true_topics': len(set(true_all[i] for i in scored)), 'coverage': round(len(kept) / len(scored), 4),
                        'purity': round(purity(tt, pp), 4) if kept else 0.0, 'ari': round(ari(tt, pp), 4) if len(kept) > 1 else 0.0,
                        'info': info})
        except Exception as e:
            res.append({'algo': a, 'error': '%s: %s' % (type(e).__name__, e)})
    return {'kind': 'topics', 'n': len(rows), 'results': res}
