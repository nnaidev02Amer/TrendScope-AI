"""Analysis engine: filters -> duplicates -> sentiment -> topics -> time series -> trend intelligence."""
import math
import random
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from . import datacentre, sentiment, topics, trends

DEFAULT_CONFIG = {
    'name': 'Untitled analysis', 'keywords': [], 'keyword_mode': 'any', 'hashtags': [], 'exclude': [],
    'date_from': '', 'date_to': '', 'languages': [], 'min_followers': 0, 'drop_duplicates': True,
    'sentiment_algo': 'vader', 'pos_threshold': 0.05, 'neg_threshold': 0.05,
    'topic_algo': 'kmeans', 'threshold': 0.3, 'min_size': 8, 'k': 0, 'max_tweets': 2500,
    'bin_minutes': 0, 'burst_z': 3.0, 'recent_bins': 3, 'emotions': True,
}
BIN_CHOICES = [300, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400]


MAX_BINS = 2500      # more time bins than this make the trend calculations slow and the charts unreadable
MAX_K = 50           # K-Means seeding cost grows with k squared


def _number(c, key, conv):
    v = c[key]
    try:
        if isinstance(v, bool):
            raise TypeError
        out = conv(v)
        if isinstance(out, float) and not math.isfinite(out):
            raise ValueError
        return out
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Setting "%s" must be a number (got %r).' % (key, v))


def clean_config(cfg):
    c = dict(DEFAULT_CONFIG)
    c.update({k: v for k, v in (cfg or {}).items() if k in DEFAULT_CONFIG})
    for k in ('keywords', 'hashtags', 'exclude', 'languages'):
        v = c[k]
        if v is None:
            v = []
        if isinstance(v, str):
            v = re.split(r'[,\n]+', v)
        if not isinstance(v, (list, tuple)):
            raise ValueError('Setting "%s" must be comma-separated text or a list (got %r).' % (k, v))
        c[k] = [str(x).strip() for x in v if str(x).strip()]
    for k in ('pos_threshold', 'neg_threshold', 'threshold', 'burst_z'):
        c[k] = _number(c, k, float)
    for k in ('min_size', 'k', 'max_tweets', 'bin_minutes', 'recent_bins', 'min_followers'):
        c[k] = _number(c, k, int)
    c['drop_duplicates'] = bool(c['drop_duplicates'])
    c['emotions'] = bool(c['emotions'])
    c['recent_bins'] = max(1, c['recent_bins'])
    sentiment.check_thresholds(c['pos_threshold'], c['neg_threshold'])
    if not 0.05 <= c['threshold'] <= 0.95:
        raise ValueError('Similarity threshold must be between 0.05 and 0.95')
    if c['min_size'] < 2:
        raise ValueError('Minimum topic size must be at least 2')
    if c['burst_z'] <= 0:
        raise ValueError('Burst z-score threshold must be greater than 0')
    if c['min_followers'] < 0:
        raise ValueError('Minimum followers cannot be negative')
    if c['max_tweets'] < 20:
        raise ValueError('Max tweets analysed must be at least 20')
    if not 0 <= c['k'] <= MAX_K:
        raise ValueError('Clusters k must be between 0 (automatic) and %d' % MAX_K)
    if not 0 <= c['bin_minutes'] <= 7 * 24 * 60:
        raise ValueError('Time bin must be between 0 (automatic) and 10080 minutes')
    for k in ('date_from', 'date_to'):
        if c[k]:
            try:
                datetime.fromisoformat(str(c[k]))
            except ValueError:
                raise ValueError('"%s" must be a date such as 2025-03-05 (got %r).' % (k, c[k]))
    if c['date_from'] and c['date_to'] and str(c['date_from']) > str(c['date_to']):
        raise ValueError('The "from" date (%s) is after the "to" date (%s).' % (c['date_from'], c['date_to']))
    return c


def _day(s, end=False):
    if not s:
        return None
    d = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return (d + timedelta(days=1)).timestamp() if end else d.timestamp()


def apply_filters(rows, c):
    """Returns (filtered [(orig_idx,row)], stage counts)."""
    stats = {'input': len(rows)}
    work = [(i, r) for i, r in enumerate(rows) if r.get('ts')]
    stats['with_timestamp'] = len(work)
    lo, hi = _day(c['date_from']), _day(c['date_to'], end=True)
    if lo:
        work = [(i, r) for i, r in work if r['ts'] >= lo]
    if hi:
        work = [(i, r) for i, r in work if r['ts'] < hi]
    stats['after_date'] = len(work)
    if c['languages']:
        langs = {l.lower() for l in c['languages']}
        work = [(i, r) for i, r in work if (r['lang'] or 'unknown') in langs]
    stats['after_language'] = len(work)
    if c['min_followers']:
        work = [(i, r) for i, r in work if r['followers'] >= c['min_followers']]
    kws = [k.lower() for k in c['keywords']]
    if kws:
        agg = all if c['keyword_mode'] == 'all' else any
        work = [(i, r) for i, r in work if agg(k in r['text'].lower() for k in kws)]
    tags = [('#' + h.lstrip('#')).lower() for h in c['hashtags']]
    if tags:
        work = [(i, r) for i, r in work if any(re.search(re.escape(t) + r'\b', r['text'].lower()) for t in tags)]
    excl = [k.lower() for k in c['exclude']]
    if excl:
        work = [(i, r) for i, r in work if not any(k in r['text'].lower() for k in excl)]
    stats['after_keywords'] = len(work)
    work.sort(key=lambda p: p[1]['ts'])
    if c['drop_duplicates'] and work:
        d = datacentre.find_duplicates([r for _, r in work])
        drop = set(d['exact']) | set(d['near'])
        stats['duplicates_removed'] = len(drop)
        work = [p for k, p in enumerate(work) if k not in drop]
    else:
        stats['duplicates_removed'] = 0
    if len(work) > c['max_tweets']:
        keep = sorted(random.Random(1).sample(range(len(work)), c['max_tweets']))
        stats['sampled_from'] = len(work)
        work = [work[k] for k in keep]
    stats['analysed'] = len(work)
    return work, stats


def choose_bin(span_seconds, minutes=0):
    if minutes:
        return minutes * 60
    for b in BIN_CHOICES:
        if span_seconds / b <= 60:
            return b
    return BIN_CHOICES[-1]


def run_analysis(rows, config, dataset):
    c = clean_config(config)
    timing = {}
    t0 = time.time()
    work, fstats = apply_filters(rows, c)
    timing['filter'] = time.time() - t0
    if len(work) < 20:
        raise ValueError('Only %d tweets remain after filtering (need at least 20). Relax the filters or choose another dataset. Stage counts: %s' % (len(work), fstats))
    tw = [r for _, r in work]
    warnings = []
    if len(tw) > 0 and sum(1 for r in tw if r['lang'] in ('', 'en')) < 0.8 * len(tw) and c['sentiment_algo'] in ('vader', 'lexicon'):
        warnings.append('More than 20% of the tweets are not English; VADER and the opinion lexicon are English-only, so their sentiment is unreliable for those tweets. Filter by language to avoid this.')

    if c['topic_algo'] == 'pivot' and len(tw) > 1500:
        warnings.append('The original document-pivot method is quadratic in the number of tweets (about %d s for %d tweets here). Use K-Means or lower "Max tweets" for faster runs.' % (round(len(tw) ** 2 / 140000), len(tw)))
    t1 = time.time()
    scores = sentiment.score(c['sentiment_algo'], [r['text'] for r in tw])
    sl = [sentiment.to_label(s, c['pos_threshold'], c['neg_threshold']) for s in scores]
    timing['sentiment'] = time.time() - t1

    t2 = time.time()
    labels, tinfo = topics.assign(tw, c['topic_algo'], c)
    timing['topics'] = time.time() - t2
    t3 = time.time()
    desc, relations = topics.describe(tw, labels)

    lo, hi = tw[0]['ts'], tw[-1]['ts']
    bsec = choose_bin(hi - lo, c['bin_minutes'])
    # Bins are anchored to the END of the data, so every recent bin is complete. Anchoring at the start left a sliver of a bin
    # at the end, which made "recent" growth, z-score and emerging flags depend on how the time span happened to divide.
    nb = max(1, math.ceil((hi - lo) / bsec - 1e-9))
    if nb > MAX_BINS:
        raise ValueError('A time bin of %g minutes would create %d bins for this period (limit %d). Choose a larger bin or "auto".' % (bsec / 60, nb, MAX_BINS))
    t0 = hi - nb * bsec
    binof = [min(nb - 1, int((r['ts'] - t0) // bsec)) for r in tw]
    volume = [0] * nb
    sent_series = [{'positive': 0, 'negative': 0, 'neutral': 0} for _ in range(nb)]
    for b, l in zip(binof, sl):
        volume[b] += 1
        sent_series[b][l] += 1

    out_topics = []
    for tid, d in sorted(desc.items()):
        mem = d['members']
        counts = [0] * nb
        for i in mem:
            counts[binof[i]] += 1
        sc = Counter(sl[i] for i in mem)
        m = trends.analyse(counts, len(tw), c['recent_bins'], c['burst_z'])
        size = len(mem)
        top_terms = [w for w, _ in d['terms'][:3]]
        item = {
            'id': tid + 1, 'label': ' · '.join(top_terms) or 'topic %d' % (tid + 1), 'size': size,
            'share': round(size / len(tw), 4), 'terms': d['terms'], 'hashtags': d['hashtags'], 'related': d['related'],
            'sentiment': {'positive': sc['positive'], 'negative': sc['negative'], 'neutral': sc['neutral'],
                          'avg': round(sum(scores[i] for i in mem) / size, 4)},
            'engagement': {'avg_retweets': round(sum(tw[i]['retweets'] for i in mem) / size, 2),
                           'avg_likes': round(sum(tw[i]['likes'] for i in mem) / size, 2),
                           'avg_followers': round(sum(tw[i]['followers'] for i in mem) / size, 1),
                           'reach': sum(tw[i]['followers'] for i in mem)},
            'series': counts, 'first_seen': min(tw[i]['ts'] for i in mem), 'last_seen': max(tw[i]['ts'] for i in mem),
            'samples': [{'text': tw[i]['text'], 'label': sl[i], 'score': round(scores[i], 3), 'ts': tw[i]['ts'],
                         'retweets': tw[i]['retweets'], 'likes': tw[i]['likes']} for i in d['rep']],
            'evolution': topics.window_terms(tw, mem),
            'sentiment_series': [round(sum(scores[i] for i in mem if binof[i] == b) / max(1, counts[b]), 3) for b in range(nb)],
        }
        item.update(m)
        item['emerging'] = bool(m['stage'] in ('new', 'rising') and (m['z'] >= c['burst_z'] or m['growth'] >= 1.0 or m['trend_growth'] >= 0.5) and size >= c['min_size'])
        if c['emotions']:
            item['emotions'] = sentiment.emotions([tw[i]['text'] for i in mem])
        out_topics.append(item)
    out_topics.sort(key=lambda t: -t['score'])
    for rank, t in enumerate(out_topics, 1):
        t['rank'] = rank
    timing['trends'] = time.time() - t3

    overall = Counter(sl)
    assigned = sum(1 for l in labels if l >= 0)
    kpis = {
        'tweets': len(tw), 'topics': len(out_topics), 'assigned': assigned, 'coverage': round(assigned / len(tw), 4),
        'emerging': sum(1 for t in out_topics if t['emerging']),
        'bursting': sum(1 for t in out_topics if t['z'] >= c['burst_z']),
        'positive': overall['positive'], 'negative': overall['negative'], 'neutral': overall['neutral'],
        'avg_sentiment': round(sum(scores) / len(scores), 4),
        'unique_users': len({r['user'] for r in tw if r['user']}),
        'total_reach': sum(r['followers'] for r in tw),
        'avg_retweets': round(sum(r['retweets'] for r in tw) / len(tw), 2),
        'avg_likes': round(sum(r['likes'] for r in tw) / len(tw), 2),
    }
    result = {
        'meta': {'config': c, 'dataset': {'id': dataset['id'], 'name': dataset['name'], 'source': dataset['source']},
                 'filters': fstats, 'topic_info': tinfo, 'warnings': warnings,
                 'timing': {k: round(v, 3) for k, v in timing.items()},
                 'date_from': lo, 'date_to': hi, 'created': time.time()},
        'bins': {'t0': t0, 'seconds': bsec, 'n': nb},
        'kpis': kpis, 'volume': volume, 'sentiment_series': sent_series, 'topics': out_topics, 'relations': [
            dict(r, a=r['a'] + 1, b=r['b'] + 1) for r in relations if r['sim'] > 0.02],
        'emotions': sentiment.emotions([r['text'] for r in tw]) if c['emotions'] else None,
        'assignments': [[work[k][0], labels[k] + 1, sl[k][:3], round(scores[k], 3)] for k in range(len(work))],
    }
    summary = {k: kpis[k] for k in ('tweets', 'topics', 'emerging', 'bursting', 'avg_sentiment')}
    summary.update({'positive_share': round(kpis['positive'] / len(tw), 3), 'negative_share': round(kpis['negative'] / len(tw), 3),
                    'topic_algo': c['topic_algo'], 'sentiment_algo': c['sentiment_algo'], 'runtime_s': round(sum(timing.values()), 2)})
    return result, summary


def compare_runs(a, b):
    """Compare two stored runs: KPI deltas and keyword-based topic matching."""
    ka, kb = a['result']['kpis'], b['result']['kpis']
    kd = {k: {'a': ka[k], 'b': kb[k], 'delta': round(kb[k] - ka[k], 4)} for k in ka if isinstance(ka[k], (int, float)) and k in kb}
    ta, tb = a['result']['topics'], b['result']['topics']

    def kw(t):
        return {w for w, _ in t['terms'][:8]}
    pairs = []
    for x in ta:
        for y in tb:
            u = kw(x) | kw(y)
            j = len(kw(x) & kw(y)) / len(u) if u else 0
            if j >= 0.25:
                pairs.append((j, x, y))
    pairs.sort(key=lambda p: -p[0])
    used_a, used_b, matched = set(), set(), []
    for j, x, y in pairs:
        if x['id'] in used_a or y['id'] in used_b:
            continue
        used_a.add(x['id']), used_b.add(y['id'])
        matched.append({'a': x['label'], 'b': y['label'], 'similarity': round(j, 3), 'size_a': x['size'], 'size_b': y['size'],
                        'stage_a': x['stage'], 'stage_b': y['stage'], 'sentiment_a': x['sentiment']['avg'],
                        'sentiment_b': y['sentiment']['avg'], 'score_a': x['score'], 'score_b': y['score']})
    return {'kpis': kd, 'matched': matched,
            'only_a': [{'label': t['label'], 'size': t['size'], 'stage': t['stage']} for t in ta if t['id'] not in used_a],
            'only_b': [{'label': t['label'], 'size': t['size'], 'stage': t['stage']} for t in tb if t['id'] not in used_b],
            'a': {'id': a['id'], 'name': a['name'], 'dataset': a['dataset_name']}, 'b': {'id': b['id'], 'name': b['name'], 'dataset': b['dataset_name']}}
