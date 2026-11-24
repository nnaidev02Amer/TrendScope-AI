"""Data Centre: parse uploads (CSV / JSON / JSONL), validate, map fields, detect duplicates,
and generate configurable synthetic data with ground-truth labels.

Normalised row schema used everywhere else:
    id, text, ts (epoch seconds, may be None), user, followers, retweets, likes, lang,
    label_sentiment ('positive'|'negative'|'neutral'|None), label_topic (str|None)
"""
import csv
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

MAX_ROWS = 20000
FIELDS = ['text', 'timestamp', 'id', 'user', 'followers', 'retweets', 'likes', 'lang', 'label_sentiment', 'label_topic']
CANDIDATES = {
    'text': ['text', 'full_text', 'content', 'tweet', 'message', 'body', 'post', 'status'],
    'timestamp': ['created_at', 'timestamp_ms', 'timestamp', 'date', 'datetime', 'time', 'created', 'posted_at'],
    'id': ['id', 'id_str', 'tweet_id', 'post_id'],
    'user': ['user.screen_name', 'username', 'screen_name', 'user', 'author', 'author_id', 'user.name'],
    'followers': ['user.followers_count', 'followers_count', 'followers', 'public_metrics.followers_count'],
    'retweets': ['retweet_count', 'retweets', 'repost_count', 'public_metrics.retweet_count', 'public_metrics.repost_count', 'shares'],
    'likes': ['favorite_count', 'like_count', 'likes', 'public_metrics.like_count', 'favorites'],
    'lang': ['lang', 'language'],
    'label_sentiment': ['label_sentiment', 'sentiment_label', 'sentiment', 'label'],
    'label_topic': ['label_topic', 'topic_label', 'topic'],
}
LABELS = {'positive': 'positive', 'pos': 'positive', '+': 'positive', '+1': 'positive', '1': 'positive', 'p': 'positive',
          'negative': 'negative', 'neg': 'negative', '-': 'negative', '-1': 'negative', 'n': 'negative',
          'neutral': 'neutral', 'neu': 'neutral', '0': 'neutral'}


# ------------------------------------------------------------------ parsing
def flatten(obj, prefix='', depth=0):
    out = {}
    for k, v in obj.items():
        key = prefix + str(k)
        if isinstance(v, dict) and depth < 3:
            out.update(flatten(v, key + '.', depth + 1))
        elif isinstance(v, (list, dict)):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def detect_format(filename, content):
    name = (filename or '').lower()
    if name.endswith('.csv') or name.endswith('.tsv'):
        return 'csv'
    if name.endswith('.jsonl') or name.endswith('.ndjson'):
        return 'jsonl'
    if name.endswith('.json'):
        # a .json file may still hold one JSON object per line (output of main.py -r)
        lines = [l for l in content.splitlines() if l.strip()]
        if len(lines) > 1 and lines[0].lstrip().startswith('{') and lines[0].rstrip().endswith('}'):
            try:
                json.loads(lines[0]), json.loads(lines[1])
                return 'jsonl'
            except ValueError:
                pass
        return 'json'
    head = content.lstrip()[:1]
    if head == '[':
        return 'json'
    if head == '{':
        return 'jsonl'
    return 'csv'


def parse_content(filename, content):
    """Return (records, fmt, parse_errors). Records are flat dicts."""
    if content.startswith('﻿'):
        content = content[1:]
    fmt = detect_format(filename, content)
    errors = 0
    records = []
    if fmt == 'csv':
        sample = content[:4096]
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
        except csv.Error:
            delim = ','
        for row in csv.DictReader(io.StringIO(content), delimiter=delim):
            records.append({(k or '').strip(): v for k, v in row.items()})
    elif fmt == 'jsonl':
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                records.extend(flatten(o) for o in (obj if isinstance(obj, list) else [obj]) if isinstance(o, dict))
            except ValueError:
                errors += 1
    else:
        try:
            obj = json.loads(content)
        except ValueError as e:
            raise ValueError('Invalid JSON: %s' % e)
        includes = {}
        if isinstance(obj, dict):
            if isinstance(obj.get('includes'), dict):
                includes = {u.get('id'): u for u in obj['includes'].get('users', []) if isinstance(u, dict)}
            for key in ('data', 'tweets', 'statuses', 'posts', 'items', 'results'):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
            else:
                obj = [obj]
        for o in obj:
            if not isinstance(o, dict):
                errors += 1
                continue
            rec = flatten(o)
            u = includes.get(o.get('author_id'))
            if u:  # X API v2 payload: attach author metrics
                rec.update(flatten(u, 'user.'))
                rec['public_metrics.followers_count'] = (u.get('public_metrics') or {}).get('followers_count')
            records.append(rec)
    if len(records) > MAX_ROWS:
        records = records[:MAX_ROWS]
    return records, fmt, errors


def suggest_mapping(columns):
    lower = {c.lower(): c for c in columns}
    mapping = {}
    for field in FIELDS:
        mapping[field] = next((lower[c] for c in CANDIDATES[field] if c in lower), None)
    return mapping


_TS_FORMATS = ['%a %b %d %H:%M:%S %z %Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M',
               '%m/%d/%Y %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y']


def parse_ts(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r'\s*-?\d+(\.\d+)?\s*', v)):
        x = float(v)
        if x > 1e11:
            x /= 1000.0
        return x if x > 0 else None
    s = str(v).strip()
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        d = None
        for f in _TS_FORMATS:
            try:
                d = datetime.strptime(s, f)
                break
            except ValueError:
                continue
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def to_int(v):
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError):
        return 0


def norm_label(v):
    if v is None:
        return None
    return LABELS.get(str(v).strip().lower())


def normalize_records(records, mapping):
    """Apply a field mapping. Returns (rows, stats)."""
    if not mapping.get('text'):
        raise ValueError('A text column must be mapped.')
    rows, missing_text, bad_ts = [], 0, 0
    for i, rec in enumerate(records):
        text = rec.get(mapping['text'])
        text = '' if text is None else str(text).strip()
        if not text:
            missing_text += 1
            continue
        g = lambda f: rec.get(mapping[f]) if mapping.get(f) else None
        ts = parse_ts(g('timestamp')) if mapping.get('timestamp') else None
        if mapping.get('timestamp') and ts is None:
            bad_ts += 1
        rid = g('id')
        rows.append({
            'id': str(rid) if rid not in (None, '') else 'row%d' % i,
            'text': text, 'ts': ts, 'user': str(g('user') or ''),
            'followers': to_int(g('followers')), 'retweets': to_int(g('retweets')), 'likes': to_int(g('likes')),
            'lang': (str(g('lang')).strip().lower() if g('lang') else ''),
            'label_sentiment': norm_label(g('label_sentiment')),
            'label_topic': (str(g('label_topic')).strip() if g('label_topic') not in (None, '') else None),
        })
    return rows, {'input_records': len(records), 'missing_text': missing_text, 'bad_timestamp': bad_ts,
                  'no_timestamp_column': not mapping.get('timestamp')}


# ------------------------------------------------------------------ duplicates
_URL = re.compile(r'https?://\S+')
_MENTION = re.compile(r'@\w+')
_WORD = re.compile(r"[a-z0-9']+")


def norm_text(text):
    t = _URL.sub(' ', text.lower())
    t = re.sub(r'^\s*rt\s+@\w+:?', ' ', t)
    t = _MENTION.sub(' ', t)
    return ' '.join(_WORD.findall(t))


def find_duplicates(rows, near=True, threshold=0.85, near_limit=4000):
    """Return {'exact': {dup_idx: keep_idx}, 'near': {dup_idx: keep_idx}}. First occurrence is kept."""
    seen, exact = {}, {}
    keys = []
    for i, r in enumerate(rows):
        k = norm_text(r['text'])
        keys.append(k)
        if k in seen:
            exact[i] = seen[k]
        else:
            seen[k] = i
    near_map = {}
    if near:
        uniq = [i for i in range(min(len(rows), near_limit)) if i not in exact]
        toks = {i: set(keys[i].split()) for i in uniq}
        post = defaultdict(list)
        for i in uniq:
            for w in toks[i]:
                post[w].append(i)
        for i in uniq:
            if i in near_map or len(toks[i]) < 4:
                continue
            cand = Counter()
            for w in toks[i]:
                lst = post[w]
                if len(lst) <= 60:
                    for j in lst:
                        if j > i:
                            cand[j] += 1
            for j, c in cand.items():
                if j in near_map or c < 0.6 * len(toks[i]):
                    continue
                inter = len(toks[i] & toks[j])
                union = len(toks[i] | toks[j])
                if union and inter / union >= threshold:
                    near_map[j] = i
    return {'exact': exact, 'near': near_map}


# ------------------------------------------------------------------ validation / preview
def build_preview(filename, content, mapping=None):
    records, fmt, errors = parse_content(filename, content)
    if not records:
        raise ValueError('No records found in file (format detected: %s).' % fmt)
    columns = []
    for r in records[:200]:
        for k in r:
            if k not in columns:
                columns.append(k)
    suggested = suggest_mapping(columns)
    mapping = mapping or suggested
    report = validate(records, mapping) if mapping.get('text') else {'error': 'No text column detected: choose one below.'}
    return {'format': fmt, 'parse_errors': errors, 'columns': columns, 'suggested_mapping': suggested, 'mapping': mapping,
            'record_count': len(records), 'sample': [{c: r.get(c) for c in columns} for r in records[:8]], 'validation': report}


def validate(records, mapping):
    rows, stats = normalize_records(records, mapping)
    dups = find_duplicates(rows)
    ts = [r['ts'] for r in rows if r['ts']]
    langs = Counter(r['lang'] or 'unknown' for r in rows)
    labelled = sum(1 for r in rows if r['label_sentiment'])
    warnings = []
    if stats['no_timestamp_column'] or not ts:
        warnings.append('No usable timestamps: trend, burst and lifecycle analysis need a timestamp column.')
    elif stats['bad_timestamp']:
        warnings.append('%d rows have unparseable timestamps and will be excluded from time-based analysis.' % stats['bad_timestamp'])
    if stats['missing_text']:
        warnings.append('%d rows have no text and will be skipped.' % stats['missing_text'])
    return {
        'valid_rows': len(rows), 'skipped_missing_text': stats['missing_text'], 'bad_timestamps': stats['bad_timestamp'],
        'exact_duplicates': len(dups['exact']), 'near_duplicates': len(dups['near']),
        'languages': dict(langs.most_common(8)), 'labelled_sentiment': labelled,
        'labelled_topic': sum(1 for r in rows if r['label_topic']),
        'date_from': min(ts) if ts else None, 'date_to': max(ts) if ts else None, 'warnings': warnings,
        'ok': bool(rows) and bool(ts),
    }


def import_rows(filename, content, mapping, drop_duplicates=False):
    records, fmt, _ = parse_content(filename, content)
    rows, stats = normalize_records(records, mapping)
    dups = find_duplicates(rows)
    removed = 0
    if drop_duplicates:
        drop = set(dups['exact']) | set(dups['near'])
        rows = [r for i, r in enumerate(rows) if i not in drop]
        removed = len(drop)
    meta = {'filename': filename, 'format': fmt, 'mapping': mapping, 'duplicates_removed': removed,
            'exact_duplicates': len(dups['exact']), 'near_duplicates': len(dups['near']), **stats}
    return rows, meta


# ------------------------------------------------------------------ synthetic data
LEAD = ['{a} {b} {c}', 'Latest on the {a}: {b} and {c}', '{a} news, {b} and {c} again', 'Thinking about {a}, {b}, {c}',
        'More {a} updates, {b}, {c}', 'Everyone discussing {a}, {b} and {c}']
POS = ['really impressed', 'love it', 'great news', 'so happy about this', 'fantastic progress', 'brilliant outcome', 'wonderful']
NEG = ['really disappointed', 'terrible', 'awful handling', 'so angry about this', 'dreadful mess', 'pathetic outcome', 'horrible']
NEU = ['update expected Thursday', 'report published this afternoon', 'details to follow', 'meeting scheduled', 'what we know so far']
HARD_POS = ['not bad at all', "can't complain really", 'better than expected']
HARD_NEG = ['not exactly thrilled', 'could be a lot better', 'supposed to be great but a letdown']
HARD_NEU = ['some say great, some say terrible', 'whatever happens, we will see']
FRAMES = {'en': {'positive': POS, 'negative': NEG, 'neutral': NEU}}
FRAMES['es'] = {'positive': ['me encanta', 'excelente noticia'], 'negative': ['muy decepcionado', 'un desastre'],
                'neutral': ['reunión el jueves', 'informe publicado']}
SUFFIX = ['', '', '', ' #news', ' #today', ' @newsdesk', ' via @localpaper', ' (thread)']

DEFAULT_TOPICS = [
    {'name': 'Transport strike', 'keywords': ['strike', 'railway', 'commuters', 'timetable', 'rail union', 'platform'], 'hashtags': ['#strike'], 'weight': 1.2,
     'shape': 'burst', 'at': 0.55, 'width': 0.10, 'magnitude': 8, 'sentiment': [0.1, 0.7, 0.2]},
    {'name': 'Climate summit', 'keywords': ['climate', 'summit', 'emissions', 'delegates', 'carbon', 'treaty'], 'hashtags': ['#cop'], 'weight': 1.0,
     'shape': 'emerging', 'at': 0.5, 'width': 0.2, 'magnitude': 1, 'sentiment': [0.35, 0.25, 0.4]},
    {'name': 'Football derby', 'keywords': ['derby', 'football', 'stadium', 'goalkeeper', 'penalty', 'referee'], 'hashtags': ['#derbyday'], 'weight': 1.0,
     'shape': 'burst', 'at': 0.2, 'width': 0.08, 'magnitude': 10, 'sentiment': [0.5, 0.3, 0.2]},
    {'name': 'Phone launch', 'keywords': ['phone', 'launch', 'camera', 'battery', 'processor', 'preorder'], 'hashtags': ['#techtalk'], 'weight': 1.1,
     'shape': 'steady', 'at': 0.5, 'width': 0.2, 'magnitude': 1, 'sentiment': [0.5, 0.2, 0.3]},
    {'name': 'University open day', 'keywords': ['university', 'campus', 'students', 'lecture', 'admissions', 'tutor'], 'hashtags': ['#openday'], 'weight': 0.8,
     'shape': 'declining', 'at': 0.5, 'width': 0.2, 'magnitude': 1, 'sentiment': [0.55, 0.1, 0.35]},
    {'name': 'Heatwave warning', 'keywords': ['heatwave', 'temperatures', 'forecast', 'drought', 'sunscreen', 'hosepipe'], 'hashtags': ['#heatwave'], 'weight': 0.9,
     'shape': 'emerging', 'at': 0.5, 'width': 0.2, 'magnitude': 1, 'sentiment': [0.05, 0.6, 0.35]},
]
DEFAULT_GEN = {'n': 2400, 'days': 7, 'start': '2025-03-03', 'seed': 42, 'noise': 0.04, 'retweet_fraction': 0.06,
               'difficulty': 0.2, 'languages': {'en': 0.93, 'es': 0.07}, 'topics': DEFAULT_TOPICS}


def _intensity(t, topic):
    shape = topic.get('shape', 'steady')
    if shape == 'emerging':
        return 0.03 + t ** 3
    if shape == 'declining':
        return 0.03 + (1 - t) ** 3
    if shape == 'burst':
        mag = float(topic.get('magnitude', 6))
        at, w = float(topic.get('at', 0.5)), max(float(topic.get('width', 0.1)), 0.01)
        return 1.0 + mag * math.exp(-0.5 * ((t - at) / w) ** 2)
    return 1.0


def _max_intensity(topic):
    return {'burst': 1.0 + float(topic.get('magnitude', 6)), 'emerging': 1.03, 'declining': 1.03}.get(topic.get('shape', 'steady'), 1.0)


def generate(cfg=None):
    """Generate labelled synthetic rows. Deterministic for a given config (seed)."""
    c = dict(DEFAULT_GEN)
    c.update(cfg or {})
    topics = c.get('topics') or DEFAULT_TOPICS
    n = max(50, min(int(c['n']), MAX_ROWS))
    rng = random.Random(int(c.get('seed', 42)))
    try:
        start = datetime.fromisoformat(str(c['start'])).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        raise ValueError('start must be an ISO date such as 2025-03-03')
    span = max(0.25, float(c['days'])) * 86400
    noise = min(0.5, max(0.0, float(c.get('noise', 0.04))))
    difficulty = min(1.0, max(0.0, float(c.get('difficulty', 0.2))))
    rt_frac = min(0.5, max(0.0, float(c.get('retweet_fraction', 0.06))))
    langs = c.get('languages') or {'en': 1.0}
    lang_names = [l for l in langs if l in FRAMES] or ['en']
    lang_w = [float(langs.get(l, 0)) for l in lang_names]
    total_w = sum(float(t.get('weight', 1)) for t in topics) or 1.0
    rows = []
    for ti, topic in enumerate(topics):
        count = int(round(n * (1 - noise) * float(topic.get('weight', 1)) / total_w))
        kws = [k.strip() for k in topic.get('keywords', []) if k.strip()] or [topic.get('name', 'topic').lower()]
        tags = [h if h.startswith('#') else '#' + h for h in topic.get('hashtags', []) if h.strip()]
        probs = topic.get('sentiment') or [0.34, 0.33, 0.33]
        s = sum(probs) or 1.0
        probs = [p / s for p in probs]
        mx = _max_intensity(topic)
        for _ in range(count):
            while True:  # rejection-sample a time following the topic's intensity curve
                t = rng.random()
                if rng.random() * mx <= _intensity(t, topic):
                    break
            lang = rng.choices(lang_names, lang_w)[0]
            label = rng.choices(['positive', 'negative', 'neutral'], probs)[0]
            picks = rng.sample(kws, min(3, len(kws)))
            while len(picks) < 3:
                picks.append(rng.choice(kws))
            frames = FRAMES[lang][label]
            if lang == 'en' and rng.random() < difficulty:
                frames = {'positive': HARD_POS, 'negative': HARD_NEG, 'neutral': HARD_NEU}[label]
            lead = rng.choice(LEAD).format(a=picks[0], b=picks[1], c=picks[2])
            tail = rng.choice(frames)
            text = (lead + ', ' + tail) if rng.random() < 0.7 else (tail.capitalize() + ': ' + lead)
            text += rng.choice(SUFFIX)
            if tags and rng.random() < 0.5:
                text += ' ' + rng.choice(tags)
            boost = 1 + 0.5 * (label != 'neutral')
            rows.append({'ts': start + t * span, 'text': text, 'lang': lang, 'label_sentiment': label,
                         'label_topic': topic.get('name', 'topic%d' % ti),
                         'followers': int(rng.lognormvariate(6.5, 1.6)),
                         'retweets': int(rng.expovariate(1 / (3.0 * boost))), 'likes': int(rng.expovariate(1 / (9.0 * boost)))})
    chatter = ['Just had a coffee', 'Anyone watching the match tonight?', 'Monday again', 'New playlist is on repeat',
               'Looking for a good book recommendation', 'Traffic was bad this morning']
    for _ in range(int(n * noise)):
        rows.append({'ts': start + rng.random() * span, 'text': rng.choice(chatter) + rng.choice(SUFFIX), 'lang': 'en',
                     'label_sentiment': 'neutral', 'label_topic': 'noise', 'followers': int(rng.lognormvariate(6, 1.5)),
                     'retweets': int(rng.expovariate(1 / 2.0)), 'likes': int(rng.expovariate(1 / 5.0))})
    rows.sort(key=lambda r: r['ts'])
    for i in range(int(len(rows) * rt_frac)):  # retweets: re-post an earlier tweet a little later
        src = rows[rng.randrange(len(rows))]
        rows.append(dict(src, text='RT @newsdesk: ' + src['text'], ts=min(start + span, src['ts'] + rng.random() * 3600)))
    rows.sort(key=lambda r: r['ts'])
    for i, r in enumerate(rows):
        r['id'] = 'syn%d' % (i + 1)
        r['user'] = 'user%d' % rng.randrange(1, 600)
    return rows[:MAX_ROWS]
