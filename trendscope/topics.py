"""Topic discovery.

  pivot       The original document-pivot method (topicDetection.useFile): TF-IDF + cosine similarity, streaming.
  kmeans      TF-IDF + spherical K-Means (pure Python, k fixed or chosen by silhouette).
  embeddings  Optional: Sentence-Transformers embeddings + K-Means (needs sentence-transformers).
  bertopic    Optional: BERTopic (needs bertopic + sentence-transformers).

Every algorithm returns one cluster id per row (-1 = not in a topic). Keywords, representative posts,
relations and evolution are computed here for all algorithms in the same way, so results are comparable.
"""
import contextlib
import importlib.util
import io
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKEN = re.compile(r"#?[a-z][a-z0-9']+")
_URL = re.compile(r'https?://\S+|@\w+')
_STOP = None


def stopset():
    global _STOP
    if _STOP is None:
        import stopwords
        _STOP = set(stopwords.stopwords) | set(stopwords.moreStopwords) | {'rt', 'amp', 'via', 'today', 'thread', 'just', 'really'}
    return _STOP


def tokenize(text):
    stop = stopset()
    return [t for t in _TOKEN.findall(_URL.sub(' ', text.lower())) if t not in stop and len(t) > 2]


def availability():
    st = importlib.util.find_spec('sentence_transformers') is not None
    bt = importlib.util.find_spec('bertopic') is not None
    return {
        'pivot': {'ok': True, 'note': 'Original document-pivot method (TF-IDF + cosine). Quadratic: slow above ~1,500 tweets'},
        'kmeans': {'ok': True, 'note': 'TF-IDF + K-Means, pure Python'},
        'embeddings': {'ok': st, 'note': 'Sentence-Transformers + K-Means' if st else 'Optional: pip install sentence-transformers'},
        'bertopic': {'ok': st and bt, 'note': 'BERTopic' if st and bt else 'Optional: pip install bertopic sentence-transformers'},
    }


# ------------------------------------------------------------------ TF-IDF vectors
def tfidf_vectors(token_lists, min_df=2):
    n = len(token_lists)
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    vocab = {w for w, c in df.items() if c >= min_df}
    idf = {w: math.log((1 + n) / (1 + df[w])) + 1 for w in vocab}
    vecs = []
    for toks in token_lists:
        tf = Counter(w for w in toks if w in vocab)
        v = {w: (1 + math.log(c)) * idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def _centroid(vecs):
    c = defaultdict(float)
    for v in vecs:
        for w, x in v.items():
            c[w] += x
    norm = math.sqrt(sum(x * x for x in c.values())) or 1.0
    return {w: x / norm for w, x in c.items()}


# ------------------------------------------------------------------ k-means
def _kmeans_once(vecs, k, seed, iters):
    n = len(vecs)
    rng = random.Random(seed)
    cents = [vecs[rng.randrange(n)]]
    while len(cents) < k:  # k-means++ seeding on cosine distance
        d = [min(1 - dot(v, c) for c in cents) ** 2 for v in vecs]
        tot = sum(d)
        if tot <= 1e-12:
            break
        r, acc = rng.random() * tot, 0.0
        for i, x in enumerate(d):
            acc += x
            if acc >= r:
                cents.append(vecs[i])
                break
    labels = [0] * n
    for it in range(iters):
        new = [max(range(len(cents)), key=lambda j: dot(v, cents[j])) for v in vecs]
        if new == labels and it > 0:
            break
        labels = new
        groups = defaultdict(list)
        for i, l in enumerate(labels):
            groups[l].append(vecs[i])
        cents = [_centroid(groups[j]) if groups.get(j) else cents[j] for j in range(len(cents))]
    score = sum(dot(v, cents[l]) for v, l in zip(vecs, labels))  # total cosine similarity to own centroid (higher = tighter)
    return labels, cents, score


def kmeans(vecs, k, seed=7, iters=20, restarts=4):
    """Spherical k-means with k-means++ seeding. Runs `restarts` times and keeps the tightest solution,
    because a single run often merges two real topics (a local optimum)."""
    n = len(vecs)
    k = max(1, min(k, n))
    best = max((_kmeans_once(vecs, k, seed + 31 * r, iters) for r in range(max(1, restarts))), key=lambda x: x[2])
    return best[0], best[1]


def silhouette(vecs, labels, sample=250, seed=3):
    idx = list(range(len(vecs)))
    random.Random(seed).shuffle(idx)
    idx = idx[:sample]
    by = defaultdict(list)
    for i in idx:
        by[labels[i]].append(i)
    if len(by) < 2:
        return -1.0
    total = 0.0
    for i in idx:
        own = [j for j in by[labels[i]] if j != i]
        a = sum(1 - dot(vecs[i], vecs[j]) for j in own) / len(own) if own else 0.0
        b = min(sum(1 - dot(vecs[i], vecs[j]) for j in g) / len(g) for l, g in by.items() if l != labels[i])
        total += (b - a) / max(a, b, 1e-9)
    return total / len(idx)


def cluster_kmeans(rows, p, texts=None):
    toks = [tokenize(r['text']) for r in rows]
    vecs = tfidf_vectors(toks)
    k = int(p.get('k') or 0)
    info = {}
    if k <= 0:
        best, best_s = 2, -2
        scores = {}
        for cand in range(2, max(3, min(12, len(rows) // 15)) + 1):
            lab, _ = kmeans(vecs, cand, restarts=2)
            s = silhouette(vecs, lab)
            scores[cand] = round(s, 3)
            if s > best_s:
                best, best_s = cand, s
        k = best
        info['silhouette_by_k'] = scores
    labels, _ = kmeans(vecs, k)
    info['k'] = k
    return labels, info


_st_models = {}


def _get_encoder(name):
    """Load a sentence-transformers model once; a failed load becomes a clear RuntimeError (no automatic retries or fallbacks)."""
    if name not in _st_models:
        try:
            from sentence_transformers import SentenceTransformer
            _st_models[name] = SentenceTransformer(name)
        except Exception as e:
            raise RuntimeError('The embedding model %s could not be loaded (it is downloaded once on first use, so it needs internet or a local copy): %s: %s'
                               % (name, type(e).__name__, e))
    return _st_models[name]


def _spherical_kmeans_np(emb, k, seed, iters=25):
    """Spherical k-means on unit vectors with k-means++ seeding. Returns (labels, total cosine similarity to own centroid)."""
    import numpy as np
    n = len(emb)
    rng = np.random.default_rng(seed)
    cents = [emb[int(rng.integers(n))]]
    while len(cents) < k:
        d = (1 - np.max(emb @ np.array(cents).T, axis=1)) ** 2
        tot = float(d.sum())
        if tot <= 1e-12:
            break
        cents.append(emb[int(rng.choice(n, p=d / tot))])
    cents = np.array(cents, dtype=float)
    lab = None
    for _ in range(iters):
        new = np.argmax(emb @ cents.T, axis=1)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        for j in range(len(cents)):
            m = emb[lab == j]
            if len(m):
                c = m.mean(axis=0)
                cents[j] = c / (np.linalg.norm(c) or 1)
    sims = emb @ cents.T
    return np.argmax(sims, axis=1), float(np.max(sims, axis=1).sum())


def cluster_embeddings(rows, p, restarts=4):
    import numpy as np
    name = p.get('embedding_model') or 'all-MiniLM-L6-v2'
    emb = np.asarray(_get_encoder(name).encode([r['text'] for r in rows], normalize_embeddings=True, show_progress_bar=False), dtype=float)
    k = max(1, min(int(p.get('k') or 0) or max(2, min(10, len(rows) // 40)), len(rows)))
    # a single random start often merges two real topics (a local optimum): keep the tightest of several k-means++ starts
    lab, _ = max((_spherical_kmeans_np(emb, k, 7 + 31 * r) for r in range(max(1, restarts))), key=lambda x: x[1])
    return [int(x) for x in lab], {'k': k, 'model': name}


def cluster_bertopic(rows, p):
    try:
        from bertopic import BERTopic
        model = BERTopic(min_topic_size=int(p.get('min_size') or 10), verbose=False)
        labels, _ = model.fit_transform([r['text'] for r in rows])
    except Exception as e:
        raise RuntimeError('BERTopic could not run (it also downloads a sentence-transformers model on first use): %s: %s' % (type(e).__name__, e))
    return [int(x) for x in labels], {'n_topics': len(set(labels) - {-1})}


def cluster_pivot(rows, p):
    """Reuse topicDetection.useFile (the original pipeline) on a temporary tweets file."""
    import json
    import topicDetection
    order = sorted(range(len(rows)), key=lambda i: rows[i]['ts'])
    fd, path = tempfile.mkstemp(suffix='.json', prefix='trendscope_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for i in order:
                r = rows[i]
                f.write(json.dumps({
                    'id': i, 'text': r['text'], 'timestamp_ms': str(int(r['ts'] * 1000)), 'entities': {'urls': []},
                    'user': {'followers_count': r['followers']}, 'retweet_count': r['retweets'], 'favorite_count': r['likes']}) + '\n')
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                topics, _non, _count, _sent, _stats = topicDetection.useFile(
                    path, float(p.get('threshold', 0.3)), int(p.get('min_size', 5)), topicDetection.VADER)
        finally:
            os.chdir(cwd)
    finally:
        os.unlink(path)
    labels = [-1] * len(rows)
    for tid, t in enumerate(topics):
        for tw in t['tweets']:
            labels[tw['id']] = tid
    return labels, {'threshold': float(p.get('threshold', 0.3))}


def assign(rows, algo, p):
    """-> (labels, info). Small clusters (< min_size) are moved to -1."""
    if len(rows) < 10:
        raise ValueError('Need at least 10 tweets after filtering (have %d).' % len(rows))
    if algo == 'pivot':
        labels, info = cluster_pivot(rows, p)
    elif algo == 'kmeans':
        labels, info = cluster_kmeans(rows, p)
    elif algo == 'embeddings':
        if not availability()['embeddings']['ok']:
            raise RuntimeError(availability()['embeddings']['note'])
        labels, info = cluster_embeddings(rows, p)
    elif algo == 'bertopic':
        if not availability()['bertopic']['ok']:
            raise RuntimeError(availability()['bertopic']['note'])
        labels, info = cluster_bertopic(rows, p)
    else:
        raise ValueError('Unknown topic algorithm: %s' % algo)
    sizes = Counter(l for l in labels if l >= 0)
    keep = sorted([l for l, c in sizes.items() if c >= int(p.get('min_size', 5))], key=lambda l: -sizes[l])
    remap = {l: i for i, l in enumerate(keep)}
    return [remap.get(l, -1) for l in labels], info


# ------------------------------------------------------------------ topic descriptions
def describe(rows, labels, n_terms=10):
    """Keywords (class-based TF-IDF), hashtags, related keywords, representative posts, relations."""
    topics = sorted({l for l in labels if l >= 0})
    toks = {i: tokenize(rows[i]['text']) for i in range(len(rows)) if labels[i] >= 0}
    members = {t: [i for i in toks if labels[i] == t] for t in topics}
    tf = {t: Counter(w for i in members[t] for w in toks[i]) for t in topics}
    glob = Counter()
    for t in topics:
        glob.update(tf[t])
    avg_len = sum(sum(c.values()) for c in tf.values()) / max(1, len(topics))
    scores = {}
    for t in topics:
        tot = sum(tf[t].values()) or 1
        scores[t] = {w: (c / tot) * math.log(1 + avg_len / glob[w]) for w, c in tf[t].items()}
    vecs = tfidf_vectors([toks[i] for i in sorted(toks)], min_df=1)
    vec_of = {i: vecs[k] for k, i in enumerate(sorted(toks))}
    out = {}
    for t in topics:
        top = sorted(scores[t].items(), key=lambda kv: -kv[1])
        words = [(w, round(s, 4)) for w, s in top if not w.startswith('#')][:n_terms]
        tags = [w for w, _ in top if w.startswith('#')][:5]
        cent = _centroid([vec_of[i] for i in members[t]])
        rep = sorted(members[t], key=lambda i: -dot(vec_of[i], cent))
        seen, reps = set(), []
        for i in rep:  # skip near-identical posts so the examples are varied
            key = ' '.join(toks[i])
            if key not in seen:
                seen.add(key)
                reps.append(i)
            if len(reps) == 5:
                break
        out[t] = {'terms': words, 'hashtags': tags, 'rep': reps, 'members': members[t],
                  'related': related_terms(words[0][0] if words else None, [toks[i] for i in members[t]])}
    relations = []
    for a in topics:
        for b in topics:
            if a < b:
                va, vb = scores[a], scores[b]
                num = sum(x * vb.get(w, 0) for w, x in va.items())
                den = math.sqrt(sum(x * x for x in va.values())) * math.sqrt(sum(x * x for x in vb.values())) or 1
                shared = sorted(set(w for w, _ in out[a]['terms']) & set(w for w, _ in out[b]['terms']))
                relations.append({'a': a, 'b': b, 'sim': round(num / den, 4), 'shared': shared})
    return out, relations


def related_terms(anchor, token_lists, n=8):
    """Terms that co-occur with the anchor keyword (cosine of their document sets)."""
    if not anchor:
        return []
    docs = defaultdict(set)
    for i, toks in enumerate(token_lists):
        for w in set(toks):
            docs[w].add(i)
    base = docs[anchor]
    res = []
    for w, d in docs.items():
        if w == anchor:
            continue
        inter = len(base & d)
        if inter >= 2:
            res.append((w, round(inter / math.sqrt(len(base) * len(d)), 3)))
    return sorted(res, key=lambda kv: -kv[1])[:n]


def window_terms(rows, member_idx, n_windows=4, n_terms=5):
    """How a topic's vocabulary changes over time: top terms per equal-width time window."""
    ts = [rows[i]['ts'] for i in member_idx]
    if not ts:
        return []
    lo, hi = min(ts), max(ts)
    width = (hi - lo) / n_windows or 1
    wins = [Counter() for _ in range(n_windows)]
    counts = [0] * n_windows
    for i in member_idx:
        w = min(n_windows - 1, int((rows[i]['ts'] - lo) / width))
        wins[w].update(tokenize(rows[i]['text']))
        counts[w] += 1
    return [{'from': lo + k * width, 'to': lo + (k + 1) * width, 'tweets': counts[k],
             'terms': [w for w, _ in wins[k].most_common(n_terms)]} for k in range(n_windows)]
