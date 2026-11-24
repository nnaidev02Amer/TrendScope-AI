"""Stage 4 verification: sentiment and NLP modules, optional-model fallbacks, and the evaluation metrics.

    .venv\\Scripts\\python -W ignore -m unittest tests.test_stage4_nlp -v

Principles
- Controlled datasets only; expected values are computed independently inside the tests.
- Nothing is downloaded and no heavy package is needed. The optional packages (transformers, sentence-transformers, bertopic) and the
  SentiStrength Java process are replaced by small FAKES that are injected only for the duration of a test. The fakes verify OUR glue
  code (availability checks, score normalisation, caching, error handling, clustering maths); they say nothing about the quality of the real models.
- When the real optional packages are missing (as on the development machine), the tests also verify the honest fallback behaviour.
"""
import contextlib
import hashlib
import importlib.machinery
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from trendscope import datacentre as dc, engine, evaluation, sentiment, server, topics  # noqa: E402

DS = {'id': 1, 'name': 'stage4', 'source': 'synthetic'}
CLASSES = ['positive', 'neutral', 'negative']


def topic(name, words, sentiment_mix=(.3, .3, .4), shape='steady', **kw):
    d = {'name': name, 'keywords': words, 'hashtags': [], 'weight': 1, 'shape': shape, 'sentiment': list(sentiment_mix)}
    d.update(kw)
    return d


TEA = ['tea', 'kettle', 'biscuit', 'mug', 'brew', 'teapot']
CARS = ['engine', 'garage', 'tyres', 'petrol', 'motor', 'gearbox']
RAIL = ['train', 'station', 'ticket', 'carriage', 'driver', 'signal']
SPACE = ['rocket', 'orbit', 'launch', 'planet', 'astronaut', 'comet']
FOOD = ['pasta', 'recipe', 'oven', 'garlic', 'sauce', 'basil']
FIVE = [topic('tea', TEA), topic('cars', CARS), topic('rail', RAIL), topic('space', SPACE), topic('food', FOOD)]


def gen(topics_, n=600, days=3, seed=1, start='2025-03-03', difficulty=0.0, noise=0.0):
    return dc.generate({'n': n, 'days': days, 'seed': seed, 'noise': noise, 'retweet_fraction': 0.0, 'difficulty': difficulty,
                        'languages': {'en': 1.0}, 'start': start, 'topics': topics_})


def indep_metrics(true, pred):
    """Accuracy / per-class precision, recall, F1 / macro-F1 / confusion, by plain counting (independent of evaluation.py)."""
    n = len(true)
    conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for t, p in zip(true, pred):
        conf[t][p] += 1
    out = {'accuracy': sum(conf[c][c] for c in CLASSES) / n, 'per': {}, 'confusion': [[conf[a][b] for b in CLASSES] for a in CLASSES]}
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in CLASSES if o != c)
        fn = sum(conf[c][o] for o in CLASSES if o != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out['per'][c] = {'precision': pr, 'recall': rc, 'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    out['macro_f1'] = sum(v['f1'] for v in out['per'].values()) / 3
    return out


# ------------------------------------------------------------------ fakes for the optional dependencies
def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class FakeEncoder:
    """Deterministic word-hash 'sentence embeddings' that weight content words above filler. Not a language model."""
    loads = 0
    content = set()
    fail = None

    def __init__(self, name):
        if FakeEncoder.fail:
            raise FakeEncoder.fail
        FakeEncoder.loads += 1
        self.name = name

    @staticmethod
    def _w(w):
        return np.random.default_rng(int(hashlib.md5(w.encode()).hexdigest(), 16) % (2 ** 32)).standard_normal(64)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            ws = [w.strip('.,:#@!?') for w in t.lower().split()]
            v = sum((self._w(w) * (1.0 if w in FakeEncoder.content else 0.04) for w in ws), np.zeros(64))
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)


class FakeBERTopic:
    """Assigns a label by the first topic keyword found; -1 (outlier) otherwise. Records its constructor arguments."""
    created = []
    fail = None
    keymap = {}

    def __init__(self, min_topic_size=10, verbose=False):
        if FakeBERTopic.fail:
            raise FakeBERTopic.fail
        FakeBERTopic.created.append(min_topic_size)

    def fit_transform(self, texts):
        labels = []
        for t in texts:
            lab = next((FakeBERTopic.keymap[w] for w in t.lower().split() if w in FakeBERTopic.keymap), -1)
            labels.append(lab)
        return labels, None


class FakeTransformers:
    constructed = 0
    fail = None
    labels = ('positive', 'neutral', 'negative')

    @staticmethod
    def pipeline(task, model=None, top_k=None, truncation=True):
        if FakeTransformers.fail:
            raise FakeTransformers.fail
        FakeTransformers.constructed += 1
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        def run(texts, batch_size=16):
            out = []
            for t in texts:
                c = sia.polarity_scores(t)['compound']
                pos, neg = max(c, 0.0) * 0.9, max(-c, 0.0) * 0.9
                lab = FakeTransformers.labels
                out.append([{'label': lab[0].upper(), 'score': pos}, {'label': lab[1].upper(), 'score': 1 - pos - neg}, {'label': lab[2].upper(), 'score': neg}])
            return out
        return run


@contextlib.contextmanager
def fake_optional_packages(transformers=False, sentence_transformers=False, bertopic=False, content=()):
    names = {}
    if transformers:
        names['transformers'] = _module('transformers', pipeline=FakeTransformers.pipeline)
        names['torch'] = _module('torch')
    if sentence_transformers:
        names['sentence_transformers'] = _module('sentence_transformers', SentenceTransformer=FakeEncoder)
    if bertopic:
        names['bertopic'] = _module('bertopic', BERTopic=FakeBERTopic)
    old = {k: sys.modules.get(k) for k in names}
    sys.modules.update(names)
    FakeEncoder.loads, FakeEncoder.content, FakeEncoder.fail = 0, set(content), None
    FakeTransformers.constructed, FakeTransformers.fail = 0, None
    FakeBERTopic.created, FakeBERTopic.fail, FakeBERTopic.keymap = [], None, {}
    sentiment._tf_pipe = None
    topics._st_models.clear()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sentiment._tf_pipe = None
        topics._st_models.clear()


REAL = {'transformer': sentiment.availability()['transformer']['ok'], 'embeddings': topics.availability()['embeddings']['ok'],
        'bertopic': topics.availability()['bertopic']['ok'], 'sentistrength': sentiment.availability()['sentistrength']['ok']}


# ------------------------------------------------------------------ 1. VADER vs opinion lexicon
class TestVaderVersusLexicon(unittest.TestCase):
    def test_vader_is_the_default_everywhere(self):
        self.assertEqual(engine.DEFAULT_CONFIG['sentiment_algo'], 'vader')
        self.assertEqual(engine.clean_config({})['sentiment_algo'], 'vader')
        res, summary = engine.run_analysis(gen(FIVE[:2], n=300), {'topic_algo': 'kmeans', 'k': 2, 'min_size': 10, 'drop_duplicates': False}, DS)
        self.assertEqual(summary['sentiment_algo'], 'vader')
        self.assertEqual(res['meta']['config']['sentiment_algo'], 'vader')

    def test_they_agree_on_plain_polarity(self):
        texts = ['I love this wonderful great day', 'terrible awful horrible service', 'the meeting is on Thursday at noon']
        for algo in ('vader', 'lexicon'):
            labels = [sentiment.to_label(s) for s in sentiment.score(algo, texts)]
            self.assertEqual(labels, ['positive', 'negative', 'neutral'], algo)

    def test_vader_handles_negation_and_intensity_the_lexicon_does_not(self):
        v = dict(zip(['good', 'not good', 'not bad', 'very good!!!', 'never happy'], sentiment.score('vader', ['good', 'not good', 'not bad', 'very good!!!', 'never happy'])))
        lx = dict(zip(v, sentiment.score('lexicon', list(v))))
        self.assertLess(v['not good'], 0)                    # negation flips the polarity
        self.assertGreater(v['not bad'], 0)
        self.assertLess(v['never happy'], 0)
        self.assertGreater(v['very good!!!'], v['good'])      # intensity raises the score
        self.assertGreater(lx['not good'], 0)                 # the lexicon counts the word 'good' and ignores 'not'
        self.assertLess(lx['not bad'], 0)
        self.assertEqual(lx['good'], lx['very good!!!'])      # no intensity: any single polar word is +-1

    def test_scores_are_normalised_to_minus_one_and_one(self):
        odd = ['', '   ', '!!!', '😀😀', 'me encanta este día', 'x' * 5000, 'good ' * 400, 'bad ' * 400, '12345', 'GREAT GREAT GREAT!!!!!!!', '#love #hate @user http://t.co/x']
        for algo in ('vader', 'lexicon'):
            s = sentiment.score(algo, odd)
            self.assertEqual(len(s), len(odd))
            self.assertTrue(all(isinstance(x, float) or isinstance(x, int) for x in s))
            self.assertTrue(all(-1.0 <= x <= 1.0 and not math.isnan(x) for x in s), (algo, s))
        self.assertEqual(sentiment.score('vader', ['', '!!!', 'me encanta este día']), [0.0, 0.0, 0.0])   # no sentiment words found -> exactly 0
        self.assertEqual(sentiment.score('vader', []), [])
        with self.assertRaises(ValueError):
            sentiment.score('nonsense', ['x'])

    def test_score_magnitude_grows_with_evidence(self):
        """There is no separate confidence field; |score| is the only strength signal, so check that it behaves like one."""
        one, two, four = sentiment.score('vader', ['good', 'good good', 'good good good good'])
        self.assertTrue(0 < one < two < four <= 1)

    def test_measured_comparison_on_labelled_data(self):
        """Real metrics on labelled synthetic tweets. Easy data: both methods work. Hard data (negation, mixed): VADER is clearly ahead."""
        easy = evaluation.evaluate_sentiment(gen(FIVE, n=900, seed=2, difficulty=0.0), ['vader', 'lexicon'])
        hard = evaluation.evaluate_sentiment(gen(FIVE, n=900, seed=2, difficulty=0.8), ['vader', 'lexicon'])
        e = {r['algo']: r for r in easy['results']}
        h = {r['algo']: r for r in hard['results']}
        self.assertGreater(e['vader']['accuracy'], 0.85)
        self.assertGreater(e['lexicon']['accuracy'], 0.85)
        self.assertLess(h['lexicon']['macro_f1'], e['lexicon']['macro_f1'] - 0.15)       # the lexicon collapses on hard cases
        self.assertGreater(h['vader']['macro_f1'], h['lexicon']['macro_f1'] + 0.05)      # VADER degrades less

    def test_run_comparison_between_algorithms_reports_exact_deltas(self):
        rows = gen(FIVE[:3], n=500, seed=4, difficulty=0.5)
        base = {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10, 'drop_duplicates': False}
        a = engine.run_analysis(rows, dict(base, sentiment_algo='vader'), DS)[0]
        b = engine.run_analysis(rows, dict(base, sentiment_algo='lexicon'), DS)[0]
        cmp = engine.compare_runs({'id': 1, 'name': 'v', 'dataset_name': 'x', 'result': a}, {'id': 2, 'name': 'l', 'dataset_name': 'x', 'result': b})
        for k in ('positive', 'negative', 'neutral'):
            self.assertEqual(cmp['kpis'][k]['delta'], b['kpis'][k] - a['kpis'][k])
        self.assertNotEqual((a['kpis']['positive'], a['kpis']['negative']), (b['kpis']['positive'], b['kpis']['negative']))
        self.assertEqual(a['kpis']['tweets'], b['kpis']['tweets'])


# ------------------------------------------------------------------ 2. thresholds
class TestThresholds(unittest.TestCase):
    def test_boundaries_are_inclusive_and_asymmetric_thresholds_work(self):
        self.assertEqual(sentiment.to_label(0.05), 'positive')
        self.assertEqual(sentiment.to_label(0.0499), 'neutral')
        self.assertEqual(sentiment.to_label(-0.05), 'negative')
        self.assertEqual(sentiment.to_label(-0.0499), 'neutral')
        self.assertEqual([sentiment.to_label(x, 0.2, 0.5) for x in (0.2, 0.19, -0.2, -0.49, -0.5)], ['positive', 'neutral', 'neutral', 'neutral', 'negative'])

    def test_a_score_of_exactly_zero_is_neutral_even_with_threshold_zero(self):
        """Regression: with thresholds of 0 (allowed by the Studio) an exact 0 used to become 'positive' and 'neutral' was impossible."""
        for z in (0.0, -0.0, 0):
            self.assertEqual(sentiment.to_label(z, 0.0, 0.0), 'neutral')
        self.assertEqual(sentiment.to_label(0.001, 0.0, 0.0), 'positive')
        self.assertEqual(sentiment.to_label(-0.001, 0.0, 0.0), 'negative')
        rows = gen(FIVE[:2], n=400, seed=3, difficulty=0.5)
        res, _ = engine.run_analysis(rows, {'topic_algo': 'kmeans', 'k': 2, 'min_size': 10, 'drop_duplicates': False, 'pos_threshold': 0, 'neg_threshold': 0}, DS)
        exact_zero = sum(1 for s in sentiment.score('vader', [r['text'] for r in rows]) if s == 0)
        self.assertGreater(exact_zero, 0)
        self.assertEqual(res['kpis']['neutral'], exact_zero)           # exactly the tweets with no sentiment found

    def test_raising_the_thresholds_never_reduces_the_neutral_share(self):
        scores = sentiment.score('vader', [r['text'] for r in gen(FIVE, n=600, seed=6, difficulty=0.3)])
        neutral = [sum(sentiment.to_label(s, t, t) == 'neutral' for s in scores) for t in (0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0)]
        self.assertEqual(neutral, sorted(neutral))
        self.assertGreater(neutral[-1], neutral[0])

    def test_invalid_thresholds_are_rejected_everywhere(self):
        for bad in ({'pos_threshold': -0.1}, {'neg_threshold': 1.5}, {'pos_threshold': 2}):
            with self.assertRaises(ValueError):
                engine.clean_config(bad)
        rows = gen(FIVE[:2], n=300)
        with self.assertRaises(ValueError):
            evaluation.evaluate_sentiment(rows, ['vader'], pos_t=-1, neg_t=0.05)
        engine.clean_config({'pos_threshold': 0, 'neg_threshold': 1})        # the closed range is accepted


# ------------------------------------------------------------------ 3. sentiment by topic and over time
class TestSentimentByTopicAndTime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # sentiment mixes are (positive, negative, neutral). 'cars' is 80% negative throughout; 'tea' is 80% negative for the first
        # 3 days and 80% positive for the last 3 days (same vocabulary), so overall sentiment flips from clearly negative to roughly neutral
        first = gen([topic('tea', TEA, (.1, .8, .1)), topic('cars', CARS, (.1, .8, .1))], n=700, seed=1, start='2025-03-03')
        second = gen([topic('tea', TEA, (.8, .1, .1)), topic('cars', CARS, (.1, .8, .1))], n=700, seed=2, start='2025-03-06')
        cls.rows = sorted(first + second, key=lambda r: r['ts'])
        cls.res, _ = engine.run_analysis(cls.rows, {'topic_algo': 'kmeans', 'k': 2, 'min_size': 20, 'bin_minutes': 360, 'drop_duplicates': False, 'max_tweets': 5000}, DS)
        cls.scores = {i: s for i, s in enumerate(sentiment.score('vader', [r['text'] for r in cls.rows]))}

    def test_every_assignment_label_matches_its_score_and_the_thresholds(self):
        for idx, tid, lab3, score in self.res['assignments']:
            self.assertEqual(lab3, sentiment.to_label(self.scores[idx])[:3])
            self.assertAlmostEqual(score, self.scores[idx], places=3)

    def test_topic_sentiment_counts_and_average_match_an_independent_recount(self):
        by = {}
        for idx, tid, lab3, score in self.res['assignments']:
            by.setdefault(tid, []).append(idx)
        for t in self.res['topics']:
            mem = by[t['id']]
            c = Counter(sentiment.to_label(self.scores[i]) for i in mem)
            self.assertEqual((t['sentiment']['positive'], t['sentiment']['negative'], t['sentiment']['neutral']), (c['positive'], c['negative'], c['neutral']))
            self.assertAlmostEqual(t['sentiment']['avg'], sum(self.scores[i] for i in mem) / len(mem), places=3)
            self.assertEqual(t['size'], len(mem))

    def test_series_over_time_match_an_independent_recount(self):
        b = self.res['bins']
        want = [Counter() for _ in range(b['n'])]
        for idx, tid, lab3, score in self.res['assignments']:
            k = min(b['n'] - 1, int((self.rows[idx]['ts'] - b['t0']) // b['seconds']))
            want[k][sentiment.to_label(self.scores[idx])] += 1
        got = self.res['sentiment_series']
        self.assertEqual(len(got), b['n'])
        for k in range(b['n']):
            self.assertEqual((got[k]['positive'], got[k]['negative'], got[k]['neutral']), (want[k]['positive'], want[k]['negative'], want[k]['neutral']), 'bin %d' % k)
        self.assertEqual(sum(sum(x.values()) for x in got), self.res['kpis']['tweets'])
        k = self.res['kpis']
        self.assertEqual((k['positive'], k['negative'], k['neutral']), tuple(sum(x[c] for x in got) for c in ('positive', 'negative', 'neutral')))

    def test_designed_sentiment_pattern_is_recovered(self):
        by_label = {}
        for idx, tid, lab3, score in self.res['assignments']:
            by_label.setdefault(tid, Counter())[self.rows[idx]['label_topic']] += 1
        names = {tid: c.most_common(1)[0][0] for tid, c in by_label.items()}
        tea = next(t for t in self.res['topics'] if names[t['id']] == 'tea')
        cars = next(t for t in self.res['topics'] if names[t['id']] == 'cars')
        self.assertLess(cars['sentiment']['avg'], -0.05)                       # cars: 80% negative in both halves
        self.assertGreater(cars['sentiment']['negative'], 4 * cars['sentiment']['positive'])
        mid = self.res['bins']['n'] // 2
        s = self.res['sentiment_series']
        net = lambda part: sum(x['positive'] - x['negative'] for x in part)
        self.assertLess(net(s[:mid - 1]), 0)                                   # first 3 days: negative-leaning overall (tea negative + cars negative)
        self.assertGreater(net(s[mid + 1:]), net(s[:mid - 1]) + 100)           # tea flips to positive in the second half
        tea_series = tea['sentiment_series']
        self.assertLess(sum(tea_series[:mid - 1]) / (mid - 1), 0)
        self.assertGreater(sum(tea_series[mid + 1:]) / (len(tea_series) - mid - 1), 0)

    def test_emotions_per_topic_match_a_recount(self):
        res, _ = engine.run_analysis(self.rows, {'topic_algo': 'kmeans', 'k': 2, 'min_size': 20, 'drop_duplicates': False, 'emotions': True}, DS)
        by = {}
        for idx, tid, _l, _s in res['assignments']:
            by.setdefault(tid, []).append(self.rows[idx]['text'])
        for t in res['topics']:
            self.assertEqual(t['emotions'], sentiment.emotions(by[t['id']]))
        analysed = [self.rows[idx]['text'] for idx, _t, _l, _s in res['assignments']]
        self.assertEqual(res['emotions'], sentiment.emotions(analysed))                  # the overall figure covers every analysed tweet
        off, _ = engine.run_analysis(self.rows, {'topic_algo': 'kmeans', 'k': 2, 'min_size': 20, 'drop_duplicates': False, 'emotions': False}, DS)
        self.assertIsNone(off['emotions'])
        self.assertTrue(all('emotions' not in t for t in off['topics']))


# ------------------------------------------------------------------ 4. SentiStrength (Java is not installed: glue code tested with a fake process)
class TestSentiStrength(unittest.TestCase):
    def setUp(self):
        import sentiStrength
        self.ss = sentiStrength
        self.calls = []

    def fake_run(self, mode='ok'):
        def run(cmd, input=None, stdout=None, encoding=None, **kw):
            self.calls.append((cmd, input))
            lines = []
            for text in (input or '').split('\n'):
                scale = 2 if 'good' in text else (-3 if 'bad' in text else (4 if 'best' in text else 0))
                lines.append('%d\t%d\t%d\t%s' % (max(scale, 0) + 1, min(scale, 0) - 1, scale, text))
            if mode == 'short':
                lines = lines[:-1]
            return types.SimpleNamespace(stdout='\n'.join(lines) + '\n', returncode=0)
        return run

    @contextlib.contextmanager
    def java_present(self, mode='ok'):
        old_which, old_run = sentiment.shutil.which, self.ss.subprocess.run
        sentiment.shutil.which = lambda name: 'C:/fake/java.exe' if name == 'java' else old_which(name)
        self.ss.subprocess.run = self.fake_run(mode)
        try:
            yield
        finally:
            sentiment.shutil.which, self.ss.subprocess.run = old_which, old_run

    def test_scores_are_scaled_to_minus_one_and_one_and_the_command_is_correct(self):
        with self.java_present():
            self.assertTrue(sentiment.availability()['sentistrength']['ok'])
            s = sentiment.score('sentistrength', ['so good today', 'bad bad bad', 'this is the best', 'nothing here'])
        self.assertEqual(s, [0.5, -0.75, 1.0, 0.0])
        cmd, sent = self.calls[0]
        self.assertEqual(cmd[:3], ['java', '-jar', './SentiStrength/SentiStrengthCom.jar'])
        self.assertIn('scale', cmd)
        self.assertIn('sentidata', cmd)
        self.assertEqual(sent.split('\n')[0], 'so+good+today')          # spaces become '+', one text per line

    def test_a_wrong_number_of_returned_scores_is_an_error_not_a_misalignment(self):
        with self.java_present('short'):
            with self.assertRaises(RuntimeError) as cm:
                sentiment.score('sentistrength', ['good one', 'bad one', 'meh'])
        self.assertIn('returned 2 scores for 3 texts', str(cm.exception))

    def test_it_does_not_alter_the_callers_texts(self):
        texts = ['so good today', 'bad\nbad']
        with self.java_present():
            sentiment.score('sentistrength', texts)
        self.assertEqual(texts, ['so good today', 'bad\nbad'])

    def test_end_to_end_run_and_evaluation_with_the_fake_process(self):
        rows = gen(FIVE[:2], n=300, seed=2)
        with self.java_present():
            res, summary = engine.run_analysis(rows, {'topic_algo': 'kmeans', 'k': 2, 'min_size': 10, 'sentiment_algo': 'sentistrength', 'drop_duplicates': False}, DS)
            ev = evaluation.evaluate_sentiment(rows, ['sentistrength'])
        self.assertEqual(summary['sentiment_algo'], 'sentistrength')
        self.assertEqual(res['kpis']['positive'] + res['kpis']['negative'] + res['kpis']['neutral'], res['kpis']['tweets'])
        self.assertNotIn('error', ev['results'][0])

    @unittest.skipIf(REAL['sentistrength'], 'Java is installed here')
    def test_without_java_it_is_unavailable_and_says_so(self):
        av = sentiment.availability()['sentistrength']
        self.assertFalse(av['ok'])
        self.assertIn('Java', av['note'])
        with self.assertRaises(RuntimeError):
            sentiment.score('sentistrength', ['good'])


# ------------------------------------------------------------------ 5. transformer sentiment
class TestTransformerSentiment(unittest.TestCase):
    @unittest.skipIf(REAL['transformer'], 'transformers/torch are installed here')
    def test_fallback_when_the_packages_are_missing(self):
        av = sentiment.availability()['transformer']
        self.assertFalse(av['ok'])
        self.assertIn('pip install transformers torch', av['note'])
        with self.assertRaises(RuntimeError) as cm:
            sentiment.score('transformer', ['hello'])
        self.assertIn('not installed', str(cm.exception))
        res = evaluation.evaluate_sentiment(gen(FIVE[:2], n=200), ['transformer', 'vader'])
        self.assertIn('error', res['results'][0])
        self.assertNotIn('error', res['results'][1])
        self.assertNotIn('accuracy', res['results'][0])                    # no fabricated numbers for an algorithm that could not run

    def test_partial_install_is_not_reported_as_available(self):
        with fake_optional_packages(transformers=True):
            sys.modules.pop('torch')
            self.assertFalse(sentiment.availability()['transformer']['ok'])

    def test_scores_normalisation_labels_and_model_is_loaded_once(self):
        with fake_optional_packages(transformers=True):
            self.assertTrue(sentiment.availability()['transformer']['ok'])
            s1 = sentiment.score('transformer', ['I love this wonderful thing', 'awful terrible mess', 'the meeting is at noon'])
            s2 = sentiment.score('transformer', ['great news'])
            self.assertEqual(FakeTransformers.constructed, 1)               # regression: the pipeline used to be rebuilt on every call
        self.assertGreater(s1[0], 0.3)
        self.assertLess(s1[1], -0.3)
        self.assertAlmostEqual(s1[2], 0.0, places=6)
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in s1 + s2))

    def test_model_load_failure_is_a_clear_runtime_error(self):
        """Regression: an OSError from the model download (offline) used to escape as a generic internal error."""
        with fake_optional_packages(transformers=True):
            FakeTransformers.fail = OSError('We couldn\'t connect to huggingface.co')
            with self.assertRaises(RuntimeError) as cm:
                sentiment.score('transformer', ['x'])
            self.assertIn('could not be loaded', str(cm.exception))
            self.assertIn('internet', str(cm.exception))
            FakeTransformers.fail = None
            self.assertEqual(len(sentiment.score('transformer', ['fine'])), 1)      # a later call can succeed: the failure was not cached

    def test_unrecognised_model_labels_are_an_error_not_silent_zeros(self):
        with fake_optional_packages(transformers=True):
            FakeTransformers.labels = ('LABEL_2', 'LABEL_1', 'LABEL_0')
            try:
                with self.assertRaises(RuntimeError) as cm:
                    sentiment.score('transformer', ['x'])
            finally:
                FakeTransformers.labels = ('positive', 'neutral', 'negative')
        self.assertIn('unrecognised labels', str(cm.exception))

    def test_engine_run_and_evaluation_with_the_fake_model(self):
        rows = gen(FIVE[:3], n=450, seed=5, difficulty=0.2)
        with fake_optional_packages(transformers=True):
            res, summary = engine.run_analysis(rows, {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10, 'sentiment_algo': 'transformer', 'drop_duplicates': False}, DS)
            ev = evaluation.evaluate_sentiment(rows, ['transformer', 'vader'])
        self.assertEqual(summary['sentiment_algo'], 'transformer')
        r = {x['algo']: x for x in ev['results']}
        self.assertEqual(r['transformer']['n'], len(rows))
        self.assertAlmostEqual(r['transformer']['accuracy'], r['vader']['accuracy'], delta=0.12)   # the fake is VADER-derived: similar, not identical


# ------------------------------------------------------------------ 6. emotions
class TestEmotions(unittest.TestCase):
    def test_each_emotion_is_detected_from_its_own_words(self):
        cases = {'joy': 'I am so happy and delighted', 'anger': 'furious and angry about it', 'sadness': 'so sad and heartbroken today',
                 'fear': 'terrified and anxious about the danger', 'surprise': 'wow what a shocking unexpected result'}
        for emo, text in cases.items():
            got = sentiment.emotions([text])
            self.assertEqual(got[emo], 1, (emo, got))
            self.assertEqual(sum(got.values()), 1, (emo, got))                 # and no other emotion is reported

    def test_counts_are_tweets_not_words_and_text_is_case_and_punctuation_insensitive(self):
        e = sentiment.emotions(['HAPPY, happy, HAPPY!!!', 'Happy.', 'nothing', 'angry? ANGRY!', 'so happy and so angry'])
        self.assertEqual((e['joy'], e['anger']), (3, 2))                # 3 tweets mention joy, 2 mention anger, repeats do not add up
        self.assertEqual(sum(sentiment.emotions([]).values()), 0)

    def test_whole_words_only(self):
        e = sentiment.emotions(['whatever', 'thankful', 'sadly mistaken', 'a madness'])
        self.assertEqual(sum(e.values()), 0)                            # 'hate' in 'whatever', 'mad' in 'madness' must not match

    def test_known_limitation_negation_is_not_understood(self):
        """Documented: the emotion lexicon is a rough indicator. 'not happy' still counts as joy."""
        self.assertEqual(sentiment.emotions(['I am not happy'])['joy'], 1)

    def test_mixed_emotions_in_one_tweet(self):
        e = sentiment.emotions(['thrilled but also worried and scared'])
        self.assertEqual((e['joy'], e['fear']), (1, 1))

    def test_emotions_agree_with_sentiment_direction_on_controlled_data(self):
        rows = gen([topic('tea', TEA, (1, 0, 0)), topic('cars', CARS, (0, 1, 0))], n=600, seed=9)
        pos = sentiment.emotions([r['text'] for r in rows if r['label_sentiment'] == 'positive'])
        neg = sentiment.emotions([r['text'] for r in rows if r['label_sentiment'] == 'negative'])
        self.assertGreater(pos['joy'], 5 * max(pos['anger'] + pos['sadness'], 1))
        self.assertGreater(neg['anger'] + neg['sadness'], 5 * max(neg['joy'], 1))


# ------------------------------------------------------------------ 7-8. sentence transformers, BERTopic, semantic clustering
class TestSemanticClustering(unittest.TestCase):
    CONTENT = set(TEA + CARS + RAIL + SPACE + FOOD)

    @unittest.skipIf(REAL['embeddings'], 'sentence-transformers is installed here')
    def test_fallback_when_sentence_transformers_is_missing(self):
        av = topics.availability()
        self.assertFalse(av['embeddings']['ok'])
        self.assertIn('pip install sentence-transformers', av['embeddings']['note'])
        self.assertFalse(av['bertopic']['ok'])
        rows = gen(FIVE[:2], n=200)
        for algo in ('embeddings', 'bertopic'):
            with self.assertRaises(RuntimeError) as cm:
                topics.assign(rows, algo, {'k': 2, 'min_size': 5})
            self.assertIn('pip install', str(cm.exception))
        res = evaluation.evaluate_topics(gen(FIVE, n=300), ['kmeans', 'embeddings', 'bertopic'], {'k': 5, 'min_size': 8})
        by = {r['algo']: r for r in res['results']}
        self.assertNotIn('error', by['kmeans'])
        self.assertIn('error', by['embeddings'])
        self.assertIn('error', by['bertopic'])
        self.assertNotIn('purity', by['embeddings'])                         # no fabricated numbers

    def test_availability_reports_each_optional_package_separately(self):
        with fake_optional_packages(sentence_transformers=True):
            a = topics.availability()
            self.assertTrue(a['embeddings']['ok'])
            self.assertFalse(a['bertopic']['ok'])                            # bertopic itself is missing
        with fake_optional_packages(sentence_transformers=True, bertopic=True):
            self.assertTrue(topics.availability()['bertopic']['ok'])
        with fake_optional_packages(bertopic=True):
            self.assertFalse(topics.availability()['bertopic']['ok'])        # bertopic without sentence-transformers

    def test_embedding_clustering_recovers_known_topics_on_every_dataset(self):
        """Regression: the old single random start merged topics on about half of these datasets (ARI about 0.7)."""
        with fake_optional_packages(sentence_transformers=True, content=self.CONTENT):
            for seed in range(8):
                rows = gen(FIVE, n=400, seed=seed)
                labels, info = topics.cluster_embeddings(rows, {'k': 5})
                truth = [r['label_topic'] for r in rows]
                self.assertEqual(evaluation.ari(truth, labels), 1.0, 'seed %d' % seed)
                self.assertEqual(info['k'], 5)

    def test_embedding_clustering_is_deterministic_and_loads_the_model_once(self):
        rows = gen(FIVE, n=300, seed=3)
        with fake_optional_packages(sentence_transformers=True, content=self.CONTENT):
            a, _ = topics.cluster_embeddings(rows, {'k': 5})
            b, _ = topics.cluster_embeddings(rows, {'k': 5})
            self.assertEqual(a, b)
            self.assertEqual(FakeEncoder.loads, 1)

    def test_embedding_k_defaults_and_edge_cases(self):
        with fake_optional_packages(sentence_transformers=True, content=self.CONTENT):
            rows = gen(FIVE, n=400, seed=3)
            labels, info = topics.cluster_embeddings(rows, {})
            self.assertEqual(info['k'], 10)                                   # default: min(10, n/40)
            self.assertEqual(len(set(labels)), 10)
            tiny = rows[:3]
            labels, info = topics.cluster_embeddings(tiny, {'k': 8})
            self.assertEqual(info['k'], 3)                                    # never more clusters than tweets
            self.assertEqual(len(labels), 3)

    def test_model_load_failure_is_a_clear_runtime_error(self):
        with fake_optional_packages(sentence_transformers=True):
            FakeEncoder.fail = OSError('offline')
            with self.assertRaises(RuntimeError) as cm:
                topics.assign(gen(FIVE[:2], n=200), 'embeddings', {'k': 2, 'min_size': 5})
            self.assertIn('could not be loaded', str(cm.exception))

    def test_assign_drops_small_clusters_and_renumbers_by_size_for_embeddings(self):
        with fake_optional_packages(sentence_transformers=True, content=self.CONTENT):
            rows = gen([topic('tea', TEA, weight=3), topic('cars', CARS, weight=2.5), topic('rail', RAIL, weight=1)], n=650, seed=2)
            sizes_true = Counter(r['label_topic'] for r in rows)
            labels, info = topics.assign(rows, 'embeddings', {'k': 3, 'min_size': 150})
        sizes = Counter(l for l in labels if l >= 0)
        self.assertEqual(sorted(sizes), [0, 1])                               # ids are renumbered 0..m-1 after the small cluster is dropped
        self.assertEqual([sizes[0], sizes[1]], [sizes_true['tea'], sizes_true['cars']])      # largest first, exactly the two big topics
        self.assertEqual(labels.count(-1), sizes_true['rail'])                # the 100-tweet topic is below min_size: outliers (-1)

    def test_full_engine_run_with_embeddings_and_semantic_topic_description(self):
        rows = gen(FIVE, n=600, seed=4)
        with fake_optional_packages(sentence_transformers=True, content=self.CONTENT):
            res, summary = engine.run_analysis(rows, {'topic_algo': 'embeddings', 'k': 5, 'min_size': 20, 'drop_duplicates': False}, DS)
        self.assertEqual(summary['topic_algo'], 'embeddings')
        self.assertEqual(len(res['topics']), 5)
        wanted = [set(TEA), set(CARS), set(RAIL), set(SPACE), set(FOOD)]
        for t in res['topics']:                                               # each discovered topic's keywords come from exactly one true topic
            kws = {w for w, _ in t['terms'][:4]}
            self.assertEqual(sum(bool(kws & w) for w in wanted), 1, kws)
            self.assertTrue(kws <= next(w for w in wanted if kws & w), kws)
        self.assertEqual(res['meta']['topic_info']['model'], 'all-MiniLM-L6-v2')

    def test_embeddings_versus_tfidf_are_compared_with_real_metrics(self):
        rows = gen(FIVE, n=500, seed=6)
        with fake_optional_packages(sentence_transformers=True, content=self.CONTENT):
            ev = evaluation.evaluate_topics(rows, ['kmeans', 'embeddings'], {'k': 5, 'min_size': 10})
        by = {r['algo']: r for r in ev['results']}
        for a in ('kmeans', 'embeddings'):
            self.assertEqual(by[a]['true_topics'], 5)
            self.assertGreater(by[a]['purity'], 0.95)
            self.assertGreater(by[a]['ari'], 0.9)
            self.assertEqual(by[a]['clusters'], 5)

    def test_bertopic_wrapper_maps_outliers_and_passes_min_size(self):
        rows = gen(FIVE[:3], n=300, seed=2)
        with fake_optional_packages(sentence_transformers=True, bertopic=True):
            FakeBERTopic.keymap = {'tea': 0, 'kettle': 0, 'engine': 1, 'garage': 1}      # trains, others -> outlier -1
            labels, info = topics.assign(rows, 'bertopic', {'min_size': 12})
            self.assertEqual(FakeBERTopic.created, [12])
        self.assertTrue(set(labels) <= {-1, 0, 1})
        self.assertGreater(labels.count(-1), 0)
        for r, l in zip(rows, labels):
            if l == 0:
                self.assertTrue({'tea', 'kettle'} & set(r['text'].lower().split()))
        self.assertEqual(info['n_topics'], 2)

    def test_bertopic_failure_is_a_clear_runtime_error(self):
        with fake_optional_packages(sentence_transformers=True, bertopic=True):
            FakeBERTopic.fail = ValueError('Cannot use UMAP with fewer than 15 samples')
            with self.assertRaises(RuntimeError) as cm:
                topics.assign(gen(FIVE[:2], n=200), 'bertopic', {'min_size': 5})
        self.assertIn('BERTopic could not run', str(cm.exception))
        self.assertIn('UMAP', str(cm.exception))


# ------------------------------------------------------------------ 10. genuine evaluation metrics
class TestEvaluationMetrics(unittest.TestCase):
    def test_metrics_on_a_hand_built_confusion(self):
        true = ['positive'] * 5 + ['neutral'] * 3 + ['negative'] * 2
        pred = ['positive'] * 4 + ['neutral'] + ['neutral'] * 2 + ['negative'] + ['negative', 'positive']
        m = evaluation.sentiment_metrics(true, pred)
        self.assertEqual(m['confusion'], [[4, 1, 0], [0, 2, 1], [1, 0, 1]])
        self.assertAlmostEqual(m['accuracy'], 0.7)
        # positive: tp 4, fp 1, fn 1 -> p=.8 r=.8 f1=.8 ; neutral: tp 2, fp 1, fn 1 -> p=2/3 r=2/3 ; negative: tp 1, fp 1, fn 1 -> p=.5 r=.5
        self.assertAlmostEqual(m['per_class']['positive']['f1'], 0.8)
        self.assertAlmostEqual(m['per_class']['neutral']['precision'], 0.6667, places=4)
        self.assertAlmostEqual(m['per_class']['negative']['recall'], 0.5)
        self.assertAlmostEqual(m['macro_f1'], round((0.8 + 2 / 3 + 0.5) / 3, 4), places=4)
        self.assertEqual([v['support'] for v in m['per_class'].values()], [5, 3, 2])

    def test_evaluate_sentiment_equals_an_independent_recomputation(self):
        rows = gen(FIVE, n=700, seed=8, difficulty=0.4)
        ev = evaluation.evaluate_sentiment(rows, ['vader', 'lexicon'], 0.1, 0.2)
        true = [r['label_sentiment'] for r in rows]
        for r in ev['results']:
            pred = [sentiment.to_label(s, 0.1, 0.2) for s in sentiment.score(r['algo'], [x['text'] for x in rows])]
            ref = indep_metrics(true, pred)
            self.assertEqual(r['confusion'], ref['confusion'])
            self.assertAlmostEqual(r['accuracy'], ref['accuracy'], places=4)
            self.assertAlmostEqual(r['macro_f1'], ref['macro_f1'], places=3)
            for c in CLASSES:
                for k in ('precision', 'recall', 'f1'):
                    self.assertAlmostEqual(r['per_class'][c][k], ref['per'][c][k], places=4, msg=(r['algo'], c, k))
            self.assertGreater(r['seconds'], 0)
            self.assertAlmostEqual(r['ms_per_1000'], r['seconds'] * 1e6 / len(rows), delta=0.0005 * 1e6 / len(rows) + 0.1)   # 'seconds' is rounded to 1 ms
        self.assertEqual(ev['label_distribution'], dict(Counter(true)))

    def test_metrics_respond_to_the_labels_they_are_given(self):
        """Nothing is fabricated: the same predictions score ~0 against inverted labels and about chance against shuffled ones."""
        rows = gen(FIVE, n=600, seed=3)
        flip = {'positive': 'negative', 'negative': 'positive', 'neutral': 'neutral'}
        good = evaluation.evaluate_sentiment(rows, ['vader'])['results'][0]
        bad = evaluation.evaluate_sentiment([dict(r, label_sentiment=flip[r['label_sentiment']]) for r in rows], ['vader'])['results'][0]
        import random
        lab = [r['label_sentiment'] for r in rows]
        random.Random(1).shuffle(lab)
        rnd = evaluation.evaluate_sentiment([dict(r, label_sentiment=l) for r, l in zip(rows, lab)], ['vader'])['results'][0]
        self.assertGreater(good['accuracy'], 0.85)
        self.assertLess(bad['accuracy'], good['accuracy'] - 0.5)
        self.assertLess(rnd['accuracy'], 0.5)

    def test_requirements_for_an_evaluation_are_enforced(self):
        rows = gen(FIVE, n=300)
        with self.assertRaises(ValueError):
            evaluation.evaluate_sentiment([dict(r, label_sentiment=None) for r in rows], ['vader'])
        with self.assertRaises(ValueError):
            evaluation.evaluate_sentiment(rows[:19], ['vader'])
        r = evaluation.evaluate_sentiment(rows, ['vader', 'nonsense'])
        self.assertNotIn('error', r['results'][0])
        self.assertIn('Unknown sentiment algorithm', r['results'][1]['error'])

    def test_ari_and_purity_against_independent_formulas(self):
        from itertools import combinations
        def rand_pairs(a, b):
            same_a = {(i, j) for i, j in combinations(range(len(a)), 2) if a[i] == a[j]}
            same_b = {(i, j) for i, j in combinations(range(len(b)), 2) if b[i] == b[j]}
            n2 = len(a) * (len(a) - 1) / 2
            ri_sum = len(same_a & same_b)
            exp = len(same_a) * len(same_b) / n2
            return (ri_sum - exp) / ((len(same_a) + len(same_b)) / 2 - exp)
        cases = [([0, 0, 1, 1], [0, 0, 1, 2]), ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]), (list('aabbccaabbcc'), [0, 0, 1, 1, 1, 2, 0, 0, 1, 2, 2, 2])]
        for t, p in cases:
            self.assertAlmostEqual(evaluation.ari(t, p), rand_pairs(t, p), places=9)
        self.assertEqual(evaluation.ari([0, 0, 1, 1], [5, 5, 7, 7]), 1.0)               # same partition, different ids
        self.assertEqual(evaluation.purity([0, 0, 1, 1], [0, 0, 0, 0]), 0.5)
        self.assertEqual(evaluation.purity([0, 0, 1, 1], [3, 3, 4, 4]), 1.0)

    def test_topic_evaluation_never_sees_the_labels(self):
        rows = gen(FIVE, n=400, seed=5)
        scrambled = [dict(r, label_topic=('x%d' % (i % 5))) for i, r in enumerate(rows)]
        a = evaluation.evaluate_topics(rows, ['kmeans'], {'k': 5, 'min_size': 10})['results'][0]
        b = evaluation.evaluate_topics(scrambled, ['kmeans'], {'k': 5, 'min_size': 10})['results'][0]
        self.assertEqual(a['clusters'], b['clusters'])
        self.assertGreater(a['purity'], 0.95)
        self.assertLess(b['purity'], 0.5)                                    # same clustering, scored against meaningless labels
        self.assertLess(b['ari'], 0.1)


# ------------------------------------------------------------------ 9. fallback through the HTTP API
class TestFallbackViaApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.httpd, cls.db = server.make_server(os.path.join(cls.tmp.name, 's4.db'), '127.0.0.1', 0)
        cls.base = 'http://127.0.0.1:%d' % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.ds = cls.call_(cls, '/api/datasets/generate', 'POST', {'name': 's4', 'config': {'n': 500, 'seed': 4, 'topics': [topic(n, w) for n, w in (('tea', TEA), ('cars', CARS), ('rail', RAIL))], 'noise': 0.0, 'difficulty': 0.2}})['id']

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.db.close()
        cls.tmp.cleanup()

    def call_(self, path, method='GET', body=None, expect=200):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                assert r.status == expect, (r.status, expect)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            data = e.read()
            assert e.code == expect, (e.code, expect, data)
            return json.loads(data or b'{}')

    def test_status_lists_every_algorithm_with_a_reason_when_unavailable(self):
        st = self.call_('/api/status')
        self.assertEqual(set(st['sentiment']), {'vader', 'lexicon', 'sentistrength', 'transformer'})
        self.assertEqual(set(st['topics']), {'pivot', 'kmeans', 'embeddings', 'bertopic'})
        self.assertTrue(st['sentiment']['vader']['ok'] and st['sentiment']['lexicon']['ok'] and st['topics']['kmeans']['ok'] and st['topics']['pivot']['ok'])
        for group in ('sentiment', 'topics'):
            for name, v in st[group].items():
                self.assertTrue(v['note'], name)                         # always explains itself
        meta = self.call_('/api/meta')
        self.assertEqual(meta['default_config']['sentiment_algo'], 'vader')

    def test_unavailable_algorithms_give_a_400_with_the_reason_never_a_500(self):
        base = {'dataset_id': self.ds, 'config': {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}}
        before = len(self.call_('/api/runs'))
        for cfg, probe in (({'sentiment_algo': 'transformer'}, REAL['transformer']), ({'sentiment_algo': 'sentistrength'}, REAL['sentistrength']),
                           ({'topic_algo': 'embeddings'}, REAL['embeddings']), ({'topic_algo': 'bertopic'}, REAL['bertopic'])):
            if probe:
                continue
            r = self.call_('/api/runs', 'POST', {'dataset_id': self.ds, 'config': dict(base['config'], **cfg)}, expect=400)
            self.assertTrue(r['error'], cfg)
            self.assertNotIn('Internal error', r['error'])
        r = self.call_('/api/runs', 'POST', {'dataset_id': self.ds, 'config': {'sentiment_algo': 'made-up'}}, expect=400)
        self.assertIn('Unknown sentiment algorithm', r['error'])
        self.assertEqual(len(self.call_('/api/runs')), before)             # failed runs leave nothing behind

    def test_evaluation_reports_unavailable_algorithms_as_errors_and_still_scores_the_rest(self):
        r = self.call_('/api/eval', 'POST', {'dataset_id': self.ds, 'kind': 'sentiment', 'algos': ['vader', 'lexicon', 'sentistrength', 'transformer']})
        by = {x['algo']: x for x in r['results']}
        self.assertNotIn('error', by['vader'])
        self.assertNotIn('error', by['lexicon'])
        for a in ('sentistrength', 'transformer'):
            if not REAL[a]:
                self.assertIn('error', by[a])
                self.assertNotIn('accuracy', by[a])
        r = self.call_('/api/eval', 'POST', {'dataset_id': self.ds, 'kind': 'topics', 'algos': ['kmeans', 'embeddings', 'bertopic'], 'k': 3, 'min_size': 10})
        by = {x['algo']: x for x in r['results']}
        self.assertNotIn('error', by['kmeans'])
        for a in ('embeddings', 'bertopic'):
            if not REAL[a]:
                self.assertIn('error', by[a])

    def test_invalid_thresholds_are_a_400_for_runs_and_evaluations(self):
        self.call_('/api/runs', 'POST', {'dataset_id': self.ds, 'config': {'pos_threshold': -0.5}}, expect=400)
        self.call_('/api/eval', 'POST', {'dataset_id': self.ds, 'kind': 'sentiment', 'algos': ['vader'], 'pos_threshold': 3}, expect=400)

    def test_optional_models_work_through_the_api_when_present_and_fail_cleanly_when_loading_fails(self):
        before = len(self.call_('/api/runs'))
        with fake_optional_packages(transformers=True, sentence_transformers=True, content=set(TEA + CARS + RAIL)):
            st = self.call_('/api/status')
            self.assertTrue(st['sentiment']['transformer']['ok'] and st['topics']['embeddings']['ok'])
            ok = self.call_('/api/runs', 'POST', {'dataset_id': self.ds, 'config': {'sentiment_algo': 'transformer', 'topic_algo': 'embeddings', 'k': 3, 'min_size': 10, 'name': 'optional'}})
            self.assertEqual(ok['summary']['sentiment_algo'], 'transformer')
            self.assertEqual(ok['summary']['topic_algo'], 'embeddings')
            FakeTransformers.fail = OSError('no internet')
            sentiment._tf_pipe = None
            r = self.call_('/api/runs', 'POST', {'dataset_id': self.ds, 'config': {'sentiment_algo': 'transformer', 'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}}, expect=400)
            self.assertIn('could not be loaded', r['error'])
        self.assertEqual(len(self.call_('/api/runs')), before + 1)         # only the successful run was stored


if __name__ == '__main__':
    unittest.main()
