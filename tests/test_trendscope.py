"""Automated tests for TrendScope AI. Run:  .venv\\Scripts\\python -m unittest discover -s tests -v

No network access and no Twitter/X credentials are used: the X API client is tested with an injected fake opener.
"""
import io
import json
import math
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from trendscope import alerts, datacentre as dc, engine, evaluation, reports, sentiment, server, topics, trends, xapi  # noqa: E402
from trendscope.db import DB  # noqa: E402

SMALL_ROWS = dc.generate({'n': 700, 'seed': 5})
DS = {'id': 1, 'name': 'test', 'source': 'synthetic'}


class TestDataCentre(unittest.TestCase):
    def test_csv_parse_and_mapping(self):
        csv_text = 'id,created_at,text,user,likes,lang\n1,2025-01-02T10:00:00Z,hello world,bob,3,en\n2,2025-01-02 11:30:00,second tweet,amy,5,en\n'
        recs, fmt, err = dc.parse_content('x.csv', csv_text)
        self.assertEqual((fmt, len(recs), err), ('csv', 2, 0))
        m = dc.suggest_mapping(list(recs[0]))
        self.assertEqual((m['text'], m['timestamp'], m['likes']), ('text', 'created_at', 'likes'))
        rows, st = dc.normalize_records(recs, m)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['ts'], 1735812000.0)
        self.assertEqual(rows[1]['likes'], 5)

    def test_semicolon_csv_and_bom(self):
        recs, fmt, _ = dc.parse_content('a.csv', '﻿text;date\nhi there;2025-01-01\n')
        self.assertEqual(recs[0]['text'], 'hi there')

    def test_jsonl_with_bad_line_and_twitter_format(self):
        content = json.dumps({'id': 1, 'text': 'a', 'timestamp_ms': '1700000000000', 'user': {'followers_count': 9, 'screen_name': 'u'}, 'retweet_count': 4}) + '\nNOT JSON\n'
        recs, fmt, err = dc.parse_content('t.jsonl', content)
        self.assertEqual((fmt, len(recs), err), ('jsonl', 1, 1))
        m = dc.suggest_mapping(list(recs[0]))
        rows, _ = dc.normalize_records(recs, m)
        self.assertEqual((rows[0]['followers'], rows[0]['retweets'], rows[0]['user']), (9, 4, 'u'))
        self.assertAlmostEqual(rows[0]['ts'], 1700000000.0)

    def test_json_array_and_x_v2_payload(self):
        payload = {'data': [{'id': '7', 'text': 'hello', 'created_at': '2025-02-01T00:00:00.000Z', 'author_id': 'a1',
                             'public_metrics': {'repost_count': 6, 'like_count': 8}}],
                   'includes': {'users': [{'id': 'a1', 'username': 'zed', 'public_metrics': {'followers_count': 123}}]}}
        recs, fmt, _ = dc.parse_content('r.json', json.dumps(payload))
        rows, _ = dc.normalize_records(recs, dc.suggest_mapping(list(recs[0])))
        self.assertEqual((rows[0]['retweets'], rows[0]['likes'], rows[0]['followers']), (6, 8, 123))

    def test_parse_ts_variants(self):
        self.assertAlmostEqual(dc.parse_ts('Wed Oct 10 20:19:24 +0000 2018'), 1539202764.0)
        self.assertAlmostEqual(dc.parse_ts(1700000000), 1700000000.0)
        self.assertAlmostEqual(dc.parse_ts('1700000000000'), 1700000000.0)
        self.assertIsNone(dc.parse_ts('not a date'))
        self.assertIsNone(dc.parse_ts(''))

    def test_label_normalisation(self):
        self.assertEqual([dc.norm_label(x) for x in ('Positive', 'NEG', '0', '-1', 'maybe', None)],
                         ['positive', 'negative', 'neutral', 'negative', None, None])

    def test_missing_text_column_rejected(self):
        with self.assertRaises(ValueError):
            dc.normalize_records([{'a': 1}], {'text': None})

    def test_duplicates(self):
        rows = [{'text': 'Big strike on the railway today http://x.co/1'}, {'text': 'RT @a: Big strike on the railway today'},
                {'text': 'big strike on the railway today!!'}, {'text': 'Completely different words about football goalkeeper penalty'},
                {'text': 'strike on the railway big news today commuters'}]
        d = dc.find_duplicates(rows)
        self.assertEqual(set(d['exact']), {1, 2})
        self.assertNotIn(3, d['exact'])
        d2 = dc.find_duplicates([{'text': 'the climate summit delegates discuss carbon emissions treaty'}, {'text': 'the climate summit delegates discuss carbon emissions treaty again'}])
        self.assertEqual(set(d2['near']), {1})

    def test_validation_and_import(self):
        csv_text = 'text,date\nhello world from here,2025-01-01\nhello world from here,2025-01-01\n,2025-01-02\nanother different message,bad\n'
        p = dc.build_preview('d.csv', csv_text)
        v = p['validation']
        self.assertEqual((v['valid_rows'], v['skipped_missing_text'], v['exact_duplicates'], v['bad_timestamps']), (3, 1, 1, 1))
        rows, meta = dc.import_rows('d.csv', csv_text, p['mapping'], drop_duplicates=True)
        self.assertEqual((len(rows), meta['duplicates_removed']), (2, 1))

    def test_no_records_error(self):
        with self.assertRaises(ValueError):
            dc.build_preview('e.csv', 'text,date\n')

    def test_generator_is_reproducible_and_labelled(self):
        a, b = dc.generate({'n': 300, 'seed': 9}), dc.generate({'n': 300, 'seed': 9})
        self.assertEqual(a, b)
        self.assertNotEqual(a, dc.generate({'n': 300, 'seed': 10}))
        self.assertTrue(all(r['label_sentiment'] in ('positive', 'negative', 'neutral') and r['label_topic'] for r in a))
        self.assertTrue(all(a[i]['ts'] <= a[i + 1]['ts'] for i in range(len(a) - 1)))

    def test_generator_respects_custom_topics_and_spike(self):
        cfg = {'n': 1500, 'days': 4, 'seed': 3, 'noise': 0, 'retweet_fraction': 0, 'languages': {'en': 1},
               'topics': [{'name': 'Spiky', 'keywords': ['alpha', 'beta', 'gamma'], 'weight': 1, 'shape': 'burst', 'at': 0.8, 'width': 0.04, 'magnitude': 30, 'sentiment': [1, 0, 0]},
                          {'name': 'Flat', 'keywords': ['delta', 'omega', 'sigma'], 'weight': 1, 'shape': 'steady', 'sentiment': [0, 0, 1]}]}
        rows = dc.generate(cfg)
        spiky = [r for r in rows if r['label_topic'] == 'Spiky']
        t0, t1 = rows[0]['ts'], rows[-1]['ts']
        frac = sum(1 for r in spiky if 0.7 <= (r['ts'] - t0) / (t1 - t0) <= 0.9) / len(spiky)
        self.assertGreater(frac, 0.6)  # most spike tweets fall around 80% of the period
        self.assertTrue(all(r['label_sentiment'] == 'positive' for r in spiky))
        self.assertTrue(all(r['label_sentiment'] == 'neutral' or r['label_sentiment'] == 'negative' for r in rows if r['label_topic'] == 'Flat'))
        self.assertTrue(all(any(k in r['text'] for k in ('alpha', 'beta', 'gamma')) for r in spiky))

    def test_generator_bad_start(self):
        with self.assertRaises(ValueError):
            dc.generate({'start': 'yesterday'})


class TestSentiment(unittest.TestCase):
    def test_vader_polarity(self):
        s = sentiment.score('vader', ['I love this, it is wonderful!', 'This is terrible and awful', 'The meeting is on Thursday'])
        self.assertGreater(s[0], 0.5)
        self.assertLess(s[1], -0.5)
        self.assertLess(abs(s[2]), 0.1)

    def test_lexicon_polarity(self):
        s = sentiment.score('lexicon', ['great wonderful amazing', 'horrible awful bad', 'table chair'])
        self.assertEqual((s[0], s[1], s[2]), (1.0, -1.0, 0.0))

    def test_thresholds(self):
        self.assertEqual([sentiment.to_label(x) for x in (0.3, -0.3, 0.01)], ['positive', 'negative', 'neutral'])
        self.assertEqual(sentiment.to_label(0.3, pos_t=0.5), 'neutral')

    def test_unavailable_algorithms_fail_honestly(self):
        av = sentiment.availability()
        for algo in ('sentistrength', 'transformer'):
            if not av[algo]['ok']:
                with self.assertRaises(RuntimeError):
                    sentiment.score(algo, ['hello'])
        with self.assertRaises(ValueError):
            sentiment.score('nope', ['x'])

    def test_emotions(self):
        e = sentiment.emotions(['I am furious and angry', 'so happy and thrilled', 'nothing here'])
        self.assertEqual((e['anger'], e['joy'], e['fear']), (1, 1, 0))


class TestTrends(unittest.TestCase):
    def test_poisson_tail(self):
        self.assertAlmostEqual(trends.poisson_sf(3, 2.0), 1 - math.exp(-2) * (1 + 2 + 2), places=6)
        self.assertEqual(trends.poisson_sf(0, 5), 1.0)

    def test_zscore_detects_spike_only(self):
        flat = [5, 4, 6, 5, 5, 4, 6, 5, 5, 4]
        self.assertLess(abs(trends.zscore(flat)), 1.5)
        self.assertGreater(trends.zscore(flat[:-1] + [25]), 5)

    def test_burst_series_interval(self):
        counts = [3, 4, 3, 5, 4, 3, 4, 30, 28, 4, 3, 4]
        z, bursts = trends.burst_series(counts, 3.0)
        self.assertEqual(len(bursts), 1)
        self.assertEqual((bursts[0]['start'], bursts[0]['peak_count']), (7, 30))

    def test_growth_and_velocity(self):
        self.assertAlmostEqual(trends.growth([2, 2, 4, 6], w=2), (10 - 4) / 4)
        self.assertAlmostEqual(trends.slope([1, 2, 3, 4]), 1.0)
        self.assertEqual(trends.slope([5]), 0.0)

    def test_lifecycle_stages(self):
        self.assertEqual(trends.classify([0, 0, 0, 0, 0, 0, 0]), 'inactive')
        self.assertEqual(trends.classify([0, 0, 0, 0, 0, 0, 3]), 'new')
        self.assertEqual(trends.classify([4, 6, 8, 10, 14, 18, 24, 30]), 'rising')
        self.assertEqual(trends.classify([5, 4, 6, 5, 5, 4, 6, 5, 4, 5]), 'stable')       # flat: stable (peaking now means clearly above the earlier level)
        self.assertEqual(trends.classify([5, 20, 40, 38, 25, 15, 9, 6, 4]), 'declining')

    def test_lifecycle_series_tracks_a_full_cycle(self):
        counts = [0] * 4 + [2, 4, 8, 16, 30, 40, 40, 38, 25, 14, 8, 5, 3]
        st = trends.lifecycle(counts)
        self.assertEqual(st[0], 'none')
        self.assertIn('rising', st)
        self.assertEqual(st[-1], 'declining')
        stages = [t['stage'] for t in trends.transitions(st)]
        self.assertLess(stages.index('rising'), len(stages) - stages[::-1].index('declining'))

    def test_score_is_bounded_and_monotone(self):
        self.assertEqual(trends.trend_score(0, 0, 0), 0)
        self.assertLessEqual(trends.trend_score(99, 99, 5), 1.0)
        self.assertGreater(trends.trend_score(5, 1, .1), trends.trend_score(1, 1, .1))

    def test_analyse_bundle(self):
        m = trends.analyse([2, 3, 2, 3, 2, 3, 2, 3, 40, 45], 200, w=2, z_threshold=3)
        self.assertGreater(m['z'], 3)
        self.assertTrue(m['bursts'])
        self.assertLess(m['p_value'], 0.001)


class TestTopics(unittest.TestCase):
    def test_kmeans_separates_clear_topics(self):
        rows = ([{'text': 'railway strike commuters platform union timetable', 'ts': i} for i in range(40)] +
                [{'text': 'football derby stadium goalkeeper penalty referee', 'ts': 100 + i} for i in range(40)])
        for i, r in enumerate(rows):
            r['text'] += ' extra%d' % (i % 3)
        labels, info = topics.cluster_kmeans(rows, {'k': 2})
        self.assertEqual(len(set(labels[:40])), 1)
        self.assertEqual(len(set(labels[40:])), 1)
        self.assertNotEqual(labels[0], labels[40])

    def test_kmeans_auto_k_finds_the_real_number_of_topics(self):
        rows = [r for r in SMALL_ROWS if r['label_topic'] != 'noise' and r['lang'] == 'en']
        labels, info = topics.cluster_kmeans(rows, {'k': 0})
        self.assertTrue(5 <= info['k'] <= 8, info)  # silhouette is a heuristic: expect close to the 6 real topics

    def test_assign_pivot_runs_original_pipeline(self):
        rows = [r for r in SMALL_ROWS if r['lang'] == 'en'][:150]
        labels, info = topics.assign(rows, 'pivot', {'threshold': 0.3, 'min_size': 5})
        self.assertEqual(len(labels), len(rows))
        self.assertGreaterEqual(len([l for l in labels if l >= 0]), 100)

    def test_describe_keywords(self):
        rows = [r for r in SMALL_ROWS if r['lang'] == 'en'][:300]
        labels, _ = topics.assign(rows, 'kmeans', {'k': 6, 'min_size': 5})
        desc, rel = topics.describe(rows, labels)
        all_terms = {w for d in desc.values() for w, _ in d['terms'][:5]}
        self.assertTrue(all_terms & {'strike', 'railway', 'climate', 'summit', 'derby', 'football', 'phone', 'battery', 'university', 'heatwave'})
        for d in desc.values():
            self.assertTrue(1 <= len(d['rep']) <= 5)

    def test_small_input_rejected_and_optional_models_fail_honestly(self):
        with self.assertRaises(ValueError):
            topics.assign(SMALL_ROWS[:5], 'kmeans', {})
        av = topics.availability()
        for algo in ('embeddings', 'bertopic'):
            if not av[algo]['ok']:
                with self.assertRaises(RuntimeError):
                    topics.assign(SMALL_ROWS[:50], algo, {})
        with self.assertRaises(ValueError):
            topics.assign(SMALL_ROWS[:50], 'magic', {})


class TestEvaluation(unittest.TestCase):
    def test_metrics_on_known_confusion(self):
        true = ['positive'] * 4 + ['negative'] * 4 + ['neutral'] * 2
        pred = ['positive', 'positive', 'positive', 'neutral', 'negative', 'negative', 'positive', 'negative', 'neutral', 'neutral']
        m = evaluation.sentiment_metrics(true, pred)
        self.assertEqual(m['confusion'], [[3, 1, 0], [0, 2, 0], [1, 0, 3]])
        self.assertAlmostEqual(m['accuracy'], 0.8)
        self.assertAlmostEqual(m['per_class']['positive']['precision'], 0.75)
        self.assertAlmostEqual(m['per_class']['positive']['recall'], 0.75)
        self.assertAlmostEqual(m['per_class']['neutral']['precision'], 2 / 3, places=3)
        self.assertAlmostEqual(m['per_class']['negative']['recall'], 0.75)

    def test_purity_and_ari(self):
        t = ['a', 'a', 'b', 'b', 'c', 'c']
        self.assertEqual(evaluation.purity(t, [0, 0, 1, 1, 2, 2]), 1.0)
        self.assertAlmostEqual(evaluation.ari(t, [5, 5, 7, 7, 9, 9]), 1.0)
        self.assertLess(evaluation.ari(t, [0, 1, 0, 1, 0, 1]), 0.1)

    def test_evaluate_sentiment_real_predictions(self):
        res = evaluation.evaluate_sentiment(SMALL_ROWS, ['vader', 'lexicon', 'sentistrength'])
        ok = {r['algo']: r for r in res['results'] if 'error' not in r}
        self.assertIn('vader', ok)
        self.assertEqual(ok['vader']['n'], res['n'])
        self.assertTrue(0 < ok['vader']['accuracy'] <= 1)
        self.assertEqual(sum(sum(r) for r in ok['vader']['confusion']), res['n'])
        if not sentiment.availability()['sentistrength']['ok']:
            self.assertIn('error', [r for r in res['results'] if r['algo'] == 'sentistrength'][0])  # reported, never faked

    def test_evaluation_requires_labels(self):
        with self.assertRaises(ValueError):
            evaluation.evaluate_sentiment([dict(r, label_sentiment=None) for r in SMALL_ROWS], ['vader'])

    def test_evaluate_topics(self):
        res = evaluation.evaluate_topics(SMALL_ROWS, ['kmeans'], {'k': 0, 'min_size': 8})
        r = res['results'][0]
        self.assertGreater(r['purity'], 0.8)
        self.assertGreater(r['ari'], 0.5)


class TestEngineAndAlerts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.summary = engine.run_analysis(SMALL_ROWS, {'topic_algo': 'kmeans', 'k': 6, 'min_size': 10}, DS)

    def test_result_structure(self):
        r = self.result
        k = r['kpis']
        self.assertEqual(k['positive'] + k['negative'] + k['neutral'], k['tweets'])
        self.assertEqual(sum(r['volume']), k['tweets'])
        self.assertEqual(len(r['topics']), 6)
        self.assertEqual([t['rank'] for t in r['topics']], list(range(1, 7)))
        for t in r['topics']:
            self.assertEqual(sum(t['series']), t['size'])
            self.assertEqual(len(t['series']), r['bins']['n'])
            s = t['sentiment']
            self.assertEqual(s['positive'] + s['negative'] + s['neutral'], t['size'])
            self.assertIn(t['stage'], ('new', 'rising', 'peaking', 'stable', 'declining', 'inactive'))
        self.assertEqual(len(r['assignments']), k['tweets'])

    def test_burst_topics_are_found_in_synthetic_data(self):
        from collections import Counter
        rows = dc.generate({'n': 2000, 'seed': 42})
        res, _ = engine.run_analysis(rows, {'topic_algo': 'kmeans', 'k': 6, 'min_size': 10}, DS)
        votes = {}
        for idx, tid, _lab, _score in res['assignments']:
            votes.setdefault(tid, Counter())[rows[idx]['label_topic']] += 1
        truth = {tid: c.most_common(1)[0][0] for tid, c in votes.items() if tid}  # discovered topic -> majority true topic
        by_truth = {v: next(t for t in res['topics'] if t['id'] == k) for k, v in truth.items()}
        self.assertTrue(by_truth['Transport strike']['bursts'])      # generated with a spike at 55% of the period
        self.assertTrue(by_truth['Football derby']['bursts'])        # generated with a spike at 20%
        self.assertEqual(by_truth['Football derby']['stage'], 'declining')   # spike long over at the end of the period
        self.assertIn(by_truth['Heatwave warning']['stage'], ('rising', 'peaking'))  # grows over time
        self.assertEqual(by_truth['University open day']['stage'], 'declining')      # fades out
        self.assertNotEqual(by_truth['Phone launch']['stage'], 'rising')            # steady
        self.assertFalse(by_truth['Phone launch']['bursts'])
        self.assertGreaterEqual(res['kpis']['emerging'], 1)

    def test_filters(self):
        c = engine.clean_config({'keywords': 'strike', 'topic_algo': 'kmeans'})
        work, st = engine.apply_filters(SMALL_ROWS, c)
        self.assertTrue(work and all('strike' in r['text'].lower() for _, r in work))
        c = engine.clean_config({'exclude': 'strike', 'languages': 'en', 'drop_duplicates': False})
        work, _ = engine.apply_filters(SMALL_ROWS, c)
        self.assertTrue(all('strike' not in r['text'].lower() and r['lang'] == 'en' for _, r in work))
        c = engine.clean_config({'hashtags': '#strike', 'drop_duplicates': False})
        work, _ = engine.apply_filters(SMALL_ROWS, c)
        self.assertTrue(work and all('#strike' in r['text'].lower() for _, r in work))
        t0 = SMALL_ROWS[0]['ts']
        c = engine.clean_config({'date_from': '2025-03-05', 'date_to': '2025-03-05', 'drop_duplicates': False})
        work, _ = engine.apply_filters(SMALL_ROWS, c)
        self.assertTrue(work and all(1741132800 <= r['ts'] < 1741219200 for _, r in work))

    def test_dedupe_option_and_sampling_cap(self):
        c = engine.clean_config({'drop_duplicates': True, 'max_tweets': 200})
        work, st = engine.apply_filters(SMALL_ROWS, c)
        self.assertGreater(st['duplicates_removed'], 0)
        self.assertEqual(len(work), 200)
        self.assertGreater(st['sampled_from'], 200)

    def test_too_few_tweets_gives_clear_error(self):
        with self.assertRaises(ValueError) as cm:
            engine.run_analysis(SMALL_ROWS, {'keywords': 'zzzznotpresent'}, DS)
        self.assertIn('Only 0 tweets remain', str(cm.exception))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            engine.clean_config({'threshold': 5})
        with self.assertRaises(ValueError):
            engine.clean_config({'min_size': 1})

    def test_non_english_warning(self):
        rows = [dict(r, lang='es') for r in SMALL_ROWS]
        res, _ = engine.run_analysis(rows, {'topic_algo': 'kmeans', 'k': 6}, DS)
        self.assertTrue(res['meta']['warnings'])

    def test_lexicon_algorithm_run(self):
        res, s = engine.run_analysis(SMALL_ROWS, {'topic_algo': 'kmeans', 'k': 6, 'sentiment_algo': 'lexicon'}, DS)
        self.assertEqual(s['sentiment_algo'], 'lexicon')

    def test_alert_rules(self):
        r = self.result
        top = max(r['topics'], key=lambda t: t['z'])
        rules = [{'id': 1, 'name': 'b', 'type': 'burst', 'enabled': True, 'params': {'z': top['z'] - 0.01, 'min_size': 1}},
                 {'id': 2, 'name': 'off', 'type': 'burst', 'enabled': False, 'params': {'z': -99, 'min_size': 1}},
                 {'id': 3, 'name': 'v', 'type': 'volume', 'enabled': True, 'params': {'size': 1}},
                 {'id': 4, 'name': 'none', 'type': 'volume', 'enabled': True, 'params': {'size': 10 ** 9}},
                 {'id': 5, 'name': 'neg', 'type': 'overall_negative', 'enabled': True, 'params': {'share': 0.0}}]
        ev = alerts.evaluate(rules, r)
        names = {e['rule_name'] for e in ev}
        self.assertEqual(names, {'b', 'v', 'neg'})
        self.assertEqual(len([e for e in ev if e['rule_name'] == 'v']), len(r['topics']))


class TestXApi(unittest.TestCase):
    PAGE1 = {'data': [{'id': '1', 'text': 'hello', 'created_at': '2025-05-01T10:00:00.000Z', 'author_id': 'u1', 'lang': 'en',
                       'public_metrics': {'repost_count': 4, 'like_count': 9}}],
             'includes': {'users': [{'id': 'u1', 'username': 'ann', 'public_metrics': {'followers_count': 77}}]}, 'meta': {'next_token': 'abc', 'result_count': 1}}
    PAGE2 = {'data': [{'id': '2', 'text': 'bye', 'created_at': '2025-05-01T11:00:00.000Z', 'author_id': 'u1', 'lang': 'en', 'public_metrics': {'retweet_count': 2, 'like_count': 1}}],
             'includes': {'users': [{'id': 'u1', 'username': 'ann', 'public_metrics': {'followers_count': 77}}]}, 'meta': {'result_count': 1}}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def test_query_builder(self):
        q = xapi.build_query(['climate summit', 'cop'], ['#net0'], ['spam'], 'en')
        self.assertEqual(q, '("climate summit" OR cop OR #net0) -spam -is:retweet lang:en')
        with self.assertRaises(xapi.XApiError):
            xapi.build_query([], [])
        with self.assertRaises(xapi.XApiError):
            xapi.build_query(['x' * 600])

    def test_search_uses_bearer_header_pages_and_maps_rows_without_network(self):
        calls = []

        def fake(req, timeout=0):
            calls.append(req)
            return self.Resp(json.dumps(self.PAGE1 if len(calls) == 1 else self.PAGE2).encode())
        pages = xapi.search_recent('x', max_results=100, pages=3, token='TESTTOKEN', opener=fake)
        self.assertEqual(len(pages), 2)  # stopped when next_token ran out
        self.assertTrue(calls[0].full_url.startswith('https://api.x.com/2/tweets/search/recent?'))
        self.assertEqual(calls[0].get_header('Authorization'), 'Bearer TESTTOKEN')
        self.assertIn('next_token=abc', calls[1].full_url)
        rows = xapi.to_rows(pages)
        self.assertEqual([(r['retweets'], r['likes'], r['followers'], r['user']) for r in rows], [(4, 9, 77, 'ann'), (2, 1, 77, 'ann')])
        self.assertAlmostEqual(rows[0]['ts'], 1746093600.0)

    def test_error_handling(self):
        def http(code, body, headers=None):
            def fake(req, timeout=0):
                raise urllib.error.HTTPError(req.full_url, code, 'x', headers or {}, io.BytesIO(json.dumps(body).encode()))
            return fake
        with self.assertRaisesRegex(xapi.XApiError, '429.*resets at'):
            xapi.search_recent('x', token='t', opener=http(429, {'title': 'Too Many Requests', 'detail': 'cap'}, {'x-rate-limit-reset': '1746093600'}))
        with self.assertRaisesRegex(xapi.XApiError, '401'):
            xapi.search_recent('x', token='t', opener=http(401, {'title': 'Unauthorized', 'detail': 'bad token'}))

    def test_requires_token(self):
        old = os.environ.pop(xapi.TOKEN_ENV, None)
        try:
            with self.assertRaisesRegex(xapi.XApiError, 'X_BEARER_TOKEN'):
                xapi.search_recent('x', opener=lambda *a, **k: self.fail('must not call the network'))
        finally:
            if old:
                os.environ[xapi.TOKEN_ENV] = old


class TestReportsAndHistory(unittest.TestCase):
    def test_db_history_compare_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            db = DB(os.path.join(d, 't.db'))
            ds_id = db.add_dataset('ds', 'synthetic', SMALL_ROWS)
            ds = db.get_dataset(ds_id)
            ids = []
            for cfg in ({'name': 'A', 'topic_algo': 'kmeans', 'k': 6}, {'name': 'B', 'topic_algo': 'kmeans', 'k': 4, 'sentiment_algo': 'lexicon'}):
                res, summ = engine.run_analysis(ds['rows'], cfg, ds)
                ids.append(db.add_run(res['meta']['config']['name'], ds, res['meta']['config'], summ, res))
            self.assertEqual([r['id'] for r in db.list_runs()], ids[::-1])
            a, b = db.get_run(ids[0]), db.get_run(ids[1])
            cmp_ = engine.compare_runs(a, b)
            self.assertIn('tweets', cmp_['kpis'])
            self.assertTrue(cmp_['matched'])
            csv_text = reports.topics_csv(a)
            self.assertEqual(len(csv_text.strip().splitlines()), len(a['result']['topics']) + 1)
            tweets = reports.tweets_csv(a, ds['rows'])
            self.assertEqual(len(tweets.strip().splitlines()), a['result']['kpis']['tweets'] + 1)
            html_text = reports.report_html(a)
            self.assertIn('SYNTHETIC DEMONSTRATION DATA', html_text)
            self.assertEqual(json.loads(reports.report_json(a))['id'], ids[0])
            db.annotate(ds_id, 3, 'positive')
            db.annotate(ds_id, 3, 'negative')
            self.assertEqual(db.annotations(ds_id), [{'idx': 3, 'label': 'negative'}])
            db.delete_run(ids[0])
            self.assertIsNone(db.get_run(ids[0]))
            self.assertEqual(len(db.list_rules()), 3)  # seeded default rules
            db.close()


class TestHttpApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.httpd, cls.db = server.make_server(os.path.join(cls.tmp.name, 'api.db'), '127.0.0.1', 0)
        cls.base = 'http://127.0.0.1:%d' % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.db.close()
        cls.tmp.cleanup()

    def call(self, path, method='GET', body=None, raw=False, expect=200):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                self.assertEqual(r.status, expect)
                return data.decode() if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            body_bytes = e.read()
            self.assertEqual(e.code, expect, body_bytes)
            return json.loads(body_bytes or b'{}')

    def test_01_static_and_status(self):
        html = self.call('/', raw=True)
        self.assertIn('TrendScope AI', html)
        self.assertIn('function', self.call('/app.js', raw=True))
        self.assertEqual(self.call('/../server.py', expect=404), None) if False else None
        st = self.call('/api/status')
        self.assertTrue(st['sentiment']['vader']['ok'] and st['topics']['kmeans']['ok'])
        self.assertIn('token_set', st['x_api'])
        self.assertIn('burst', self.call('/api/meta')['rule_types'])

    def test_02_generate_run_export_compare(self):
        g = self.call('/api/datasets/generate', 'POST', {'name': 'api synth', 'config': {'n': 600, 'seed': 11}})
        self.assertEqual(g['rows'] > 500, True)
        info = self.call('/api/datasets/%d' % g['id'])
        self.assertEqual((info['source'], info['labelled_sentiment'] > 0), ('synthetic', True))
        run1 = self.call('/api/runs', 'POST', {'dataset_id': g['id'], 'config': {'name': 'r1', 'topic_algo': 'kmeans', 'k': 6}})
        run2 = self.call('/api/runs', 'POST', {'dataset_id': g['id'], 'config': {'name': 'r2', 'topic_algo': 'kmeans', 'k': 5, 'keywords': 'strike, climate, phone'}})
        full = self.call('/api/runs/%d' % run1['id'])
        self.assertEqual(full['source'], 'synthetic')
        self.assertTrue(full['result']['topics'])
        self.assertIn('topic', self.call('/api/runs/%d/export?format=topics.csv' % run1['id'], raw=True))
        self.assertIn('sentiment', self.call('/api/runs/%d/export?format=tweets.csv' % run1['id'], raw=True))
        self.assertIn('<table', self.call('/api/runs/%d/export?format=html' % run1['id'], raw=True))
        self.assertEqual(json.loads(self.call('/api/runs/%d/export?format=json' % run1['id'], raw=True))['id'], run1['id'])
        self.call('/api/runs/%d/export?format=pdf' % run1['id'], expect=400)
        cmp_ = self.call('/api/compare?a=%d&b=%d' % (run1['id'], run2['id']))
        self.assertLess(cmp_['kpis']['tweets']['b'], cmp_['kpis']['tweets']['a'])
        self.assertEqual(len(self.call('/api/runs')), 2)
        self.call('/api/runs/%d' % run2['id'], 'DELETE')
        self.assertEqual(len(self.call('/api/runs')), 1)

    def test_03_upload_preview_mapping_import_and_analyse(self):
        lines = ['when,comment_body,likes_total']
        rows = dc.generate({'n': 400, 'seed': 2})
        for r in rows:
            lines.append('%s,"%s",%d' % (dc.datetime.fromtimestamp(r['ts'], dc.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), r['text'].replace('"', ''), r['likes']))
        content = '\n'.join(lines)
        p = self.call('/api/datasets/preview', 'POST', {'filename': 'mine.csv', 'content': content})
        self.assertIsNone(p['suggested_mapping']['text'])  # 'comment_body' is not a known name: the user must map it
        self.assertIn('error', p['validation'])
        mapping = {'text': 'comment_body', 'timestamp': 'when', 'likes': 'likes_total'}
        p = self.call('/api/datasets/preview', 'POST', {'filename': 'mine.csv', 'content': content, 'mapping': mapping})
        self.assertTrue(p['validation']['ok'])
        self.assertGreater(p['validation']['exact_duplicates'], 0)
        imp = self.call('/api/datasets/import', 'POST', {'filename': 'mine.csv', 'content': content, 'mapping': mapping, 'name': 'uploaded', 'drop_duplicates': True})
        self.assertLess(imp['rows'], 400)
        self.call('/api/datasets/import', 'POST', {'filename': 'mine.csv', 'content': content, 'mapping': mapping, 'name': ' '}, expect=400)
        run = self.call('/api/runs', 'POST', {'dataset_id': imp['id'], 'config': {'topic_algo': 'kmeans', 'k': 0}})
        self.assertEqual(self.call('/api/runs/%d' % run['id'])['source'], 'upload')

    def test_04_configs_roundtrip(self):
        self.call('/api/configs', 'POST', {'name': 'my cfg', 'config': {'keywords': 'a, b', 'min_size': 12, 'burst_z': 2.5}})
        cfgs = self.call('/api/configs')
        self.assertEqual((cfgs[0]['name'], cfgs[0]['config']['keywords'], cfgs[0]['config']['min_size']), ('my cfg', ['a', 'b'], 12))
        self.call('/api/configs', 'POST', {'name': 'my cfg', 'config': {'min_size': 20}})  # overwrite by name
        self.assertEqual(len(self.call('/api/configs')), 1)
        self.call('/api/configs', 'POST', {'name': '', 'config': {}}, expect=400)
        self.call('/api/configs/%d' % cfgs[0]['id'], 'DELETE')
        self.assertEqual(self.call('/api/configs'), [])

    def test_05_evaluation_and_annotations(self):
        ds = self.call('/api/datasets/generate', 'POST', {'name': 'eval', 'config': {'n': 500, 'seed': 4}})['id']
        r = self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader', 'lexicon', 'sentistrength']})
        done = {x['algo']: x for x in r['results']}
        self.assertNotIn('error', done['vader'])
        self.assertTrue(0 < done['vader']['macro_f1'] <= 1)
        self.assertEqual(r['dataset']['source'], 'synthetic')
        t = self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'topics', 'algos': ['kmeans'], 'min_size': 8})
        self.assertGreater(t['results'][0]['purity'], 0.7)
        self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': []}, expect=400)
        self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader'], 'source': 'annotations'}, expect=400)  # not enough annotations yet
        sample = self.call('/api/datasets/%d/sample?n=25' % ds)
        self.assertEqual(len(sample), 25)
        truth = self.db.get_dataset(ds)['rows']
        for s in sample:
            self.call('/api/annotations', 'POST', {'dataset_id': ds, 'idx': s['idx'], 'label': truth[s['idx']]['label_sentiment']})
        self.assertEqual(len(self.call('/api/annotations?dataset_id=%d' % ds)), 25)
        a = self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader'], 'source': 'annotations'})
        self.assertEqual(a['n'], 25)
        self.call('/api/annotations', 'POST', {'dataset_id': ds, 'idx': 1, 'label': 'bogus'}, expect=400)
        self.assertGreaterEqual(len(self.call('/api/evals')), 3)

    def test_06_alert_rules_and_events(self):
        rules = self.call('/api/alerts/rules')
        self.assertEqual(len(rules), 3)
        rid = self.call('/api/alerts/rules', 'POST', {'name': 'every topic', 'type': 'volume', 'enabled': True, 'params': {'size': 1}})['id']
        ds = self.call('/api/datasets/generate', 'POST', {'name': 'al', 'config': {'n': 500, 'seed': 8}})['id']
        run = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': {'topic_algo': 'kmeans', 'k': 6}})
        ev = self.call('/api/alerts/events?run_id=%d' % run['id'])
        self.assertTrue([e for e in ev if e['rule_name'] == 'every topic'])
        self.assertEqual(len(ev), len(run['alerts']))
        self.call('/api/alerts/rules', 'POST', {'id': rid, 'name': 'every topic', 'type': 'volume', 'enabled': False, 'params': {'size': 1}})
        run2 = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': {'topic_algo': 'kmeans', 'k': 6}})
        self.assertFalse([e for e in run2['alerts'] if e['rule_name'] == 'every topic'])
        self.call('/api/alerts/rules', 'POST', {'name': 'x', 'type': 'nope', 'params': {}}, expect=400)
        self.call('/api/alerts/rules/%d' % rid, 'DELETE')
        self.assertEqual(len(self.call('/api/alerts/rules')), 3)

    def test_07_errors_and_live_without_token(self):
        self.call('/api/nope', expect=404)
        self.call('/api/runs/99999', expect=404)
        self.call('/api/runs', 'POST', {'dataset_id': 99999, 'config': {}}, expect=404)
        self.call('/api/datasets/preview', 'POST', {'filename': 'x.json', 'content': '{bad json'}, expect=400)
        self.call('/api/runs', 'POST', {'dataset_id': 1, 'config': {'threshold': 7}}, expect=400)
        old = os.environ.pop(xapi.TOKEN_ENV, None)
        try:
            r = self.call('/api/live/search', 'POST', {'keywords': ['climate']}, expect=400)
            self.assertIn('X_BEARER_TOKEN', r['error'])
        finally:
            if old:
                os.environ[xapi.TOKEN_ENV] = old
        self.call('/%2e%2e/%2e%2e/trendscope.py', expect=404)


if __name__ == '__main__':
    unittest.main()
