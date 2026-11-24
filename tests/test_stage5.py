"""Stage 5 verification: evaluation history, run history, SQLite storage, report exports, alerts, and error handling.

    .venv\\Scripts\\python -W ignore -m unittest tests.test_stage5 -v

All destructive tests use temporary databases. One test opens a COPY of the real database (made with SQLite's read-only backup API) and
checks that the original file is byte-for-byte unchanged. Expected values are computed independently from the stored results or the raw
tweets, never by calling the function under test. No network, credentials or optional packages are used.
"""
import csv
import hashlib
import html.parser
import io
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from trendscope import alerts, datacentre as dc, engine, evaluation, reports, sentiment, server, topics  # noqa: E402
from trendscope.db import DB  # noqa: E402

CLASSES = ['positive', 'neutral', 'negative']
TEA = ['tea', 'kettle', 'biscuit', 'mug', 'brew', 'teapot']
CARS = ['engine', 'garage', 'tyres', 'petrol', 'motor', 'gearbox']
RAIL = ['train', 'station', 'ticket', 'carriage', 'driver', 'signal']
SURGE = ['solar', 'panel', 'battery', 'inverter', 'rooftop', 'grid']
QUAKE = ['earthquake', 'tremor', 'rescue', 'aftershock', 'magnitude', 'rubble']
REAL_DB = os.path.join(ROOT, 'trendscope_data', 'trendscope.db')


def topic(name, words, mix=(.3, .3, .4), shape='steady', **kw):
    d = {'name': name, 'keywords': words, 'hashtags': [], 'weight': 1, 'shape': shape, 'sentiment': list(mix)}   # mix = (positive, negative, neutral)
    d.update(kw)
    return d


def gen(topics_, n=500, days=4, seed=1, difficulty=0.0):
    return dc.generate({'n': n, 'days': days, 'seed': seed, 'noise': 0.0, 'retweet_fraction': 0.0, 'difficulty': difficulty,
                        'languages': {'en': 1.0}, 'start': '2025-03-03', 'topics': topics_})


QUIET = [topic('tea', TEA, (.1, .1, .8)), topic('rail', RAIL, (.1, .1, .8)), topic('cars', CARS, (.1, .1, .8))]   # steady, mostly neutral
ALERTY = [topic('tea', TEA, (.3, .1, .6)), topic('cars', CARS, (.03, .9, .07)), topic('surge', SURGE, (.6, .1, .3), 'emerging'),
          topic('quake', QUAKE, (.1, .8, .1), 'burst', at=0.995, width=0.01, magnitude=40)]


def indep_metrics(true, pred):
    n = len(true)
    conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for t, p in zip(true, pred):
        conf[t][p] += 1
    out = {'accuracy': sum(conf[c][c] for c in CLASSES) / n, 'confusion': [[conf[a][b] for b in CLASSES] for a in CLASSES], 'per': {}}
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in CLASSES if o != c)
        fn = sum(conf[c][o] for o in CLASSES if o != c)
        pr, rc = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        out['per'][c] = {'precision': pr, 'recall': rc, 'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    out['macro_f1'] = sum(v['f1'] for v in out['per'].values()) / 3
    return out


def pair_ari(a, b):
    n2 = len(a) * (len(a) - 1) / 2
    sa = {(i, j) for i, j in combinations(range(len(a)), 2) if a[i] == a[j]}
    sb = {(i, j) for i, j in combinations(range(len(b)), 2) if b[i] == b[j]}
    exp = len(sa) * len(sb) / n2
    return (len(sa & sb) - exp) / ((len(sa) + len(sb)) / 2 - exp)


class HttpCase(unittest.TestCase):
    """One in-process server per test class, on a temporary database."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.httpd, cls.db = server.make_server(os.path.join(cls.tmp.name, 's5.db'), '127.0.0.1', 0)
        cls.base = 'http://127.0.0.1:%d' % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.db.close()
        cls.tmp.cleanup()

    def req(self, path, method='GET', body=None):
        r = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(r, timeout=180) as resp:
                return resp.status, dict(resp.headers), resp.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read().decode('utf-8')

    def call(self, path, method='GET', body=None, expect=200):
        st, _h, text = self.req(path, method, body)
        self.assertEqual(st, expect, (path, text[:300]))
        return json.loads(text) if text else None

    def make_dataset(self, topics_=QUIET, name='ds', source='synthetic', **kw):
        return self.db.add_dataset(name, source, gen(topics_, **kw))

    def analyse(self, ds, **cfg):
        base = {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10, 'drop_duplicates': False}
        base.update(cfg)
        r = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': base})
        return r['id'], self.call('/api/runs/%d' % r['id'])


# =============================================================================================== 5. alerts: rules (pure)
def T(label, size, z=0.0, neg=0, pos=0, avg=0.0, emerging=False, p=0.5):
    return {'label': label, 'size': size, 'z': z, 'p_value': p, 'emerging': emerging, 'sentiment': {'negative': neg, 'positive': pos, 'neutral': size - neg - pos, 'avg': avg}}


def result(topics_, tweets=1000, negative=0):
    return {'topics': topics_, 'kpis': {'tweets': tweets, 'negative': negative}}


def rule(typ, name='r', enabled=True, **params):
    return {'id': 7, 'name': name, 'type': typ, 'enabled': enabled, 'params': params}


class TestAlertRules(unittest.TestCase):
    def test_burst_fires_at_the_threshold_and_respects_min_size(self):
        res = result([T('a', 20, z=4.0), T('b', 20, z=3.99), T('c', 14, z=9.0), T('d', 50, z=12.5, p=0.00001)])
        ev = alerts.evaluate([rule('burst', z=4.0, min_size=15)], res)
        self.assertEqual([e['message'].split('"')[1] for e in ev], ['a', 'd'])         # a: exactly at the threshold; b: just below; c: too small
        self.assertTrue(all(e['severity'] == 'high' and e['rule_id'] == 7 and e['rule_name'] == 'r' for e in ev))
        self.assertIn('z=12.5', ev[1]['message'])
        self.assertIn('50 tweets', ev[1]['message'])

    def test_negative_share_boundaries(self):
        res = result([T('exact', 50, neg=30), T('below', 50, neg=29), T('small', 20, neg=20), T('all', 100, neg=100)])
        ev = alerts.evaluate([rule('negative_share', share=0.6, min_size=30)], res)
        self.assertEqual([e['message'].split('"')[1] for e in ev], ['exact', 'all'])
        self.assertEqual(ev[0]['message'], 'Topic "exact": 60% negative (30 of 50 tweets)')
        self.assertEqual(ev[0]['severity'], 'medium')

    def test_sentiment_drop_boundaries(self):
        res = result([T('at', 40, avg=-0.4), T('above', 40, avg=-0.39), T('deep', 60, avg=-0.8), T('tiny', 5, avg=-0.9)])
        ev = alerts.evaluate([rule('sentiment_drop', avg=-0.4, min_size=30)], res)
        self.assertEqual([e['message'].split('"')[1] for e in ev], ['at', 'deep'])
        self.assertIn('-0.80', ev[1]['message'])

    def test_volume_rule(self):
        ev = alerts.evaluate([rule('volume', size=300)], result([T('big', 300), T('almost', 299), T('huge', 900)]))
        self.assertEqual([e['message'] for e in ev], ['Topic "big": 300 tweets', 'Topic "huge": 900 tweets'])
        self.assertEqual({e['severity'] for e in ev}, {'low'})

    def test_new_trend_rule_lists_emerging_topics(self):
        res = result([T('a', 30, emerging=True), T('b', 30), T('c', 30, emerging=True)])
        ev = alerts.evaluate([rule('emerging_count', count=1)], res)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]['message'], '2 emerging trend(s): a; c')
        self.assertEqual(len(alerts.evaluate([rule('emerging_count', count=2)], res)), 1)
        self.assertEqual(alerts.evaluate([rule('emerging_count', count=3)], res), [])
        self.assertEqual(alerts.evaluate([rule('emerging_count', count=1)], result([T('x', 30)])), [])      # nothing emerging: nothing reported

    def test_overall_negative_rule_and_empty_results(self):
        self.assertEqual(len(alerts.evaluate([rule('overall_negative', share=0.5)], result([], tweets=200, negative=100))), 1)
        self.assertEqual(alerts.evaluate([rule('overall_negative', share=0.5)], result([], tweets=200, negative=99)), [])
        self.assertEqual(alerts.evaluate([rule('overall_negative', share=0.5)], result([], tweets=0, negative=0)), [])     # no division by zero
        for typ in alerts.RULE_TYPES:
            self.assertEqual(alerts.evaluate([rule(typ, **alerts.clean_params(typ, {}))], result([])), [] if typ != 'overall_negative' else alerts.evaluate([rule(typ)], result([])))

    def test_disabled_rules_never_fire_and_enabled_defaults_to_true(self):
        res = result([T('a', 500, z=50, neg=500, avg=-1)])
        self.assertEqual(alerts.evaluate([rule('burst', enabled=False, z=1), rule('volume', enabled=False, size=1)], res), [])
        r = rule('volume', size=1)
        del r['enabled']
        self.assertEqual(len(alerts.evaluate([r], res)), 1)

    def test_missing_parameters_fall_back_to_the_documented_defaults(self):
        res = result([T('a', 40, z=4.5, neg=30, avg=-0.5)])
        self.assertEqual(len(alerts.evaluate([rule('burst')], res)), 1)                 # default z=4, min size 0
        self.assertEqual(len(alerts.evaluate([rule('negative_share')], res)), 1)        # default share 0.6
        self.assertEqual(len(alerts.evaluate([rule('sentiment_drop')], res)), 1)        # default avg -0.4
        self.assertEqual(alerts.evaluate([rule('volume')], res), [])                    # default 300 tweets

    def test_several_rules_are_reported_in_rule_order_with_their_own_ids(self):
        res = result([T('a', 400, z=9, neg=300, avg=-0.7, emerging=True)], tweets=400, negative=300)
        rs = [dict(rule('volume', 'vol', size=100), id=1), dict(rule('burst', 'bur', z=3), id=2), dict(rule('overall_negative', 'neg', share=0.5), id=3)]
        ev = alerts.evaluate(rs, res)
        self.assertEqual([(e['rule_id'], e['rule_name']) for e in ev], [(1, 'vol'), (2, 'bur'), (3, 'neg')])

    def test_a_rule_that_cannot_be_evaluated_is_reported_and_does_not_stop_the_others(self):
        res = result([T('a', 400, z=9)])
        rs = [dict(rule('burst', 'corrupt', z='abc'), id=1), dict(rule('volume', 'fine', size=100), id=2), dict(rule('mystery', 'old'), id=3)]
        ev = alerts.evaluate(rs, res)
        self.assertEqual([e['rule_id'] for e in ev], [1, 2, 3])
        self.assertIn('could not be evaluated', ev[0]['message'])
        self.assertIn('could not be evaluated', ev[2]['message'])
        self.assertEqual(ev[1]['message'], 'Topic "a": 400 tweets')

    def test_parameter_validation(self):
        self.assertEqual(alerts.clean_params('burst', {}), {'z': 4.0, 'min_size': 15})
        self.assertEqual(alerts.clean_params('burst', {'z': 2}), {'z': 2, 'min_size': 15})
        self.assertEqual(alerts.clean_params('negative_share', {'share': 0, 'min_size': 0}), {'share': 0, 'min_size': 0})       # inclusive bounds
        self.assertEqual(alerts.clean_params('negative_share', {'share': 1}), {'share': 1, 'min_size': 30})
        for typ, bad in (('burst', {'z': 'abc'}), ('burst', {'z': None}), ('burst', {'z': True}), ('burst', {'z': float('nan')}), ('burst', {'z': float('inf')}),
                         ('burst', {'z': -1}), ('burst', {'min_size': -2}), ('negative_share', {'share': 1.01}), ('sentiment_drop', {'avg': -1.5}),
                         ('volume', {'size': 0}), ('emerging_count', {'count': 0}), ('overall_negative', {'share': -0.1}), ('volume', {'nonsense': 1}),
                         ('burst', ['z', 4]), ('burst', 'z=4')):
            with self.assertRaises(ValueError, msg=(typ, bad)):
                alerts.clean_params(typ, bad)
        with self.assertRaises(ValueError):
            alerts.clean_params('mystery', {})


# =============================================================================================== 5. alerts: end to end
class TestAlertsEndToEnd(HttpCase):
    def setUp(self):
        for r in self.db.list_rules():
            self.db.delete_rule(r['id'])

    def add_rule(self, name, typ, enabled=True, **params):
        return self.call('/api/alerts/rules', 'POST', {'name': name, 'type': typ, 'enabled': enabled, 'params': params})['id']

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ds = cls.db.add_dataset('alerty', 'synthetic', gen(ALERTY, n=2400, days=4, seed=5))
        cls.quiet = cls.db.add_dataset('quiet', 'synthetic', gen(QUIET, n=600, days=4, seed=6))

    def labels_of(self, events):
        return sorted(re.search(r'Topic "(.*?)"', e['message']).group(1) for e in events)

    def test_alerts_match_an_independent_evaluation_of_the_stored_result(self):
        self.add_rule('burst3', 'burst', z=3, min_size=15)
        self.add_rule('neg', 'negative_share', share=0.6, min_size=30)
        self.add_rule('drop', 'sentiment_drop', avg=-0.3, min_size=30)
        self.add_rule('trend', 'emerging_count', count=1)
        self.add_rule('overall', 'overall_negative', share=0.4)
        rid, run = self.analyse(self.ds, k=4, min_size=30, bin_minutes=60, burst_z=3)
        T_ = run['result']['topics']
        k = run['result']['kpis']
        events = self.call('/api/alerts/events?run_id=%d' % rid)
        by_rule = {}
        for e in events:
            by_rule.setdefault(e['rule_name'], []).append(e)
        self.assertEqual(self.labels_of(by_rule.get('burst3', [])), sorted(t['label'] for t in T_ if t['z'] >= 3 and t['size'] >= 15))
        self.assertEqual(self.labels_of(by_rule.get('neg', [])), sorted(t['label'] for t in T_ if t['sentiment']['negative'] / t['size'] >= 0.6 and t['size'] >= 30))
        self.assertEqual(self.labels_of(by_rule.get('drop', [])), sorted(t['label'] for t in T_ if t['sentiment']['avg'] <= -0.3 and t['size'] >= 30))
        self.assertEqual(len(by_rule.get('trend', [])), 1 if any(t['emerging'] for t in T_) else 0)
        self.assertEqual(len(by_rule.get('overall', [])), 1 if k['negative'] / k['tweets'] >= 0.4 else 0)
        # the designed situations are really detected
        quake = next(t for t in T_ if {'earthquake', 'rescue', 'tremor'} & {w for w, _ in t['terms'][:6]})
        cars = next(t for t in T_ if {'engine', 'garage', 'tyres'} & {w for w, _ in t['terms'][:6]})
        self.assertIn(quake['label'], self.labels_of(by_rule['burst3']))              # burst at the end of the period
        self.assertIn(cars['label'], self.labels_of(by_rule['neg']))                  # 90% negative topic
        self.assertIn(cars['label'], self.labels_of(by_rule['drop']))
        self.assertTrue(any(t['emerging'] for t in T_) and 'trend' in by_rule)        # the growing topic is reported as a new trend

    def test_the_run_response_carries_the_same_events_that_are_stored(self):
        self.add_rule('vol', 'volume', size=10)
        r = self.call('/api/runs', 'POST', {'dataset_id': self.quiet, 'config': {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10, 'drop_duplicates': False}})
        stored = self.call('/api/alerts/events?run_id=%d' % r['id'])
        self.assertEqual([(e['rule_name'], e['severity'], e['message']) for e in stored], [(e['rule_name'], e['severity'], e['message']) for e in r['alerts']])
        self.assertEqual(len(stored), 3)
        self.assertTrue(all(e['run_id'] == r['id'] and e['created'] > 0 for e in stored))

    def test_no_false_alerts_on_steady_neutral_data_with_the_default_rules(self):
        for name, typ, params in ((n, t, p) for n, t, p in __import__('trendscope.db', fromlist=['DEFAULT_RULES']).DEFAULT_RULES):
            self.add_rule(name, typ, **params)
        rid, run = self.analyse(self.quiet, k=3, min_size=10)
        self.assertEqual([t['emerging'] for t in run['result']['topics']], [False] * 3)
        self.assertEqual(self.call('/api/alerts/events?run_id=%d' % rid), [])

    def test_rule_edits_apply_to_later_runs_only(self):
        rule_id = self.add_rule('vol', 'volume', size=1)
        rid1, _ = self.analyse(self.quiet)
        self.call('/api/alerts/rules', 'POST', {'id': rule_id, 'name': 'vol', 'type': 'volume', 'enabled': True, 'params': {'size': 100000}})
        rid2, _ = self.analyse(self.quiet)
        self.call('/api/alerts/rules', 'POST', {'id': rule_id, 'name': 'vol renamed', 'type': 'volume', 'enabled': False, 'params': {'size': 1}})
        rid3, _ = self.analyse(self.quiet)
        self.assertEqual(len(self.call('/api/alerts/events?run_id=%d' % rid1)), 3)
        self.assertEqual(len(self.call('/api/alerts/events?run_id=%d' % rid2)), 0)        # threshold raised
        self.assertEqual(len(self.call('/api/alerts/events?run_id=%d' % rid3)), 0)        # rule disabled
        self.assertEqual(len(self.call('/api/alerts/events?run_id=%d' % rid1)), 3)        # the past is not rewritten
        rules = {r['id']: r for r in self.call('/api/alerts/rules')}
        self.assertEqual((rules[rule_id]['name'], rules[rule_id]['enabled']), ('vol renamed', 0))
        self.call('/api/alerts/rules/%d' % rule_id, 'DELETE')
        self.assertNotIn(rule_id, [r['id'] for r in self.call('/api/alerts/rules')])
        self.assertEqual(len(self.call('/api/alerts/events?run_id=%d' % rid1)), 3)        # events survive their rule

    def test_events_listing_and_deleting_a_run_removes_its_events(self):
        self.add_rule('vol', 'volume', size=1)
        a, _ = self.analyse(self.quiet)
        b, _ = self.analyse(self.quiet)
        allev = self.call('/api/alerts/events')
        self.assertEqual({e['run_id'] for e in allev} & {a, b}, {a, b})
        self.assertEqual([e['id'] for e in allev], sorted((e['id'] for e in allev), reverse=True))           # newest first
        self.call('/api/runs/%d' % a, 'DELETE')
        self.assertEqual(self.call('/api/alerts/events?run_id=%d' % a), [])
        self.assertEqual(len(self.call('/api/alerts/events?run_id=%d' % b)), 3)

    def test_invalid_rules_are_rejected_and_nothing_is_stored(self):
        before = self.call('/api/alerts/rules')
        bad = [{'name': 'x', 'type': 'burst', 'params': {'z': 'abc'}}, {'name': 'x', 'type': 'burst', 'params': {'z': -5}},
               {'name': 'x', 'type': 'volume', 'params': {'nonsense': 1}}, {'name': 'x', 'type': 'emerging_count', 'params': {'count': 0}},
               {'name': '  ', 'type': 'volume', 'params': {}}, {'name': 'x', 'type': 'mystery', 'params': {}}, {'type': 'volume', 'params': {}},
               {'name': 'x', 'type': 'negative_share', 'params': {'share': 7}}]
        for body in bad:
            r = self.call('/api/alerts/rules', 'POST', body, expect=400)
            self.assertTrue(r['error'], body)
        self.assertEqual(self.call('/api/alerts/rules'), before)

    def test_a_corrupt_rule_already_in_the_database_does_not_lose_the_run(self):
        """Regression: a rule with bad parameters used to make the run fail with a 400 AFTER the run had been saved."""
        self.db.save_rule({'name': 'corrupt', 'type': 'burst', 'params': {'z': 'abc'}})
        before = len(self.call('/api/runs'))
        r = self.call('/api/runs', 'POST', {'dataset_id': self.quiet, 'config': {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}})
        self.assertEqual(len(self.call('/api/runs')), before + 1)
        self.assertEqual(len(r['alerts']), 1)
        self.assertIn('could not be evaluated', r['alerts'][0]['message'])


# =============================================================================================== 2-3. history and SQLite storage
class TestHistoryAndStorage(HttpCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ds = cls.db.add_dataset('quiet', 'synthetic', gen(QUIET, n=500, seed=2))

    def test_a_saved_run_loads_back_exactly_and_the_list_summarises_it(self):
        rid, run = self.analyse(self.ds, name='first run')
        self.assertEqual((run['name'], run['dataset_id'], run['dataset_name'], run['source']), ('first run', self.ds, 'quiet', 'synthetic'))
        k = run['result']['kpis']
        row = next(r for r in self.call('/api/runs') if r['id'] == rid)
        self.assertEqual(row['summary']['tweets'], k['tweets'])
        self.assertEqual(row['summary']['topics'], len(run['result']['topics']))
        self.assertEqual(row['summary']['avg_sentiment'], k['avg_sentiment'])
        self.assertEqual(run['config']['topic_algo'], 'kmeans')
        self.assertEqual(run['config'], run['result']['meta']['config'])
        self.assertEqual(run['summary'], self.db.get_run(rid)['summary'])
        self.assertEqual(self.db.get_run(rid)['result'], run['result'])                  # nothing lost or altered by SQLite/JSON

    def test_listing_is_newest_first(self):
        ids = [self.analyse(self.ds, name='n%d' % i)[0] for i in range(3)]
        listed = [r['id'] for r in self.call('/api/runs')]
        self.assertEqual([i for i in listed if i in ids], sorted(ids, reverse=True))

    def test_delete_removes_only_that_run(self):
        a, _ = self.analyse(self.ds, name='keep')
        b, _ = self.analyse(self.ds, name='drop')
        self.call('/api/runs/%d' % b, 'DELETE')
        self.assertNotIn(b, [r['id'] for r in self.call('/api/runs')])
        self.call('/api/runs/%d' % b, expect=404)
        self.call('/api/runs/%d/export?format=json' % b, expect=404)
        self.assertEqual(self.call('/api/runs/%d' % a)['name'], 'keep')
        self.assertEqual(self.call('/api/datasets/%d' % self.ds)['id'], self.ds)           # the dataset is untouched
        self.call('/api/runs/%d' % b, 'DELETE')                                            # deleting again is harmless
        self.call('/api/runs/abc', expect=400)

    def test_comparison_deltas_are_b_minus_a_and_swap_sign(self):
        a, ra = self.analyse(self.ds, name='a', k=3)
        b, rb = self.analyse(self.ds, name='b', k=2, sentiment_algo='lexicon')
        ab, ba = self.call('/api/compare?a=%d&b=%d' % (a, b)), self.call('/api/compare?a=%d&b=%d' % (b, a))
        for key, v in ab['kpis'].items():
            self.assertAlmostEqual(v['delta'], ra['result']['kpis'][key] * -1 + rb['result']['kpis'][key], places=3, msg=key)
            self.assertAlmostEqual(v['delta'], -ba['kpis'][key]['delta'], places=3, msg=key)
        self.assertEqual((ab['a']['id'], ab['b']['id'], ab['a']['name']), (a, b, 'a'))
        self.assertNotEqual(ab['kpis']['topics']['delta'], 0)
        same = self.call('/api/compare?a=%d&b=%d' % (a, a))
        self.assertTrue(all(v['delta'] == 0 for v in same['kpis'].values()))
        self.assertEqual(len(same['matched']), len(ra['result']['topics']))

    def test_comparison_errors(self):
        a, _ = self.analyse(self.ds)
        self.call('/api/compare', expect=400)
        self.call('/api/compare?a=%d' % a, expect=400)
        self.call('/api/compare?a=%d&b=999999' % a, expect=404)
        self.call('/api/compare?a=x&b=y', expect=400)

    def test_a_run_survives_deleting_its_dataset(self):
        ds = self.db.add_dataset('temporary', 'upload', gen(QUIET, n=300, seed=3))
        rid, run = self.analyse(ds)
        self.call('/api/datasets/%d' % ds, 'DELETE')
        again = self.call('/api/runs/%d' % rid)
        self.assertEqual((again['dataset_name'], again['source']), ('temporary', 'upload'))
        for fmt in ('topics.csv', 'json', 'html'):
            self.assertEqual(self.req('/api/runs/%d/export?format=%s' % (rid, fmt))[0], 200)
        st, _h, text = self.req('/api/runs/%d/export?format=tweets.csv' % rid)
        self.assertEqual(st, 400)
        self.assertIn('dataset of this run was deleted', text)                                # explained, not a crash

    def test_data_persists_across_reopening_the_database_file(self):
        path = os.path.join(self.tmp.name, 'persist.db')
        d = DB(path)
        ds = d.add_dataset('p', 'upload', gen(QUIET, n=300, seed=4), {'filename': 'p.csv'})
        res, summary = engine.run_analysis(d.get_dataset(ds)['rows'], {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}, d.get_dataset(ds))
        rid = d.add_run('persisted', d.get_dataset(ds), res['meta']['config'], summary, res)
        d.save_config('cfg', res['meta']['config'])
        d.annotate(ds, 3, 'positive')
        d.close()
        d2 = DB(path)
        self.assertEqual(d2.get_run(rid)['result'], json.loads(json.dumps(res)))   # JSON turns tuples into lists
        self.assertEqual(d2.list_datasets()[0]['meta']['filename'], 'p.csv')
        self.assertEqual(d2.list_configs()[0]['config'], res['meta']['config'])
        self.assertEqual(d2.annotations(ds), [{'idx': 3, 'label': 'positive'}])
        self.assertEqual(d2.get_dataset(ds)['n'], 300)
        d2.close()

    def test_databases_made_before_the_alert_and_evaluation_tables_existed_still_open(self):
        path = os.path.join(self.tmp.name, 'old.db')
        con = sqlite3.connect(path)
        con.executescript("""CREATE TABLE datasets(id INTEGER PRIMARY KEY, name TEXT, source TEXT, created REAL, n INTEGER, meta TEXT, rows TEXT);
            CREATE TABLE configs(id INTEGER PRIMARY KEY, name TEXT UNIQUE, config TEXT, created REAL);
            CREATE TABLE runs(id INTEGER PRIMARY KEY, name TEXT, created REAL, dataset_id INTEGER, dataset_name TEXT, source TEXT, config TEXT, summary TEXT, result TEXT);""")
        rows = gen(QUIET, n=300, seed=4)
        ds = {'id': 1, 'name': 'old ds', 'source': 'synthetic'}
        res, summary = engine.run_analysis(rows, {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}, ds)
        con.execute('INSERT INTO datasets VALUES(1,?,?,?,?,?,?)', ('old ds', 'synthetic', 1.0, len(rows), '{}', json.dumps(rows)))
        con.execute('INSERT INTO runs VALUES(1,?,?,?,?,?,?,?,?)', ('old run', 1.0, 1, 'old ds', 'synthetic', json.dumps(res['meta']['config']), json.dumps(summary), json.dumps(res)))
        con.commit()
        con.close()
        d = DB(path)
        self.assertEqual(d.get_run(1)['result'], json.loads(json.dumps(res)))
        self.assertEqual(len(d.list_rules()), 3)                                           # alert tables created and seeded once
        self.assertEqual(d.list_evals(), [])
        self.assertIn('rank', reports.topics_csv(d.get_run(1)))
        self.assertIn('old ds', reports.report_html(d.get_run(1)))
        d.close()
        d = DB(path)
        self.assertEqual(len(d.list_rules()), 3)
        d.close()

    def test_the_same_alert_is_never_stored_twice_for_one_run(self):
        d = DB(os.path.join(self.tmp.name, 'dup.db'))
        ev = [{'rule_id': 1, 'rule_name': 'a', 'severity': 'high', 'message': 'm1'}, {'rule_id': 1, 'rule_name': 'a', 'severity': 'high', 'message': 'm1'},
              {'rule_id': 2, 'rule_name': 'b', 'severity': 'low', 'message': 'm1'}]
        d.add_events(5, ev)
        d.add_events(5, ev)                                                               # a repeated delivery changes nothing
        self.assertEqual([(e['rule_id'], e['message']) for e in d.list_events(5)], [(1, 'm1'), (2, 'm1')])
        d.add_events(6, ev[:1])                                                           # another run may raise the same alert again
        self.assertEqual(len(d.list_events()), 3)
        d.close()

    def test_default_alert_rules_are_not_resurrected_after_the_user_deletes_them(self):
        """Regression: deleting every rule used to bring the three defaults back the next time the app started."""
        path = os.path.join(self.tmp.name, 'rules.db')
        d = DB(path)
        self.assertEqual([r['name'] for r in d.list_rules()], ['Statistical burst', 'High negative share in a topic', 'Emerging trends detected'])
        for r in d.list_rules():
            d.delete_rule(r['id'])
        d.close()
        d = DB(path)
        self.assertEqual(d.list_rules(), [])
        d.close()

    def test_unusual_text_is_stored_literally_and_never_interpreted(self):
        nasty = 'O\'Brien "q" %s %(name)s <b>&amp;</b> 😀 \'; DROP TABLE runs;--'
        g = self.call('/api/datasets/generate', 'POST', {'name': nasty, 'config': {'n': 300, 'seed': 2, 'topics': QUIET}})
        self.call('/api/configs', 'POST', {'name': nasty, 'config': {'k': 3}})
        rule_id = self.call('/api/alerts/rules', 'POST', {'name': nasty, 'type': 'volume', 'params': {'size': 5}})['id']
        rid, run = self.analyse(g['id'], name=nasty)
        self.assertEqual(run['name'], nasty)
        self.assertEqual(next(d for d in self.call('/api/datasets') if d['id'] == g['id'])['name'], nasty)
        self.assertIn(nasty, [c['name'] for c in self.call('/api/configs')])
        self.assertIn(nasty, [e['rule_name'] for e in self.call('/api/alerts/events?run_id=%d' % rid)])
        self.assertTrue(self.db.q("SELECT name FROM sqlite_master WHERE name='runs'"))     # the injection attempt did nothing
        page = self.req('/api/runs/%d/export?format=html' % rid)[2]
        self.assertIn(html.escape(nasty), page) if False else None
        self.assertNotIn('<b>&amp;</b>', page)
        self.call('/api/alerts/rules/%d' % rule_id, 'DELETE')

    def test_concurrent_writes_are_all_stored_without_corruption(self):
        d = DB(os.path.join(self.tmp.name, 'conc.db'))
        ds = {'id': 1, 'name': 'c', 'source': 'synthetic'}
        ids, errors = [], []

        def worker(w):
            try:
                for i in range(15):
                    ids.append(d.add_run('w%d-%d' % (w, i), ds, {'w': w}, {'tweets': i}, {'payload': 'x' * 2000, 'w': w, 'i': i}))
                    d.list_runs()
            except Exception as e:  # pragma: no cover
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len(set(ids)), 120)
        self.assertEqual(len(d.list_runs()), 120)
        self.assertTrue(all(d.get_run(i)['result']['payload'] == 'x' * 2000 for i in ids[::17]))
        d.close()

    def test_parallel_runs_over_http(self):
        ds = self.db.add_dataset('par', 'synthetic', gen(QUIET, n=300, seed=8))
        out = []
        ths = [threading.Thread(target=lambda i=i: out.append(self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10, 'name': 'par%d' % i}})['id'])) for i in range(4)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        self.assertEqual(len(set(out)), 4)
        names = {r['id']: r['name'] for r in self.call('/api/runs')}
        self.assertEqual(sorted(names[i] for i in out), ['par0', 'par1', 'par2', 'par3'])

    @unittest.skipUnless(os.path.exists(REAL_DB), 'no existing database')
    def test_a_copy_of_the_real_database_works_and_the_original_is_never_touched(self):
        before = hashlib.sha256(open(REAL_DB, 'rb').read()).hexdigest()
        src = sqlite3.connect('file:' + REAL_DB.replace('\\', '/') + '?mode=ro', uri=True)
        copy = os.path.join(self.tmp.name, 'real_copy.db')
        dst = sqlite3.connect(copy)
        src.backup(dst)
        dst.close()
        src.close()
        d = DB(copy)
        runs, datasets = d.list_runs(), d.list_datasets()
        self.assertTrue(runs and datasets)
        for r in runs:
            full = d.get_run(r['id'])
            self.assertEqual(len(full['result']['topics']), r['summary']['topics'])
            self.assertEqual(full['result']['kpis']['tweets'], r['summary']['tweets'])
            self.assertIn('rank', reports.topics_csv(full))
            self.assertEqual(json.loads(reports.report_json(full))['id'], full['id'])
            self.assertIn('<table', reports.report_html(full))
        big = max(datasets, key=lambda x: x['n'])
        new_id = server.do_run(d, big['id'], {'topic_algo': 'kmeans', 'k': 4, 'max_tweets': 600})['id']        # a new analysis on existing stored data
        self.assertEqual(len(d.list_runs()), len(runs) + 1)
        self.assertEqual(engine.compare_runs(d.get_run(runs[-1]['id']), d.get_run(new_id))['a']['id'], runs[-1]['id'])
        d.delete_run(new_id)
        self.assertEqual(len(d.list_runs()), len(runs))
        d.close()
        self.assertEqual(hashlib.sha256(open(REAL_DB, 'rb').read()).hexdigest(), before)


# =============================================================================================== 4. reports
class _Page(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.cells, self.text, self.ban, self._in_ban, self.title, self._in_title, self._cell = [], [], [], '', False, '', False, None

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == 'div' and ('class', 'ban') in attrs:
            self._in_ban = True
        if tag == 'title':
            self._in_title = True
        if tag in ('td', 'th'):
            self._cell = ''

    def handle_endtag(self, tag):
        if tag == 'div':
            self._in_ban = False
        if tag == 'title':
            self._in_title = False
        if tag in ('td', 'th') and self._cell is not None:
            self.cells.append(self._cell)
            self._cell = None

    def handle_data(self, data):
        self.text.append(data)
        if self._in_ban:
            self.ban += data
        if self._in_title:
            self.title += data
        if self._cell is not None:
            self._cell += data


def fmt(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M')


class TestReports(HttpCase):
    HOSTILE = '<script>alert(1)</script> & "q" \'s\' %s %(name)s'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rows = gen(ALERTY, n=1200, days=4, seed=7)
        for i in (0, 1, 2, 3, 4, 5):          # texts that stress CSV quoting and HTML escaping
            rows[i]['text'] += [', "double quotes" and, commas', '\nsecond line\r\nthird line', ' emoji 😀 ünïcode', ' <b>bold</b> <script>x</script>', ' ;semi;colons;', " 'single' quotes"][i]
        cls.rows = rows
        cls.ds = cls.db.add_dataset(cls.HOSTILE, 'upload', rows)
        cls.rid, cls.rep = cls.run_(cls('run_'), name=cls.HOSTILE)

    def run_(self, name=None):
        r = self.call('/api/runs', 'POST', {'dataset_id': self.ds, 'config': {'topic_algo': 'kmeans', 'k': 4, 'min_size': 20, 'bin_minutes': 60, 'drop_duplicates': False, 'name': name}})
        return r['id'], self.call('/api/runs/%d' % r['id'])

    def test_topics_csv_equals_the_stored_result(self):
        st, h, text = self.req('/api/runs/%d/export?format=topics.csv' % self.rid)
        self.assertEqual(st, 200)
        self.assertTrue(h['Content-Type'].startswith('text/csv'))
        self.assertEqual(h['Content-Disposition'], 'attachment; filename=trendscope_run%d_topics.csv' % self.rid)
        rows = list(csv.DictReader(io.StringIO(text)))
        T_ = self.rep['result']['topics']
        self.assertEqual(len(rows), len(T_))
        for row, t in zip(rows, T_):
            self.assertEqual((int(row['rank']), row['topic'], row['stage'], row['emerging'], int(row['tweets'])), (t['rank'], t['label'], t['stage'], str(t['emerging']), t['size']))
            self.assertEqual((float(row['share']), float(row['trend_score']), float(row['burst_z']), float(row['growth']), float(row['velocity'])), (t['share'], t['score'], t['z'], t['growth'], t['velocity']))
            s = t['sentiment']
            self.assertEqual((int(row['positive']), int(row['negative']), int(row['neutral']), float(row['avg_sentiment'])), (s['positive'], s['negative'], s['neutral'], s['avg']))
            self.assertEqual((row['first_seen'], row['last_seen']), (fmt(t['first_seen']), fmt(t['last_seen'])))
            self.assertEqual(row['keywords'].split(), [w for w, _ in t['terms']])
            self.assertEqual(int(row['reach']), t['engagement']['reach'])

    def test_tweets_csv_matches_the_raw_tweets_and_independent_sentiment(self):
        st, h, text = self.req('/api/runs/%d/export?format=tweets.csv' % self.rid)
        self.assertEqual(st, 200)
        rows = list(csv.DictReader(io.StringIO(text)))
        A = self.rep['result']['assignments']
        self.assertEqual(len(rows), len(A))
        names = {t['id']: t['label'] for t in self.rep['result']['topics']}
        for out, (idx, tid, lab, score) in zip(rows, A):
            src = self.rows[idx]
            self.assertEqual((out['id'], out['text'], out['language'], out['time_utc']), (src['id'], src['text'], src['lang'], fmt(src['ts'])))     # exact, incl. quotes, newlines, emoji
            self.assertEqual(out['topic'], names.get(tid, '(no topic)'))
        texts = [self.rows[idx]['text'] for idx, *_ in A]
        scores = sentiment.score('vader', texts)
        for out, sc in zip(rows, scores):
            self.assertAlmostEqual(float(out['sentiment_score']), sc, places=3)
            self.assertEqual(out['sentiment'], sentiment.to_label(sc))
        self.assertEqual(Counter(r['sentiment'] for r in rows), Counter({'positive': self.rep['result']['kpis']['positive'], 'negative': self.rep['result']['kpis']['negative'], 'neutral': self.rep['result']['kpis']['neutral']}))
        for i in range(6):                                                    # the six stress texts are present and intact
            self.assertTrue(any(r['text'] == self.rows[i]['text'] for r in rows), i)

    def test_json_export_carries_the_analysis_metadata_and_the_full_result(self):
        st, h, text = self.req('/api/runs/%d/export?format=json' % self.rid)
        self.assertEqual(st, 200)
        self.assertEqual(h['Content-Type'].split(';')[0], 'application/json')
        self.assertEqual(h['Content-Disposition'], 'attachment; filename=trendscope_run%d.json' % self.rid)
        d = json.loads(text)
        self.assertEqual((d['id'], d['name'], d['dataset'], d['source']), (self.rid, self.HOSTILE, self.HOSTILE, 'upload'))
        self.assertEqual(d['config'], self.rep['config'])
        self.assertEqual(d['result'], self.rep['result'])
        m = d['result']['meta']
        self.assertEqual((m['dataset']['id'], m['dataset']['source'], m['config']['topic_algo'], m['config']['sentiment_algo']), (self.ds, 'upload', 'kmeans', 'vader'))
        self.assertEqual((m['date_from'], m['date_to']), (self.rep['result']['meta']['date_from'], self.rep['result']['meta']['date_to']))
        self.assertEqual(m['filters']['input'], len(self.rows))
        self.assertGreater(m['created'], 0)

    def test_html_report_shows_the_right_numbers_and_escapes_everything(self):
        st, h, text = self.req('/api/runs/%d/export?format=html' % self.rid)
        self.assertEqual(st, 200)
        self.assertEqual(h['Content-Disposition'], 'attachment; filename=trendscope_run%d_report.html' % self.rid)
        p = _Page()
        p.feed(text)
        self.assertNotIn('script', p.tags)                                    # hostile names became text, not markup
        self.assertEqual(p.title, 'TrendScope report: ' + self.HOSTILE)
        self.assertIn('USER-UPLOADED DATASET', p.ban)
        k, T_ = self.rep['result']['kpis'], self.rep['result']['topics']
        body = ' '.join(p.text)
        for needle in ('%d' % k['tweets'], 'run #%d' % self.rid, 'topic algorithm: kmeans', 'sentiment: vader', 'period %s to %s' % (fmt(self.rep['result']['meta']['date_from']), fmt(self.rep['result']['meta']['date_to']))):
            self.assertIn(needle, re.sub(r'\s+', ' ', body), needle)
        self.assertIn('%.0f%% / %.0f%%' % (100 * k['positive'] / k['tweets'], 100 * k['negative'] / k['tweets']), body)
        table = [c for c in p.cells][9:]                                       # skip the 9 header cells of the ranking table
        ranking = [table[i:i + 9] for i in range(0, len(table), 9)]
        ranking = [r for r in ranking if r[0].isdigit()]
        self.assertEqual(len(ranking), len(T_))
        for r, t in zip(ranking, T_):
            self.assertEqual((int(r[0]), r[1], int(r[3]), r[4], r[5]), (t['rank'], t['label'], t['size'], '%.2f' % t['score'], '%.1f' % t['z']))
            self.assertTrue(r[2].startswith(t['stage']))
            self.assertEqual('(emerging)' in r[2], t['emerging'])
        self.assertIn(html.escape(json.dumps(self.rep['config'], indent=1)).replace('&#x27;', "'"), text.replace('&#x27;', "'"))      # configuration echoed, escaped

    def test_banner_names_the_data_source(self):
        want = {'synthetic': 'SYNTHETIC DEMONSTRATION DATA', 'upload': 'USER-UPLOADED DATASET', 'live': 'LIVE X API DATA'}
        for src, label in want.items():
            ds = self.db.add_dataset('src-' + src, src, gen(QUIET, n=300, seed=3))
            r = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}})
            page = self.req('/api/runs/%d/export?format=html' % r['id'])[2]
            p = _Page()
            p.feed(page)
            self.assertIn(label, p.ban, src)
            bg = re.search(r'\.ban\{background:(#\w+)', page).group(1)
            self.assertEqual(bg, {'synthetic': '#fff3cd', 'upload': '#e4e7eb', 'live': '#d9f2e3'}[src])

    def test_exports_work_for_a_run_with_no_topics(self):
        r = self.call('/api/runs', 'POST', {'dataset_id': self.ds, 'config': {'topic_algo': 'kmeans', 'k': 3, 'min_size': 100000}})
        self.assertEqual(r['summary']['topics'], 0)
        self.assertEqual(len(list(csv.reader(io.StringIO(self.req('/api/runs/%d/export?format=topics.csv' % r['id'])[2])))), 1)       # header only
        page = self.req('/api/runs/%d/export?format=html' % r['id'])[2]
        self.assertIn('Topic ranking', page)
        self.assertEqual(json.loads(self.req('/api/runs/%d/export?format=json' % r['id'])[2])['result']['topics'], [])
        tw = list(csv.DictReader(io.StringIO(self.req('/api/runs/%d/export?format=tweets.csv' % r['id'])[2])))
        self.assertTrue(all(row['topic'] == '(no topic)' for row in tw))
        self.assertEqual(self.call('/api/alerts/events?run_id=%d' % r['id']), [] if not any(x['type'] == 'overall_negative' for x in self.call('/api/alerts/rules')) else self.call('/api/alerts/events?run_id=%d' % r['id']))

    def test_export_errors(self):
        self.assertEqual(self.req('/api/runs/%d/export?format=pdf' % self.rid)[0], 400)
        self.assertEqual(self.req('/api/runs/%d/export' % self.rid)[0], 200)                    # default format is json
        self.assertEqual(self.req('/api/runs/999999/export?format=json')[0], 404)

    def test_csv_cells_that_could_run_as_spreadsheet_formulas_are_neutralised_without_changing_stored_data(self):
        rows = gen(QUIET, n=300, seed=9)
        for i, t in enumerate(['=1+1 tweet about tea', '+cmd|x tea', '-2+3 tea', '@SUM(1) tea', '	TAB tea', 'safe = tea +1']):
            rows[i]['text'] = t
        ds = self.db.add_dataset('formula', 'upload', rows)
        r = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': {'topic_algo': 'kmeans', 'k': 3, 'min_size': 10, 'drop_duplicates': False}})
        out = list(csv.DictReader(io.StringIO(self.req('/api/runs/%d/export?format=tweets.csv' % r['id'])[2])))
        texts = {x['id']: x['text'] for x in out}
        for i in range(5):
            self.assertEqual(texts[str(rows[i]['id'])], "'" + rows[i]['text'])
        self.assertEqual(texts[str(rows[5]['id'])], rows[5]['text'])                           # only a leading trigger character matters
        for x in out:
            self.assertNotIn(x['text'][:1], ['=', '+', '-', '@', '\t', '\r'])
        self.assertEqual(self.db.get_dataset(ds)['rows'][0]['text'], '=1+1 tweet about tea')    # stored source data is untouched
        self.assertEqual(json.loads(self.req('/api/runs/%d/export?format=json' % r['id'])[2])['id'], r['id'])

    def test_numbers_and_normal_cells_are_never_altered_by_the_guard(self):
        self.assertEqual([reports.safe_cell(v) for v in (-0.4, 0, 3, True, None, 'tea', '')], [-0.4, 0, 3, True, None, 'tea', ''])
        self.assertEqual(reports.safe_cell('=x'), "'=x")
        text = reports.topics_csv(self.call('/api/runs/%d' % self.rid))
        self.assertTrue(all(not c.startswith("'") for row in list(csv.reader(io.StringIO(text)))[1:] for c in row[:1]))
        neg = [r for r in csv.DictReader(io.StringIO(text)) if float(r['avg_sentiment']) < 0]
        self.assertTrue(all(not r['avg_sentiment'].startswith("'") for r in neg))               # negative numbers stay numbers


# =============================================================================================== 1. evaluation lab: persistence, annotations, errors
class TestEvaluationLab(HttpCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rows = gen(QUIET + [topic('surge', SURGE, (.4, .4, .2))], n=800, seed=3, difficulty=0.3)
        cls.ds = cls.db.add_dataset('labelled', 'synthetic', cls.rows)

    def test_sentiment_evaluation_is_stored_and_equals_an_independent_calculation(self):
        r = self.call('/api/eval', 'POST', {'dataset_id': self.ds, 'kind': 'sentiment', 'algos': ['vader', 'lexicon'], 'pos_threshold': 0.1, 'neg_threshold': 0.2})
        true = [x['label_sentiment'] for x in self.rows]
        for res in r['results']:
            pred = [sentiment.to_label(s, 0.1, 0.2) for s in sentiment.score(res['algo'], [x['text'] for x in self.rows])]
            ref = indep_metrics(true, pred)
            self.assertEqual(res['confusion'], ref['confusion'])
            self.assertAlmostEqual(res['accuracy'], ref['accuracy'], places=4)
            self.assertAlmostEqual(res['macro_f1'], ref['macro_f1'], places=3)
            for c in CLASSES:
                for m in ('precision', 'recall', 'f1'):
                    self.assertAlmostEqual(res['per_class'][c][m], ref['per'][c][m], places=4)
        stored = next(e for e in self.call('/api/evals') if e['id'] == r['id'])
        self.assertEqual(stored['result'], {k: v for k, v in r.items() if k != 'id'})                                   # what is shown later is exactly what was computed
        self.assertEqual((stored['kind'], stored['dataset_id'], stored['source'], stored['params']['algos']), ('sentiment', self.ds, 'dataset', ['vader', 'lexicon']))

    def test_topic_evaluation_purity_and_ari_equal_an_independent_calculation(self):
        r = self.call('/api/eval', 'POST', {'dataset_id': self.ds, 'kind': 'topics', 'algos': ['kmeans'], 'k': 4, 'min_size': 10})
        res = r['results'][0]
        rows = [x for x in self.rows if x.get('label_topic') and x.get('ts')]
        labels, _ = topics.assign(rows, 'kmeans', {'k': 4, 'min_size': 10, 'threshold': 0.3})
        kept = [i for i, x in enumerate(rows) if x['label_topic'] != 'noise' and labels[i] >= 0]
        tt, pp = [rows[i]['label_topic'] for i in kept], [labels[i] for i in kept]
        pur = sum(max(Counter(t for t, p in zip(tt, pp) if p == c).values()) for c in set(pp)) / len(tt)
        self.assertAlmostEqual(res['purity'], pur, places=4)
        self.assertAlmostEqual(res['ari'], pair_ari(tt, pp), places=4)
        self.assertEqual((res['true_topics'], res['clusters']), (4, 4))
        self.assertGreater(res['purity'], 0.95)

    def test_history_is_newest_first_and_limited_to_30(self):
        ds = self.db.add_dataset('hist', 'synthetic', gen(QUIET, n=200, seed=4))
        ids = [self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader']})['id'] for _ in range(32)]
        listed = self.call('/api/evals')
        self.assertEqual(len(listed), 30)
        self.assertEqual([e['id'] for e in listed], sorted(ids, reverse=True)[:30])

    def test_manual_annotations_become_the_ground_truth(self):
        ds = self.db.add_dataset('annot', 'upload', [dict(r, label_sentiment=None, label_topic=None) for r in self.rows[:200]])
        self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader']}, expect=400)          # no labels at all
        truth = {i: self.rows[i]['label_sentiment'] for i in range(0, 90, 3)}                                           # 30 annotated tweets
        for i, lab in truth.items():
            self.call('/api/annotations', 'POST', {'dataset_id': ds, 'idx': i, 'label': lab})
        r = self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader'], 'source': 'annotations'})
        self.assertEqual((r['n'], r['label_distribution']), (30, dict(Counter(truth.values()))))
        pred = [sentiment.to_label(s) for s in sentiment.score('vader', [self.rows[i]['text'] for i in truth])]
        ref = indep_metrics(list(truth.values()), pred)
        self.assertEqual(r['results'][0]['confusion'], ref['confusion'])
        self.assertAlmostEqual(r['results'][0]['macro_f1'], ref['macro_f1'], places=3)
        self.call('/api/annotations', 'POST', {'dataset_id': ds, 'idx': 0, 'label': None})                              # un-annotate
        self.assertEqual(self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader'], 'source': 'annotations'})['n'], 29)
        sample = self.call('/api/datasets/%d/sample?n=10&seed=3' % ds)
        self.assertTrue(len(sample) == 10 and not ({s['idx'] for s in sample} & (set(truth) - {0})))                    # annotated tweets are not offered again

    def test_invalid_annotations_are_rejected(self):
        """Regression: any index was accepted. A negative one silently labelled the wrong tweet and an out-of-range one broke the evaluation with a 500."""
        ds = self.db.add_dataset('annot2', 'upload', [dict(r, label_sentiment=None) for r in self.rows[:100]])
        n = 100
        for idx in (-1, -3, n, 10 ** 6):
            r = self.call('/api/annotations', 'POST', {'dataset_id': ds, 'idx': idx, 'label': 'positive'}, expect=400)
            self.assertIn('outside this dataset', r['error'])
        self.call('/api/annotations', 'POST', {'dataset_id': 99999, 'idx': 0, 'label': 'positive'}, expect=404)
        self.call('/api/annotations', 'POST', {'dataset_id': ds, 'idx': 0, 'label': 'happy'}, expect=400)
        self.assertEqual(self.call('/api/annotations?dataset_id=%d' % ds), [])
        self.call('/api/annotations', 'POST', {'dataset_id': ds, 'idx': n - 1, 'label': 'negative'})                    # the last valid index is accepted
        self.assertEqual(self.call('/api/annotations?dataset_id=%d' % ds), [{'idx': n - 1, 'label': 'negative'}])

    def test_bad_annotations_stored_by_an_older_version_do_not_break_evaluation(self):
        ds = self.db.add_dataset('annot3', 'upload', [dict(r, label_sentiment=None) for r in self.rows[:100]])
        for i in range(25):
            self.db.annotate(ds, i, self.rows[i]['label_sentiment'])
        self.db.annotate(ds, 10 ** 6, 'positive')                                                                       # bypasses the API, like an old database
        r = self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader'], 'source': 'annotations'})
        self.assertEqual(r['n'], 25)

    def test_evaluation_errors(self):
        d = self.ds
        self.call('/api/eval', 'POST', {'dataset_id': 99999, 'kind': 'sentiment', 'algos': ['vader']}, expect=404)
        self.call('/api/eval', 'POST', {'dataset_id': d, 'kind': 'sentiment', 'algos': []}, expect=400)
        self.call('/api/eval', 'POST', {'dataset_id': d, 'kind': 'bogus', 'algos': ['vader']}, expect=400)
        self.call('/api/eval', 'POST', {'dataset_id': d, 'kind': 'sentiment', 'algos': ['vader'], 'pos_threshold': 5}, expect=400)
        self.call('/api/eval', 'POST', {'kind': 'sentiment', 'algos': ['vader']}, expect=400)
        empty = self.db.add_dataset('empty', 'upload', [])
        self.assertIn('have 0', self.call('/api/eval', 'POST', {'dataset_id': empty, 'kind': 'sentiment', 'algos': ['vader']}, expect=400)['error'])
        unlabeled = self.db.add_dataset('nolabels', 'upload', [dict(r, label_topic=None, label_sentiment=None) for r in self.rows[:200]])
        self.call('/api/eval', 'POST', {'dataset_id': unlabeled, 'kind': 'topics', 'algos': ['kmeans']}, expect=400)
        self.assertEqual([r['algo'] for r in self.call('/api/eval', 'POST', {'dataset_id': d, 'kind': 'sentiment', 'algos': ['vader', 'nonsense']})['results']], ['vader', 'nonsense'])


# =============================================================================================== 7. invalid configurations and empty datasets
class TestInvalidConfigurations(HttpCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ds = cls.db.add_dataset('cfg', 'synthetic', gen(QUIET, n=400, seed=3))
        cls.empty = cls.db.add_dataset('empty', 'upload', [])

    BAD = [
        ({'bin_minutes': -5}, 'Time bin must be'), ({'bin_minutes': 10 ** 7}, 'Time bin must be'), ({'bin_minutes': 1}, 'would create'),
        ({'max_tweets': 0}, 'at least 20'), ({'max_tweets': -5}, 'at least 20'), ({'burst_z': -1}, 'greater than 0'), ({'burst_z': 0}, 'greater than 0'),
        ({'min_followers': -10}, 'cannot be negative'), ({'k': -4}, 'between 0'), ({'k': 999}, 'between 0'), ({'min_size': 1}, 'at least 2'),
        ({'threshold': 2.0}, 'between 0.05 and 0.95'), ({'pos_threshold': -0.1}, 'between 0 and 1'), ({'neg_threshold': 3}, 'between 0 and 1'),
        ({'date_from': 'garbage'}, 'must be a date'), ({'date_from': '2025-04-01', 'date_to': '2025-01-01'}, 'is after'),
        ({'burst_z': 'abc'}, 'must be a number'), ({'k': None}, 'must be a number'), ({'min_size': True}, 'must be a number'), ({'burst_z': float('nan')}, None),
        ({'keywords': 5}, 'comma-separated'), ({'sentiment_algo': 'nope'}, 'Unknown sentiment'), ({'topic_algo': 'nope'}, 'Unknown topic'),
        ({'languages': ['zz']}, 'Only 0 tweets'), ({'keywords': 'zzzznomatch'}, 'Only 0 tweets'),
    ]

    def test_every_invalid_configuration_is_a_clear_400_and_stores_nothing(self):
        before = len(self.call('/api/runs'))
        for cfg, msg in self.BAD:
            body = json.dumps({'dataset_id': self.ds, 'config': dict({'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}, **cfg)}, allow_nan=True)
            st, _h, text = self.req('/api/runs', 'POST', json.loads(body))
            self.assertEqual(st, 400, (cfg, text[:200]))
            err = json.loads(text)['error']
            if msg:
                self.assertIn(msg, err, cfg)
            self.assertNotIn('Internal error', err)
        self.assertEqual(len(self.call('/api/runs')), before)

    def test_boundary_values_are_accepted(self):
        for cfg in ({'bin_minutes': 5}, {'bin_minutes': 0}, {'k': 0}, {'k': 1}, {'max_tweets': 20}, {'burst_z': 0.1}, {'min_followers': 0}, {'recent_bins': 0}, {'pos_threshold': 0, 'neg_threshold': 1}):
            st, _h, text = self.req('/api/runs', 'POST', {'dataset_id': self.ds, 'config': dict({'topic_algo': 'kmeans', 'k': 3, 'min_size': 10}, **cfg)})
            if cfg.get('max_tweets') == 20:
                self.assertIn(st, (200, 400))                                  # 20 tweets is allowed by validation; the 20-tweet minimum after filtering decides
            else:
                self.assertEqual(st, 200, (cfg, text[:200]))

    def test_a_missing_dataset_and_a_missing_dataset_id(self):
        self.assertIn('not found', self.call('/api/runs', 'POST', {'dataset_id': 99999, 'config': {}}, expect=404)['error'])
        self.assertIn('dataset_id is missing', self.call('/api/runs', 'POST', {'config': {}}, expect=400)['error'])

    def test_empty_dataset_everywhere(self):
        self.assertIn('Only 0 tweets', self.call('/api/runs', 'POST', {'dataset_id': self.empty, 'config': {}}, expect=400)['error'])
        info = self.call('/api/datasets/%d' % self.empty)
        self.assertEqual((info['n'], info['preview'], info['date_from']), (0, [], None))
        self.assertEqual(self.call('/api/datasets/%d/sample' % self.empty), [])
        self.assertEqual(self.call('/api/datasets/99999/sample', expect=404)['error'], 'Dataset not found')
        self.call('/api/datasets/%d' % self.empty, 'DELETE')

    def test_saved_configurations_are_validated_too(self):
        self.call('/api/configs', 'POST', {'name': 'bad', 'config': {'k': 500}}, expect=400)
        self.call('/api/configs', 'POST', {'name': '', 'config': {}}, expect=400)
        self.assertNotIn('bad', [c['name'] for c in self.call('/api/configs')])

    def test_the_limits_protect_against_runaway_computation(self):
        rows = gen(QUIET, n=300, days=14, seed=2)
        with self.assertRaises(ValueError) as cm:
            engine.run_analysis(rows, {'bin_minutes': 5, 'topic_algo': 'kmeans', 'k': 3}, {'id': 1, 'name': 'x', 'source': 'synthetic'})
        self.assertIn('4030 bins' if False else 'bins for this period', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
