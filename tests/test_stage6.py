r"""Stage 6: end-to-end workflows over HTTP, optional-model fallbacks, saved-run compatibility and the real startup command.

    .venv\Scripts\python -W ignore -m unittest tests.test_stage6 -v

Everything uses temporary databases and synthetic data; the real database is only read through a copy. No network beyond localhost.
"""
import csv
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage5 import HttpCase, ROOT, gen, ALERTY, REAL_DB  # noqa: E402
from trendscope import reports  # noqa: E402


def csv_text(rows):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(['id', 'created_at', 'text', 'lang', 'likes', 'retweets', 'sentiment', 'topic'])
    for r in rows:
        w.writerow([r['id'], datetime.fromtimestamp(r['ts'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), r['text'], r['lang'], r['likes'], r['retweets'],
                    r['label_sentiment'], r['label_topic']])
    return out.getvalue()


class TestFullWorkflow(HttpCase):
    """Import -> configure -> run -> inspect -> evaluate -> compare -> report -> alerts, all from one uploaded file."""

    def test_the_whole_journey_uses_one_consistent_set_of_real_results(self):
        rows = gen(ALERTY, n=1000, days=4, seed=21)
        content = csv_text(rows)
        # 1. import: preview auto-maps, import stores an UPLOAD dataset
        pv = self.call('/api/datasets/preview', 'POST', {'filename': 'journey.csv', 'content': content})
        self.assertEqual(pv['mapping']['text'], 'text')
        imp = self.call('/api/datasets/import', 'POST', {'filename': 'journey.csv', 'content': content, 'mapping': pv['mapping'], 'name': 'Journey upload', 'drop_duplicates': False})
        self.assertEqual(imp['rows'], 1000)
        ds = imp['id']
        self.assertEqual([d['source'] for d in self.call('/api/datasets') if d['id'] == ds], ['upload'])
        # 2. configure + run (and save the configuration)
        cfg = {'topic_algo': 'kmeans', 'k': 4, 'min_size': 20, 'bin_minutes': 60, 'drop_duplicates': False, 'name': 'Journey A'}
        self.call('/api/configs', 'POST', {'name': 'journey cfg', 'config': cfg})
        self.assertEqual(self.call('/api/configs')[0]['name'], 'journey cfg')
        a = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': cfg})
        run = self.call('/api/runs/%d' % a['id'])
        self.assertEqual(run['source'], 'upload')
        res = run['result']
        # 3. every section reads the same stored result
        k = res['kpis']
        self.assertEqual(k['positive'] + k['negative'] + k['neutral'], k['tweets'])
        in_topics = sum(t['size'] for t in res['topics'])
        self.assertLessEqual(in_topics, k['tweets'])
        self.assertEqual(sum(t['sentiment']['positive'] + t['sentiment']['negative'] + t['sentiment']['neutral'] for t in res['topics']), in_topics)
        self.assertEqual((run['summary']['tweets'], run['summary']['topics']), (k['tweets'], len(res['topics'])))
        self.assertEqual(run['summary']['emerging'], sum(1 for t in res['topics'] if t['emerging']))
        listed = [r for r in self.call('/api/runs') if r['id'] == a['id']][0]
        self.assertEqual((listed['source'], listed['dataset_name']), ('upload', 'Journey upload'))
        # 4. evaluate models on the uploaded labels; the evaluation is kept in history
        ev = self.call('/api/eval', 'POST', {'dataset_id': ds, 'kind': 'sentiment', 'algos': ['vader', 'lexicon']})
        self.assertEqual((ev['n'], ev['dataset']['source']), (1000, 'upload'))
        self.assertIn(ev['id'], [e['id'] for e in self.call('/api/evals')])
        # 5. a second run, then compare history
        b = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': dict(cfg, name='Journey B', k=3)})
        rb = self.call('/api/runs/%d' % b['id'])
        cmp_ = self.call('/api/compare?a=%d&b=%d' % (a['id'], b['id']))
        self.assertEqual((cmp_['kpis']['topics']['a'], cmp_['kpis']['topics']['b']), (len(res['topics']), len(rb['result']['topics'])))
        self.assertEqual(cmp_['kpis']['topics']['delta'], len(rb['result']['topics']) - len(res['topics']))
        self.assertEqual(cmp_['kpis']['tweets']['delta'], rb['result']['kpis']['tweets'] - k['tweets'])
        # 6. reports match the stored run
        csv_rows = list(csv.DictReader(io.StringIO(self.req('/api/runs/%d/export?format=topics.csv' % a['id'])[2])))
        self.assertEqual([r['topic'] for r in csv_rows], [t['label'] for t in res['topics']])
        self.assertEqual([int(r['tweets']) for r in csv_rows], [t['size'] for t in res['topics']])
        page = self.req('/api/runs/%d/export?format=html' % a['id'])[2]
        for needle in ('USER-UPLOADED DATASET', 'Journey upload', 'kmeans'):
            self.assertIn(needle, page)
        meta = json.loads(self.req('/api/runs/%d/export?format=json' % a['id'])[2])
        self.assertEqual((meta['source'], meta['dataset'], meta['config']['topic_algo']), ('upload', 'Journey upload', 'kmeans'))
        # 7. alerts: a new rule fires on the next run only, and the stored events equal the run response
        self.call('/api/alerts/rules', 'POST', {'name': 'big', 'type': 'volume', 'enabled': True, 'params': {'size': 1}})
        c = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': dict(cfg, name='Journey C')})
        fired = [e for e in c['alerts'] if e['rule_name'] == 'big']
        self.assertEqual(len(fired), len(self.call('/api/runs/%d' % c['id'])['result']['topics']))
        self.assertEqual(len(self.call('/api/alerts/events?run_id=%d' % c['id'])), len(c['alerts']))
        self.assertEqual([e for e in self.call('/api/alerts/events?run_id=%d' % a['id']) if e['rule_name'] == 'big'], [])
        # 8. deleting a run removes it, its alerts and its comparisons
        self.call('/api/runs/%d' % c['id'], 'DELETE')
        self.assertNotIn(c['id'], [r['id'] for r in self.call('/api/runs')])
        self.assertEqual(self.call('/api/alerts/events?run_id=%d' % c['id']), [])
        self.call('/api/compare?a=%d&b=%d' % (a['id'], c['id']), expect=404)


class TestOptionalModelsAndErrors(HttpCase):
    def test_unavailable_optional_models_give_clear_errors_and_store_nothing(self):
        st = self.call('/api/status')
        ds = self.db.add_dataset('d', 'synthetic', gen(ALERTY, n=300, seed=2))
        missing = [k for k, v in st['sentiment'].items() if isinstance(v, dict) and not v.get('available', True)]
        for algo in missing:
            err = self.call('/api/runs', 'POST', {'dataset_id': ds, 'config': {'sentiment_algo': algo}}, expect=400)['error']
            self.assertTrue(err and 'Traceback' not in err)
        self.assertEqual(self.call('/api/runs'), [])

    def test_malformed_requests_are_clear_client_errors_and_pages_load(self):
        self.assertEqual(self.req('/api/nope')[0], 404)
        self.assertEqual(self.req('/api/runs/abc')[0], 400)
        self.assertEqual(self.req('/api/runs', 'POST', {})[0], 400)
        self.assertEqual(self.req('/api/runs/1/export?format=json')[0], 404)
        for path in ('/', '/app.js', '/style.css'):
            self.assertEqual(self.req(path)[0], 200)
        self.assertIn('Loading', self.req('/')[2])                      # loading state is in the page before the script runs


class TestCompatibilityAndStartup(unittest.TestCase):
    def test_a_copy_of_the_real_database_opens_and_every_saved_run_exports(self):
        if not os.path.exists(REAL_DB):
            self.skipTest('no real database')
        before = open(REAL_DB, 'rb').read()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            copy = os.path.join(tmp, 'copy.db')
            src = sqlite3.connect('file:%s?mode=ro' % REAL_DB.replace('\\', '/'), uri=True)
            dst = sqlite3.connect(copy)
            src.backup(dst)
            src.close()
            dst.close()
            from trendscope.db import DB
            d = DB(copy)
            self.assertEqual(d.conn.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            for r in d.list_runs():
                run = d.get_run(r['id'])
                self.assertTrue(reports.topics_csv(run).startswith('rank,topic'))
                self.assertIn('TrendScope', reports.report_html(run))
                json.loads(reports.report_json(run))
            d.close()
        self.assertEqual(open(REAL_DB, 'rb').read(), before)

    def test_the_documented_startup_command_serves_the_demo(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            port = 18765
            p = subprocess.Popen([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'trendscope.py'), '--port', str(port), '--no-browser', '--db', os.path.join(tmp, 'boot.db')],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                status = None
                for _ in range(120):
                    try:
                        status = json.loads(urllib.request.urlopen('http://127.0.0.1:%d/api/status' % port, timeout=2).read())
                        break
                    except Exception:
                        time.sleep(1)
                self.assertIsNotNone(status, 'server did not start')
                self.assertEqual((status['counts']['datasets'], status['counts']['runs']), (1, 1))      # the demo was seeded on first start
                self.assertIn('TrendScope AI', urllib.request.urlopen('http://127.0.0.1:%d/' % port).read().decode())
            finally:
                p.terminate()
                p.wait(10)


if __name__ == '__main__':
    unittest.main()
