"""HTTP server (standard library): JSON API under /api plus the static single-page app."""
import json
import mimetypes
import os
import random
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__, alerts, datacentre, engine, evaluation, reports, sentiment, topics, xapi
from .db import DB

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
MAX_BODY = 80 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


def ensure_demo(db):
    """Create the default synthetic dataset and a first analysis so the dashboard is never empty."""
    if db.list_runs():
        return
    rows = datacentre.generate(datacentre.DEFAULT_GEN)
    ds_id = db.add_dataset('Synthetic demo (6 topics, 7 days)', 'synthetic', rows, {'generator': {k: v for k, v in datacentre.DEFAULT_GEN.items()}})
    do_run(db, ds_id, {'name': 'Demo analysis'})


def do_run(db, ds_id, config):
    ds = db.get_dataset(int(ds_id))
    if not ds:
        raise ApiError('Dataset %s not found' % ds_id, 404)
    result, summary = engine.run_analysis(ds['rows'], config, ds)
    name = result['meta']['config']['name']
    run_id = db.add_run(name, ds, result['meta']['config'], summary, result)
    events = alerts.evaluate(db.list_rules(), result)
    db.add_events(run_id, events)
    return {'id': run_id, 'summary': summary, 'alerts': events, 'warnings': result['meta']['warnings']}


def dataset_info(ds, with_rows=15):
    rows = ds['rows']
    ts = [r['ts'] for r in rows if r['ts']]
    langs = {}
    for r in rows:
        langs[r['lang'] or 'unknown'] = langs.get(r['lang'] or 'unknown', 0) + 1
    return {'id': ds['id'], 'name': ds['name'], 'source': ds['source'], 'n': ds['n'], 'meta': ds['meta'], 'languages': langs,
            'date_from': min(ts) if ts else None, 'date_to': max(ts) if ts else None,
            'labelled_sentiment': sum(1 for r in rows if r['label_sentiment']), 'labelled_topic': sum(1 for r in rows if r['label_topic']),
            'preview': rows[:with_rows]}


class Handler(BaseHTTPRequestHandler):
    db = None
    ui_dir = None
    server_version = 'TrendScope/' + __version__

    def log_message(self, fmt, *args):
        if os.environ.get('TRENDSCOPE_VERBOSE'):
            sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    # ---- plumbing
    def _send(self, status, body, ctype='application/json', headers=None):
        try:
            self._send_raw(status, body, ctype, headers)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass  # the browser went away mid-response; nothing to do

    def _send_raw(self, status, body, ctype, headers):
        if not isinstance(body, bytes):
            body = body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', ctype + ('; charset=utf-8' if ctype.startswith('text') or ctype == 'application/json' else ''))
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj))

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_BODY:
            raise ApiError('Upload too large (limit %d MB)' % (MAX_BODY // 1048576), 413)
        raw = self.rfile.read(n) if n else b''
        return json.loads(raw.decode('utf-8')) if raw else {}

    def do_GET(self):
        self._route('GET')

    def do_POST(self):
        self._route('POST')

    def do_DELETE(self):
        self._route('DELETE')

    def _route(self, method):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path.startswith('/api/'):
                body = self._body() if method == 'POST' else {}
                res = self.api(method, u.path[5:].strip('/'), q, body)
                if isinstance(res, tuple):  # (bytes/str, content-type, headers)
                    self._send(200, res[0], res[1], res[2] if len(res) > 2 else None)
                else:
                    self._json(res)
            elif method == 'GET':
                self.static(u.path)
            else:
                self._json({'error': 'Not found'}, 404)
        except ApiError as e:
            self._json({'error': str(e)}, e.status)
        except (ValueError, KeyError, TypeError, xapi.XApiError, RuntimeError) as e:
            self._json({'error': str(e) or type(e).__name__}, 400)
        except Exception as e:  # unexpected: report, keep server alive
            traceback.print_exc()
            self._json({'error': 'Internal error: %s: %s' % (type(e).__name__, e)}, 500)

    def static(self, path):
        base = STATIC
        if self.ui_dir and path.startswith('/_ui/'):  # browser-test files, only when the test runner enables them
            base, path = self.ui_dir, path[4:]
        rel = 'index.html' if path in ('', '/') else path.lstrip('/')
        full = os.path.normpath(os.path.join(base, rel))
        if not full.startswith(base) or not os.path.isfile(full):
            return self._json({'error': 'Not found'}, 404)
        with open(full, 'rb') as f:
            self._send(200, f.read(), mimetypes.guess_type(full)[0] or 'application/octet-stream')

    # ---- API
    def api(self, method, path, q, b):
        db = self.db
        parts = path.split('/')
        head = parts[0]

        if path == 'status' and method == 'GET':
            return {'version': __version__, 'sentiment': sentiment.availability(), 'topics': topics.availability(),
                    'x_api': {'token_set': xapi.token_available(), 'env_var': xapi.TOKEN_ENV},
                    'counts': {'datasets': len(db.list_datasets()), 'runs': len(db.list_runs())}}
        if path == 'meta' and method == 'GET':
            return {'default_config': engine.DEFAULT_CONFIG, 'default_generator': datacentre.DEFAULT_GEN,
                    'bin_choices': engine.BIN_CHOICES, 'rule_types': alerts.RULE_TYPES, 'fields': datacentre.FIELDS}

        # datasets / data centre
        if head == 'datasets':
            if path == 'datasets' and method == 'GET':
                return db.list_datasets()
            if path == 'datasets/preview' and method == 'POST':
                return datacentre.build_preview(b.get('filename', ''), b['content'], b.get('mapping'))
            if path == 'datasets/import' and method == 'POST':
                if not (b.get('name') or '').strip():
                    raise ApiError('Give the dataset a name.')
                rows, meta = datacentre.import_rows(b.get('filename', ''), b['content'], b['mapping'], bool(b.get('drop_duplicates')))
                if not rows:
                    raise ApiError('No valid rows to import.')
                return {'id': db.add_dataset(b['name'].strip(), 'upload', rows, meta), 'rows': len(rows), 'meta': meta}
            if path == 'datasets/generate' and method == 'POST':
                rows = datacentre.generate(b.get('config'))
                name = (b.get('name') or 'Synthetic dataset').strip()
                return {'id': db.add_dataset(name, 'synthetic', rows, {'generator': b.get('config')}), 'rows': len(rows)}
            if len(parts) >= 2 and parts[1].isdigit():
                ds_id = int(parts[1])
                if len(parts) == 2 and method == 'GET':
                    ds = db.get_dataset(ds_id)
                    if not ds:
                        raise ApiError('Dataset not found', 404)
                    return dataset_info(ds)
                if len(parts) == 2 and method == 'DELETE':
                    db.delete_dataset(ds_id)
                    return {'ok': True}
                if len(parts) == 3 and parts[2] == 'sample' and method == 'GET':
                    ds = db.get_dataset(ds_id)
                    if not ds:
                        raise ApiError('Dataset not found', 404)
                    done = {a['idx'] for a in db.annotations(ds_id)}
                    pool = [i for i in range(ds['n']) if i not in done]
                    rng = random.Random(int(q.get('seed', 0)) or None)
                    pick = rng.sample(pool, min(int(q.get('n', 10)), len(pool)))
                    return [{'idx': i, 'text': ds['rows'][i]['text']} for i in pick]
            raise ApiError('Unknown datasets route', 404)

        if path == 'annotations':
            if method == 'POST':
                label = b.get('label')
                if label not in (None, 'positive', 'negative', 'neutral'):
                    raise ApiError('label must be positive, negative, neutral or null')
                ds_id, idx = int(b['dataset_id']), int(b['idx'])
                ds = db.q('SELECT n FROM datasets WHERE id=?', (ds_id,))
                if not ds:
                    raise ApiError('Dataset not found', 404)
                if not 0 <= idx < ds[0]['n']:
                    raise ApiError('Tweet index %d is outside this dataset (0 to %d).' % (idx, ds[0]['n'] - 1))
                db.annotate(ds_id, idx, label)
                return {'ok': True}
            return db.annotations(int(q['dataset_id']))

        if path == 'live/search' and method == 'POST':
            query = xapi.build_query(b.get('keywords', []), b.get('hashtags', []), b.get('exclude', []), b.get('lang') or None)
            pages = xapi.search_recent(query, int(b.get('max_results', 100)), int(b.get('pages', 1)))
            rows = xapi.to_rows(pages)
            if not rows:
                raise ApiError('The search returned no posts.')
            name = (b.get('name') or 'Live X search: ' + query)[:80]
            return {'id': db.add_dataset(name, 'live', rows, {'query': query}), 'rows': len(rows), 'query': query}

        # configs
        if head == 'configs':
            if method == 'GET':
                return db.list_configs()
            if method == 'POST':
                name = (b.get('name') or '').strip()
                if not name:
                    raise ApiError('Give the configuration a name.')
                cfg = engine.clean_config(b.get('config'))
                cfg['name'] = name
                db.save_config(name, cfg)
                return {'ok': True}
            if method == 'DELETE':
                db.delete_config(int(parts[1]))
                return {'ok': True}

        # runs
        if head == 'runs':
            if path == 'runs' and method == 'GET':
                return db.list_runs()
            if path == 'runs' and method == 'POST':
                if b.get('dataset_id') in (None, ''):
                    raise ApiError('Choose a dataset (dataset_id is missing).')
                return do_run(db, b['dataset_id'], b.get('config') or {})
            run_id = int(parts[1])
            if len(parts) == 2 and method == 'GET':
                run = db.get_run(run_id)
                if not run:
                    raise ApiError('Run not found', 404)
                return run
            if len(parts) == 2 and method == 'DELETE':
                db.delete_run(run_id)
                return {'ok': True}
            if len(parts) == 3 and parts[2] == 'export':
                run = db.get_run(run_id)
                if not run:
                    raise ApiError('Run not found', 404)
                fmt = q.get('format', 'json')
                fname = 'trendscope_run%d' % run_id
                if fmt == 'topics.csv':
                    return reports.topics_csv(run), 'text/csv', {'Content-Disposition': 'attachment; filename=%s_topics.csv' % fname}
                if fmt == 'tweets.csv':
                    ds = db.get_dataset(run['dataset_id'])
                    if not ds:
                        raise ApiError('The dataset of this run was deleted, so per-tweet export is unavailable.')
                    return reports.tweets_csv(run, ds['rows']), 'text/csv', {'Content-Disposition': 'attachment; filename=%s_tweets.csv' % fname}
                if fmt == 'html':
                    return reports.report_html(run), 'text/html', {'Content-Disposition': 'attachment; filename=%s_report.html' % fname}
                if fmt == 'json':
                    return reports.report_json(run), 'application/json', {'Content-Disposition': 'attachment; filename=%s.json' % fname}
                raise ApiError('Unknown export format')
        if path == 'compare' and method == 'GET':
            if 'a' not in q or 'b' not in q:
                raise ApiError('Give the two runs to compare as ?a=<run id>&b=<run id>.')
            a, c = db.get_run(int(q['a'])), db.get_run(int(q['b']))
            if not a or not c:
                raise ApiError('Run not found', 404)
            return engine.compare_runs(a, c)

        # evaluation
        if path == 'eval' and method == 'POST':
            ds_id = int(b['dataset_id'])
            ds = db.get_dataset(ds_id)
            if not ds:
                raise ApiError('Dataset not found', 404)
            kind, algos = b.get('kind', 'sentiment'), b.get('algos') or []
            if not algos:
                raise ApiError('Select at least one algorithm.')
            if kind == 'sentiment':
                rows = ds['rows']
                source = b.get('source', 'dataset')
                if source == 'annotations':
                    ann = [a for a in db.annotations(ds_id) if 0 <= a['idx'] < len(ds['rows'])]   # ignore any out-of-range annotation stored by an older version
                    rows = [dict(ds['rows'][a['idx']], label_sentiment=a['label']) for a in ann]
                res = evaluation.evaluate_sentiment(rows, algos, float(b.get('pos_threshold', 0.05)), float(b.get('neg_threshold', 0.05)))
                res['source'] = source
            elif kind == 'topics':
                params = {'threshold': float(b.get('threshold', 0.3)), 'min_size': int(b.get('min_size', 8)), 'k': int(b.get('k', 0))}
                res = evaluation.evaluate_topics(ds['rows'], algos, params)
                res['params'], res['source'] = params, 'dataset'
            else:
                raise ApiError('kind must be sentiment or topics')
            res['dataset'] = {'id': ds_id, 'name': ds['name'], 'source': ds['source']}
            res['id'] = db.add_eval(kind, ds_id, res['source'], {'algos': algos}, res)
            return res
        if path == 'evals' and method == 'GET':
            return db.list_evals()

        # alerts
        if path == 'alerts/rules':
            if method == 'GET':
                return db.list_rules()
            if method == 'POST':
                if b.get('type') not in alerts.RULE_TYPES:
                    raise ApiError('Unknown rule type')
                if not (b.get('name') or '').strip():
                    raise ApiError('Give the rule a name.')
                return {'id': db.save_rule(dict(b, name=b['name'].strip(), params=alerts.clean_params(b['type'], b.get('params'))))}
        if head == 'alerts' and len(parts) == 3 and parts[1] == 'rules' and method == 'DELETE':
            db.delete_rule(int(parts[2]))
            return {'ok': True}
        if path == 'alerts/events' and method == 'GET':
            return db.list_events(int(q['run_id']) if q.get('run_id') else None)

        raise ApiError('Unknown API route: %s %s' % (method, path), 404)


def make_server(db_path, host='127.0.0.1', port=8765):
    db = DB(db_path)

    class H(Handler):
        pass
    H.db = db
    return ThreadingHTTPServer((host, port), H), db
