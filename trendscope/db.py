"""SQLite persistence: datasets, saved configurations, analysis history, annotations, evaluations, alerts."""
import json
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY, name TEXT, source TEXT, created REAL, n INTEGER, meta TEXT, rows TEXT);
CREATE TABLE IF NOT EXISTS configs(id INTEGER PRIMARY KEY, name TEXT UNIQUE, config TEXT, created REAL);
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, name TEXT, created REAL, dataset_id INTEGER, dataset_name TEXT,
    source TEXT, config TEXT, summary TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS annotations(dataset_id INTEGER, idx INTEGER, label TEXT, PRIMARY KEY(dataset_id, idx));
CREATE TABLE IF NOT EXISTS evals(id INTEGER PRIMARY KEY, created REAL, kind TEXT, dataset_id INTEGER, source TEXT, params TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS alert_rules(id INTEGER PRIMARY KEY, name TEXT, type TEXT, params TEXT, enabled INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS alert_events(id INTEGER PRIMARY KEY, run_id INTEGER, rule_id INTEGER, rule_name TEXT,
    severity TEXT, message TEXT, created REAL);
"""

DEFAULT_RULES = [
    ('Statistical burst', 'burst', {'z': 4.0, 'min_size': 15}),
    ('High negative share in a topic', 'negative_share', {'share': 0.6, 'min_size': 30}),
    ('Emerging trends detected', 'emerging_count', {'count': 1}),
]


class DB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        # the default alert rules are seeded only when the rules table is first created, so rules the user deleted do not come back
        first_time = not self.q("SELECT name FROM sqlite_master WHERE type='table' AND name='alert_rules'")
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        if first_time:
            for name, typ, params in DEFAULT_RULES:
                self.x('INSERT INTO alert_rules(name,type,params,enabled) VALUES(?,?,?,1)', (name, typ, json.dumps(params)))

    def close(self):
        with self.lock:
            self.conn.close()

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # ---- datasets
    def add_dataset(self, name, source, rows, meta=None):
        meta = dict(meta or {}, labelled_sentiment=sum(1 for r in rows if r.get('label_sentiment')),
                    labelled_topic=sum(1 for r in rows if r.get('label_topic')))
        return self.x('INSERT INTO datasets(name,source,created,n,meta,rows) VALUES(?,?,?,?,?,?)',
                      (name, source, time.time(), len(rows), json.dumps(meta), json.dumps(rows)))

    def list_datasets(self):
        out = self.q('SELECT id,name,source,created,n,meta FROM datasets ORDER BY id DESC')
        for d in out:
            d['meta'] = json.loads(d['meta'] or '{}')
        return out

    def get_dataset(self, ds_id):
        r = self.q('SELECT * FROM datasets WHERE id=?', (ds_id,))
        if not r:
            return None
        r = r[0]
        r['meta'] = json.loads(r['meta'] or '{}')
        r['rows'] = json.loads(r['rows'])
        return r

    def delete_dataset(self, ds_id):
        self.x('DELETE FROM annotations WHERE dataset_id=?', (ds_id,))
        self.x('DELETE FROM datasets WHERE id=?', (ds_id,))

    # ---- configs
    def save_config(self, name, config):
        self.x('INSERT INTO configs(name,config,created) VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET config=excluded.config, created=excluded.created',
               (name, json.dumps(config), time.time()))

    def list_configs(self):
        out = self.q('SELECT * FROM configs ORDER BY name')
        for c in out:
            c['config'] = json.loads(c['config'])
        return out

    def delete_config(self, cid):
        self.x('DELETE FROM configs WHERE id=?', (cid,))

    # ---- runs
    def add_run(self, name, dataset, config, summary, result):
        return self.x('INSERT INTO runs(name,created,dataset_id,dataset_name,source,config,summary,result) VALUES(?,?,?,?,?,?,?,?)',
                      (name, time.time(), dataset['id'], dataset['name'], dataset['source'],
                       json.dumps(config), json.dumps(summary), json.dumps(result)))

    def list_runs(self):
        out = self.q('SELECT id,name,created,dataset_id,dataset_name,source,summary FROM runs ORDER BY id DESC')
        for r in out:
            r['summary'] = json.loads(r['summary'] or '{}')
        return out

    def get_run(self, run_id):
        r = self.q('SELECT * FROM runs WHERE id=?', (run_id,))
        if not r:
            return None
        r = r[0]
        r['config'] = json.loads(r['config'])
        r['summary'] = json.loads(r['summary'])
        r['result'] = json.loads(r['result'])
        return r

    def delete_run(self, run_id):
        self.x('DELETE FROM alert_events WHERE run_id=?', (run_id,))
        self.x('DELETE FROM runs WHERE id=?', (run_id,))

    # ---- annotations
    def annotate(self, ds_id, idx, label):
        if label is None:
            self.x('DELETE FROM annotations WHERE dataset_id=? AND idx=?', (ds_id, idx))
        else:
            self.x('INSERT OR REPLACE INTO annotations(dataset_id,idx,label) VALUES(?,?,?)', (ds_id, idx, label))

    def annotations(self, ds_id):
        return self.q('SELECT idx,label FROM annotations WHERE dataset_id=? ORDER BY idx', (ds_id,))

    # ---- evals
    def add_eval(self, kind, ds_id, source, params, result):
        return self.x('INSERT INTO evals(created,kind,dataset_id,source,params,result) VALUES(?,?,?,?,?,?)',
                      (time.time(), kind, ds_id, source, json.dumps(params), json.dumps(result)))

    def list_evals(self):
        out = self.q('SELECT * FROM evals ORDER BY id DESC LIMIT 30')
        for e in out:
            e['params'] = json.loads(e['params'])
            e['result'] = json.loads(e['result'])
        return out

    # ---- alerts
    def list_rules(self):
        out = self.q('SELECT * FROM alert_rules ORDER BY id')
        for r in out:
            r['params'] = json.loads(r['params'])
        return out

    def save_rule(self, rule):
        params = json.dumps(rule.get('params', {}))
        if rule.get('id'):
            self.x('UPDATE alert_rules SET name=?,type=?,params=?,enabled=? WHERE id=?',
                   (rule['name'], rule['type'], params, 1 if rule.get('enabled', True) else 0, rule['id']))
            return rule['id']
        return self.x('INSERT INTO alert_rules(name,type,params,enabled) VALUES(?,?,?,?)',
                      (rule['name'], rule['type'], params, 1 if rule.get('enabled', True) else 0))

    def delete_rule(self, rid):
        self.x('DELETE FROM alert_rules WHERE id=?', (rid,))

    def add_events(self, run_id, events):
        seen = {(x['rule_id'], x['message']) for x in self.q('SELECT rule_id,message FROM alert_events WHERE run_id=?', (run_id,))}
        for e in events:
            key = (e.get('rule_id'), e['message'])
            if key in seen:                       # the same rule never records the same alert twice for one run
                continue
            seen.add(key)
            self.x('INSERT INTO alert_events(run_id,rule_id,rule_name,severity,message,created) VALUES(?,?,?,?,?,?)',
                   (run_id, e.get('rule_id'), e['rule_name'], e['severity'], e['message'], time.time()))

    def list_events(self, run_id=None):
        if run_id:
            return self.q('SELECT * FROM alert_events WHERE run_id=? ORDER BY id', (run_id,))
        return self.q('SELECT * FROM alert_events ORDER BY id DESC LIMIT 200')
