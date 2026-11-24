"""Exports: topic ranking CSV, per-tweet CSV, full JSON, standalone HTML report."""
import csv
import html
import io
import json
from datetime import datetime, timezone

SOURCE_TEXT = {
    'synthetic': 'SYNTHETIC DEMONSTRATION DATA: generated tweets, not real Twitter/X content',
    'upload': 'USER-UPLOADED DATASET: not verified as real Twitter/X data',
    'live': 'LIVE X API DATA: retrieved with the official X API v2 recent search',
}


def fmt_ts(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M') if ts else ''


def safe_cell(v):
    """Spreadsheet formula-injection guard for CSV exports: a text cell starting with = + - @ tab or CR gets a leading apostrophe, which
    spreadsheets hide and treat as 'text'. Numbers and booleans are untouched; stored data is never changed, only the exported copy."""
    if isinstance(v, str) and v[:1] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + v
    return v


class _Writer:
    def __init__(self, out):
        self.w = csv.writer(out)

    def writerow(self, row):
        self.w.writerow([safe_cell(c) for c in row])


def topics_csv(run):
    out = io.StringIO()
    w = _Writer(out)
    w.writerow(['rank', 'topic', 'stage', 'emerging', 'tweets', 'share', 'trend_score', 'burst_z', 'p_value', 'growth', 'velocity',
                'positive', 'negative', 'neutral', 'avg_sentiment', 'avg_retweets', 'avg_likes', 'reach', 'first_seen', 'last_seen', 'keywords'])
    for t in run['result']['topics']:
        s, e = t['sentiment'], t['engagement']
        w.writerow([t['rank'], t['label'], t['stage'], t['emerging'], t['size'], t['share'], t['score'], t['z'], t['p_value'], t['growth'],
                    t['velocity'], s['positive'], s['negative'], s['neutral'], s['avg'], e['avg_retweets'], e['avg_likes'], e['reach'],
                    fmt_ts(t['first_seen']), fmt_ts(t['last_seen']), ' '.join(x for x, _ in t['terms'])])
    return out.getvalue()


def tweets_csv(run, dataset_rows):
    topic_label = {t['id']: t['label'] for t in run['result']['topics']}
    out = io.StringIO()
    w = _Writer(out)
    w.writerow(['id', 'time_utc', 'text', 'language', 'sentiment', 'sentiment_score', 'topic'])
    for idx, tid, lab, score in run['result']['assignments']:
        r = dataset_rows[idx]
        w.writerow([r['id'], fmt_ts(r['ts']), r['text'], r['lang'], {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}[lab], score,
                    topic_label.get(tid, '(no topic)')])
    return out.getvalue()


def report_json(run):
    return json.dumps({'id': run['id'], 'name': run['name'], 'dataset': run['dataset_name'], 'source': run['source'],
                       'config': run['config'], 'result': run['result']}, indent=1)


def report_html(run):
    e = html.escape
    r, k = run['result'], run['result']['kpis']
    src = run['source']
    cls = {'synthetic': '#fff3cd', 'live': '#d9f2e3'}.get(src, '#e4e7eb')
    rows = ''
    for t in r['topics']:
        s = t['sentiment']
        n = max(t['size'], 1)
        bar = ('<div class="b"><i style="width:%.1f%%;background:#2e8b57"></i><i style="width:%.1f%%;background:#c0392b"></i>'
               '<i style="width:%.1f%%;background:#8a94a0"></i></div>') % (100 * s['positive'] / n, 100 * s['negative'] / n, 100 * s['neutral'] / n)
        rows += '<tr><td>%d</td><td>%s</td><td>%s%s</td><td>%d</td><td>%.2f</td><td>%.1f</td><td>%+.0f%%</td><td>%s</td><td>%s</td></tr>' % (
            t['rank'], e(t['label']), e(t['stage']), ' (emerging)' if t['emerging'] else '', t['size'], t['score'], t['z'],
            t['growth'] * 100, bar, e(', '.join(w for w, _ in t['terms'][:6])))
    cfg = e(json.dumps(run['config'], indent=1))
    return '''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>TrendScope report: %(name)s</title><style>
body{font:14px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 14px;color:#1f2933}
.ban{background:%(bg)s;padding:10px 14px;font-weight:600;border-radius:6px}.k{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.k div{border:1px solid #ddd;border-radius:8px;padding:10px 16px}.k b{display:block;font-size:22px}
table{border-collapse:collapse;width:100%%;font-size:13px}td,th{border-bottom:1px solid #e4e7eb;padding:6px;text-align:left}
.b{display:flex;height:10px;width:110px;background:#eee}.b i{display:block}pre{background:#f4f5f7;padding:10px;overflow:auto;font-size:12px}
</style></head><body><h1>TrendScope AI report: %(name)s</h1><div class="ban">%(banner)s</div>
<p>Dataset: <b>%(ds)s</b> &middot; run #%(id)s &middot; topic algorithm: %(ta)s &middot; sentiment: %(sa)s &middot; period %(f)s to %(t)s (UTC)</p>
<div class="k"><div><b>%(tw)d</b>tweets analysed</div><div><b>%(nt)d</b>topics</div><div><b>%(em)d</b>emerging</div><div><b>%(bu)d</b>bursting</div>
<div><b>%(pos).0f%% / %(neg).0f%%</b>positive / negative</div></div>
<h2>Topic ranking</h2><table><tr><th>#</th><th>Topic</th><th>Stage</th><th>Tweets</th><th>Score</th><th>Burst z</th><th>Growth</th><th>Sentiment</th><th>Keywords</th></tr>%(rows)s</table>
<h2>Configuration</h2><pre>%(cfg)s</pre><p style="color:#616e7c">Generated by TrendScope AI. Trend stages and scores are computed from the data; see README for definitions.</p></body></html>''' % {
        'name': e(run['name']), 'bg': cls, 'banner': e(SOURCE_TEXT.get(src, src)), 'ds': e(run['dataset_name']), 'id': run['id'],
        'ta': e(run['config'].get('topic_algo', '')), 'sa': e(run['config'].get('sentiment_algo', '')),
        'f': fmt_ts(r['meta']['date_from']), 't': fmt_ts(r['meta']['date_to']), 'tw': k['tweets'], 'nt': k['topics'], 'em': k['emerging'],
        'bu': k['bursting'], 'pos': 100 * k['positive'] / max(k['tweets'], 1), 'neg': 100 * k['negative'] / max(k['tweets'], 1),
        'rows': rows, 'cfg': cfg}
