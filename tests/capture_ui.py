r"""Stage 6 UI capture: screenshots of every tab (desktop 1280px and phone 390px), console-error check and horizontal-overflow check.

    .venv\Scripts\python tests\capture_ui.py

Uses throw-away databases (your data is untouched) and headless Chrome/Edge. Writes PNGs to ./screenshots and prints a PASS/FAIL summary.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trendscope import server, datacentre  # noqa: E402

TABS = ['dashboard', 'studio', 'data', 'trends', 'topics', 'sentiment', 'eval', 'history', 'alerts']
CANDIDATES = [os.environ.get('CHROME', ''), r'C:\Program Files\Google\Chrome\Application\chrome.exe', r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', shutil.which('chrome') or '']
OUT = os.path.join(ROOT, 'screenshots')


def chrome_path():
    c = next((c for c in CANDIDATES if c and os.path.isfile(c)), None)
    if not c:
        sys.exit('Chrome/Edge not found. Set CHROME.')
    return c


def shot(chrome, url, png, w, h, budget=25000):
    p = subprocess.run([chrome, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--enable-logging=stderr', '--v=0', '--window-size=%d,%d' % (w, h),
                        '--virtual-time-budget=%d' % budget, '--screenshot=' + png, url], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)
    return [l for l in p.stderr.splitlines() if 'CONSOLE' in l and ('error' in l.lower() or 'uncaught' in l.lower())]


def probe(chrome, base, tab):
    out = subprocess.run([chrome, '--headless=new', '--disable-gpu', '--window-size=500,900', '--virtual-time-budget=20000', '--dump-dom',
                          '%s/_ui/mobile.html?tab=%s' % (base, tab)], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240).stdout
    m = re.search(r'<pre id="T"[^>]*>(\{.*?\})</pre>', out)
    return m.group(1) if m else None


def serve(tmp, name, demo=True):
    server.Handler.ui_dir = os.path.join(ROOT, 'tests', 'browser')
    httpd, db = server.make_server(os.path.join(tmp, name), '127.0.0.1', 0)
    if demo:
        server.ensure_demo(db)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, db, 'http://127.0.0.1:%d' % httpd.server_address[1]


def main():
    chrome = chrome_path()
    os.makedirs(OUT, exist_ok=True)
    problems, n = [], 0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        httpd, db, base = serve(tmp, 'demo.db')
        for tab in TABS:
            for label, w, h in (('desktop', 1280, 1500), ('mobile', 390, 2400)):
                # headless Chrome cannot open windows narrower than ~500px, so phone shots use the 390px same-origin iframe page
                url = '%s/#%s' % (base, tab) if label == 'desktop' else '%s/_ui/mobile.html?tab=%s' % (base, tab)
                errs = shot(chrome, url, os.path.join(OUT, '%s_%s.png' % (tab, label)), w if label == 'desktop' else 390, h)
                n += 1
                problems += ['%s %s console: %s' % (tab, label, e) for e in errs]
            r = probe(chrome, base, tab)
            if r is None or '"overflowX":true' in r:
                problems.append('mobile overflow on %s: %s' % (tab, r))
            print('captured', tab, r, flush=True)
        # an uploaded dataset run becomes the newest run -> dashboard must show the grey UPLOAD banner
        rows = datacentre.generate(dict(datacentre.DEFAULT_GEN, n=600, seed=11))
        up = db.add_dataset('Uploaded: my_tweets.csv', 'upload', rows, {'filename': 'my_tweets.csv'})
        server.do_run(db, up, {'name': 'Uploaded data run', 'topic_algo': 'kmeans', 'k': 5})
        errs = shot(chrome, base + '/#dashboard', os.path.join(OUT, 'dashboard_uploaded_desktop.png'), 1280, 1100)
        problems += ['uploaded dashboard console: ' + e for e in errs]
        httpd.shutdown(); httpd.server_close(); db.close()
        # empty database: every analysis tab must show its empty state, not an error
        httpd, db, base = serve(tmp, 'empty.db', demo=False)
        for tab in ('dashboard', 'trends', 'history'):
            errs = shot(chrome, '%s/#%s' % (base, tab), os.path.join(OUT, 'empty_%s_desktop.png' % tab), 1280, 700)
            problems += ['empty %s console: %s' % (tab, e) for e in errs]
        httpd.shutdown(); httpd.server_close(); db.close()
    shots = sorted(f for f in os.listdir(OUT) if f.endswith('.png'))
    small = [f for f in shots if os.path.getsize(os.path.join(OUT, f)) < 5000]
    problems += ['blank-looking screenshot: ' + f for f in small]
    print('\n%d screenshots written to %s (%d files)' % (n + 4, OUT, len(shots)))
    print('FAIL' if problems else 'PASS', *problems, sep='\n  ')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
