"""Browser test: drives the real TrendScope UI in headless Chrome (clicks, forms, uploads, exports).

    .venv\Scripts\python tests\run_ui_tests.py

Starts its own server with a throw-away database (the demo is seeded first), so your data is untouched.
Needs Chrome or Edge; set the CHROME environment variable if it is installed in an unusual place.
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
from trendscope import server  # noqa: E402

CANDIDATES = [os.environ.get('CHROME', ''), r'C:\Program Files\Google\Chrome\Application\chrome.exe',
              r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe', r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
              shutil.which('chrome') or '', shutil.which('google-chrome') or '', shutil.which('chromium') or '']


def main():
    chrome = next((c for c in CANDIDATES if c and os.path.isfile(c)), None)
    if not chrome:
        sys.exit('Chrome/Edge not found. Set the CHROME environment variable.')
    with tempfile.TemporaryDirectory() as tmp:
        server.Handler.ui_dir = os.path.join(ROOT, 'tests', 'browser')
        httpd, db = server.make_server(os.path.join(tmp, 'ui.db'), '127.0.0.1', 0)
        print('Seeding demo data...', flush=True)
        server.ensure_demo(db)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print('Running UI tests in headless browser (about a minute)...', flush=True)
        out = subprocess.run([chrome, '--headless=new', '--disable-gpu', '--window-size=1280,900', '--virtual-time-budget=900000',
                              '--dump-dom', 'http://127.0.0.1:%d/_ui/uitest.html' % port], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900).stdout
        httpd.shutdown()
        httpd.server_close()
        db.close()
    m = re.search(r'<pre id="T">(.*?)</pre>', out, re.S)
    if not m or 'DONE' not in m.group(1):
        print(out[-1500:])
        sys.exit('UI test did not finish.')
    print(m.group(1))
    lines = m.group(1).splitlines()
    bad = [l for l in lines if l.startswith(('FAIL', 'ERROR'))]
    print('\n%d passed, %d failed' % (sum(l.startswith('PASS') for l in lines), len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
