#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
# TrendScope AI launcher.
#   python trendscope.py                 start the app on http://localhost:8765 (seeds a synthetic demo on first run)
#   python trendscope.py --port 9000     choose the port (the next free one is used if busy)
#   python trendscope.py --no-demo       start with an empty database
#   python trendscope.py --no-browser    do not open a browser tab
#
# Data (SQLite) lives in ./trendscope_data/. Live X data is optional: set X_BEARER_TOKEN before starting.

import argparse
import os
import socket
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # the existing pipeline (topicDetection, opinionLexicon) uses paths relative to the project root
sys.path.insert(0, ROOT)


def free_port(start):
    for port in range(start, start + 30):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    sys.exit('No free port found')


def main():
    ap = argparse.ArgumentParser(description='TrendScope AI: social media trend intelligence')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--db', default=os.path.join(ROOT, 'trendscope_data', 'trendscope.db'))
    ap.add_argument('--no-demo', action='store_true')
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    from trendscope import server
    port = free_port(args.port)
    httpd, db = server.make_server(args.db, args.host, port)
    if not args.no_demo:
        print('Preparing the synthetic demo (first run only)...', flush=True)
        server.ensure_demo(db)
    url = 'http://localhost:%d/' % port
    print('TrendScope AI running at %s   (Ctrl+C to stop)' % url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
