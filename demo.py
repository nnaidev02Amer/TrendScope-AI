#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
# One-command demonstration of the trending topics pipeline on SYNTHETIC tweets.
# No Twitter/X credentials are needed.
#
#   python demo.py              generate data, analyse it, build the dashboard, serve it on localhost
#   python demo.py --no-serve   do everything except starting the web server
#   python demo.py --port 9000  choose the port (default 8765; the next free port is used if busy)
#
# Steps: 1. synthetic tweets -> demo/tweets_synthetic.json   (fixed random seed: reproducible)
#        2. python main.py -a demo/tweets_synthetic.json demo/out   (the unchanged analysis pipeline)
#        3. python dashboard.py demo/out                            (builds demo/out/dashboard.html)
#        4. serve demo/out on http://localhost:<port>/dashboard.html

import argparse
import functools
import http.server
import json
import os
import random
import socket
import subprocess
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, 'demo')
OUT_DIR = os.path.join(DEMO_DIR, 'out')
TWEETS = os.path.join(DEMO_DIR, 'tweets_synthetic.json')

SUBJECTS = ['London transport strike', 'Plymouth university campus', 'climate summit', 'football derby', 'new phone launch']
TEMPLATES = {
    'positive': ['Absolutely love the new {x}, great work and amazing results!',
                 'So happy about the {x} announcement, wonderful news today',
                 '{x} is fantastic, best thing I have seen this year'],
    'negative': ['Terrible {x} experience, awful and disappointing',
                 'I hate the {x} situation, this is a disaster and so sad',
                 '{x} is horrible, worst decision ever, angry and upset'],
    'neutral': ['The {x} meeting is scheduled for Tuesday at noon',
                'Reading a report about the {x} release this afternoon',
                '{x} update published, details in the document'],
}


def generateTweets(path, count = 400, seed = 42):
    """Write `count` fake tweets in the structure main.py expects (one JSON object per line)."""
    random.seed(seed)
    t0 = 1_700_000_000_000
    with open(path, 'w', encoding = 'utf-8') as f:
        for i in range(count):
            text = random.choice(TEMPLATES[random.choice(list(TEMPLATES))]).format(x = SUBJECTS[i % len(SUBJECTS)])
            if random.random() < 0.5:
                text += random.choice([' #news', ' @bbc', ' today', ' again', ' really'])
            tweet = {
                'id': 1000 + i, 'id_str': str(1000 + i), 'lang': 'en',
                'timestamp_ms': str(t0 + i * 3000),
                'entities': {'urls': [{'url': 'https://t.co/abc%d' % i}] if i % 7 == 0 else []},
                'user': {'id': 50 + i % 40, 'followers_count': random.randint(10, 50000)},
                'retweet_count': random.randint(0, 200), 'favorite_count': random.randint(0, 500),
            }
            if i % 7 == 0:
                text += ' https://t.co/abc%d' % i
            if i % 10 == 0:
                text = 'RT @bbc: ' + text
                tweet['retweeted_status'] = {'id': 9000 + i % 3, 'user': {'followers_count': 100000}}
            tweet['text'] = text
            f.write(json.dumps(tweet) + '\n')


def run(cmd):
    env = dict(os.environ, PYTHONIOENCODING = 'utf-8')
    res = subprocess.run(cmd, cwd = HERE, env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, encoding = 'utf-8', errors = 'replace')
    if res.returncode != 0:
        print(res.stdout)
        sys.exit('Command failed: ' + ' '.join(cmd))
    return res.stdout


def freePort(start):
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    sys.exit('No free port found')


def main():
    ap = argparse.ArgumentParser(description = 'Synthetic-data demo of the trending topics project.')
    ap.add_argument('--no-serve', action = 'store_true', help = 'build everything but do not start the web server')
    ap.add_argument('--no-browser', action = 'store_true', help = 'do not open a browser tab')
    ap.add_argument('--port', type = int, default = 8765)
    ap.add_argument('--tweets', type = int, default = 400, help = 'number of synthetic tweets (default 400)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok = True)
    print('[1/3] Generating %d synthetic tweets (seed 42)...' % args.tweets)
    generateTweets(TWEETS, args.tweets)

    print('[2/3] Running the analysis pipeline (main.py -a)...')
    run([sys.executable, 'main.py', '-a', TWEETS, OUT_DIR])
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding = 'utf-8') as f:
        json.dump({'source': 'synthetic', 'tweets': args.tweets, 'seed': 42}, f)

    print('[3/3] Building the dashboard...')
    print(run([sys.executable, 'dashboard.py', OUT_DIR]).strip())

    if args.no_serve:
        print('Open', os.path.join(OUT_DIR, 'dashboard.html'))
        return
    port = freePort(args.port)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory = OUT_DIR)
    url = 'http://localhost:%d/dashboard.html' % port
    print('\nDashboard running at ' + url + '   (Ctrl+C to stop)')
    if not args.no_browser:
        webbrowser.open(url)
    try:
        with http.server.ThreadingHTTPServer(('127.0.0.1', port), handler) as srv:
            srv.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
