"""Optional live retrieval from the official X API v2 (recent search).

Based on docs.x.com (checked 2026-11-20):
  GET https://api.x.com/2/tweets/search/recent      header  Authorization: Bearer <token>
  query (required, <= 512 chars self-serve), max_results 10..100, start_time/end_time (last 7 days only),
  next_token for paging, tweet.fields=created_at,lang,public_metrics,author_id, expansions=author_id,
  user.fields=public_metrics,username. Response: data[], includes.users[], meta.next_token.
  Errors: 400 / 401 / 403 / 429 (x-rate-limit-reset header), JSON body with title/detail.
  The spec names the retweet counter `repost_count`; older payloads say `retweet_count`: both are accepted.

The bearer token is read from the X_BEARER_TOKEN environment variable only. It is never stored, logged,
or accepted from the browser. Network access goes through an injectable `opener` so tests never call X.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = 'https://api.x.com'
PATH = '/2/tweets/search/recent'
TOKEN_ENV = 'X_BEARER_TOKEN'


class XApiError(Exception):
    pass


def token_available():
    return bool(os.environ.get(TOKEN_ENV))


def build_query(keywords=(), hashtags=(), exclude=(), lang=None, exclude_retweets=True):
    """Compose an X search query string from simple lists (keywords are OR-ed)."""
    terms = ['"%s"' % k.replace('"', '') if ' ' in k else k for k in keywords if k.strip()]
    terms += [('#' + h.lstrip('#')) for h in hashtags if h.strip()]
    if not terms:
        raise XApiError('Provide at least one keyword or hashtag.')
    q = '(' + ' OR '.join(terms) + ')'
    for x in exclude:
        if x.strip():
            q += ' -' + ('"%s"' % x if ' ' in x else x)
    if exclude_retweets:
        q += ' -is:retweet'
    if lang:
        q += ' lang:' + lang
    if len(q) > 512:
        raise XApiError('Query is %d characters; the self-serve limit is 512.' % len(q))
    return q


def _get(url, token, opener):
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token, 'User-Agent': 'trendscope-academic-prototype'})
    try:
        with opener(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode('utf-8'))
        except Exception:
            body = {}
        detail = body.get('detail') or body.get('title') or (body.get('errors') or [{}])[0].get('message') or e.reason
        if e.code == 429:
            reset = e.headers.get('x-rate-limit-reset') if e.headers else None
            raise XApiError('Rate limit or usage cap reached (HTTP 429)%s: %s' % (
                ' - resets at %s UTC' % datetime.fromtimestamp(int(reset), timezone.utc).strftime('%H:%M:%S') if reset else '', detail))
        if e.code in (401, 403):
            raise XApiError('X API refused the credentials (HTTP %d): %s' % (e.code, detail))
        raise XApiError('X API error HTTP %d: %s' % (e.code, detail))
    except urllib.error.URLError as e:
        raise XApiError('Could not reach the X API: %s' % e.reason)


def search_recent(query, max_results=100, pages=1, token=None, opener=urllib.request.urlopen):
    """Return a list of raw page payloads (dicts). Only the last 7 days are searchable."""
    token = token or os.environ.get(TOKEN_ENV)
    if not token:
        raise XApiError('No bearer token: set the %s environment variable before starting the server.' % TOKEN_ENV)
    pages_out, next_token = [], None
    for _ in range(max(1, min(int(pages), 5))):
        params = {'query': query, 'max_results': max(10, min(int(max_results), 100)),
                  'tweet.fields': 'created_at,lang,public_metrics,author_id', 'expansions': 'author_id',
                  'user.fields': 'public_metrics,username'}
        if next_token:
            params['next_token'] = next_token
        payload = _get(BASE + PATH + '?' + urllib.parse.urlencode(params), token, opener)
        pages_out.append(payload)
        next_token = (payload.get('meta') or {}).get('next_token')
        if not next_token:
            break
    return pages_out


def to_rows(pages):
    """Convert raw X API v2 pages into normalised TrendScope rows."""
    from .datacentre import parse_ts
    rows = []
    for p in pages:
        users = {u['id']: u for u in (p.get('includes') or {}).get('users', [])}
        for t in p.get('data') or []:
            m = t.get('public_metrics') or {}
            u = users.get(t.get('author_id'), {})
            rows.append({
                'id': str(t.get('id')), 'text': t.get('text', ''), 'ts': parse_ts(t.get('created_at')),
                'user': u.get('username', str(t.get('author_id', ''))),
                'followers': int((u.get('public_metrics') or {}).get('followers_count', 0)),
                'retweets': int(m.get('retweet_count', m.get('repost_count', 0))), 'likes': int(m.get('like_count', 0)),
                'lang': t.get('lang', ''), 'label_sentiment': None, 'label_topic': None})
    return rows
