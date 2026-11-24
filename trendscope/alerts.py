"""Configurable alert rules, evaluated against an analysis result after every run."""
import math

RULE_TYPES = {
    'burst': {'label': 'Statistical burst: topic z-score at or above', 'params': [
        {'key': 'z', 'label': 'z-score', 'default': 4.0, 'step': 0.5, 'min': 0, 'max': 100}, {'key': 'min_size', 'label': 'min topic size', 'default': 15, 'step': 1, 'min': 0, 'max': 1e9}]},
    'negative_share': {'label': 'Negative share of a topic at or above', 'params': [
        {'key': 'share', 'label': 'share (0-1)', 'default': 0.6, 'step': 0.05, 'min': 0, 'max': 1}, {'key': 'min_size', 'label': 'min topic size', 'default': 30, 'step': 1, 'min': 0, 'max': 1e9}]},
    'sentiment_drop': {'label': 'Average topic sentiment at or below', 'params': [
        {'key': 'avg', 'label': 'avg sentiment (-1..1)', 'default': -0.4, 'step': 0.05, 'min': -1, 'max': 1}, {'key': 'min_size', 'label': 'min topic size', 'default': 30, 'step': 1, 'min': 0, 'max': 1e9}]},
    'volume': {'label': 'Topic volume at or above', 'params': [
        {'key': 'size', 'label': 'tweets', 'default': 300, 'step': 10, 'min': 1, 'max': 1e9}]},
    'emerging_count': {'label': 'Number of emerging trends at or above', 'params': [
        {'key': 'count', 'label': 'topics', 'default': 1, 'step': 1, 'min': 1, 'max': 1e6}]},
    'overall_negative': {'label': 'Overall negative share at or above', 'params': [
        {'key': 'share', 'label': 'share (0-1)', 'default': 0.5, 'step': 0.05, 'min': 0, 'max': 1}]},
}


def clean_params(typ, params):
    """Validate a rule's parameters against RULE_TYPES: known keys only, finite numbers inside the allowed range; missing keys get the default."""
    if typ not in RULE_TYPES:
        raise ValueError('Unknown rule type')
    spec = {p['key']: p for p in RULE_TYPES[typ]['params']}
    params = params or {}
    if not isinstance(params, dict):
        raise ValueError('Rule parameters must be an object')
    unknown = sorted(set(params) - set(spec))
    if unknown:
        raise ValueError('Unknown parameter(s) for this rule: %s (allowed: %s)' % (', '.join(unknown), ', '.join(spec)))
    out = {}
    for key, p in spec.items():
        v = params.get(key, p['default'])
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError('"%s" must be a number' % p['label'])
        if not p['min'] <= v <= p['max']:
            raise ValueError('"%s" must be between %g and %g' % (p['label'], p['min'], p['max']))
        out[key] = v
    return out


def evaluate(rules, result):
    """Alert events for one run. A rule that cannot be evaluated (for example a corrupt stored rule) produces an explicit event
    instead of aborting the run or being skipped silently."""
    events = []
    topics = result['topics']
    k = result['kpis']
    for r in rules:
        if not r.get('enabled', True):
            continue
        try:
            events.extend(_evaluate_rule(r, topics, k))
        except (TypeError, KeyError, ValueError, ZeroDivisionError) as e:
            events.append({'rule_id': r.get('id'), 'rule_name': r.get('name', '?'), 'severity': 'low',
                           'message': 'This rule could not be evaluated (%s: %s). Edit or delete it in the Alerts tab.' % (type(e).__name__, e)})
    return events


def _evaluate_rule(r, topics, k):
    events = []
    p, typ = r['params'], r['type']

    def ev(sev, msg):
        events.append({'rule_id': r.get('id'), 'rule_name': r['name'], 'severity': sev, 'message': msg})
    if typ == 'burst':
        for t in topics:
            if t['z'] >= p.get('z', 4) and t['size'] >= p.get('min_size', 0):
                ev('high', 'Topic "%s": burst z=%.1f (p=%.4f), %d tweets' % (t['label'], t['z'], t['p_value'], t['size']))
    elif typ == 'negative_share':
        for t in topics:
            s = t['sentiment']
            share = s['negative'] / t['size']
            if share >= p.get('share', 0.6) and t['size'] >= p.get('min_size', 0):
                ev('medium', 'Topic "%s": %.0f%% negative (%d of %d tweets)' % (t['label'], share * 100, s['negative'], t['size']))
    elif typ == 'sentiment_drop':
        for t in topics:
            if t['sentiment']['avg'] <= p.get('avg', -0.4) and t['size'] >= p.get('min_size', 0):
                ev('medium', 'Topic "%s": average sentiment %.2f' % (t['label'], t['sentiment']['avg']))
    elif typ == 'volume':
        for t in topics:
            if t['size'] >= p.get('size', 300):
                ev('low', 'Topic "%s": %d tweets' % (t['label'], t['size']))
    elif typ == 'emerging_count':
        names = [t['label'] for t in topics if t['emerging']]
        if len(names) >= p.get('count', 1):
            ev('medium', '%d emerging trend(s): %s' % (len(names), '; '.join(names)))
    elif typ == 'overall_negative':
        share = k['negative'] / max(k['tweets'], 1)
        if share >= p.get('share', 0.5):
            ev('medium', 'Overall negative share %.0f%%' % (share * 100))
    else:
        raise ValueError('unknown rule type %s' % typ)
    return events
