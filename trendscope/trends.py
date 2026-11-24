"""Trend intelligence on per-topic time series (tweet counts per time bin).

Definitions (all computed from the data, nothing hard-coded per topic):

 burst z-score   (mean of the last `w` bins - baseline mean) / max(baseline std, sqrt(baseline mean)).
                 The Poisson floor sqrt(mean) stops tiny baselines from producing absurd z values.
                 Baseline = all earlier bins. A *burst* is a bin whose z (against its own trailing baseline) >= z_threshold.
 p-value         Poisson tail P(X >= observed recent count | baseline rate), a second, distribution-based view.
 growth rate     (sum of last w bins - sum of the w bins before) / max(previous sum, 1).
 trend growth    smoother version over the longer window L = max(6, n/5) bins; catches slow, steady ramps.
 velocity        least-squares slope of the last L bins (tweets per bin, per bin).
 lifecycle       new | rising | peaking | stable | declining | inactive, evaluated at every bin (see classify()).
 trend score     0.5*clip(z,0,10)/10 + 0.3*clip(growth,0,5)/5 + 0.2*share of tweets   (documented, monotone, in [0,1]).
"""
import math


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def poisson_sf(k, lam):
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    lam = max(lam, 1e-9)
    cdf, term = 0.0, math.exp(-lam)
    for i in range(int(k)):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, min(1.0, 1.0 - cdf))


def slope(ys):
    n = len(ys)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, mean(ys)
    den = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den if den else 0.0


def zscore(counts, w=1, min_base=3):
    """z-score of the last w bins against all earlier bins."""
    if len(counts) < w + min_base:
        return 0.0
    base = counts[:-w]
    m, s = mean(base), std(base)
    return (mean(counts[-w:]) - m) / max(s, math.sqrt(max(m, 0.25)))


def burst_series(counts, z_threshold=3.0, w=1):
    """z-score at each bin using only the bins before it, plus merged burst intervals."""
    z = []
    for t in range(len(counts)):
        z.append(round(zscore(counts[:t + 1], w), 3))
    bursts, start = [], None
    for t, v in enumerate(z):
        if v >= z_threshold and start is None:
            start = t
        if v < z_threshold and start is not None:
            bursts.append((start, t - 1))
            start = None
    if start is not None:
        bursts.append((start, len(z) - 1))
    out = []
    for a, b in bursts:
        seg = counts[a:b + 1]
        out.append({'start': a, 'end': b, 'peak': a + seg.index(max(seg)), 'peak_count': max(seg), 'z': max(z[a:b + 1])})
    return z, out


def growth(counts, w=1):
    if len(counts) < 2 * w:
        return 0.0
    recent, prev = sum(counts[-w:]), sum(counts[-2 * w:-w])
    return (recent - prev) / max(prev, 1)


def trend_growth(counts, L=6):
    """Smoother growth: mean of the last L bins vs the L bins before (L = max(6, n/5))."""
    L = max(L, len(counts) // 5)
    if len(counts) < 2 * L:
        return 0.0
    prev = mean(counts[-2 * L:-L])
    return (mean(counts[-L:]) - prev) / max(prev, 0.5)


def velocity(counts, L=6):
    return slope(counts[-L:])


def elevated(counts, L=6, min_base=3, k=2.0, rel=1.3, skip=2):
    """Is the recent level (mean of the last L bins) clearly above the topic's own earlier level?

    The earlier level is the mean of the bins from the topic's first active bin up to the recent window, so the silent bins before
    the topic existed do not count. The first `skip` bins are left out as well: lifecycle() smooths over 3 bins, so those two
    values are pulled down by the silence before the topic appeared, which made flat new topics look like peaks.
    'Clearly' means both statistically (more than k standard errors, Poisson variance ~ level) and materially (at least `rel`
    times the earlier level). With fewer than `min_base` earlier bins there is no evidence of a peak."""
    first = next((i for i, c in enumerate(counts) if c > 0), None)
    if first is None:
        return False
    base = counts[first + skip:-L]
    if len(base) < min_base:
        return False
    b, m = mean(base), mean(counts[-L:])
    se = math.sqrt(max(b, 0.5) / len(base) + max(m, 0.5) / L)
    return m - b > k * se and m >= rel * b


def classify(counts, new_bins=2, L=6):
    """Lifecycle stage using only `counts` (the series up to the current bin).

    L is at least 6 bins and grows with the series (n/5). A slope over the last L bins counts as real only if it is more than 2 standard errors away from zero
    (Poisson noise: variance ~ level), so random wobble in a flat series is not read as growth or decline.

    inactive  no tweets in the last L bins
    new       first seen fewer than `new_bins` bins ago, after at least 2 silent bins (it did not exist before)
    rising    significantly increasing and the current level is >= 50% of the peak
    declining significantly decreasing, or the current level (mean of last 2 bins) is below 50% of the peak by more than 2 Poisson standard deviations of the peak
    peaking   neither rising nor declining, and the level is clearly above the topic's own earlier level (see elevated()): it is at a peak
    stable    neither rising nor declining, and the level is not clearly above its earlier level: flat or low growth (also: not enough history to show a peak)
    """
    if not counts or sum(counts[-L:]) == 0:
        return 'inactive'
    first = next(i for i, c in enumerate(counts) if c > 0)
    t = len(counts) - 1
    if first >= 2 and t - first < new_bins:
        return 'new'
    L = max(L, len(counts) // 5)  # longer series -> longer window, so a slow ramp can still be significant
    recent = counts[-L:]
    peak = max(counts)
    cur = mean(counts[-2:])
    m = mean(recent)
    sxx = sum((i - (len(recent) - 1) / 2) ** 2 for i in range(len(recent))) or 1.0
    se = math.sqrt(max(m, 0.5) / sxx)
    sl = slope(recent)
    if sl > 2 * se and cur >= 0.5 * peak:
        return 'rising'
    # "below half of the peak" only counts if the drop is larger than Poisson noise on the peak level; otherwise a steady
    # low-volume topic (a few tweets per bin) is called declining just because its noisy maximum is twice its average
    if sl < -2 * se or (cur < 0.5 * peak and peak - cur > 2 * math.sqrt(max(peak, 1))):
        return 'declining'
    return 'peaking' if elevated(counts, L) else 'stable'


def lifecycle(counts, smooth=3):
    """Stage at every bin (smoothed counts to avoid flicker from random fluctuation)."""
    sm = [mean(counts[max(0, i - smooth + 1):i + 1]) for i in range(len(counts))]
    stages = [classify(sm[:i + 1]) for i in range(len(sm))]
    first = next((i for i, c in enumerate(counts) if c > 0), None)
    if first is not None:
        stages = ['none'] * first + stages[first:]
    return stages


def transitions(stages):
    out, prev = [], None
    for i, s in enumerate(stages):
        if s != prev and s != 'none':
            out.append({'bin': i, 'stage': s})
        prev = s if s != 'none' else prev
    return out


def trend_score(z, g, share):
    return round(0.5 * min(max(z, 0), 10) / 10 + 0.3 * min(max(g, 0), 5) / 5 + 0.2 * min(max(share, 0), 1), 4)


def analyse(counts, total_tweets, w=1, z_threshold=3.0):
    """All trend metrics for one topic's series."""
    z_all, bursts = burst_series(counts, z_threshold, w)
    stages = lifecycle(counts)
    z = zscore(counts, w)
    g = growth(counts, w)
    base = counts[:-w] if len(counts) > w else counts
    lam = mean(base) * w
    return {
        'z': round(z, 3), 'p_value': round(poisson_sf(sum(counts[-w:]), lam), 6), 'growth': round(g, 3), 'trend_growth': round(trend_growth(counts), 3),
        'velocity': round(velocity(counts), 3), 'z_series': z_all, 'bursts': bursts, 'stages': stages,
        'stage': stages[-1] if stages else 'inactive', 'transitions': transitions(stages),
        'peak_bin': counts.index(max(counts)) if counts else 0, 'peak_count': max(counts) if counts else 0,
        'score': trend_score(z, g, sum(counts) / max(total_tweets, 1)),
    }
