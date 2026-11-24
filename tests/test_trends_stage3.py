"""Stage 3 verification of trend detection and topic lifecycle, on controlled data with known answers.

    .venv\\Scripts\\python -W ignore -m unittest tests.test_trends_stage3 -v      (or: unittest discover -s tests)

Every expected value is either computed by hand in the test (independent of trendscope.trends) or follows from how the
synthetic data was generated (a topic generated as 'burst at 50% of the period' must be found bursting at about 50%).
No network, no credentials, no browser.
"""
import math
import os
import random
import sys
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from trendscope import datacentre as dc, engine, topics, trends  # noqa: E402

DS = {'id': 1, 'name': 'stage3', 'source': 'synthetic'}


def poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def topic(name, words, shape='steady', **kw):
    d = {'name': name, 'keywords': words, 'hashtags': [], 'weight': 1, 'shape': shape, 'sentiment': [.3, .3, .4]}
    d.update(kw)
    return d


TEA = topic('tea', ['tea', 'kettle', 'biscuit', 'mug', 'brew', 'teapot'])
CARS = topic('cars', ['engine', 'garage', 'tyres', 'petrol', 'motor', 'gearbox'])


def gen(topics_, n=3000, days=6, seed=1, start='2025-03-03'):
    return dc.generate({'n': n, 'days': days, 'seed': seed, 'noise': 0.0, 'retweet_fraction': 0.0, 'difficulty': 0.0,
                        'languages': {'en': 1.0}, 'start': start, 'topics': topics_})


def run(rows, **cfg):
    base = {'topic_algo': 'kmeans', 'min_size': 20, 'drop_duplicates': False, 'max_tweets': 20000, 'emotions': False}
    base.update(cfg)
    return engine.run_analysis(rows, base, DS)[0]


# ----------------------------------------------------------------------------------------------- 1-3: maths, bursts
class TestTrendMaths(unittest.TestCase):
    def test_zscore_hand_computed(self):
        c = [4, 6, 5, 5, 4, 6, 5, 5]           # baseline mean 5, sample variance 4/7, std 0.756 < sqrt(5)=2.236 (Poisson floor wins)
        self.assertAlmostEqual(trends.zscore(c + [20]), (20 - 5) / math.sqrt(5), places=9)
        self.assertAlmostEqual(trends.zscore(c + [20, 22], w=2), (21 - 5) / math.sqrt(5), places=9)
        self.assertAlmostEqual(trends.zscore(c + [2]), (2 - 5) / math.sqrt(5), places=9)   # drops are negative
        self.assertEqual(trends.zscore([5, 5, 5], w=1), 0.0)                              # not enough baseline -> 0, never an error

    def test_zscore_uses_sample_std_when_it_exceeds_the_poisson_floor(self):
        base = [1, 30, 2, 28, 3, 31, 1, 29]
        m = sum(base) / 8
        s = math.sqrt(sum((x - m) ** 2 for x in base) / 7)
        self.assertGreater(s, math.sqrt(m))
        self.assertAlmostEqual(trends.zscore(base + [40]), (40 - m) / s, places=9)

    def test_growth_velocity_trend_growth_hand_computed(self):
        self.assertAlmostEqual(trends.growth([10, 20]), 1.0)                     # +100%
        self.assertAlmostEqual(trends.growth([10, 5]), -0.5)
        self.assertAlmostEqual(trends.growth([0, 7]), 7.0)                       # previous 0 -> divided by 1, never by zero
        self.assertAlmostEqual(trends.growth([2, 2, 4, 6], w=2), (10 - 4) / 4)
        ramp = list(range(1, 13))
        self.assertAlmostEqual(trends.velocity(ramp), 1.0)                       # +1 tweet per bin
        self.assertAlmostEqual(trends.velocity(ramp[::-1]), -1.0)
        self.assertAlmostEqual(trends.velocity([5] * 10), 0.0)
        c = [4] * 14 + [10] * 6                                                  # n=20 -> L=6: mean(last 6)=(4*0+...)
        L = 6
        self.assertAlmostEqual(trends.trend_growth(c), (sum(c[-L:]) / L - sum(c[-2 * L:-L]) / L) / (sum(c[-2 * L:-L]) / L))

    def test_poisson_tail_matches_direct_summation(self):
        for lam in (0.5, 3.0, 20.0):
            for k in (1, 4, 9, 25):
                direct = 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
                self.assertAlmostEqual(trends.poisson_sf(k, lam), max(0.0, direct), places=9)

    def test_p_value_in_bundle_is_the_poisson_tail_of_the_recent_window(self):
        c = [3, 4, 3, 5, 4, 3, 4, 3, 4, 20, 22]
        m = trends.analyse(c, 100, w=2, z_threshold=3)
        lam = sum(c[:-2]) / 9 * 2
        self.assertAlmostEqual(m['p_value'], round(trends.poisson_sf(42, lam), 6), places=6)
        self.assertLess(m['p_value'], 1e-6)

    def test_burst_threshold_is_monotone(self):
        rng = random.Random(4)
        c = [poisson(rng, 8) for _ in range(40)]
        c[25] += 30
        c[30] += 12
        c[33] += 6
        counts = [len(trends.burst_series(c, z)[1]) for z in (1, 2, 3, 5, 8, 20)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertGreater(counts[0], counts[-1])
        self.assertEqual(counts[-1], 0)

    def test_burst_detection_has_no_lookahead(self):
        rng = random.Random(9)
        c = [poisson(rng, 10) for _ in range(30)]
        z_short, _ = trends.burst_series(c[:20], 3)
        z_long, _ = trends.burst_series(c + [90, 80], 3)
        self.assertEqual(z_short, z_long[:20])        # later data never changes an earlier bin's z-score

    def test_false_positive_rate_on_pure_noise_is_low(self):
        for lam in (3, 20):
            flagged = total = 0
            for seed in range(120):
                rng = random.Random(seed)
                c = [poisson(rng, lam) for _ in range(60)]
                z, _ = trends.burst_series(c, 3.0)
                flagged += sum(1 for v in z[10:] if v >= 3.0)
                total += len(z[10:])
            self.assertLess(flagged / total, 0.012, 'lambda=%d: %d/%d bins flagged' % (lam, flagged, total))

    def test_injected_spikes_are_always_found(self):
        hits = 0
        for seed in range(60):
            rng = random.Random(seed)
            c = [poisson(rng, 12) for _ in range(50)]
            c[35] += 40
            _, b = trends.burst_series(c, 3.0)
            hits += any(x['start'] <= 35 <= x['end'] for x in b)
        self.assertEqual(hits, 60)

    def test_burst_interval_merges_consecutive_bins_and_reports_peak(self):
        c = [5, 4, 6, 5, 5, 4, 6, 5, 30, 40, 35, 5, 5, 4]
        _, b = trends.burst_series(c, 3.0)
        self.assertEqual(len(b), 1)
        self.assertEqual((b[0]['start'], b[0]['peak'], b[0]['peak_count']), (8, 9, 40))
        self.assertLessEqual(b[0]['end'], 10)


# ----------------------------------------------------------------------------------------------- 4-5: lifecycle
class TestLifecycle(unittest.TestCase):
    def test_canonical_shapes(self):
        self.assertEqual(trends.classify([0] * 8), 'inactive')
        self.assertEqual(trends.classify([0, 0, 0, 0, 0, 0, 0, 4]), 'new')
        self.assertEqual(trends.classify([0, 0, 0, 0, 0, 0, 4, 6]), 'new')
        self.assertEqual(trends.classify([3, 5, 8, 12, 18, 26, 38, 55]), 'rising')
        self.assertEqual(trends.classify([60, 55, 38, 26, 18, 12, 8, 5]), 'declining')
        self.assertEqual(trends.classify([5, 20, 40, 38, 25, 15, 9, 6, 4]), 'declining')   # spike long over
        self.assertEqual(trends.classify([10] * 10 + [40, 42, 41, 40, 39, 41, 40, 40]), 'peaking')   # high plateau well above its earlier level
        self.assertEqual(trends.classify([10, 12, 11, 40, 42, 41, 40, 39]), 'rising')       # the jump is still inside the 6-bin window

    def test_flat_noisy_series_is_not_mistaken_for_a_trend(self):
        wrong = 0
        for seed in range(100):
            rng = random.Random(seed)
            c = [poisson(rng, 20) for _ in range(48)]
            wrong += trends.lifecycle(c)[-1] != 'stable'
        self.assertLessEqual(wrong, 6, '%d/100 steady series got a rising/declining/peaking label' % wrong)

    def test_noisy_growth_and_decay_are_recognised(self):
        """+8% per bin (4 -> 80 over 40 bins) with Poisson noise. Slower ramps are deliberately NOT called (see the next test)."""
        up = down = 0
        for seed in range(100):
            rng = random.Random(seed)
            growth = [poisson(rng, 4 * 1.08 ** i) for i in range(40)]
            decay = [poisson(rng, 4 * 1.08 ** (39 - i)) for i in range(40)]
            up += trends.lifecycle(growth)[-1] == 'rising'
            down += trends.lifecycle(decay)[-1] == 'declining'
        self.assertGreaterEqual(up, 90, up)
        self.assertGreaterEqual(down, 90, down)

    def test_slow_ramp_is_conservatively_not_called_rising(self):
        """Documented limit: the rule needs the last-L-bins slope to be > 2 standard errors from zero, so a ramp of about
        +2.5% per bin on Poisson noise is mostly reported as 'peaking' (it ends at its highest level) rather than 'rising'.
        It is never reported as 'declining'."""
        labels = Counter()
        for seed in range(100):
            rng = random.Random(seed)
            labels[trends.lifecycle([poisson(rng, 4 + 1.2 * i) for i in range(40)])[-1]] += 1
        self.assertLessEqual(labels['declining'], 2, labels)
        self.assertGreater(labels['peaking'], labels['rising'], labels)

    def test_full_cycle_moves_through_the_stages_in_order(self):
        c = [0] * 5 + [2, 4, 8, 16, 30, 44, 46, 45, 30, 16, 8, 4, 2]
        st = trends.lifecycle(c)
        self.assertEqual(st[:5], ['none'] * 5)
        seq = [t['stage'] for t in trends.transitions(st)]
        self.assertEqual(seq[0], 'new')
        self.assertLess(seq.index('rising'), seq.index('declining'))
        self.assertEqual(seq[-1], 'declining')
        self.assertTrue(all(t['bin'] >= 5 for t in trends.transitions(st)))

    def test_lifecycle_never_uses_future_data(self):
        rng = random.Random(2)
        c = [0, 0, 0] + [poisson(rng, 3 + i) for i in range(25)] + [poisson(rng, 9) for _ in range(12)]
        full = trends.lifecycle(c)
        for k in (10, 18, 27, 36):
            self.assertEqual(trends.lifecycle(c[:k])[-1], full[k - 1], 'stage at bin %d depends on later bins' % (k - 1))

    def test_stage_order_over_two_time_windows(self):
        """The same topic observed over a longer window must be consistent with its earlier window."""
        c = [0] * 4 + [3, 5, 9, 15, 24, 36, 35, 37, 36, 20, 11, 6, 3]
        first = trends.lifecycle(c[:11])
        longer = trends.lifecycle(c)
        self.assertEqual(longer[:11], first)
        self.assertEqual(first[-1], 'rising')
        self.assertEqual(longer[-1], 'declining')


# ----------------------------------------------------------------------------------------------- end-to-end controlled data
class TestEndToEndKnownTopics(unittest.TestCase):
    """One dataset, five topics with known behaviour: the analysis must recover each one's trend character."""

    @classmethod
    def setUpClass(cls):
        words = {
            'steady': ['tea', 'kettle', 'biscuit', 'mug', 'brew', 'teapot'],
            'surge': ['solar', 'panel', 'battery', 'inverter', 'rooftop', 'grid'],
            'fading': ['vinyl', 'cassette', 'stereo', 'turntable', 'tape', 'walkman'],
            'spike_mid': ['marathon', 'runners', 'finish', 'medal', 'course', 'pacer'],
            'spike_end': ['earthquake', 'tremor', 'rescue', 'aftershock', 'magnitude', 'rubble'],
        }
        cls.words = words
        tp = [topic('steady', words['steady']), topic('surge', words['surge'], 'emerging'), topic('fading', words['fading'], 'declining'),
              topic('spike_mid', words['spike_mid'], 'burst', at=0.45, width=0.03, magnitude=25),
              topic('spike_end', words['spike_end'], 'burst', at=0.995, width=0.01, magnitude=40)]
        cls.rows = gen(tp, n=4000, days=6, seed=11)
        cls.res = run(cls.rows, k=5, bin_minutes=60)
        votes = {}
        for idx, tid, _l, _s in cls.res['assignments']:
            votes.setdefault(tid, Counter())[cls.rows[idx]['label_topic']] += 1
        cls.purity = {tid: c.most_common(1)[0][1] / sum(c.values()) for tid, c in votes.items() if tid}
        truth = {tid: c.most_common(1)[0][0] for tid, c in votes.items() if tid}
        cls.by = {truth[t['id']]: t for t in cls.res['topics'] if t['id'] in truth}

    def test_clusters_are_clean_enough_to_judge(self):
        self.assertEqual(set(self.by), {'steady', 'surge', 'fading', 'spike_mid', 'spike_end'})
        self.assertTrue(all(p > 0.95 for p in self.purity.values()), self.purity)

    def test_steady_topic_has_no_trend(self):
        t = self.by['steady']
        self.assertEqual(t['stage'], 'stable')
        self.assertFalse(t['emerging'])
        self.assertFalse(t['bursts'])
        self.assertLess(abs(t['growth']), 0.6)
        self.assertLess(abs(t['z']), 2.5)

    def test_emerging_topic_is_rising_and_flagged(self):
        t = self.by['surge']
        self.assertEqual(t['stage'], 'rising')
        self.assertTrue(t['emerging'])
        self.assertGreater(t['trend_growth'], 0.5)
        self.assertGreater(t['velocity'], 0)

    def test_declining_topic(self):
        t = self.by['fading']
        self.assertEqual(t['stage'], 'declining')
        self.assertFalse(t['emerging'])
        self.assertLess(t['velocity'], 0)
        self.assertLess(t['trend_growth'], 0)

    def test_mid_period_spike_is_located_and_is_over(self):
        t = self.by['spike_mid']
        n = self.res['bins']['n']
        self.assertTrue(t['bursts'])
        peak = max(t['bursts'], key=lambda b: b['peak_count'])
        self.assertAlmostEqual((peak['peak'] + 0.5) / n, 0.45, delta=0.04)        # generated at 45% of the period
        self.assertEqual(t['stage'], 'declining')                                  # activity has returned to baseline
        self.assertFalse(t['emerging'])
        self.assertLess(t['z'], 1.0)
        self.assertIn('declining', t['stages'][-5:])

    def test_end_of_period_spike_is_bursting_now_and_ranked_first(self):
        t = self.by['spike_end']
        self.assertGreaterEqual(t['z'], 3.0)
        self.assertLess(t['p_value'], 0.001)
        self.assertTrue(t['emerging'])
        self.assertIn(t['stage'], ('rising', 'new'))
        self.assertEqual(t['rank'], 1)
        self.assertGreaterEqual(self.res['kpis']['bursting'], 1)
        self.assertIn(t['id'], [x['id'] for x in self.res['topics'] if x['emerging']])

    def test_ranking_follows_the_documented_score(self):
        T = self.res['topics']
        for t in T:
            exp = 0.5 * min(max(t['z'], 0), 10) / 10 + 0.3 * min(max(t['growth'], 0), 5) / 5 + 0.2 * t['share']
            self.assertAlmostEqual(t['score'], round(exp, 4), places=3)
        self.assertEqual([t['rank'] for t in T], list(range(1, len(T) + 1)))
        self.assertEqual([t['score'] for t in T], sorted([t['score'] for t in T], reverse=True))

    def test_emerging_flag_and_kpis_follow_their_definition(self):
        cfg = self.res['meta']['config']
        for t in self.res['topics']:
            want = t['stage'] in ('new', 'rising') and (t['z'] >= cfg['burst_z'] or t['growth'] >= 1.0 or t['trend_growth'] >= 0.5) and t['size'] >= cfg['min_size']
            self.assertEqual(t['emerging'], want, t['label'])
        k = self.res['kpis']
        self.assertEqual(k['emerging'], sum(t['emerging'] for t in self.res['topics']))
        self.assertEqual(k['bursting'], sum(t['z'] >= cfg['burst_z'] for t in self.res['topics']))

    def test_series_equal_an_independent_recount_from_the_raw_tweets(self):
        b = self.res['bins']
        hist = {}
        total = [0] * b['n']
        for idx, tid, _l, _s in self.res['assignments']:
            k = min(b['n'] - 1, int((self.rows[idx]['ts'] - b['t0']) // b['seconds']))
            hist.setdefault(tid, [0] * b['n'])[k] += 1
            total[k] += 1
        self.assertEqual(self.res['volume'], total)
        for t in self.res['topics']:
            self.assertEqual(t['series'], hist[t['id']], t['label'])

    def test_derived_metrics_match_independent_formulas(self):
        w = self.res['meta']['config']['recent_bins']
        for t in self.res['topics']:
            c = t['series']
            base = c[:-w]
            m = sum(base) / len(base)
            s = math.sqrt(sum((x - m) ** 2 for x in base) / (len(base) - 1))
            self.assertAlmostEqual(t['z'], (sum(c[-w:]) / w - m) / max(s, math.sqrt(max(m, 0.25))), places=2)
            self.assertAlmostEqual(t['growth'], (sum(c[-w:]) - sum(c[-2 * w:-w])) / max(sum(c[-2 * w:-w]), 1), places=2)
            last = c[-6:]
            mx, my = 2.5, sum(last) / 6
            self.assertAlmostEqual(t['velocity'], sum((i - mx) * (y - my) for i, y in enumerate(last)) / sum((i - mx) ** 2 for i in range(6)), places=2)
            self.assertEqual(t['peak_count'], max(c))
            self.assertEqual(t['peak_bin'], c.index(max(c)))
            self.assertEqual(len(t['stages']), len(c))
            self.assertEqual(len(t['z_series']), len(c))
            self.assertEqual(t['stage'], t['stages'][-1])

    def test_transitions_describe_the_stage_strip(self):
        for t in self.res['topics']:
            rebuilt, prev = [], None
            for i, s in enumerate(t['stages']):
                if s != prev and s != 'none':
                    rebuilt.append({'bin': i, 'stage': s})
                prev = s if s != 'none' else prev
            self.assertEqual(t['transitions'], rebuilt)

    def test_first_and_last_seen_bound_the_series(self):
        b = self.res['bins']
        for t in self.res['topics']:
            first = next(i for i, c in enumerate(t['series']) if c)
            last = max(i for i, c in enumerate(t['series']) if c)
            self.assertEqual(int((t['first_seen'] - b['t0']) // b['seconds']), first)
            self.assertEqual(min(b['n'] - 1, int((t['last_seen'] - b['t0']) // b['seconds'])), last)


# ----------------------------------------------------------------------------------------------- the five lifecycle states
class TestFiveLifecycleStates(unittest.TestCase):
    """new, rising, peaking, stable, declining (plus inactive): controlled series, noisy steady series, and a dataset with one topic per state."""

    def test_canonical_series_for_every_state(self):
        cases = {
            'new': [0, 0, 0, 0, 0, 0, 0, 5],
            'rising': [3, 5, 8, 12, 18, 26, 38, 55],
            'peaking': [10] * 12 + [38, 41, 40, 42, 39, 41, 40, 40],       # now clearly above its own earlier level, not changing
            'stable': [20, 22, 19, 21, 20, 18, 22, 21, 19, 20, 21, 20],     # flat
            'declining': [60, 55, 38, 26, 18, 12, 8, 5],
            'inactive': [4, 3, 0, 0, 0, 0, 0, 0],
        }
        for want, series in cases.items():
            self.assertEqual(trends.classify(series), want, series)

    def test_stable_vs_peaking_boundary(self):
        flat = [20, 21, 19, 20, 22, 20, 19, 21, 20, 20, 21, 19, 20, 20]
        self.assertEqual(trends.classify(flat), 'stable')
        self.assertEqual(trends.classify([20] * 8 + [26] * 6), 'peaking')          # +30%: clearly above (and still not rising: flat inside the window)
        self.assertEqual(trends.classify([20] * 8 + [22] * 6), 'stable')           # +10%: low growth, not a peak
        self.assertEqual(trends.classify([100] * 8 + [124] * 6), 'stable')         # +24% is significant but below the 30% materiality floor
        self.assertEqual(trends.classify([20, 40, 20, 40, 20, 40]), 'stable')      # short history: no evidence of a peak
        self.assertFalse(trends.elevated([0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 30, 30, 30, 30]))   # silence before the topic existed is not a baseline

    def test_topic_that_started_late_and_is_flat_is_stable_not_peaking(self):
        rng = random.Random(0)
        c = [0] * 20 + [poisson(rng, 15) for _ in range(30)]
        self.assertEqual(trends.lifecycle(c)[-1], 'stable')

    def test_steady_series_are_stable_at_every_volume(self):
        """Regression: a steady low-volume topic used to be called 'declining' in ~40% of runs because its noisy maximum
        is about twice its mean; and every flat topic used to be 'peaking'."""
        for lam in (1, 3, 10, 20, 100):
            labels = Counter()
            for seed in range(120):
                rng = random.Random(1000 * lam + seed)
                labels[trends.lifecycle([poisson(rng, lam) for _ in range(48)])[-1]] += 1
            self.assertGreaterEqual(labels['stable'], 108, 'lambda=%d: %s' % (lam, dict(labels)))          # >= 90% stable
            self.assertLessEqual(labels['declining'] + labels['rising'], 9, 'lambda=%d: %s' % (lam, dict(labels)))   # false trends <= 7.5%
            self.assertLessEqual(labels['peaking'], 6, 'lambda=%d: %s' % (lam, dict(labels)))

    def test_real_patterns_are_still_found_in_noise(self):
        def share(make, want, n=100):
            return sum(trends.lifecycle(make(random.Random(s)))[-1] == want for s in range(n)) / n
        self.assertGreaterEqual(share(lambda r: [poisson(r, 4 * 1.08 ** i) for i in range(40)], 'rising'), 0.9)
        self.assertGreaterEqual(share(lambda r: [poisson(r, 4 * 1.08 ** (39 - i)) for i in range(40)], 'declining'), 0.9)
        self.assertGreaterEqual(share(lambda r: [poisson(r, 10) for _ in range(30)] + [poisson(r, 40) for _ in range(18)], 'peaking'), 0.9)
        self.assertGreaterEqual(share(lambda r: [poisson(r, 10) for _ in range(30)] + [poisson(r, 60) for _ in range(3)] + [poisson(r, 10) for _ in range(10)], 'declining'), 0.9)
        self.assertGreaterEqual(share(lambda r: [poisson(r, 20) for _ in range(30)] + [poisson(r, 6) for _ in range(12)], 'declining'), 0.9)

    def test_lifecycle_series_uses_all_five_states_over_a_story(self):
        rng = random.Random(3)
        story = ([0] * 4 + [poisson(rng, 12) for _ in range(20)]                  # appears, then flat -> stable
                 + [poisson(rng, 12 + 6 * i) for i in range(10)]                  # climbs -> rising
                 + [poisson(rng, 70) for _ in range(14)]                          # plateau well above earlier level -> peaking
                 + [poisson(rng, 70 * 0.8 ** i) for i in range(16)])              # decay -> declining
        seen = {s for s in trends.lifecycle(story) if s != 'none'}
        self.assertTrue({'new', 'stable', 'rising', 'peaking', 'declining'} <= seen, seen)
        order = [t['stage'] for t in trends.transitions(trends.lifecycle(story))]
        self.assertEqual(order[0], 'new')
        self.assertEqual(order[-1], 'declining')
        first = {s: order.index(s) for s in ('new', 'stable', 'rising', 'peaking', 'declining')}
        self.assertLess(first['new'], first['stable'])
        self.assertLess(first['stable'], first['peaking'])
        self.assertLess(first['peaking'], first['declining'])
        self.assertLess(order.index('rising', first['stable']), first['peaking'])        # it climbs after its flat phase, before the plateau
        self.assertNotIn('peaking', order[:order.index('stable') + 1])                   # no false peak while a new flat topic warms up

    def test_every_stage_label_is_one_of_the_documented_set(self):
        rng = random.Random(8)
        allowed = {'new', 'rising', 'peaking', 'stable', 'declining', 'inactive', 'none'}
        for _ in range(60):
            c = [poisson(rng, rng.choice([0.3, 2, 15])) for _ in range(rng.randint(3, 60))]
            self.assertTrue(set(trends.lifecycle(c)) <= allowed)


class TestFiveStatesEndToEnd(unittest.TestCase):
    """One dataset with one topic built for each state; the analysis must label each one correctly and the flags must follow."""

    @classmethod
    def setUpClass(cls):
        w = {'stable': ['tea', 'kettle', 'biscuit', 'mug', 'brew', 'teapot'], 'rising': ['solar', 'panel', 'battery', 'inverter', 'rooftop', 'grid'],
             'declining': ['vinyl', 'cassette', 'stereo', 'turntable', 'tape', 'walkman'], 'peaking': ['marathon', 'runners', 'finish', 'medal', 'course', 'pacer']}
        tp = [topic('stable', w['stable']), topic('rising', w['rising'], 'emerging'), topic('declining', w['declining'], 'declining'),
              topic('peaking', w['peaking'], 'burst', at=0.9, width=0.5, magnitude=8)]     # broad plateau at its highest level over the last fifth
        rows = gen(tp, n=4000, days=6, seed=11)
        hi = max(r['ts'] for r in rows)
        rng = random.Random(5)
        words = ['volcano', 'eruption', 'ash', 'lava', 'crater', 'evacuate']
        for i in range(150):   # a topic that exists only in the last two hours
            rows.append(dict(rows[0], id='new%d' % i, ts=hi - 7000 + rng.random() * 7000, text=' '.join(rng.sample(words, 3)) + ' today', label_topic='new', lang='en'))
        rows.sort(key=lambda r: r['ts'])
        cls.rows = rows
        cls.res = run(rows, k=5, bin_minutes=60)
        votes = {}
        for idx, tid, _l, _s in cls.res['assignments']:
            votes.setdefault(tid, Counter())[rows[idx]['label_topic']] += 1
        truth = {tid: c.most_common(1)[0][0] for tid, c in votes.items() if tid}
        cls.purity = {tid: c.most_common(1)[0][1] / sum(c.values()) for tid, c in votes.items() if tid}
        cls.by = {truth[t['id']]: t for t in cls.res['topics'] if t['id'] in truth}

    def test_dataset_separates_into_the_five_known_topics(self):
        self.assertEqual(set(self.by), {'stable', 'rising', 'declining', 'peaking', 'new'})
        self.assertTrue(all(p > 0.95 for p in self.purity.values()), self.purity)

    def test_each_topic_gets_its_state(self):
        got = {k: t['stage'] for k, t in self.by.items()}
        self.assertEqual(got, {'stable': 'stable', 'rising': 'rising', 'declining': 'declining', 'peaking': 'peaking', 'new': 'new'})

    def test_flags_follow_the_states(self):
        self.assertTrue(self.by['rising']['emerging'])
        self.assertFalse(self.by['stable']['emerging'])
        self.assertFalse(self.by['declining']['emerging'])
        self.assertFalse(self.by['peaking']['emerging'])
        self.assertTrue(self.by['new']['z'] > 3 or self.by['new']['growth'] >= 1.0)         # new topics are emerging by definition of the flag
        self.assertTrue(self.by['new']['emerging'])

    def test_strip_of_stages_is_consistent(self):
        for t in self.res['topics']:
            self.assertEqual(t['stage'], t['stages'][-1])
            self.assertTrue(set(t['stages']) <= {'none', 'new', 'rising', 'peaking', 'stable', 'declining', 'inactive'})
        self.assertIn('stable', self.by['stable']['stages'])
        self.assertGreater(self.by['stable']['stages'].count('stable'), 0.7 * len(self.by['stable']['stages']))
        self.assertEqual(self.by['new']['stages'].count('none'), len(self.by['new']['stages']) - 2)   # 'none' before it existed

    def test_old_saved_runs_with_peaking_flat_topics_still_load_and_compare(self):
        """Saved-run compatibility: results stored before the 'stable' state existed only use the older labels; nothing depends on 'stable'."""
        old = {'id': 1, 'name': 'old', 'dataset_name': 'x', 'result': {'kpis': dict(self.res['kpis']), 'topics': [dict(t, stage='peaking' if t['stage'] == 'stable' else t['stage']) for t in self.res['topics']]}}
        new = {'id': 2, 'name': 'new', 'dataset_name': 'x', 'result': self.res}
        cmp = engine.compare_runs(old, new)
        self.assertEqual(len(cmp['matched']), len(self.res['topics']))
        self.assertTrue(any(m['stage_a'] == 'peaking' and m['stage_b'] == 'stable' for m in cmp['matched']))
        from trendscope import reports
        csv_text = reports.topics_csv({'result': old['result']})
        self.assertIn('peaking', csv_text)


# ----------------------------------------------------------------------------------------------- regression: partial last bin
class TestBinAnchoring(unittest.TestCase):
    """Regression: bins used to start at the first tweet, so the last bin could be a sliver and a perfectly
    steady topic showed negative 'recent growth' and a depressed z-score depending on the time span chosen."""

    def test_steady_topics_look_steady_for_any_time_span(self):
        # 7.01 and 7.05 days leave only ~5% and ~37% of a 3-hour bin at the end under start-anchored binning
        for days in (7.0, 7.01, 7.05, 7.3):
            res = run(gen([TEA, CARS], n=2400, days=days, seed=3), k=2, recent_bins=1)
            for t in res['topics']:
                prior = sorted(t['series'][-12:-1])
                median = prior[len(prior) // 2]
                self.assertGreater(t['series'][-1], 0.4 * median, 'days=%s last bin %s vs median %s' % (days, t['series'][-1], median))
                self.assertGreater(t['z'], -1.5, 'days=%s z %.2f' % (days, t['z']))
                self.assertGreater(t['growth'], -0.45, 'days=%s growth %.2f' % (days, t['growth']))
                self.assertFalse(t['emerging'])

    def test_last_bin_ends_at_the_last_tweet(self):
        for days in (7.0, 7.01, 7.3):
            res = run(gen([TEA, CARS], n=1500, days=days, seed=3), k=2)
            b = res['bins']
            self.assertAlmostEqual(res['meta']['date_to'], b['t0'] + b['n'] * b['seconds'], places=3)

    def test_bins_cover_all_tweets_exactly_once(self):
        rows = gen([TEA, CARS], n=1500, days=5.02, seed=5)
        res = run(rows, k=2, bin_minutes=180)
        b = res['bins']
        self.assertEqual(sum(res['volume']), res['kpis']['tweets'])
        self.assertLessEqual(b['t0'], res['meta']['date_from'])
        self.assertGreater(b['t0'] + b['seconds'], res['meta']['date_from'])       # the first bin contains the first tweet
        self.assertEqual(b['n'], math.ceil((res['meta']['date_to'] - res['meta']['date_from']) / b['seconds'] - 1e-9))

    def test_spike_in_the_final_bin_is_detected_whatever_the_span(self):
        tp = [TEA, topic('quake', ['earthquake', 'tremor', 'rescue', 'aftershock', 'magnitude', 'rubble'], 'burst', at=0.997, width=0.004, magnitude=60)]
        for days in (6.0, 6.02, 6.4):
            res = run(gen(tp, n=3000, days=days, seed=8), k=2, bin_minutes=60, recent_bins=1)
            q = next(t for t in res['topics'] if 'earthquake' in [w for w, _ in t['terms']] or 'rescue' in [w for w, _ in t['terms']])
            self.assertGreaterEqual(q['z'], 3.0, 'days=%s z=%.2f' % (days, q['z']))
            self.assertTrue(q['emerging'], 'days=%s' % days)


# ----------------------------------------------------------------------------------------------- 6: matching and evolution between periods
class TestPeriodMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        late = topic('late', ['volcano', 'eruption', 'ash', 'lava', 'crater', 'evacuate'], 'burst', at=0.93, width=0.02, magnitude=400)
        cls.rows = gen([TEA, CARS, late], n=5000, days=8, seed=21, start='2025-03-03')
        cfg = {'k': 2, 'bin_minutes': 120, 'min_size': 25}
        cls.a = run(cls.rows, date_from='2025-03-03', date_to='2025-03-06', **cfg)
        cls.b = run(cls.rows, date_from='2025-03-07', date_to='2025-03-10', **dict(cfg, k=3))
        wrap = lambda r, n: {'id': n, 'name': 'P%d' % n, 'dataset_name': 'x', 'result': r}
        cls.cmp = engine.compare_runs(wrap(cls.a, 1), wrap(cls.b, 2))
        cls.same = engine.compare_runs(wrap(cls.a, 1), wrap(cls.a, 2))

    def words(self, label):
        return set(label.replace(' · ', ' ').split())

    def test_periods_cover_the_right_data(self):
        self.assertLess(self.a['meta']['date_to'], self.b['meta']['date_from'])
        self.assertGreater(self.a['kpis']['tweets'], 500)
        self.assertGreater(self.b['kpis']['tweets'], 500)

    def test_recurring_topics_are_matched_across_periods(self):
        self.assertEqual(len(self.cmp['matched']), 2)
        for m in self.cmp['matched']:
            self.assertGreaterEqual(m['similarity'], 0.5)
            self.assertTrue(self.words(m['a']) & self.words(m['b']), m)       # same keywords describe both
            tea = {'tea', 'kettle', 'biscuit', 'mug', 'brew', 'teapot'}
            cars = {'engine', 'garage', 'tyres', 'petrol', 'motor', 'gearbox'}
            self.assertTrue(self.words(m['a']) <= tea and self.words(m['b']) <= tea or self.words(m['a']) <= cars and self.words(m['b']) <= cars, m)

    def test_topic_that_exists_only_in_the_later_period_is_reported_as_new(self):
        self.assertEqual(len(self.cmp['only_b']), 1)
        self.assertTrue(self.words(self.cmp['only_b'][0]['label']) <= {'volcano', 'eruption', 'ash', 'lava', 'crater', 'evacuate'})
        self.assertEqual(self.cmp['only_a'], [])
        late = next(t for t in self.b['topics'] if t['label'] == self.cmp['only_b'][0]['label'])
        self.assertTrue(late['bursts'])                                    # its eruption burst is found inside period B
        self.assertGreater(max(late['series']), 8 * (sum(late['series'][:len(late['series']) // 2]) / (len(late['series']) // 2) + 1))
        self.assertLess(late['first_seen'], max(b['peak'] * self.b['bins']['seconds'] + self.b['bins']['t0'] for b in late['bursts']))

    def test_kpi_deltas_are_b_minus_a(self):
        for k, v in self.cmp['kpis'].items():
            self.assertAlmostEqual(v['delta'], round(v['b'] - v['a'], 4), places=4)
            self.assertEqual(v['a'], self.a['kpis'][k])
            self.assertEqual(v['b'], self.b['kpis'][k])

    def test_comparing_a_run_with_itself_matches_everything_exactly(self):
        n = len(self.a['topics'])
        self.assertEqual(len(self.same['matched']), n)
        self.assertEqual(self.same['only_a'] + self.same['only_b'], [])
        self.assertTrue(all(m['similarity'] == 1.0 and m['a'] == m['b'] for m in self.same['matched']))
        self.assertTrue(all(v['delta'] == 0 for v in self.same['kpis'].values()))

    def test_each_topic_is_matched_at_most_once(self):
        pairs = [(m['a'], m['b']) for m in self.cmp['matched']]
        self.assertEqual(len({p[0] for p in pairs}), len(pairs))
        self.assertEqual(len({p[1] for p in pairs}), len(pairs))


class TestTopicEvolution(unittest.TestCase):
    def test_window_terms_follow_a_vocabulary_shift(self):
        rows, idx = [], []
        for i in range(120):
            late = i >= 60
            rows.append({'ts': 1000.0 + i * 100, 'text': 'harbour council meeting ' + ('ferry fares protest' if late else 'dredging permit approved')})
            idx.append(i)
        w = topics.window_terms(rows, idx, n_windows=4, n_terms=6)
        self.assertEqual([x['tweets'] for x in w], [30, 30, 30, 30])
        self.assertEqual(sum(x['tweets'] for x in w), 120)
        early, late = set(w[0]['terms']), set(w[3]['terms'])
        self.assertTrue({'dredging', 'permit', 'approved'} <= early and not ({'ferry', 'fares', 'protest'} & early))
        self.assertTrue({'ferry', 'fares', 'protest'} <= late and not ({'dredging', 'permit', 'approved'} & late))
        self.assertTrue(all(w[i]['to'] == w[i + 1]['from'] or abs(w[i]['to'] - w[i + 1]['from']) < 1e-6 for i in range(3)))

    def test_window_terms_edge_cases(self):
        self.assertEqual(topics.window_terms([], []), [])
        one = [{'ts': 5.0, 'text': 'solo tweet here'}]
        w = topics.window_terms(one, [0])
        self.assertEqual(sum(x['tweets'] for x in w), 1)


# ----------------------------------------------------------------------------------------------- 7: comparison / relationships
class TestTopicRelations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(1)
        A = ['rail', 'train', 'strike', 'platform']
        B = ['rail', 'train', 'timetable', 'delay']
        C = ['volcano', 'lava', 'crater', 'ash']
        cls.rows, cls.labels = [], []
        for lab, words in enumerate((A, B, C)):
            for _ in range(40):
                cls.rows.append({'ts': 1.0, 'text': ' '.join(rng.sample(words, 3)) + ' today'})
                cls.labels.append(lab)
        cls.out, cls.rel = topics.describe(cls.rows, cls.labels)
        cls.sim = {(r['a'], r['b']): r for r in cls.rel}

    def test_topics_sharing_vocabulary_are_more_related_than_unrelated_ones(self):
        self.assertGreater(self.sim[(0, 1)]['sim'], 0.2)
        self.assertEqual(self.sim[(0, 2)]['sim'], 0.0)
        self.assertEqual(self.sim[(1, 2)]['sim'], 0.0)
        self.assertGreater(self.sim[(0, 1)]['sim'], self.sim[(0, 2)]['sim'])

    def test_shared_keywords_are_reported(self):
        self.assertIn('rail', self.sim[(0, 1)]['shared'])
        self.assertIn('train', self.sim[(0, 1)]['shared'])
        self.assertEqual(self.sim[(0, 2)]['shared'], [])

    def test_similarity_is_bounded_and_every_pair_listed_once(self):
        self.assertEqual(len(self.rel), 3)
        self.assertTrue(all(0.0 <= r['sim'] <= 1.0 and r['a'] < r['b'] for r in self.rel))

    def test_topic_keywords_come_from_their_own_documents(self):
        self.assertTrue({'volcano', 'lava', 'crater', 'ash'} <= {w for w, _ in self.out[2]['terms']})
        self.assertFalse({'volcano', 'lava'} & {w for w, _ in self.out[0]['terms']})

    def test_related_keywords_use_cooccurrence(self):
        docs = [['solar', 'panel', 'roof']] * 6 + [['solar', 'panel']] * 3 + [['wind', 'turbine']] * 6 + [['solar', 'cat']]
        rel = dict(topics.related_terms('solar', docs))
        self.assertIn('panel', rel)
        self.assertGreater(rel['panel'], rel.get('roof', 0))
        self.assertNotIn('turbine', rel)
        self.assertNotIn('cat', rel)                      # co-occurs only once: below the evidence floor
        self.assertEqual(topics.related_terms(None, docs), [])

    def test_engine_relations_use_one_based_ids_that_match_the_topic_list(self):
        rows = gen([TEA, CARS, topic('rail', ['train', 'station', 'ticket', 'carriage', 'driver', 'signal'])], n=1500, seed=2)
        res = run(rows, k=3)
        ids = {t['id'] for t in res['topics']}
        for r in res['relations']:
            self.assertTrue(r['a'] in ids and r['b'] in ids and r['a'] != r['b'])


if __name__ == '__main__':
    unittest.main()
