'use strict';
/* TrendScope AI front end: plain JavaScript, hand-written SVG charts, no external libraries. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtN = (n, d = 0) => Number(n).toLocaleString('en-GB', {maximumFractionDigits: d});
const pct = (a, b) => b ? 100 * a / b : 0;
const fmtTs = ts => ts ? new Date(ts * 1000).toISOString().slice(0, 16).replace('T', ' ') : '';
const COLORS = ['#2c5aa0', '#e0a030', '#2e8b57', '#c0392b', '#7b5ea7', '#16a3b8', '#d96c9e', '#6b7a2a'];
const SENT = {positive: '#2e8b57', negative: '#c0392b', neutral: '#8a94a0'};
const STAGE = {new: '#7b5ea7', rising: '#2e8b57', peaking: '#e0a030', stable: '#2c7da0', declining: '#c0392b', inactive: '#9aa5b1', none: '#eef0f3'};
const TABS = [['dashboard', 'Dashboard'], ['studio', 'Analysis Studio'], ['data', 'Data Centre'], ['trends', 'Trends'], ['topics', 'Topic Explorer'],
  ['sentiment', 'Sentiment'], ['eval', 'Evaluation Lab'], ['history', 'History & Reports'], ['alerts', 'Alerts']];
const SRC = {synthetic: 'SYNTHETIC DEMONSTRATION DATA: generated tweets, not real Twitter/X content',
  upload: 'USER-UPLOADED DATASET: not verified as real Twitter/X data', live: 'LIVE X API DATA: retrieved with the official X API v2 (recent search)'};
const SRC_SHORT = {synthetic: 'SYNTHETIC', upload: 'UPLOADED', live: 'LIVE X'};

const state = {status: null, meta: null, datasets: [], runs: [], run: null, tab: 'dashboard', topic: null, dash: {sort: 'score', dir: -1},
  compare: new Set(), sel: new Set(), upload: null, genTopics: null, lastAlerts: null, annot: []};

async function api(path, o = {}) {
  const r = await fetch('/api/' + path, {method: o.method || 'GET', headers: {'Content-Type': 'application/json'}, body: o.body ? JSON.stringify(o.body) : undefined});
  const t = await r.text();
  let j = null; try { j = JSON.parse(t); } catch (e) {}
  if (!r.ok) throw new Error((j && j.error) || ('HTTP ' + r.status));
  return j;
}
function toast(msg, type = 'ok') {
  const d = document.createElement('div'); d.className = type; d.textContent = msg; $('#toast').appendChild(d);
  setTimeout(() => d.remove(), type === 'err' ? 9000 : 3500);
}
async function busy(btn, fn) {
  const old = btn.textContent; btn.disabled = true; btn.textContent = 'Working…';
  try { await fn(); } catch (e) { toast(e.message, 'err'); } finally { btn.disabled = false; btn.textContent = old; }
}
const tag = s => `<span class="tag" style="background:${STAGE[s] || '#999'}">${esc(s)}</span>`;
const srcTag = s => `<span class="tag" style="background:${s === 'synthetic' ? '#b8860b' : s === 'live' ? '#2e8b57' : '#52606d'}">${SRC_SHORT[s] || s}</span>`;
const sbar = s => { const n = (s.positive + s.negative + s.neutral) || 1;
  return `<div class="sbar" title="+${s.positive} / -${s.negative} / ~${s.neutral}"><i style="width:${pct(s.positive, n)}%;background:var(--pos)"></i><i style="width:${pct(s.negative, n)}%;background:var(--neg)"></i><i style="width:${pct(s.neutral, n)}%;background:var(--neu)"></i></div>`; };
const legend = items => `<div class="legend">${items.map(([n, c]) => `<span><i style="background:${c}"></i>${esc(n)}</span>`).join('')}</div>`;
const note = (t, c = '') => `<div class="note ${c}">${t}</div>`;
const empty = t => `<div class="empty">${t}</div>`;

/* ---------------------------------------------------------------- charts */
function binLabel(run, i) {
  const b = run.result.bins, d = new Date((b.t0 + i * b.seconds) * 1000);
  return b.seconds >= 86400 ? d.toISOString().slice(5, 10) : d.toISOString().slice(5, 16).replace('T', ' ');
}
function axes(W, h, L, R, T, B, max, labels) {
  let s = '';
  for (let i = 0; i <= 4; i++) { const v = max * i / 4, y = h - B - v / max * (h - B - T);
    s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#e4e7eb"/><text x="${L - 5}" y="${y + 4}" text-anchor="end">${fmtN(v, max < 5 ? 1 : 0)}</text>`; }
  const n = labels.length, step = Math.max(1, Math.ceil(n / 7));
  for (let i = 0; i < n; i += step) { const x = L + (n < 2 ? 0 : i / (n - 1)) * (W - L - R);
    s += `<text x="${x}" y="${h - B + 15}" text-anchor="middle">${esc(labels[i])}</text>`; }
  return s;
}
function lineChart(labels, series, o = {}) {
  const W = 720, h = o.h || 250, L = 42, R = 12, T = 10, B = 30, n = labels.length;
  const max = Math.max(1e-9, ...series.flatMap(s => s.values)) * 1.08;
  const X = i => L + (n < 2 ? 0 : i / (n - 1)) * (W - L - R), Y = v => h - B - v / max * (h - B - T);
  let s = `<svg viewBox="0 0 ${W} ${h}" role="img" aria-label="${esc(o.aria || 'line chart')}">` + axes(W, h, L, R, T, B, max, labels);
  series.forEach(se => {
    s += `<polyline fill="none" stroke="${se.color}" stroke-width="${se.w || 2}" ${se.dash ? 'stroke-dasharray="5 3"' : ''} points="${se.values.map((v, i) => X(i) + ',' + Y(v)).join(' ')}"/>`;
    if (n <= 70) se.values.forEach((v, i) => { s += `<circle cx="${X(i)}" cy="${Y(v)}" r="4" fill="${se.color}" fill-opacity="0"><title>${esc(se.name)} · ${esc(labels[i])}: ${fmtN(v, 2)}</title></circle>`; });
  });
  (o.marks || []).forEach(m => { s += `<rect x="${X(m.a) - 2}" y="${T}" width="${Math.max(4, X(m.b) - X(m.a) + 4)}" height="${h - B - T}" fill="#c0392b" opacity=".12"><title>${esc(m.title || '')}</title></rect>`; });
  return s + '</svg>' + (series.length > 1 || o.legend ? legend(series.map(x => [x.name, x.color])) : '');
}
function stackedBars(labels, series, o = {}) {
  const W = 720, h = o.h || 230, L = 42, R = 12, T = 10, B = 30, n = labels.length;
  const tot = labels.map((_, i) => series.reduce((a, s) => a + s.values[i], 0)), max = Math.max(1, ...tot) * 1.05;
  const bw = (W - L - R) / n, Y = v => h - B - v / max * (h - B - T);
  let s = `<svg viewBox="0 0 ${W} ${h}" role="img" aria-label="${esc(o.aria || 'stacked bar chart')}">` + axes(W, h, L, R, T, B, max, labels);
  labels.forEach((lab, i) => { let acc = 0;
    series.forEach(se => { const v = se.values[i]; if (v) s += `<rect x="${L + i * bw + 0.5}" y="${Y(acc + v)}" width="${Math.max(1, bw - 1)}" height="${Y(acc) - Y(acc + v)}" fill="${se.color}"><title>${esc(lab)} · ${esc(se.name)}: ${v}</title></rect>`; acc += v; }); });
  return s + '</svg>' + legend(series.map(x => [x.name, x.color]));
}
function donut(items) {
  const tot = items.reduce((a, i) => a + i.value, 0) || 1, cx = 100, cy = 100, R = 82, r = 52; let a0 = -Math.PI / 2, s = '<svg viewBox="0 0 420 200" role="img" aria-label="sentiment donut">';
  items.forEach((it, k) => { const a1 = a0 + 2 * Math.PI * it.value / tot - 1e-4, l = a1 - a0 > Math.PI ? 1 : 0, P = (q, a) => [cx + q * Math.cos(a), cy + q * Math.sin(a)];
    const [x0, y0] = P(R, a0), [x1, y1] = P(R, a1), [x2, y2] = P(r, a1), [x3, y3] = P(r, a0);
    if (it.value) s += `<path d="M${x0},${y0}A${R},${R} 0 ${l} 1 ${x1},${y1}L${x2},${y2}A${r},${r} 0 ${l} 0 ${x3},${y3}Z" fill="${it.color}"><title>${esc(it.name)}: ${it.value}</title></path>`;
    a0 += 2 * Math.PI * it.value / tot;
    s += `<rect x="215" y="${58 + k * 34}" width="13" height="13" rx="2" fill="${it.color}"/><text x="236" y="${69 + k * 34}" style="font-size:13px;fill:#1f2933">${esc(it.name)}: ${fmtN(it.value)} (${fmtN(pct(it.value, tot), 1)}%)</text>`; });
  return s + `<text x="${cx}" y="${cy + 2}" text-anchor="middle" style="font-size:22px;font-weight:700;fill:#1f2933">${fmtN(tot)}</text><text x="${cx}" y="${cy + 18}" text-anchor="middle">tweets</text></svg>`;
}
const spark = (vals, color = '#2c5aa0') => { const mx = Math.max(1, ...vals), n = vals.length;
  return `<svg viewBox="0 0 90 24" width="90" height="24"><polyline fill="none" stroke="${color}" stroke-width="1.6" points="${vals.map((v, i) => (n < 2 ? 0 : i / (n - 1) * 88 + 1) + ',' + (22 - v / mx * 20)).join(' ')}"/></svg>`; };
const strip = stages => `<div class="strip" title="Lifecycle stage per time bin">${stages.map((s, i) => `<i style="background:${STAGE[s] || '#fff'}" title="bin ${i}: ${s}"></i>`).join('')}</div>`;
function heat(m, cls, title) {
  const mx = Math.max(1, ...m.flat()); const rowT = m.map(r => r.reduce((a, b) => a + b, 0) || 1);
  return `<table class="cm"><tr><td class="l" style="border:0">${esc(title || '')}</td>${cls.map(c => `<th>pred ${esc(c.slice(0, 3))}</th>`).join('')}</tr>` +
    m.map((r, i) => `<tr><th class="l">true ${esc(cls[i].slice(0, 3))}</th>${r.map((v, j) => `<td style="background:rgba(44,90,160,${(v / mx * 0.85 + .05).toFixed(2)});color:${v / mx > .5 ? '#fff' : '#1f2933'}" title="${v} of ${rowT[i]} (${fmtN(pct(v, rowT[i]), 1)}% of the true class)">${v}</td>`).join('')}</tr>`).join('') + '</table>';
}

/* ---------------------------------------------------------------- shell */
function needRun() { return state.run ? false : true; }
function noRun() { return empty('No analysis yet. Open <a href="#studio">Analysis Studio</a> and run one, or use the Quick start button below.') + `<p style="text-align:center"><button id="qs">Quick start: generate demo data and analyse</button></p>`; }
function bindQuick() { const b = $('#qs'); if (b) b.onclick = () => busy(b, quickStart); }
async function quickStart() {
  const g = await api('datasets/generate', {method: 'POST', body: {name: 'Synthetic demo (6 topics, 7 days)', config: state.meta.default_generator}});
  const r = await api('runs', {method: 'POST', body: {dataset_id: g.id, config: {name: 'Demo analysis'}}});
  await refresh(); await selectRun(r.id); toast('Demo analysis ready.');
}
function setBanner() {
  const b = $('#banner'), s = state.run ? state.run.source : 'none';
  b.className = 'banner ' + s; b.textContent = state.run ? SRC[s] + '   ·   dataset: ' + state.run.dataset_name : 'No analysis selected. Data source labels appear here.';
}
function drawNav() { $('#nav').innerHTML = TABS.map(([k, n]) => `<a href="#${k}" class="${k === state.tab ? 'on' : ''}">${n}</a>`).join(''); }
async function refresh() {
  [state.datasets, state.runs, state.status] = await Promise.all([api('datasets'), api('runs'), api('status')]);
  const sel = $('#runSel');
  sel.innerHTML = state.runs.length ? state.runs.map(r => `<option value="${r.id}">#${r.id} ${esc(r.name)} [${SRC_SHORT[r.source] || r.source}] ${esc(r.dataset_name)}</option>`).join('') : '<option value="">(no runs yet)</option>';
  if (state.run) sel.value = state.run.id;
}
async function selectRun(id) {
  state.run = id ? await api('runs/' + id) : null; state.topic = null; state.compare.clear();
  if (state.run) $('#runSel').value = state.run.id;
  setBanner(); render();
}
function show(tab) {
  state.tab = TABS.some(t => t[0] === tab) ? tab : 'dashboard';
  $$('main > section').forEach(s => s.classList.toggle('on', s.id === 'tab-' + state.tab));
  drawNav(); render();
}
function render() { ({dashboard: rDash, studio: rStudio, data: rData, trends: rTrends, topics: rTopics, sentiment: rSent, eval: rEval, history: rHistory, alerts: rAlerts})[state.tab](); }
const R = () => state.run.result;

/* ---------------------------------------------------------------- dashboard */
function dominant(t) { const s = t.sentiment, n = t.size; return s.positive / n > .45 ? 'positive' : s.negative / n > .45 ? 'negative' : 'mixed'; }
function rDash() {
  const el = $('#tab-dashboard');
  if (needRun()) { el.innerHTML = noRun(); return bindQuick(); }
  const r = R(), k = r.kpis, f = state.dash;
  const warn = (r.meta.warnings || []).map(w => note(esc(w), 'warn')).join('');
  el.innerHTML = `${warn}<div class="kpis">
    <div class="kpi"><span>Tweets analysed</span><b>${fmtN(k.tweets)}</b><small>of ${fmtN(r.meta.filters.input)} in dataset</small></div>
    <div class="kpi"><span>Topics</span><b>${k.topics}</b><small>${fmtN(k.coverage * 100, 1)}% of tweets covered</small></div>
    <div class="kpi"><span>Emerging trends</span><b style="color:var(--pos)">${k.emerging}</b><small>new/rising and growing</small></div>
    <div class="kpi"><span>Bursting now</span><b style="color:var(--neg)">${k.bursting}</b><small>z ≥ ${r.meta.config.burst_z}</small></div>
    <div class="kpi"><span>Avg sentiment</span><b>${k.avg_sentiment > 0 ? '+' : ''}${k.avg_sentiment.toFixed(3)}</b><small>${fmtN(pct(k.positive, k.tweets))}% pos · ${fmtN(pct(k.negative, k.tweets))}% neg · ${fmtN(pct(k.neutral, k.tweets))}% neu</small></div>
    <div class="kpi"><span>Total reach</span><b>${fmtN(k.total_reach)}</b><small>${fmtN(k.unique_users)} users</small></div>
    <div class="kpi"><span>Avg engagement</span><b>${fmtN(k.avg_retweets + k.avg_likes, 1)}</b><small>${fmtN(k.avg_retweets, 1)} RT + ${fmtN(k.avg_likes, 1)} likes</small></div></div>
  <div class="card"><h2>Trend ranking</h2><p class="sub">Ranked by trend score (burst z-score, growth and share). Click a column to sort, click a row to explore the topic.</p>
    <div class="inline"><div><label>Search topic or keyword</label><input type="text" id="d_q" value="${esc(f.q || '')}" placeholder="e.g. strike"></div>
    <div><label>Stage</label><select id="d_stage"><option value="">all</option>${['new', 'rising', 'peaking', 'stable', 'declining', 'inactive'].map(s => `<option ${f.stage === s ? 'selected' : ''}>${s}</option>`).join('')}</select></div>
    <div><label>Sentiment</label><select id="d_sent"><option value="">all</option>${['positive', 'negative', 'mixed'].map(s => `<option ${f.sent === s ? 'selected' : ''}>${s}</option>`).join('')}</select></div>
    <div><label>Min tweets</label><input type="number" id="d_min" value="${f.min || 0}" min="0" style="width:90px"></div>
    <label style="margin-top:16px"><input type="checkbox" id="d_em" ${f.em ? 'checked' : ''}> emerging only</label></div>
    <div class="scroll" id="d_table"></div></div>
  <div class="grid"><div class="card"><h2>Volume over time</h2><p class="sub">All tweets and the five top-ranked topics per ${r.bins.seconds >= 86400 ? 'day' : fmtN(r.bins.seconds / 3600, 2) + ' h'}.</p><div id="d_vol"></div></div>
    <div class="card"><h2>Overall sentiment</h2><p class="sub">${esc(r.meta.config.sentiment_algo)} · positive ≥ ${r.meta.config.pos_threshold}, negative ≤ -${r.meta.config.neg_threshold}</p>${donut([{name: 'Positive', value: k.positive, color: SENT.positive}, {name: 'Negative', value: k.negative, color: SENT.negative}, {name: 'Neutral', value: k.neutral, color: SENT.neutral}])}</div>
    <div class="card"><h2>Sentiment over time</h2><div id="d_st"></div></div>
    <div class="card"><h2>Engagement per topic</h2><p class="sub">Average retweets and likes per tweet.</p><div id="d_eng"></div></div></div>
  <div class="card"><h2>Alerts for this run</h2><div id="d_alerts"></div></div>`;
  const labels = Array.from({length: r.bins.n}, (_, i) => binLabel(state.run, i));
  const top = r.topics.slice(0, 5);
  $('#d_vol').innerHTML = lineChart(labels, [{name: 'All tweets', color: '#1f2933', values: r.volume, w: 2.5}, ...top.map((t, i) => ({name: t.label, color: COLORS[i], values: t.series}))], {aria: 'volume over time'});
  $('#d_st').innerHTML = stackedBars(labels, ['positive', 'negative', 'neutral'].map(s => ({name: s, color: SENT[s], values: r.sentiment_series.map(x => x[s])})));
  const me = Math.max(1, ...r.topics.map(t => t.engagement.avg_retweets + t.engagement.avg_likes));
  $('#d_eng').innerHTML = r.topics.slice(0, 12).map(t => `<div style="display:grid;grid-template-columns:150px 1fr 70px;gap:8px;align-items:center;margin:4px 0;font-size:12.5px"><div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(t.label)}">${esc(t.label)}</div><div class="sbar" style="background:none"><i style="width:${pct(t.engagement.avg_retweets, me)}%;background:#2c5aa0"></i><i style="width:${pct(t.engagement.avg_likes, me)}%;background:#e0a030"></i></div><div>${fmtN(t.engagement.avg_retweets + t.engagement.avg_likes, 1)}</div></div>`).join('') + legend([['avg retweets', '#2c5aa0'], ['avg likes', '#e0a030']]);
  api('alerts/events?run_id=' + state.run.id).then(ev => { $('#d_alerts').innerHTML = ev.length ? `<table><tr><th class="l">Severity</th><th class="l">Rule</th><th class="l">Message</th></tr>${ev.map(e => `<tr><td class="l">${tag2(e.severity)}</td><td class="l">${esc(e.rule_name)}</td><td class="l wrap">${esc(e.message)}</td></tr>`).join('')}</table>` : empty('No alerts fired for this run. Configure rules in the Alerts tab.'); });
  ['d_q', 'd_stage', 'd_sent', 'd_min', 'd_em'].forEach(id => $('#' + id).oninput = () => { f.q = $('#d_q').value; f.stage = $('#d_stage').value; f.sent = $('#d_sent').value; f.min = +$('#d_min').value; f.em = $('#d_em').checked; dashTable(); });
  dashTable();
}
const pv = (p, bare) => (bare ? '' : 'p ') + (p < 0.0001 ? '< 0.0001' : (bare ? '' : '= ') + p.toFixed(4));
const tag2 = s => `<span class="tag" style="background:${{high: '#c0392b', medium: '#e0a030', low: '#52606d'}[s] || '#52606d'}">${esc(s)}</span>`;
function dashTable() {
  const f = state.dash, r = R();
  let rows = r.topics.filter(t => (!f.q || (t.label + ' ' + t.terms.map(x => x[0]).join(' ') + ' ' + t.hashtags.join(' ')).toLowerCase().includes(f.q.toLowerCase())) && (!f.stage || t.stage === f.stage) && (!f.sent || dominant(t) === f.sent) && t.size >= (f.min || 0) && (!f.em || t.emerging));
  const get = {score: t => t.score, size: t => t.size, z: t => t.z, growth: t => t.growth, sent: t => t.sentiment.avg, eng: t => t.engagement.avg_retweets + t.engagement.avg_likes, label: t => t.label, stage: t => t.stage};
  rows.sort((a, b) => f.dir * ((get[f.sort](a) > get[f.sort](b)) - (get[f.sort](a) < get[f.sort](b))));
  const th = (k, n, l) => `<th class="s ${l ? 'l' : ''}" data-k="${k}">${n}${f.sort === k ? (f.dir < 0 ? ' ▼' : ' ▲') : ''}</th>`;
  $('#d_table').innerHTML = rows.length ? `<table><tr><th>#</th>${th('label', 'Topic', 1)}${th('stage', 'Stage', 1)}${th('score', 'Score')}${th('size', 'Tweets')}${th('z', 'Burst z')}${th('growth', 'Growth')}${th('sent', 'Sentiment')}<th>Mix</th>${th('eng', 'RT+likes')}<th>Trend</th></tr>` +
    rows.map(t => `<tr class="click" data-id="${t.id}"><td>${t.rank}</td><td class="l wrap"><b>${esc(t.label)}</b>${t.emerging ? ' <span class="tag" style="background:#2e8b57">emerging</span>' : ''}<br><small>${esc(t.hashtags.join(' '))}</small></td><td class="l">${tag(t.stage)}</td>
    <td><div class="pbar" title="${t.score}"><i style="width:${t.score * 100}%"></i></div>${t.score.toFixed(2)}</td><td>${t.size}</td><td>${t.z.toFixed(1)}</td><td>${t.growth > 0 ? '+' : ''}${fmtN(t.growth * 100)}%</td><td>${t.sentiment.avg > 0 ? '+' : ''}${t.sentiment.avg.toFixed(2)}</td><td>${sbar(t.sentiment)}</td><td>${fmtN(t.engagement.avg_retweets + t.engagement.avg_likes, 1)}</td><td>${spark(t.series)}</td></tr>`).join('') + '</table>' : empty('No topics match the filters.');
  $$('#d_table th.s').forEach(h => h.onclick = () => { f.dir = f.sort === h.dataset.k ? -f.dir : -1; f.sort = h.dataset.k; dashTable(); });
  $$('#d_table tr.click').forEach(tr => tr.onclick = () => { state.topic = +tr.dataset.id; location.hash = '#topics'; });
}

/* ---------------------------------------------------------------- analysis studio */
const F = (id, label, inner) => `<div><label for="${id}">${label}</label>${inner}</div>`;
let studioBuilt = false;
function rStudio() {
  const el = $('#tab-studio'), d = state.meta.default_config, av = state.status;
  if (!studioBuilt) {
    studioBuilt = true;
    el.innerHTML = `<div class="grid"><div class="card"><h2>1. Data and filters</h2><p class="sub">Choose a dataset, then narrow it down.</p>
      <div class="row">${F('s_ds', 'Dataset', '<select id="s_ds"></select>')}${F('s_name', 'Analysis name', `<input type="text" id="s_name" value="${esc(d.name)}">`)}</div>
      <div id="s_dsinfo" class="sub"></div>
      <div class="row">${F('s_kw', 'Keywords (comma separated)', '<input type="text" id="s_kw" placeholder="strike, climate">')}${F('s_mode', 'Keyword match', '<select id="s_mode"><option value="any">any keyword</option><option value="all">all keywords</option></select>')}</div>
      <div class="row">${F('s_ht', 'Hashtags must include one of', '<input type="text" id="s_ht" placeholder="#strike, #cop">')}${F('s_ex', 'Exclude words', '<input type="text" id="s_ex" placeholder="spam, giveaway">')}</div>
      <div class="row">${F('s_from', 'From date', '<input type="date" id="s_from">')}${F('s_to', 'To date (inclusive)', '<input type="date" id="s_to">')}${F('s_lang', 'Languages (e.g. en, es; empty = all)', '<input type="text" id="s_lang" placeholder="en">')}${F('s_minf', 'Min author followers', '<input type="number" id="s_minf" min="0" value="0">')}</div>
      <div class="inline"><label><input type="checkbox" id="s_dedupe" checked> remove exact and near-duplicate tweets</label></div></div>
    <div class="card"><h2>2. Algorithms and thresholds</h2><p class="sub">Unavailable options explain what to install.</p>
      <div class="row">${F('s_sa', 'Sentiment algorithm', '<select id="s_sa"></select>')}${F('s_pos', 'Positive if score ≥', '<input type="number" id="s_pos" step="0.01" min="0" max="1">')}${F('s_neg', 'Negative if score ≤ −', '<input type="number" id="s_neg" step="0.01" min="0" max="1">')}</div>
      <div id="s_sanote" class="sub"></div>
      <div class="row">${F('s_ta', 'Topic algorithm', '<select id="s_ta"></select>')}${F('s_thr', 'Cosine threshold (pivot)', '<input type="number" id="s_thr" step="0.05" min="0.05" max="0.95">')}${F('s_k', 'Clusters k (K-Means; 0 = auto)', '<input type="number" id="s_k" min="0">')}${F('s_min', 'Min topic size', '<input type="number" id="s_min" min="2">')}</div>
      <div id="s_tanote" class="sub"></div>
      <div class="row">${F('s_bin', 'Time bin', '<select id="s_bin"></select>')}${F('s_z', 'Burst z-score threshold', '<input type="number" id="s_z" step="0.5" min="1">')}${F('s_rec', 'Recent window (bins)', '<input type="number" id="s_rec" min="1">')}${F('s_max', 'Max tweets analysed', '<input type="number" id="s_max" min="50">')}</div>
      <div class="inline"><label><input type="checkbox" id="s_emo"> include experimental emotion analysis</label></div></div></div>
    <div class="card"><h2>3. Run and save</h2><div class="inline"><button id="s_run">Run analysis</button>
      <input type="text" id="s_cfgname" placeholder="configuration name" style="width:220px"><button class="ghost" id="s_save">Save configuration</button>
      <select id="s_cfgs" style="width:220px"></select><button class="ghost" id="s_load">Load</button><button class="danger" id="s_del">Delete</button></div>
      <div id="s_result"></div></div>`;
    $('#s_ds').onchange = dsInfo; $('#s_run').onclick = e => busy(e.target, runStudio);
    $('#s_save').onclick = async () => { const n = $('#s_cfgname').value.trim(); if (!n) return toast('Enter a configuration name', 'err'); await api('configs', {method: 'POST', body: {name: n, config: readStudio()}}); toast('Configuration saved'); loadCfgs(); };
    $('#s_load').onclick = async () => { const c = (await api('configs')).find(x => x.id == $('#s_cfgs').value); if (c) { fillStudio(c.config); $('#s_cfgname').value = c.name; toast('Loaded ' + c.name); } };
    $('#s_del').onclick = async () => { const v = $('#s_cfgs').value; if (v) { await api('configs/' + v, {method: 'DELETE'}); toast('Deleted'); loadCfgs(); } };
    $('#s_sa').onchange = $('#s_ta').onchange = algoNotes;
    fillStudio(d);
  }
  const keep = $('#s_ds').value;
  $('#s_ds').innerHTML = state.datasets.map(x => `<option value="${x.id}">#${x.id} ${esc(x.name)} [${SRC_SHORT[x.source] || x.source}, ${x.n} tweets]</option>`).join('') || '<option value="">(no datasets: open the Data Centre)</option>';
  if (keep) $('#s_ds').value = keep; else if (state.run) $('#s_ds').value = state.run.dataset_id;
  const opts = (obj, cur) => Object.entries(obj).map(([k, v]) => `<option value="${k}" ${v.ok ? '' : 'disabled'} ${k === cur ? 'selected' : ''}>${k}${v.ok ? '' : ' (unavailable)'}</option>`).join('');
  $('#s_sa').innerHTML = opts(av.sentiment, $('#s_sa').dataset.v || d.sentiment_algo); $('#s_ta').innerHTML = opts(av.topics, $('#s_ta').dataset.v || d.topic_algo);
  algoNotes(); dsInfo(); loadCfgs();
}
function algoNotes() { const s = state.status; $('#s_sa').dataset.v = $('#s_sa').value; $('#s_ta').dataset.v = $('#s_ta').value;
  $('#s_sanote').textContent = (s.sentiment[$('#s_sa').value] || {}).note || ''; $('#s_tanote').textContent = (s.topics[$('#s_ta').value] || {}).note || ''; }
async function dsInfo() {
  const id = $('#s_ds').value; if (!id) return;
  const d = await api('datasets/' + id);
  $('#s_dsinfo').innerHTML = `${srcTag(d.source)} ${d.n} tweets · ${fmtTs(d.date_from)} to ${fmtTs(d.date_to)} UTC · languages: ${Object.entries(d.languages).map(([l, n]) => `<span class="chip btn2" data-l="${esc(l)}">${esc(l)} ${n}</span>`).join('')}`;
  $$('#s_dsinfo .chip').forEach(c => c.onclick = () => $('#s_lang').value = c.dataset.l);
}
async function loadCfgs() { const c = await api('configs'); $('#s_cfgs').innerHTML = c.length ? c.map(x => `<option value="${x.id}">${esc(x.name)}</option>`).join('') : '<option value="">(no saved configurations)</option>'; }
function fillStudio(c) {
  const set = (id, v) => { const e = $('#' + id); if (e.type === 'checkbox') e.checked = !!v; else e.value = v ?? ''; };
  set('s_name', c.name); set('s_kw', (c.keywords || []).join(', ')); set('s_mode', c.keyword_mode); set('s_ht', (c.hashtags || []).join(', ')); set('s_ex', (c.exclude || []).join(', '));
  set('s_from', c.date_from); set('s_to', c.date_to); set('s_lang', (c.languages || []).join(', ')); set('s_minf', c.min_followers); set('s_dedupe', c.drop_duplicates);
  $('#s_sa').dataset.v = c.sentiment_algo; $('#s_sa').value = c.sentiment_algo; set('s_pos', c.pos_threshold); set('s_neg', c.neg_threshold);
  $('#s_ta').dataset.v = c.topic_algo; $('#s_ta').value = c.topic_algo; set('s_thr', c.threshold); set('s_k', c.k); set('s_min', c.min_size);
  $('#s_bin').innerHTML = `<option value="0">auto</option>` + state.meta.bin_choices.map(b => `<option value="${b / 60}">${b >= 86400 ? '1 day' : b >= 3600 ? b / 3600 + ' h' : b / 60 + ' min'}</option>`).join('');
  set('s_bin', c.bin_minutes); set('s_z', c.burst_z); set('s_rec', c.recent_bins); set('s_max', c.max_tweets); set('s_emo', c.emotions);
}
function readStudio() {
  const v = id => $('#' + id).value, n = id => +$('#' + id).value;
  return {name: v('s_name') || 'Untitled analysis', keywords: v('s_kw'), keyword_mode: v('s_mode'), hashtags: v('s_ht'), exclude: v('s_ex'), date_from: v('s_from'), date_to: v('s_to'),
    languages: v('s_lang'), min_followers: n('s_minf'), drop_duplicates: $('#s_dedupe').checked, sentiment_algo: v('s_sa'), pos_threshold: n('s_pos'), neg_threshold: n('s_neg'),
    topic_algo: v('s_ta'), threshold: n('s_thr'), k: n('s_k'), min_size: n('s_min'), bin_minutes: n('s_bin'), burst_z: n('s_z'), recent_bins: n('s_rec'), max_tweets: n('s_max'), emotions: $('#s_emo').checked};
}
async function runStudio() {
  const ds = $('#s_ds').value; if (!ds) throw new Error('Choose a dataset first.');
  $('#s_result').innerHTML = note('Analysis running… the original document-pivot method can take a few seconds on thousands of tweets.');
  const r = await api('runs', {method: 'POST', body: {dataset_id: +ds, config: readStudio()}});
  await refresh(); await selectRun(r.id);
  const s = r.summary;
  $('#s_result').innerHTML = note(`<b>Run #${r.id} finished in ${s.runtime_s}s:</b> ${s.tweets} tweets, ${s.topics} topics, ${s.emerging} emerging, ${s.bursting} bursting. ${r.alerts.length} alert(s) fired. <a href="#dashboard">Open dashboard</a> · <a href="#trends">Trends</a> · <a href="#topics">Topic Explorer</a>`) + r.warnings.map(w => note(esc(w), 'warn')).join('');
  toast('Analysis complete: run #' + r.id);
}

/* ---------------------------------------------------------------- data centre */
let dataBuilt = false;
function rData() {
  const el = $('#tab-data');
  if (!dataBuilt) {
    dataBuilt = true; state.genTopics = JSON.parse(JSON.stringify(state.meta.default_generator.topics));
    el.innerHTML = `<div class="card"><h2>Datasets</h2><p class="sub">Every dataset carries a source label that follows it into the dashboard banner and reports.</p><div class="scroll" id="dc_list"></div><div id="dc_detail"></div></div>
    <div class="grid"><div class="card"><h2>Upload CSV / JSON / JSONL</h2><p class="sub">Works with exports from X, or any file with a text column. Nothing leaves your computer.</p>
      <input type="file" id="up_file" accept=".csv,.tsv,.json,.jsonl,.ndjson,.txt"><div id="up_out"></div></div>
    <div class="card"><h2>Live data from the X API (optional)</h2><div id="live_out"></div></div></div>
    <div class="card"><h2>Synthetic data generator</h2><p class="sub">Generates labelled tweets (ground-truth sentiment and topic) with the activity curve you choose per topic. Fully reproducible for a given seed.</p>
      <div class="row">${F('g_n', 'Tweets', '<input type="number" id="g_n" value="2400" min="50" max="20000">')}${F('g_days', 'Days', '<input type="number" id="g_days" value="7" min="0.25" step="0.25">')}${F('g_start', 'Start date', '<input type="date" id="g_start" value="2025-03-03">')}${F('g_seed', 'Random seed', '<input type="number" id="g_seed" value="42">')}</div>
      <div class="row">${F('g_noise', 'Off-topic chatter (0-0.5)', '<input type="number" id="g_noise" value="0.04" step="0.01" min="0" max="0.5">')}${F('g_rt', 'Retweet duplicates (0-0.5)', '<input type="number" id="g_rt" value="0.06" step="0.01" min="0" max="0.5">')}${F('g_diff', 'Hard sentiment cases (0-1)', '<input type="number" id="g_diff" value="0.2" step="0.05" min="0" max="1">')}${F('g_es', 'Spanish share (0-0.5)', '<input type="number" id="g_es" value="0.07" step="0.01" min="0" max="0.5">')}</div>
      <h3>Topics</h3><div class="scroll"><table id="g_topics"></table></div>
      <div class="inline"><button class="ghost sm" id="g_add">+ Add topic</button><button class="ghost sm" id="g_reset">Reset to defaults</button>
      <input type="text" id="g_name" placeholder="dataset name" value="Custom synthetic dataset" style="width:260px"><button id="g_go">Generate dataset</button></div>
      <p class="sub">Shape: <b>steady</b> flat · <b>burst</b> activity spike at "at" (0-1 of the period) with width and magnitude · <b>emerging</b> grows over time · <b>declining</b> fades. Sentiment: share positive / negative / neutral.</p></div>`;
    $('#up_file').onchange = onUpload; $('#g_add').onclick = () => { readGenTopics(); state.genTopics.push({name: 'New topic', keywords: ['alpha', 'beta', 'gamma'], hashtags: [], weight: 1, shape: 'steady', at: .5, width: .1, magnitude: 5, sentiment: [.34, .33, .33]}); drawGen(); };
    $('#g_reset').onclick = () => { state.genTopics = JSON.parse(JSON.stringify(state.meta.default_generator.topics)); drawGen(); };
    $('#g_go').onclick = e => busy(e.target, genGo); drawGen(); drawLive();
  }
  drawDatasets();
}
function drawDatasets() {
  $('#dc_list').innerHTML = state.datasets.length ? `<table><tr><th class="l">ID</th><th class="l">Name</th><th class="l">Source</th><th>Tweets</th><th class="l">Created</th><th></th></tr>${state.datasets.map(d => `<tr><td class="l">${d.id}</td><td class="l wrap">${esc(d.name)}</td><td class="l">${srcTag(d.source)}</td><td>${d.n}</td><td class="l">${fmtTs(d.created)}</td><td><button class="sm ghost" data-view="${d.id}">Inspect</button> <button class="sm" data-use="${d.id}">Analyse</button> <button class="sm danger" data-del="${d.id}">Delete</button></td></tr>`).join('')}</table>` : empty('No datasets yet. Upload a file or generate synthetic data below.');
  $$('#dc_list [data-view]').forEach(b => b.onclick = async () => { const d = await api('datasets/' + b.dataset.view);
    $('#dc_detail').innerHTML = `<h3>#${d.id} ${esc(d.name)} ${srcTag(d.source)}</h3><p class="sub">${d.n} tweets · ${fmtTs(d.date_from)} to ${fmtTs(d.date_to)} UTC · labelled sentiment: ${d.labelled_sentiment} · labelled topic: ${d.labelled_topic} · languages ${esc(JSON.stringify(d.languages))}</p><div class="scroll"><table><tr><th class="l">Time</th><th class="l">Text</th><th>Lang</th><th>RT</th><th>Likes</th><th>Followers</th><th class="l">Label</th></tr>${d.preview.map(r => `<tr><td class="l">${fmtTs(r.ts)}</td><td class="l wrap">${esc(r.text)}</td><td>${esc(r.lang)}</td><td>${r.retweets}</td><td>${r.likes}</td><td>${r.followers}</td><td class="l">${esc(r.label_sentiment || '')} ${esc(r.label_topic || '')}</td></tr>`).join('')}</table></div>`; });
  $$('#dc_list [data-use]').forEach(b => b.onclick = () => { $('#s_ds') && ($('#s_ds').value = b.dataset.use); studioBuilt && (state.pendingDs = b.dataset.use); location.hash = '#studio'; setTimeout(() => { $('#s_ds').value = b.dataset.use; dsInfo(); }, 50); });
  $$('#dc_list [data-del]').forEach(b => b.onclick = async () => { if (!confirm('Delete this dataset? Runs made from it keep their results.')) return; await api('datasets/' + b.dataset.del, {method: 'DELETE'}); await refresh(); drawDatasets(); toast('Dataset deleted'); });
}
async function onUpload(e) {
  const f = e.target.files[0]; if (!f) return;
  const content = await f.text(); state.upload = {filename: f.name, content};
  await previewUpload();
}
async function previewUpload(mapping) {
  const u = state.upload, out = $('#up_out');
  try {
    const p = await api('datasets/preview', {method: 'POST', body: {filename: u.filename, content: u.content, mapping}});
    const v = p.validation, m = p.mapping;
    out.innerHTML = `<p class="sub"><b>${esc(u.filename)}</b>: ${p.format.toUpperCase()}, ${p.record_count} records${p.parse_errors ? `, ${p.parse_errors} unparseable lines` : ''}</p>
    <h3>Field mapping</h3><div class="row">${state.meta.fields.map(fl => F('m_' + fl, fl, `<select id="m_${fl}" data-f="${fl}"><option value="">(none)</option>${p.columns.map(c => `<option ${m[fl] === c ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>`)).join('')}</div>
    <button class="ghost sm" id="up_val">Re-validate with this mapping</button>
    <h3>Validation</h3>${v.error ? note(esc(v.error), 'err') : `<div class="kpis"><div class="kpi"><span>Valid rows</span><b>${v.valid_rows}</b></div><div class="kpi"><span>Exact duplicates</span><b>${v.exact_duplicates}</b></div><div class="kpi"><span>Near duplicates</span><b>${v.near_duplicates}</b></div><div class="kpi"><span>Bad timestamps</span><b>${v.bad_timestamps}</b></div><div class="kpi"><span>Labelled sentiment</span><b>${v.labelled_sentiment}</b></div></div>
    <p class="sub">Period: ${fmtTs(v.date_from)} to ${fmtTs(v.date_to)} · languages: ${esc(JSON.stringify(v.languages))}</p>${v.warnings.map(w => note(esc(w), 'warn')).join('')}`}
    <h3>Preview</h3><div class="scroll"><table><tr>${p.columns.slice(0, 8).map(c => `<th class="l">${esc(c)}</th>`).join('')}</tr>${p.sample.map(r => `<tr>${p.columns.slice(0, 8).map(c => `<td class="l wrap">${esc(String(r[c] ?? '').slice(0, 90))}</td>`).join('')}</tr>`).join('')}</table></div>
    <div class="inline"><input type="text" id="up_name" placeholder="dataset name" value="${esc(u.filename.replace(/\.[^.]+$/, ''))}" style="width:240px"><label><input type="checkbox" id="up_dd"> drop duplicates on import</label><button id="up_go" ${v.ok ? '' : 'disabled'}>Import dataset</button></div>`;
    $('#up_val').onclick = () => previewUpload(curMap());
    $('#up_go').onclick = ev => busy(ev.target, async () => { const r = await api('datasets/import', {method: 'POST', body: {filename: u.filename, content: u.content, mapping: curMap(), name: $('#up_name').value, drop_duplicates: $('#up_dd').checked}});
      toast(`Imported ${r.rows} tweets` + (r.meta.duplicates_removed ? ` (${r.meta.duplicates_removed} duplicates removed)` : '')); out.innerHTML = ''; $('#up_file').value = ''; await refresh(); drawDatasets(); });
  } catch (err) { out.innerHTML = note(esc(err.message), 'err'); }
}
const curMap = () => Object.fromEntries($$('#up_out select[data-f]').map(s => [s.dataset.f, s.value || null]));
function drawGen() {
  const t = state.genTopics;
  $('#g_topics').innerHTML = `<tr><th class="l">Name</th><th class="l">Keywords (comma)</th><th class="l">Hashtags</th><th>Weight</th><th class="l">Shape</th><th>At</th><th>Width</th><th>Magn.</th><th>% pos</th><th>% neg</th><th>% neu</th><th></th></tr>` + t.map((x, i) => `<tr data-i="${i}">
    <td><input type="text" data-k="name" value="${esc(x.name)}" style="width:130px"></td><td><input type="text" data-k="keywords" value="${esc(x.keywords.join(', '))}" style="width:210px"></td><td><input type="text" data-k="hashtags" value="${esc((x.hashtags || []).join(', '))}" style="width:90px"></td>
    <td><input type="number" data-k="weight" value="${x.weight}" step="0.1" min="0.1" style="width:64px"></td><td><select data-k="shape">${['steady', 'burst', 'emerging', 'declining'].map(s => `<option ${x.shape === s ? 'selected' : ''}>${s}</option>`).join('')}</select></td>
    <td><input type="number" data-k="at" value="${x.at}" step="0.05" min="0" max="1" style="width:60px"></td><td><input type="number" data-k="width" value="${x.width}" step="0.02" min="0.01" style="width:60px"></td><td><input type="number" data-k="magnitude" value="${x.magnitude}" step="1" min="1" style="width:60px"></td>
    ${[0, 1, 2].map(j => `<td><input type="number" data-s="${j}" value="${Math.round(x.sentiment[j] * 100)}" min="0" max="100" style="width:58px"></td>`).join('')}<td><button class="danger sm" data-rm="${i}">×</button></td></tr>`).join('');
  $$('#g_topics [data-rm]').forEach(b => b.onclick = () => { readGenTopics(); state.genTopics.splice(+b.dataset.rm, 1); drawGen(); });
}
function readGenTopics() {
  state.genTopics = $$('#g_topics tr[data-i]').map(tr => { const g = k => $(`[data-k=${k}]`, tr).value, list = s => s.split(',').map(x => x.trim()).filter(Boolean);
    return {name: g('name') || 'topic', keywords: list(g('keywords')), hashtags: list(g('hashtags')), weight: +g('weight') || 1, shape: g('shape'), at: +g('at'), width: +g('width'), magnitude: +g('magnitude'), sentiment: $$('[data-s]', tr).map(i => +i.value)}; });
}
async function genGo() {
  readGenTopics(); const es = +$('#g_es').value;
  if (!state.genTopics.length) throw new Error('Add at least one topic.');
  const bad = state.genTopics.find(t => !t.keywords.length); if (bad) throw new Error('Topic "' + bad.name + '" needs keywords.');
  const cfg = {n: +$('#g_n').value, days: +$('#g_days').value, start: $('#g_start').value, seed: +$('#g_seed').value, noise: +$('#g_noise').value, retweet_fraction: +$('#g_rt').value, difficulty: +$('#g_diff').value, languages: {en: 1 - es, es}, topics: state.genTopics};
  const r = await api('datasets/generate', {method: 'POST', body: {name: $('#g_name').value, config: cfg}});
  toast(`Generated ${r.rows} synthetic tweets (dataset #${r.id})`); await refresh(); drawDatasets();
}
function drawLive() {
  const s = state.status.x_api;
  $('#live_out').innerHTML = (s.token_set ? note('Bearer token found in the <span class="mono">' + s.env_var + '</span> environment variable.') : note('No token. Live retrieval is disabled until you set <span class="mono">' + s.env_var + '</span> and restart the app. Synthetic and uploaded data work without it.', 'warn')) +
    `<p class="sub">Uses the official X API v2 <span class="mono">GET /2/tweets/search/recent</span> (last 7 days only; subject to your plan, rate limits and per-read costs). The token is never typed into this page or stored.</p>
    <div class="row">${F('l_kw', 'Keywords (comma)', '<input type="text" id="l_kw" placeholder="climate, summit">')}${F('l_ht', 'Hashtags', '<input type="text" id="l_ht">')}${F('l_ex', 'Exclude', '<input type="text" id="l_ex">')}${F('l_lang', 'Language', '<input type="text" id="l_lang" value="en">')}</div>
    <div class="row">${F('l_max', 'Posts per page (10-100)', '<input type="number" id="l_max" value="100" min="10" max="100">')}${F('l_pages', 'Pages (1-5)', '<input type="number" id="l_pages" value="1" min="1" max="5">')}</div>
    <button id="l_go" ${s.token_set ? '' : 'disabled'}>Fetch from X</button>`;
  $('#l_go').onclick = e => busy(e.target, async () => { const l = id => $('#' + id).value.split(',').map(x => x.trim()).filter(Boolean);
    const r = await api('live/search', {method: 'POST', body: {keywords: l('l_kw'), hashtags: l('l_ht'), exclude: l('l_ex'), lang: $('#l_lang').value, max_results: +$('#l_max').value, pages: +$('#l_pages').value}});
    toast('Fetched ' + r.rows + ' posts'); await refresh(); drawDatasets(); });
}

/* ---------------------------------------------------------------- trends */
function rTrends() {
  const el = $('#tab-trends'); if (needRun()) { el.innerHTML = noRun(); return bindQuick(); }
  const r = R(), T = r.topics, labels = Array.from({length: r.bins.n}, (_, i) => binLabel(state.run, i));
  const emerging = T.filter(t => t.emerging);
  if (!state.trendTopic || !T.some(t => t.id === state.trendTopic)) state.trendTopic = (emerging[0] || T[0] || {}).id;
  el.innerHTML = `<div class="card"><h2>Emerging trends</h2><p class="sub">Stage <i>new</i> or <i>rising</i>, and a burst z-score ≥ ${r.meta.config.burst_z}, or growth ≥ +100% over the last ${r.meta.config.recent_bins} bin(s), or a steady ramp of ≥ +50% over the longer trend window.</p>
    ${emerging.length ? `<div class="grid3">${emerging.map(t => `<div class="card" style="margin:0;cursor:pointer" data-t="${t.id}"><b>${esc(t.label)}</b> ${tag(t.stage)}<br><small>${t.size} tweets · burst z=${t.z.toFixed(1)} · recent growth ${t.growth > 0 ? '+' : ''}${fmtN(t.growth * 100)}% · trend ramp ${t.trend_growth > 0 ? '+' : ''}${fmtN(t.trend_growth * 100)}% · ${pv(t.p_value)}</small>${spark(t.series, '#2e8b57')}</div>`).join('')}</div>` : empty('No emerging trends in this run.')}</div>
  <div class="card"><h2>Topic ranking and comparison</h2><p class="sub">Tick two or more topics to compare their activity. Score = 0.5·burst + 0.3·growth + 0.2·share (see README).</p>
    <div class="scroll"><table><tr><th></th><th>#</th><th class="l">Topic</th><th class="l">Stage</th><th>Score</th><th>Tweets</th><th>Burst z</th><th>p-value</th><th>Growth</th><th>Trend ramp</th><th>Velocity</th><th>Peak</th><th class="l">Lifecycle</th></tr>
    ${T.map(t => `<tr class="click" data-t="${t.id}"><td><input type="checkbox" data-c="${t.id}" ${state.compare.has(t.id) ? 'checked' : ''}></td><td>${t.rank}</td><td class="l wrap"><b>${esc(t.label)}</b></td><td class="l">${tag(t.stage)}</td><td>${t.score.toFixed(2)}</td><td>${t.size}</td><td>${t.z.toFixed(1)}</td><td>${pv(t.p_value, 1)}</td><td>${t.growth > 0 ? '+' : ''}${fmtN(t.growth * 100)}%</td><td>${t.trend_growth > 0 ? '+' : ''}${fmtN(t.trend_growth * 100)}%</td><td>${t.velocity > 0 ? '+' : ''}${t.velocity.toFixed(2)}</td><td>${esc(binLabel(state.run, t.peak_bin))}</td><td style="min-width:140px">${strip(t.stages)}</td></tr>`).join('')}</table></div>
    <div id="t_cmp"></div></div>
  <div class="card" id="t_detail"></div>`;
  $$('#tab-trends [data-t]').forEach(x => x.onclick = e => { if (e.target.tagName === 'INPUT') return; state.trendTopic = +x.dataset.t; rTrends(); });
  $$('#tab-trends [data-c]').forEach(c => c.onchange = () => { c.checked ? state.compare.add(+c.dataset.c) : state.compare.delete(+c.dataset.c); drawCompare(labels); });
  drawCompare(labels); drawTrendDetail(labels);
}
function drawCompare(labels) {
  const T = R().topics.filter(t => state.compare.has(t.id)), el = $('#t_cmp');
  if (T.length < 2) { el.innerHTML = note('Select two or more topics above to compare them.'); return; }
  const idx = $('#t_idx') ? $('#t_idx').checked : false;
  el.innerHTML = `<h3>Comparison</h3><label><input type="checkbox" id="t_idx" ${idx ? 'checked' : ''}> normalise each topic to its own peak (compare shapes, not volumes)</label>` +
    lineChart(labels, T.map((t, i) => ({name: t.label, color: COLORS[i % 8], values: idx ? t.series.map(v => v / Math.max(1, t.peak_count)) : t.series})), {aria: 'topic comparison'}) +
    `<table><tr><th class="l">Topic</th><th>Tweets</th><th>Score</th><th>Burst z</th><th>Growth</th><th>Sentiment</th><th>RT+likes</th><th>Reach</th><th class="l">Shared keywords with others</th></tr>${T.map(t => `<tr><td class="l">${esc(t.label)}</td><td>${t.size}</td><td>${t.score.toFixed(2)}</td><td>${t.z.toFixed(1)}</td><td>${fmtN(t.growth * 100)}%</td><td>${t.sentiment.avg.toFixed(2)}</td><td>${fmtN(t.engagement.avg_retweets + t.engagement.avg_likes, 1)}</td><td>${fmtN(t.engagement.reach)}</td><td class="l wrap">${esc(T.filter(o => o !== t).flatMap(o => t.terms.map(x => x[0]).filter(w => o.terms.some(y => y[0] === w))).join(', ') || 'none')}</td></tr>`).join('')}</table>`;
  $('#t_idx').onchange = () => drawCompare(labels);
}
function drawTrendDetail(labels) {
  const t = R().topics.find(x => x.id === state.trendTopic), el = $('#t_detail'); if (!t) return;
  const marks = t.bursts.map(b => ({a: b.start, b: b.end, title: `burst ${binLabel(state.run, b.start)}: peak ${b.peak_count} tweets, z=${b.z.toFixed(1)}`}));
  el.innerHTML = `<h2>Lifecycle: ${esc(t.label)} ${tag(t.stage)}</h2><p class="sub">Stage is re-evaluated at every time bin. Shaded bands are statistical bursts (z ≥ ${R().meta.config.burst_z}).</p>
    ${lineChart(labels, [{name: 'tweets per bin', color: '#2c5aa0', values: t.series}], {marks, aria: 'topic lifecycle'})}
    <h3>Lifecycle stages over time</h3>${strip(t.stages)}<div class="legend" style="margin-top:4px">${['new', 'rising', 'peaking', 'stable', 'declining', 'inactive'].map(s => `<span><i style="background:${STAGE[s]}"></i>${s}</span>`).join('')}</div>
    <h3>Burst z-score over time</h3>${lineChart(labels, [{name: 'z-score', color: '#c0392b', values: t.z_series.map(v => Math.max(0, v))}], {aria: 'z-score'})}
    <div class="grid3"><div><h3>Bursts</h3>${t.bursts.length ? `<table><tr><th class="l">From</th><th class="l">Peak</th><th>Tweets</th><th>z</th></tr>${t.bursts.map(b => `<tr><td class="l">${esc(binLabel(state.run, b.start))}</td><td class="l">${esc(binLabel(state.run, b.peak))}</td><td>${b.peak_count}</td><td>${b.z.toFixed(1)}</td></tr>`).join('')}</table>` : '<p class="sub">No statistical bursts detected.</p>'}</div>
    <div><h3>Stage transitions</h3><table><tr><th class="l">Time</th><th class="l">Stage</th></tr>${t.transitions.map(x => `<tr><td class="l">${esc(binLabel(state.run, x.bin))}</td><td class="l">${tag(x.stage)}</td></tr>`).join('')}</table></div>
    <div><h3>Metrics</h3><p class="sub">z ${t.z.toFixed(2)} · p ${t.p_value} · recent growth ${fmtN(t.growth * 100)}% · trend ramp ${fmtN(t.trend_growth * 100)}% · velocity ${t.velocity.toFixed(2)} tweets/bin² · peak ${t.peak_count} tweets at ${esc(binLabel(state.run, t.peak_bin))} · first ${fmtTs(t.first_seen)} · last ${fmtTs(t.last_seen)}</p></div></div>`;
}

/* ---------------------------------------------------------------- topic explorer */
function rTopics() {
  const el = $('#tab-topics'); if (needRun()) { el.innerHTML = noRun(); return bindQuick(); }
  const r = R(), T = r.topics; if (!T.length) { el.innerHTML = empty('This run found no topics. Lower the minimum topic size or relax the filters.'); return; }
  if (!T.some(t => t.id === state.topic)) state.topic = T[0].id;
  const t = T.find(x => x.id === state.topic), mx = Math.max(...t.terms.map(x => x[1]), 1e-9);
  const rel = r.relations.filter(x => x.a === t.id || x.b === t.id).map(x => ({o: T.find(y => y.id === (x.a === t.id ? x.b : x.a)), sim: x.sim, shared: x.shared})).filter(x => x.o).sort((a, b) => b.sim - a.sim);
  el.innerHTML = `<div class="two"><div><label>Topics (${r.meta.config.topic_algo})</label><div class="list">${T.map(x => `<div data-t="${x.id}" class="${x.id === t.id ? 'on' : ''}">${x.rank}. ${esc(x.label)}<br><small>${x.size} tweets · ${x.stage}</small></div>`).join('')}</div></div>
  <div><div class="card"><h2>${esc(t.label)} ${tag(t.stage)} ${t.emerging ? '<span class="tag" style="background:#2e8b57">emerging</span>' : ''}</h2><p class="sub">${t.size} tweets (${fmtN(t.share * 100, 1)}% of analysed) · sentiment ${t.sentiment.avg.toFixed(2)} · reach ${fmtN(t.engagement.reach)} · ${esc(t.hashtags.join(' '))}</p>${sbar(t.sentiment)}
    <div class="grid" style="margin-top:12px"><div><h3>Keywords (class-based TF-IDF)</h3>${t.terms.map(([w, s]) => `<div style="display:grid;grid-template-columns:110px 1fr;gap:8px;align-items:center;font-size:12.5px"><span>${esc(w)}</span><div class="pbar"><i style="width:${pct(s, mx)}%"></i></div></div>`).join('')}
      <h3>Related keywords</h3><p>${t.related.length ? t.related.map(([w, s]) => `<span class="chip" title="co-occurrence ${s}">${esc(w)}</span>`).join('') : '<span class="sub">none found</span>'}</p></div>
    <div><h3>Activity</h3>${lineChart(Array.from({length: r.bins.n}, (_, i) => binLabel(state.run, i)), [{name: 'tweets per bin', color: '#2c5aa0', values: t.series}], {h: 190})}</div></div></div>
  <div class="card"><h3>Representative posts</h3><p class="sub">Closest to the topic centroid in TF-IDF space, near-identical posts removed.</p>${t.samples.map(s => `<div class="post"><b style="color:${SENT[s.label]}">${s.label === 'positive' ? '+' : s.label === 'negative' ? '−' : '~'}${s.score.toFixed(2)}</b>${esc(s.text)} <small>(${fmtTs(s.ts)} · ${s.retweets} RT · ${s.likes} likes)</small></div>`).join('')}</div>
  <div class="grid"><div class="card"><h3>Topic evolution</h3><p class="sub">Top terms in four equal time windows: how the conversation shifts.</p><table><tr><th class="l">Window (UTC)</th><th>Tweets</th><th class="l">Top terms</th></tr>${t.evolution.map(w => `<tr><td class="l">${fmtTs(w.from)}</td><td>${w.tweets}</td><td class="l wrap">${esc(w.terms.join(', '))}</td></tr>`).join('')}</table></div>
  <div class="card"><h3>Related topics</h3><p class="sub">Cosine similarity of keyword weights.</p>${rel.length ? `<table><tr><th class="l">Topic</th><th>Similarity</th><th class="l">Shared keywords</th></tr>${rel.slice(0, 8).map(x => `<tr class="click" data-t="${x.o.id}"><td class="l wrap">${esc(x.o.label)}</td><td>${x.sim.toFixed(2)}</td><td class="l wrap">${esc(x.shared.join(', ') || '-')}</td></tr>`).join('')}</table>` : '<p class="sub">No related topics above the similarity floor.</p>'}</div></div></div></div>`;
  $$('#tab-topics [data-t]').forEach(x => x.onclick = () => { state.topic = +x.dataset.t; rTopics(); });
}

/* ---------------------------------------------------------------- sentiment */
function rSent() {
  const el = $('#tab-sentiment'); if (needRun()) { el.innerHTML = noRun(); return bindQuick(); }
  const r = R(), k = r.kpis, cfg = r.meta.config, T = r.topics, labels = Array.from({length: r.bins.n}, (_, i) => binLabel(state.run, i));
  const avgSeries = r.sentiment_series.map(x => { const n = x.positive + x.negative + x.neutral; return n ? (x.positive - x.negative) / n : 0; });
  const emo = r.emotions, emoT = emo ? Math.max(1, ...Object.values(emo)) : 1;
  el.innerHTML = `<div class="note">Algorithm: <b>${esc(cfg.sentiment_algo)}</b> (VADER baseline unless changed in the Studio) · positive if score ≥ ${cfg.pos_threshold}, negative if ≤ −${cfg.neg_threshold}. Re-run with another algorithm in the Studio and compare runs in History, or measure accuracy in the Evaluation Lab.</div>
  <div class="grid"><div class="card"><h2>Positive vs negative vs neutral</h2>${donut([{name: 'Positive', value: k.positive, color: SENT.positive}, {name: 'Negative', value: k.negative, color: SENT.negative}, {name: 'Neutral', value: k.neutral, color: SENT.neutral}])}</div>
  <div class="card"><h2>Net sentiment over time</h2><p class="sub">(positive − negative) / tweets per time bin.</p>${lineChart(labels, [{name: 'net sentiment', color: '#2c5aa0', values: avgSeries.map(v => v + 1)}], {aria: 'net sentiment'})}<p class="sub">Shifted by +1 for plotting: 1.0 is balanced, above 1 is net positive.</p></div></div>
  <div class="card"><h2>Sentiment by topic</h2><div class="scroll"><table><tr><th class="l">Topic</th><th>Tweets</th><th>Avg score</th><th>% positive</th><th>% negative</th><th>% neutral</th><th class="l">Mix</th></tr>${T.map(t => `<tr><td class="l wrap">${esc(t.label)}</td><td>${t.size}</td><td>${t.sentiment.avg.toFixed(2)}</td><td>${fmtN(pct(t.sentiment.positive, t.size))}</td><td>${fmtN(pct(t.sentiment.negative, t.size))}</td><td>${fmtN(pct(t.sentiment.neutral, t.size))}</td><td>${sbar(t.sentiment)}</td></tr>`).join('')}</table></div></div>
  <div class="card"><h2>Sentiment over time</h2>${stackedBars(labels, ['positive', 'negative', 'neutral'].map(s => ({name: s, color: SENT[s], values: r.sentiment_series.map(x => x[s])})))}</div>
  <div class="card"><h2>Emotions <span class="tag" style="background:#b8860b">experimental</span></h2><p class="sub">Counts tweets containing words from a small hand-written lexicon (joy, anger, sadness, fear, surprise). It is a rough indicator, not a validated emotion model.</p>
  ${emo ? `<div class="grid"><div>${Object.entries(emo).map(([e, v]) => `<div style="display:grid;grid-template-columns:90px 1fr 60px;gap:8px;align-items:center;margin:4px 0"><span>${e}</span><div class="pbar"><i style="width:${pct(v, emoT)}%"></i></div><span>${v}</span></div>`).join('')}</div>
  <div><table><tr><th class="l">Topic</th><th class="l">Dominant emotion</th></tr>${T.map(t => { const e = Object.entries(t.emotions || {}).sort((a, b) => b[1] - a[1])[0]; return `<tr><td class="l wrap">${esc(t.label)}</td><td class="l">${e && e[1] ? esc(e[0]) + ' (' + e[1] + ')' : 'none detected'}</td></tr>`; }).join('')}</table></div></div>` : empty('Emotion analysis was switched off for this run.')}</div>`;
}

/* ---------------------------------------------------------------- evaluation lab */
let evalBuilt = false;
function rEval() {
  const el = $('#tab-eval'), st = state.status;
  if (!evalBuilt) {
    evalBuilt = true;
    el.innerHTML = `<div class="note warn">Only real predictions are scored against real labels. Synthetic labels are the generator's ground truth, and the synthetic text is templated, so scores on it are optimistic. Use your own labelled data or annotate tweets below for a fair test.</div>
    <div class="card"><h2>Compare algorithms</h2><div class="row">${F('e_ds', 'Dataset', '<select id="e_ds"></select>')}${F('e_kind', 'What to evaluate', '<select id="e_kind"><option value="sentiment">Sentiment classifiers</option><option value="topics">Topic algorithms</option></select>')}${F('e_src', 'Labels from', '<select id="e_src"><option value="dataset">dataset label column / synthetic ground truth</option><option value="annotations">my manual annotations</option></select>')}</div>
      <div class="row" id="e_sp">${F('e_pos', 'Positive if ≥', '<input type="number" id="e_pos" value="0.05" step="0.01">')}${F('e_neg', 'Negative if ≤ −', '<input type="number" id="e_neg" value="0.05" step="0.01">')}</div>
      <div class="row" id="e_tp" style="display:none">${F('e_thr', 'Cosine threshold (pivot)', '<input type="number" id="e_thr" value="0.3" step="0.05">')}${F('e_k', 'k (K-Means, 0 = auto)', '<input type="number" id="e_k" value="0">')}${F('e_min', 'Min topic size', '<input type="number" id="e_min" value="8">')}</div>
      <div id="e_algos" class="inline"></div><button id="e_go">Run evaluation</button><div id="e_out"></div></div>
    <div class="card"><h2>Annotate tweets manually</h2><p class="sub">Label a random sample as positive, negative or neutral. Your labels become the ground truth for "my manual annotations" above.</p>
      <div class="inline"><button class="ghost" id="a_load">Load 10 unlabelled tweets</button><span id="a_count" class="sub"></span></div><div id="a_list"></div></div>
    <div class="card"><h2>Past evaluations</h2><div id="e_hist"></div></div>`;
    $('#e_kind').onchange = drawAlgos; $('#e_go').onclick = e => busy(e.target, runEval); $('#a_load').onclick = loadAnnot; $('#e_ds').onchange = annotCount;
  }
  const keep = $('#e_ds').value;
  $('#e_ds').innerHTML = state.datasets.map(d => `<option value="${d.id}">#${d.id} ${esc(d.name)} [${SRC_SHORT[d.source] || d.source}, ${(d.meta || {}).labelled_sentiment || 0} labelled]</option>`).join('') || '<option value="">(no datasets)</option>';
  const lab = state.datasets.find(d => ((d.meta || {}).labelled_sentiment || 0) >= 20);
  if (keep) $('#e_ds').value = keep; else if (lab) $('#e_ds').value = lab.id;
  drawAlgos(); annotCount(); evalHist();
}
function drawAlgos() {
  const kind = $('#e_kind').value, av = kind === 'sentiment' ? state.status.sentiment : state.status.topics;
  $('#e_sp').style.display = kind === 'sentiment' ? '' : 'none'; $('#e_tp').style.display = kind === 'topics' ? '' : 'none'; $('#e_src').disabled = kind === 'topics';
  $('#e_algos').innerHTML = '<b>Algorithms:</b> ' + Object.entries(av).map(([k, v]) => `<label title="${esc(v.note)}"><input type="checkbox" value="${k}" ${v.ok ? (k === 'transformer' || k === 'embeddings' || k === 'bertopic' ? '' : 'checked') : 'disabled'}> ${k}${v.ok ? '' : ' (unavailable)'}</label>`).join(' ');
}
async function runEval() {
  const kind = $('#e_kind').value, algos = $$('#e_algos input:checked').map(i => i.value);
  const body = {kind, algos, dataset_id: +$('#e_ds').value, source: $('#e_src').value, pos_threshold: +$('#e_pos').value, neg_threshold: +$('#e_neg').value, threshold: +$('#e_thr').value, k: +$('#e_k').value, min_size: +$('#e_min').value};
  $('#e_out').innerHTML = note('Running…'); const r = await api('eval', {method: 'POST', body});
  $('#e_out').innerHTML = evalHtml(r); evalHist();
}
function evalHtml(r) {
  const ds = r.dataset ? `${esc(r.dataset.name)} ${srcTag(r.dataset.source)}` : '';
  if (r.kind === 'sentiment') return `<h3>Sentiment results on ${r.n} labelled tweets · ${ds}</h3><p class="sub">Label distribution: ${esc(JSON.stringify(r.label_distribution))}</p>
    <div class="scroll"><table><tr><th class="l">Algorithm</th><th>Accuracy</th><th>Macro precision</th><th>Macro recall</th><th>Macro F1</th><th>P/R/F1 positive</th><th>P/R/F1 neutral</th><th>P/R/F1 negative</th><th>Time (s)</th><th>ms / 1000 tweets</th></tr>
    ${r.results.map(x => x.error ? `<tr><td class="l">${esc(x.algo)}</td><td class="l wrap" colspan="9" style="color:var(--neg)">${esc(x.error)}</td></tr>` : `<tr><td class="l"><b>${esc(x.algo)}</b></td><td>${fmtN(x.accuracy * 100, 1)}%</td><td>${x.macro_precision.toFixed(3)}</td><td>${x.macro_recall.toFixed(3)}</td><td><b>${x.macro_f1.toFixed(3)}</b></td>${['positive', 'neutral', 'negative'].map(c => `<td>${x.per_class[c].precision.toFixed(2)} / ${x.per_class[c].recall.toFixed(2)} / ${x.per_class[c].f1.toFixed(2)}</td>`).join('')}<td>${x.seconds}</td><td>${x.ms_per_1000}</td></tr>`).join('')}</table></div>
    <div class="grid3">${r.results.filter(x => !x.error).map(x => `<div><h3>${esc(x.algo)}: confusion matrix</h3>${heat(x.confusion, x.classes, 'rows = truth')}</div>`).join('')}</div>`;
  return `<h3>Topic results · ${ds}</h3><p class="sub">Purity and ARI are computed on tweets the algorithm assigned to a topic; coverage is the share of on-topic tweets it assigned. Off-topic chatter is excluded from scoring.</p>
    <div class="scroll"><table><tr><th class="l">Algorithm</th><th>Clusters</th><th>True topics</th><th>Coverage</th><th>Purity</th><th>ARI</th><th>Time (s)</th></tr>${r.results.map(x => x.error ? `<tr><td class="l">${esc(x.algo)}</td><td class="l wrap" colspan="6" style="color:var(--neg)">${esc(x.error)}</td></tr>` : `<tr><td class="l"><b>${esc(x.algo)}</b></td><td>${x.clusters}</td><td>${x.true_topics}</td><td>${fmtN(x.coverage * 100, 1)}%</td><td>${x.purity.toFixed(3)}</td><td>${x.ari.toFixed(3)}</td><td>${x.seconds}</td></tr>`).join('')}</table></div>`;
}
async function evalHist() { const h = await api('evals'); $('#e_hist').innerHTML = h.length ? h.slice(0, 6).map(e => `<details><summary>#${e.id} ${e.kind} · ${esc((e.result.dataset || {}).name || '')} · ${fmtTs(e.created)} · ${esc((e.params.algos || []).join(', '))}</summary>${evalHtml(e.result)}</details>`).join('') : empty('No evaluations run yet.'); }
async function annotCount() { const id = $('#e_ds').value; if (!id) return; const a = await api('annotations?dataset_id=' + id); $('#a_count').textContent = a.length + ' tweet(s) annotated in this dataset'; }
async function loadAnnot() {
  const id = $('#e_ds').value; if (!id) return toast('Choose a dataset', 'err');
  const s = await api(`datasets/${id}/sample?n=10`);
  $('#a_list').innerHTML = s.length ? s.map(x => `<div class="post" data-i="${x.idx}">${esc(x.text)}<div class="inline">${['positive', 'neutral', 'negative'].map(l => `<button class="sm ghost" data-l="${l}" style="color:${SENT[l]};border-color:${SENT[l]}">${l}</button>`).join('')}<span class="sub"></span></div></div>`).join('') : empty('Every tweet in this dataset is already annotated.');
  $$('#a_list .post').forEach(p => $$('button', p).forEach(b => b.onclick = async () => { await api('annotations', {method: 'POST', body: {dataset_id: +id, idx: +p.dataset.i, label: b.dataset.l}}); $('span.sub', p).textContent = 'saved as ' + b.dataset.l; annotCount(); }));
}

/* ---------------------------------------------------------------- history & reports */
function rHistory() {
  const el = $('#tab-history'), runs = state.runs;
  el.innerHTML = `<div class="card"><h2>Analysis history</h2><p class="sub">Every analysis is stored in SQLite. Tick two runs to compare them.</p>${runs.length ? `<div class="scroll"><table><tr><th></th><th class="l">Run</th><th class="l">Dataset</th><th class="l">Created</th><th class="l">Algorithms</th><th>Tweets</th><th>Topics</th><th>Emerging</th><th>Avg sent.</th><th class="l">Export</th><th></th></tr>
    ${runs.map(r => `<tr><td><input type="checkbox" data-s="${r.id}" ${state.sel.has(r.id) ? 'checked' : ''}></td><td class="l"><a href="#dashboard" data-open="${r.id}">#${r.id} ${esc(r.name)}</a></td><td class="l wrap">${srcTag(r.source)} ${esc(r.dataset_name)}</td><td class="l">${fmtTs(r.created)}</td><td class="l">${esc(r.summary.sentiment_algo)} / ${esc(r.summary.topic_algo)}</td><td>${r.summary.tweets}</td><td>${r.summary.topics}</td><td>${r.summary.emerging}</td><td>${r.summary.avg_sentiment}</td>
    <td class="l"><a class="btn sm ghost" href="/api/runs/${r.id}/export?format=topics.csv">topics.csv</a> <a class="btn sm ghost" href="/api/runs/${r.id}/export?format=tweets.csv">tweets.csv</a> <a class="btn sm ghost" href="/api/runs/${r.id}/export?format=json">json</a> <a class="btn sm ghost" href="/api/runs/${r.id}/export?format=html" target="_blank">html report</a></td><td><button class="sm danger" data-d="${r.id}">Delete</button></td></tr>`).join('')}</table></div>
    <div class="inline"><button id="h_cmp">Compare selected runs</button></div>` : empty('No runs yet.')}<div id="h_out"></div></div>`;
  $$('#tab-history [data-s]').forEach(c => c.onchange = () => c.checked ? state.sel.add(+c.dataset.s) : state.sel.delete(+c.dataset.s));
  $$('#tab-history [data-open]').forEach(a => a.onclick = () => selectRun(+a.dataset.open));
  $$('#tab-history [data-d]').forEach(b => b.onclick = async () => { if (!confirm('Delete this run and its alerts?')) return; await api('runs/' + b.dataset.d, {method: 'DELETE'}); state.sel.delete(+b.dataset.d); const cur = state.run && state.run.id === +b.dataset.d; await refresh(); if (cur) await selectRun(state.runs[0] ? state.runs[0].id : null); else render(); toast('Run deleted'); });
  const cb = $('#h_cmp'); if (cb) cb.onclick = () => busy(cb, async () => { const ids = [...state.sel]; if (ids.length !== 2) throw new Error('Tick exactly two runs.'); const c = await api(`compare?a=${Math.min(...ids)}&b=${Math.max(...ids)}`); $('#h_out').innerHTML = compareHtml(c); });
}
function compareHtml(c) {
  const lab = {tweets: 'Tweets analysed', topics: 'Topics', emerging: 'Emerging trends', bursting: 'Bursting topics', avg_sentiment: 'Avg sentiment', positive: 'Positive', negative: 'Negative', neutral: 'Neutral', coverage: 'Coverage', total_reach: 'Reach', avg_retweets: 'Avg retweets', avg_likes: 'Avg likes', unique_users: 'Users', assigned: 'Assigned tweets'};
  return `<h3>Run #${c.a.id} (${esc(c.a.name)}) vs run #${c.b.id} (${esc(c.b.name)})</h3><table><tr><th class="l">Metric</th><th>#${c.a.id}</th><th>#${c.b.id}</th><th>Change</th></tr>${Object.entries(c.kpis).filter(([k]) => lab[k]).map(([k, v]) => `<tr><td class="l">${lab[k]}</td><td>${fmtN(v.a, 3)}</td><td>${fmtN(v.b, 3)}</td><td style="color:${v.delta > 0 ? 'var(--pos)' : v.delta < 0 ? 'var(--neg)' : 'inherit'}">${v.delta > 0 ? '+' : ''}${fmtN(v.delta, 3)}</td></tr>`).join('')}</table>
  <h3>Matched topics (keyword overlap)</h3>${c.matched.length ? `<table><tr><th class="l">Run #${c.a.id}</th><th class="l">Run #${c.b.id}</th><th>Similarity</th><th>Size</th><th class="l">Stage</th><th>Sentiment</th></tr>${c.matched.map(m => `<tr><td class="l wrap">${esc(m.a)}</td><td class="l wrap">${esc(m.b)}</td><td>${m.similarity}</td><td>${m.size_a} → ${m.size_b}</td><td class="l">${esc(m.stage_a)} → ${esc(m.stage_b)}</td><td>${m.sentiment_a.toFixed(2)} → ${m.sentiment_b.toFixed(2)}</td></tr>`).join('')}</table>` : '<p class="sub">No topics matched.</p>'}
  <div class="grid"><div><h3>Only in run #${c.a.id}</h3>${c.only_a.map(t => `<span class="chip">${esc(t.label)} (${t.size})</span>`).join('') || '<span class="sub">none</span>'}</div><div><h3>Only in run #${c.b.id}</h3>${c.only_b.map(t => `<span class="chip">${esc(t.label)} (${t.size})</span>`).join('') || '<span class="sub">none</span>'}</div></div>`;
}

/* ---------------------------------------------------------------- alerts */
let alertsBuilt = false;
async function rAlerts() {
  const el = $('#tab-alerts'), types = state.meta.rule_types;
  const rules = await api('alerts/rules'), events = await api('alerts/events');
  const pin = (r, p) => `<label style="display:inline-block;margin-right:8px">${esc(p.label)} <input type="number" step="${p.step}" value="${r.params[p.key] ?? p.default}" data-p="${p.key}" style="width:80px"></label>`;
  el.innerHTML = `<div class="card"><h2>Alert rules</h2><p class="sub">Rules are checked automatically after every analysis run.</p><div class="scroll"><table><tr><th class="l">On</th><th class="l">Name</th><th class="l">Condition</th><th class="l">Parameters</th><th></th></tr>
    ${rules.map(r => `<tr data-r="${r.id}"><td class="l"><input type="checkbox" data-e ${r.enabled ? 'checked' : ''}></td><td class="l"><input type="text" data-n value="${esc(r.name)}" style="width:210px"></td><td class="l wrap">${esc((types[r.type] || {}).label || r.type)}</td><td class="l wrap">${((types[r.type] || {}).params || []).map(p => pin(r, p)).join('')}</td><td><button class="sm" data-save>Save</button> <button class="sm danger" data-del>Delete</button></td></tr>`).join('')}</table></div>
    <h3>New rule</h3><div class="inline"><select id="n_type" style="width:330px">${Object.entries(types).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join('')}</select><input type="text" id="n_name" placeholder="rule name" style="width:200px"><span id="n_params"></span><button id="n_add">Add rule</button></div></div>
  <div class="card"><h2>Recent alert events</h2>${events.length ? `<div class="scroll"><table><tr><th class="l">Run</th><th class="l">When</th><th class="l">Severity</th><th class="l">Rule</th><th class="l">Message</th></tr>${events.map(e => `<tr><td class="l">#${e.run_id}</td><td class="l">${fmtTs(e.created)}</td><td class="l">${tag2(e.severity)}</td><td class="l">${esc(e.rule_name)}</td><td class="l wrap">${esc(e.message)}</td></tr>`).join('')}</table></div>` : empty('No alerts have fired yet.')}</div>`;
  const readP = scope => Object.fromEntries($$('[data-p]', scope).map(i => [i.dataset.p, +i.value]));
  $$('#tab-alerts tr[data-r]').forEach(tr => { const r = rules.find(x => x.id == tr.dataset.r);
    $('[data-save]', tr).onclick = async () => { try { await api('alerts/rules', {method: 'POST', body: {id: r.id, name: $('[data-n]', tr).value, type: r.type, enabled: $('[data-e]', tr).checked, params: readP(tr)}}); toast('Rule saved'); } catch (e) { toast(e.message, 'err'); } };
    $('[data-del]', tr).onclick = async () => { try { await api('alerts/rules/' + r.id, {method: 'DELETE'}); toast('Rule deleted'); rAlerts(); } catch (e) { toast(e.message, 'err'); } }; });
  const drawNew = () => { $('#n_params').innerHTML = types[$('#n_type').value].params.map(p => pin({params: {}}, p)).join(''); };
  $('#n_type').onchange = drawNew; drawNew();
  $('#n_add').onclick = async () => { try { await api('alerts/rules', {method: 'POST', body: {name: $('#n_name').value, type: $('#n_type').value, enabled: true, params: readP($('#n_params'))}}); toast('Rule added'); rAlerts(); } catch (e) { toast(e.message, 'err'); } };
}

/* ---------------------------------------------------------------- boot */
async function boot() {
  try {
    [state.meta, state.status] = await Promise.all([api('meta'), api('status')]);
    await refresh();
    const first = state.runs[0]; if (first) state.run = await api('runs/' + first.id);
    $('#runSel').value = state.run ? state.run.id : '';
    $('#runSel').onchange = e => selectRun(e.target.value ? +e.target.value : null);
    setBanner();
    window.addEventListener('hashchange', () => show(location.hash.slice(1)));
    show(location.hash.slice(1) || 'dashboard');
  } catch (e) { const ld = $('#loading'); if (ld) ld.remove(); document.body.insertAdjacentHTML('afterbegin', `<div class="note err"><b>Could not start: ${esc(e.message)}</b><br>The TrendScope server may have stopped. Start it again with <span class="mono">python trendscope.py</span> and reload this page.</div>`); }
}
boot();
