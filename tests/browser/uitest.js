const sleep = ms => new Promise(r => setTimeout(r, ms));
async function until(fn, label, ms = 90000) { const t0 = Date.now(); while (Date.now() - t0 < ms) { try { const v = fn(); if (v) return v; } catch (e) {} await sleep(100); } throw new Error('timeout: ' + label); }
const results = [];
const ok = (name, cond, extra) => results.push((cond ? 'PASS ' : 'FAIL ') + name + (extra !== undefined ? ' [' + extra + ']' : ''));
const go = async tab => { location.hash = '#' + tab; await sleep(50); await until(() => $('#tab-' + tab).classList.contains('on'), 'tab ' + tab); };
const type = (sel, v) => { const e = $(sel); e.value = v; e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true})); };
(async () => {
  try {
    await until(() => state.run && $('#tab-dashboard').querySelector('.kpi'), 'dashboard');
    ok('dashboard KPIs render', $$('#tab-dashboard .kpi').length === 7);
    ok('ranking table has rows', $$('#d_table tr.click').length === 6, $$('#d_table tr.click').length);
    type('#d_q', 'strike'); ok('search filter narrows table', $$('#d_table tr.click').length === 1);
    type('#d_q', ''); type('#d_stage', 'declining'); ok('stage filter', $$('#d_table tr.click').length === 3, $$('#d_table tr.click').length);
    type('#d_stage', ''); $('#d_em').checked = true; $('#d_em').dispatchEvent(new Event('input')); ok('emerging-only filter', $$('#d_table tr.click').length === 1, $$('#d_table tr.click').length);
    $('#d_em').checked = false; $('#d_em').dispatchEvent(new Event('input'));
    $$('#d_table th.s')[2].click(); ok('column sort toggles', /▼|▲/.test($('#d_table').textContent));
    ok('synthetic banner shown', /SYNTHETIC DEMONSTRATION DATA/.test($('#banner').textContent));

    await go('studio'); await until(() => $('#s_ds').value, 'studio form');
    type('#s_kw', 'strike, climate'); type('#s_lang', 'en'); type('#s_name', 'UI filtered run'); type('#s_min', '5'); type('#s_k', '2');
    type('#s_cfgname', 'ui cfg'); $('#s_save').click();
    await until(() => [...$('#s_cfgs').options].some(o => o.text === 'ui cfg'), 'config saved');
    ok('save configuration', true);
    $('#s_run').click();
    await until(() => /Run #\d+ finished/.test($('#s_result').textContent), 'run finished', 150000);
    ok('studio run completes', true, $('#s_result').textContent.slice(0, 70));
    ok('new run selected and filtered', state.run.name === 'UI filtered run' && state.run.result.kpis.tweets < 2330, state.run.result.kpis.tweets);
    ok('keyword filter stored in config', state.run.config.keywords.join() === 'strike,climate');
    type('#s_kw', ''); $('#s_load').click(); await until(() => $('#s_kw').value === 'strike, climate', 'load config');
    ok('load saved configuration refills form', true);
    $('#s_del').click(); await until(() => /no saved/.test($('#s_cfgs').textContent), 'delete config'); ok('delete configuration', true);

    await go('trends'); await until(() => $('#tab-trends [data-c]'), 'trends');
    const boxes = $$('#tab-trends [data-c]');
    ok('trends tab lists topics', boxes.length >= 2, boxes.length);
    boxes[0].checked = true; boxes[0].dispatchEvent(new Event('change')); boxes[1].checked = true; boxes[1].dispatchEvent(new Event('change'));
    ok('topic comparison draws chart', !!$('#t_cmp svg'));
    await go('topics'); const items = $$('#tab-topics .list div');
    items[items.length - 1].click(); ok('topic selection changes detail', $('#tab-topics .list div.on') === $$('#tab-topics .list div')[items.length - 1]);
    ok('explorer shows posts and evolution', $$('#tab-topics .post').length >= 1 && /Topic evolution/.test($('#tab-topics').textContent));
    await go('sentiment'); ok('sentiment tab shows experimental emotions', /experimental/.test($('#tab-sentiment').textContent));

    await go('data'); await until(() => $('#g_topics tr[data-i]'), 'generator');
    $('#g_add').click(); const trs = $$('#g_topics tr[data-i]'); const last = trs[trs.length - 1];
    $('[data-k=name]', last).value = 'Custom spike'; $('[data-k=keywords]', last).value = 'quantum, qubit, lattice'; $('[data-k=shape]', last).value = 'burst';
    type('#g_n', '600'); type('#g_name', 'UI generated'); $('#g_go').click();
    await until(() => /UI generated/.test($('#dc_list').textContent), 'dataset generated', 60000);
    ok('generator creates dataset with custom topic', /SYNTHETIC/.test($('#dc_list').textContent));
    const lines = ['stamp,body,n_likes']; for (let i = 0; i < 80; i++) lines.push('2025-04-0' + (1 + i % 5) + ' ' + (10 + i % 10) + ':00:00,"custom upload tweet about transit ' + (i % 4) + ' delays and the metro line",' + i);
    const dt = new DataTransfer(); dt.items.add(new File([lines.join('\n')], 'mydata.csv', {type: 'text/csv'})); $('#up_file').files = dt.files; $('#up_file').dispatchEvent(new Event('change'));
    await until(() => $('#m_text'), 'upload preview');
    ok('upload preview lists columns and auto-maps text', /stamp/.test($('#up_out').textContent) && $('#m_text').value === 'body');
    type('#m_text', 'body'); type('#m_timestamp', 'stamp'); type('#m_likes', 'n_likes'); $('#up_val').click();
    await until(() => $('#up_go') && !$('#up_go').disabled, 'validation');
    ok('field mapping validates 80 rows', /80/.test($('#up_out .kpis').textContent), $('#up_out .kpis').textContent.replace(/\s+/g, ' ').slice(0, 60));
    type('#up_name', 'UI upload'); $('#up_go').click(); await until(() => /UI upload/.test($('#dc_list').textContent), 'uploaded');
    ok('import adds UPLOADED dataset', /UPLOADED/.test($('#dc_list').textContent));
    $$('#dc_list [data-view]')[0].click(); await until(() => /Time/.test($('#dc_detail').textContent), 'inspect'); ok('inspect shows rows', true);
    ok('live X panel disabled without token', $('#l_go').disabled && /No token/.test($('#live_out').textContent));

    await go('eval'); await until(() => $('#e_ds').value, 'eval form');
    $('#e_go').click(); await until(() => /Sentiment results/.test($('#e_out').textContent), 'eval result', 90000);
    ok('sentiment evaluation table + confusion matrices', $$('#e_out table.cm').length >= 2 && /Macro F1/.test($('#e_out').textContent), $$('#e_out table.cm').length);
    ok('unavailable algorithms are disabled', $$('#e_algos input:disabled').length >= 2);
    type('#e_kind', 'topics'); $('#e_go').click(); await until(() => /Topic results/.test($('#e_out').textContent), 'topic eval', 150000);
    ok('topic evaluation shows purity and ARI', /Purity/.test($('#e_out').textContent) && /ARI/.test($('#e_out').textContent));
    $('#a_load').click(); await until(() => $$('#a_list .post').length === 10, 'annotation sample');
    $$('#a_list .post')[0].querySelector('button[data-l=positive]').click(); await until(() => /saved as positive/.test($('#a_list').textContent), 'annotation saved');
    ok('manual annotation saved', true);

    await go('alerts'); await until(() => $('#n_add'), 'alerts');
    type('#n_name', 'UI rule'); type('#n_type', 'volume'); $('#n_params [data-p=size]').value = '5'; $('#n_add').click();
    await until(() => $$('#tab-alerts [data-n]').some(i => i.value === 'UI rule'), 'rule added'); ok('add alert rule', true);
    const tr = $$('#tab-alerts tr[data-r]').find(t => $('[data-n]', t).value === 'UI rule'); $('[data-e]', tr).checked = false; $('[data-save]', tr).click(); await sleep(500);
    $('[data-del]', tr).click(); await until(() => !$$('#tab-alerts [data-n]').some(i => i.value === 'UI rule'), 'rule deleted'); ok('disable, save and delete alert rule', true);

    await go('history'); await until(() => $$('#tab-history [data-s]').length >= 2, 'history rows');
    const cbs = $$('#tab-history [data-s]'); cbs[0].checked = true; cbs[0].dispatchEvent(new Event('change')); cbs[1].checked = true; cbs[1].dispatchEvent(new Event('change'));
    $('#h_cmp').click(); await until(() => /Matched topics/.test($('#h_out').textContent), 'compare');
    ok('run comparison renders KPI deltas', /Tweets analysed/.test($('#h_out').textContent));
    ok('export links present', $$('#tab-history a[href*="export?format="]').length >= 8);
    const csv = await (await fetch($$('#tab-history a[href*="topics.csv"]')[0].href)).text(); ok('topics.csv export downloads', /rank,topic,stage/.test(csv));
    const htm = await (await fetch($$('#tab-history a[href*="format=html"]')[0].href)).text(); ok('html report downloads with data-source banner', /DEMONSTRATION DATA|UPLOAD|LIVE/.test(htm));
    window.confirm = () => true; const before = state.runs.length; $$('#tab-history [data-d]')[0].click(); await until(() => state.runs.length === before - 1, 'delete run'); ok('delete run from history', true);
  } catch (e) { results.push('ERROR ' + e.message); }
  $('#T').textContent = results.join('\n') + '\nDONE';
})();
