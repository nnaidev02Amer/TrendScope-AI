#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
# Dashboard generator for the Twitter trending topics project.
#
# Reads the files written by `main.py -a <tweets> <outdir>`:
#   topics.json, stats.csv, statsnontrending.csv, statstime.csv, sentimentStats.json, summary.txt
# and writes <outdir>/dashboard.html, a single self-contained page (inline SVG + JavaScript,
# no external libraries, works offline).
#
# Usage:
#   python dashboard.py <analysis directory> [--source synthetic|live|unspecified] [-o file.html]
#
# The data-source label is shown in a banner so synthetic demo data can never be mistaken for
# real Twitter/X data. If <outdir>/meta.json exists (written by demo.py) it takes precedence.

import argparse
import csv
import json
import os
import sys

SOURCES = {
    'synthetic': 'SYNTHETIC DEMONSTRATION DATA: generated tweets, not real Twitter/X content',
    'live': 'LIVE TWITTER/X DATA: tweets retrieved with main.py -r',
    'unspecified': 'DATA SOURCE NOT SPECIFIED: analysis of a user-supplied tweets file',
}


def readTsv(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def num(x):
    try:
        v = float(x)
        return int(v) if v == int(v) else v
    except (ValueError, TypeError):
        return 0


def build(outDir, source):
    meta = {'source': source}
    metaPath = os.path.join(outDir, 'meta.json')
    if os.path.isfile(metaPath):
        with open(metaPath, 'r', encoding='utf-8') as f:
            meta.update(json.load(f))
    if meta.get('source') not in SOURCES:
        meta['source'] = 'unspecified'

    with open(os.path.join(outDir, 'sentimentStats.json'), encoding='utf-8') as f:
        sentiment = json.loads(f.readline())

    with open(os.path.join(outDir, 'topics.json'), encoding='utf-8') as f:
        topicsRaw = [json.loads(line) for line in f if line.strip()]
    rows = readTsv(os.path.join(outDir, 'stats.csv'))
    if len(rows) != len(topicsRaw):
        sys.exit('stats.csv and topics.json disagree on the number of topics')

    topics = []
    for i, (t, r) in enumerate(zip(topicsRaw, rows)):
        terms = sorted(t['terms'].items(), key=lambda kv: kv[1], reverse=True)
        samples = sorted(t['tweets'], key=lambda tw: tw['occurrences'], reverse=True)[:5]
        topics.append({
            'id': i + 1,
            'terms': [k for k, _ in terms[:6]],
            'tweets': num(r['tweetCount']),
            'unique': num(r['singleTweetCount']),
            'views': num(r['views']),
            'avgViews': num(r['avgViews']),
            'avgRetweets': num(r['avgRetweets']),
            'avgFavorites': num(r['avgFavorites']),
            'avgUrls': num(r['avgURLsCount']),
            'avgSentiment': num(r['avgSentiment']),
            'nsr': num(r['nsr']),
            'pos': num(r['positiveCount']),
            'neg': num(r['negativeCount']),
            'neu': num(r['neutralCount']),
            'samples': [{'text': s['text'], 'n': s['occurrences'], 'sent': s['sentiment'] / s['occurrences']}
                        for s in samples],
        })

    nonTrending = readTsv(os.path.join(outDir, 'statsnontrending.csv'))
    nonTrendingTweets = num(nonTrending[0]['tweetCount']) if nonTrending else 0

    timeline = {}
    timePath = os.path.join(outDir, 'statstime.csv')
    if os.path.isfile(timePath):
        for r in readTsv(timePath):
            s = num(r['secs'])
            entry = timeline.setdefault(s, {'secs': s, 'topics': 0, 'tweets': 0})
            entry['topics'] += 1
            entry['tweets'] += num(r['tweetCount'])

    summary = ''
    summaryPath = os.path.join(outDir, 'summary.txt')
    if os.path.isfile(summaryPath):
        with open(summaryPath, 'r', encoding='utf-8') as f:
            summary = f.read()

    return {
        'summary': summary,
        'meta': meta,
        'banner': SOURCES[meta['source']],
        'sentiment': sentiment,
        'topics': topics,
        'nonTrendingTweets': nonTrendingTweets,
        'timeline': sorted(timeline.values(), key=lambda e: e['secs']),
    }


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Twitter Trending Topics: analysis dashboard</title>
<style>
:root{--pos:#2e8b57;--neg:#c0392b;--neu:#8a94a0;--ink:#1f2933;--mute:#616e7c;--line:#e4e7eb;--bg:#f5f7fa;--card:#fff;--acc:#2c5aa0}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
header{background:#14213d;color:#fff;padding:20px 32px}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:#c5cde0}
.banner{padding:10px 32px;font-weight:600;font-size:13px;letter-spacing:.02em}
.banner.synthetic{background:#fff3cd;color:#7a5b00;border-bottom:2px solid #f0c33c}
.banner.live{background:#d9f2e3;color:#14532d;border-bottom:2px solid #2e8b57}
.banner.unspecified{background:#e4e7eb;color:#323f4b;border-bottom:2px solid #8a94a0}
main{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.kpi b{display:block;font-size:26px;line-height:1.2}.kpi span{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(480px,100%),1fr));gap:18px;margin-bottom:18px}
h2{font-size:16px;margin:0 0 4px}.sub{color:var(--mute);font-size:12.5px;margin:0 0 12px}
.legend{display:flex;gap:14px;font-size:12.5px;margin-bottom:8px;flex-wrap:wrap}.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{cursor:pointer;color:var(--mute);font-weight:600;user-select:none;position:sticky;top:0;background:#fff}th:hover{color:var(--acc)}
th:first-child,td:first-child,th.l,td.l{text-align:left}td.l{white-space:normal}tr.row{cursor:pointer}tr.row:hover{background:#f0f4fa}
.sbar{display:flex;height:12px;border-radius:3px;overflow:hidden;min-width:110px;background:var(--line)}.sbar span{display:block;height:100%}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11.5px;font-weight:600;color:#fff}
.samples{background:#fafbfc;padding:10px 14px;text-align:left;white-space:normal}.samples div{padding:3px 0;border-bottom:1px dashed var(--line)}
.bars .r{display:grid;grid-template-columns:150px 1fr 92px;gap:8px;align-items:center;margin:5px 0;font-size:12.5px}
.bars .lab{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
svg text{font:11px system-ui,sans-serif;fill:#52606d}
details{margin-top:6px}summary{cursor:pointer;color:var(--acc);font-weight:600}pre{background:#f0f2f5;padding:12px;border-radius:6px;max-height:340px;overflow:auto;white-space:pre-wrap;font-size:12px}
@media(max-width:600px){.terms{display:none}td.l{white-space:nowrap}header,.banner{padding-left:16px;padding-right:16px}main{padding:16px 10px 40px}.bars .r{grid-template-columns:110px 1fr 70px}.card{padding:12px}}
ul.notes{margin:6px 0 0 18px;padding:0;color:var(--mute);font-size:13px}
</style></head><body>
<header><h1>Twitter Trending Topics: analysis dashboard</h1>
<p>Document-pivot topic detection (TF-IDF + cosine similarity) with VADER sentiment</p></header>
<div id="banner" class="banner"></div>
<main>
<section class="kpis" id="kpis"></section>
<section class="grid">
 <div class="card"><h2>Overall sentiment</h2><p class="sub">All analysed tweets (retweets counted each time).</p><div id="donut"></div></div>
 <div class="card"><h2>Sentiment by topic</h2><p class="sub">Share of positive, negative and neutral tweets in each trending topic.</p>
  <div class="legend"><span><i style="background:var(--pos)"></i>Positive</span><span><i style="background:var(--neg)"></i>Negative</span><span><i style="background:var(--neu)"></i>Neutral</span></div><div id="topicBars" class="bars"></div></div>
</section>
<section class="grid">
 <div class="card"><h2>Topic size vs sentiment</h2><p class="sub">Each bubble is a topic. Horizontal: net sentiment rate (positive minus negative share). Vertical: tweets. Bubble area: audience (followers reached).</p><div id="bubble"></div></div>
 <div class="card"><h2>Engagement per topic</h2><p class="sub">Average retweets and favourites per tweet.</p>
  <div class="legend"><span><i style="background:#2c5aa0"></i>Avg retweets</span><span><i style="background:#e0a030"></i>Avg favourites</span></div><div id="engage" class="bars"></div></div>
</section>
<section class="card" style="margin-bottom:18px"><h2>Detected topics</h2><p class="sub">Click a column header to sort, click a row to see its most repeated tweets. Topic labels are the highest TF-IDF terms (Porter-stemmed).</p><div style="overflow-x:auto"><table id="tbl"></table></div></section>
<section class="grid">
 <div class="card"><h2>Trending topics over time</h2><p class="sub">Tweets inside trending topics as the stream is processed (a snapshot every 100 tweets).</p><div id="timeline"></div></div>
 <div class="card"><h2>About this analysis</h2><ul class="notes" id="notes"></ul></div>
</section>
<section class="card"><h2>Text summary</h2><details><summary>Show summary.txt</summary><pre id="summary"></pre></details></section>
</main>
<script>
const D=__DATA__;
const C={pos:'#2e8b57',neg:'#c0392b',neu:'#8a94a0'};
const $=id=>document.getElementById(id);
const fmt=n=>Number(n).toLocaleString('en-GB',{maximumFractionDigits:2});
const label=t=>t.terms.slice(0,3).join(' ');
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs,text){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;return e}
function pct(a,b){return b?(100*a/b):0}

// banner + KPIs
$('banner').textContent=D.banner;$('banner').className='banner '+D.meta.source;
const S=D.sentiment,T=D.topics;
const inTopics=T.reduce((a,t)=>a+t.tweets,0);
const k=[[S.total,'Tweets analysed'],[T.length,'Trending topics'],[fmt(pct(inTopics,S.total))+'%','Tweets in trending topics'],
[fmt(pct(S.positive,S.total))+'%','Positive'],[fmt(pct(S.negative,S.total))+'%','Negative'],[fmt(pct(S.neutral,S.total))+'%','Neutral']];
$('kpis').innerHTML=k.map(x=>'<div class="card kpi"><b>'+x[0]+'</b><span>'+x[1]+'</span></div>').join('');

// donut
(function(){const w=440,h=230,cx=110,cy=115,R=90,r=55;const svg=el('svg',{viewBox:'0 0 '+w+' '+h,width:'100%',role:'img','aria-label':'Sentiment donut chart'});
let a0=-Math.PI/2;[['Positive',S.positive,C.pos],['Negative',S.negative,C.neg],['Neutral',S.neutral,C.neu]].forEach(([n,v,c],i)=>{
const a1=a0+2*Math.PI*v/S.total,l=a1-a0>Math.PI?1:0,p=(R,a)=>[cx+R*Math.cos(a),cy+R*Math.sin(a)];
const [x0,y0]=p(R,a0),[x1,y1]=p(R,a1-1e-4),[x2,y2]=p(r,a1-1e-4),[x3,y3]=p(r,a0);
const path=el('path',{d:`M${x0},${y0}A${R},${R} 0 ${l} 1 ${x1},${y1}L${x2},${y2}A${r},${r} 0 ${l} 0 ${x3},${y3}Z`,fill:c});path.appendChild(el('title',{},n+': '+v));svg.appendChild(path);a0=a1;
svg.appendChild(el('rect',{x:240,y:70+i*38,width:14,height:14,fill:c,rx:2}));
svg.appendChild(el('text',{x:262,y:82+i*38,style:'font-size:14px;fill:#1f2933'},n+': '+v+' ('+fmt(pct(v,S.total))+'%)'))});
svg.appendChild(el('text',{x:cx,y:cy-2,'text-anchor':'middle',style:'font-size:22px;font-weight:700;fill:#1f2933'},S.total));
svg.appendChild(el('text',{x:cx,y:cy+16,'text-anchor':'middle'},'tweets'));$('donut').appendChild(svg)})();

// per-topic stacked bars + engagement bars
const maxEng=Math.max(1,...T.map(t=>Math.max(t.avgRetweets,t.avgFavorites)));
$('topicBars').innerHTML=T.map(t=>{const n=t.pos+t.neg+t.neu;
return '<div class="r"><div class="lab" title="'+label(t)+'">#'+t.id+' '+label(t)+'</div><div class="sbar" title="+'+t.pos+' / -'+t.neg+' / ~'+t.neu+'"><span style="width:'+pct(t.pos,n)+'%;background:var(--pos)"></span><span style="width:'+pct(t.neg,n)+'%;background:var(--neg)"></span><span style="width:'+pct(t.neu,n)+'%;background:var(--neu)"></span></div><div>'+n+'</div></div>'}).join('');
$('engage').innerHTML=T.map(t=>'<div class="r"><div class="lab">#'+t.id+' '+label(t)+'</div><div><div class="sbar" style="height:7px;margin-bottom:2px;background:none"><span style="width:'+pct(t.avgRetweets,maxEng)+'%;background:#2c5aa0"></span></div><div class="sbar" style="height:7px;background:none"><span style="width:'+pct(t.avgFavorites,maxEng)+'%;background:#e0a030"></span></div></div><div>'+fmt(t.avgRetweets)+' / '+fmt(t.avgFavorites)+'</div></div>').join('');

// bubble chart
(function(){const w=520,h=330,L=50,B=40,Tp=14,Rr=14;const svg=el('svg',{viewBox:'0 0 '+w+' '+h,width:'100%',role:'img','aria-label':'Topic size versus sentiment'});
const maxY=Math.max(...T.map(t=>t.tweets))*1.15,maxV=Math.max(...T.map(t=>t.views));
const X=v=>L+(v+1.2)/2.4*(w-L-Rr),Y=v=>h-B-v/maxY*(h-B-Tp);
svg.appendChild(el('rect',{x:L,y:Tp,width:X(-.2)-L,height:h-B-Tp,fill:'#c0392b',opacity:.05}));
svg.appendChild(el('rect',{x:X(.2),y:Tp,width:w-Rr-X(.2),height:h-B-Tp,fill:'#2e8b57',opacity:.07}));
[-1,-.5,0,.5,1].forEach(v=>{svg.appendChild(el('line',{x1:X(v),x2:X(v),y1:Tp,y2:h-B,stroke:'#e4e7eb'}));svg.appendChild(el('text',{x:X(v),y:h-B+15,'text-anchor':'middle'},v))});
for(let i=0;i<=4;i++){const v=maxY*i/4;svg.appendChild(el('line',{x1:L,x2:w-Rr,y1:Y(v),y2:Y(v),stroke:'#e4e7eb'}));svg.appendChild(el('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'},Math.round(v)))}
svg.appendChild(el('text',{x:(L+w)/2,y:h-6,'text-anchor':'middle'},'Net sentiment rate  (negative  <  0  <  positive)'));
const yl=el('text',{x:14,y:h/2,'text-anchor':'middle',transform:'rotate(-90 14 '+h/2+')'},'Tweets in topic');svg.appendChild(yl);
T.forEach(t=>{const c=t.nsr>.2?C.pos:t.nsr<-.2?C.neg:C.neu,r=8+22*Math.sqrt(t.views/maxV);
const g=el('g',{});const ci=el('circle',{cx:X(t.nsr),cy:Y(t.tweets),r:r,fill:c,'fill-opacity':.55,stroke:c});
ci.appendChild(el('title',{},'#'+t.id+' '+label(t)+'\n'+t.tweets+' tweets, nsr '+t.nsr+', audience '+fmt(t.views)));g.appendChild(ci);
g.appendChild(el('text',{x:X(t.nsr),y:Y(t.tweets)+4,'text-anchor':'middle',style:'font-weight:700;fill:#1f2933'},t.id));svg.appendChild(g)});$('bubble').appendChild(svg)})();

// timeline
(function(){const tl=D.timeline;if(tl.length<2){$('timeline').textContent='Not enough snapshots to draw a timeline.';return}
const w=520,h=270,L=46,B=38,Tp=14,R=70;const svg=el('svg',{viewBox:'0 0 '+w+' '+h,width:'100%',role:'img','aria-label':'Trending topics over time'});
const maxX=tl[tl.length-1].secs||1,maxY=Math.max(...tl.map(e=>e.tweets))*1.1;
const X=v=>L+v/maxX*(w-L-R),Y=v=>h-B-v/maxY*(h-B-Tp);
for(let i=0;i<=4;i++){const v=maxY*i/4;svg.appendChild(el('line',{x1:L,x2:w-R,y1:Y(v),y2:Y(v),stroke:'#e4e7eb'}));svg.appendChild(el('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'},Math.round(v)))}
for(let i=0;i<=4;i++){const v=maxX*i/4;svg.appendChild(el('text',{x:X(v),y:h-B+15,'text-anchor':'middle'},Math.round(v)))}
svg.appendChild(el('text',{x:(L+w-R)/2,y:h-6,'text-anchor':'middle'},'Seconds since first tweet'));
svg.appendChild(el('text',{x:14,y:h/2,'text-anchor':'middle',transform:'rotate(-90 14 '+h/2+')'},'Tweets in trending topics'));
svg.appendChild(el('polyline',{points:tl.map(e=>X(e.secs)+','+Y(e.tweets)).join(' '),fill:'none',stroke:'#2c5aa0','stroke-width':2.5}));
tl.forEach(e=>{const c=el('circle',{cx:X(e.secs),cy:Y(e.tweets),r:4,fill:'#2c5aa0'});c.appendChild(el('title',{},e.tweets+' tweets in '+e.topics+' trending topics at '+e.secs+'s'));svg.appendChild(c)});
const last=tl[tl.length-1];svg.appendChild(el('text',{x:X(last.secs)+6,y:Y(last.tweets)+4},last.topics+' topics'));$('timeline').appendChild(svg)})();

// sortable table
let sortKey='tweets',sortDir=-1;const open=new Set();
const cols=[['id','Topic',1],['tweets','Tweets'],['views','Audience'],['avgRetweets','Avg RT'],['avgFavorites','Avg fav'],['nsr','Net sentiment'],['mix','Sentiment mix',0]];
function dom(t){return t.nsr>.2?['Positive',C.pos]:t.nsr<-.2?['Negative',C.neg]:['Mixed / neutral',C.neu]}
function render(){const rows=[...T].sort((a,b)=>sortDir*((a[sortKey]>b[sortKey])-(a[sortKey]<b[sortKey])));
let h='<thead><tr>'+cols.map(c=>'<th'+(c[0]==='id'?' class="l"':'')+(c[2]===0?'':' data-k="'+c[0]+'"')+'>'+c[1]+(sortKey===c[0]?(sortDir<0?' ▼':' ▲'):'')+'</th>').join('')+'<th class="l terms">Top terms</th></tr></thead><tbody>';
rows.forEach(t=>{const n=t.pos+t.neg+t.neu,d=dom(t);
h+='<tr class="row" data-id="'+t.id+'"><td class="l">#'+t.id+' <span class="tag" style="background:'+d[1]+'">'+d[0]+'</span></td><td>'+t.tweets+'</td><td>'+fmt(t.views)+'</td><td>'+fmt(t.avgRetweets)+'</td><td>'+fmt(t.avgFavorites)+'</td><td>'+t.nsr.toFixed(2)+'</td>'+
'<td><div class="sbar"><span style="width:'+pct(t.pos,n)+'%;background:var(--pos)"></span><span style="width:'+pct(t.neg,n)+'%;background:var(--neg)"></span><span style="width:'+pct(t.neu,n)+'%;background:var(--neu)"></span></div></td><td class="l terms" style="white-space:normal">'+t.terms.slice(0,5).join(', ')+'</td></tr>';
if(open.has(t.id))h+='<tr><td colspan="8" class="samples">'+t.samples.map(s=>'<div><b style="color:'+(s.sent>0?C.pos:s.sent<0?C.neg:C.neu)+'">'+(s.sent>0?'+':s.sent<0?'−':'~')+'</b> ('+s.n+'×) '+s.text.replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))+'</div>').join('')+'</td></tr>'});
$('tbl').innerHTML=h+'</tbody>';
$('tbl').querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const kk=th.dataset.k;if(kk===sortKey)sortDir*=-1;else{sortKey=kk;sortDir=-1}render()});
$('tbl').querySelectorAll('tr.row').forEach(tr=>tr.onclick=()=>{const id=+tr.dataset.id;open.has(id)?open.delete(id):open.add(id);render()})}
render();

// notes
$('notes').innerHTML=[
'Data source: <b>'+D.banner.split(':')[0].toLowerCase()+'</b>.',
D.nonTrendingTweets?D.nonTrendingTweets+' tweets belong to groups smaller than the minimum topic size and are not shown as topics.':'Every tweet belongs to a topic of at least the minimum size.',
'Sentiment uses VADER compound scores: a tweet is positive above 0, negative below 0.',
'Net sentiment rate = (positive tweets minus negative tweets) / tweets in the topic.',
'Audience is the sum of the followers of the authors of the tweets in a topic.',
D.meta.source==='synthetic'?'The synthetic tweets are template sentences, so topics are cleaner than real Twitter data would give.':'Results depend on the tweet sample and on the similarity threshold and minimum topic size used.'
].map(x=>'<li>'+x+'</li>').join('');
$('summary').textContent=D.summary||'summary.txt not available';
</script></body></html>
'''


def main():
    ap = argparse.ArgumentParser(description='Build dashboard.html from a main.py -a output directory.')
    ap.add_argument('outdir')
    ap.add_argument('--source', choices=list(SOURCES), default='unspecified')
    ap.add_argument('-o', '--output', help='output file (default: <outdir>/dashboard.html)')
    args = ap.parse_args()
    if not os.path.isdir(args.outdir):
        sys.exit('Not a directory: ' + args.outdir)
    data = build(args.outdir, args.source)
    blob = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    target = args.output or os.path.join(args.outdir, 'dashboard.html')
    with open(target, 'w', encoding='utf-8') as f:
        f.write(TEMPLATE.replace('__DATA__', blob))
    print('Dashboard written to', target)


if __name__ == '__main__':
    main()
