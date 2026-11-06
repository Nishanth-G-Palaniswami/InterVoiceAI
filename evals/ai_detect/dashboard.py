"""Cheat-detection dashboard: one self-contained HTML page from the stress test and the analyzed calls.

    uv run python evals/ai_detect/dashboard.py            # writes evals/ai_detect/dashboard.html
    uv run python evals/ai_detect/dashboard.py --open     # and opens it

Sections: headline numbers with their do-nothing baselines, catch vs false-flag per detector variant, the four
tiers, every analyzed call (calls/*.events.jsonl, or --calls <dir|file>) with per-answer verdicts and evidence,
and how it works plus the caveats. Works offline; nothing is fetched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import subprocess
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
try:  # .env.local may hold DATABASE_URL_READONLY
    from dotenv import load_dotenv
    load_dotenv(HERE.parents[1] / '.env.local')
except ImportError:
    pass
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from analyze_call import load_events  # noqa: E402
from run import AI_TIERS, HUMAN_TIERS, evaluate, load_fixtures, rate  # noqa: E402

from interviewer.ai_text import (  # noqa: E402
    answer_confidence,
    answer_texts,
    call_confidence,
    observe,
    problem_areas,
)
from interviewer.delivery import answer_delivery  # noqa: E402


def detector_table(rows: list[dict]) -> list[dict]:
    rng = random.Random(2026)
    coin = {r["id"]: rng.random() < 0.5 for r in rows}
    median_words = sorted(r["words"] for r in rows)[len(rows) // 2] if rows else 0
    detectors = [
        ("Never flag (do nothing)", "baseline", lambda r: False),
        ("Always flag", "baseline", lambda r: True),
        ("Coin flip", "baseline", lambda r: coin[r["id"]]),
        (f"Long answer only (> {median_words} words)", "baseline", lambda r: r["words"] > median_words),
        ("Text signals only", "detector", lambda r: r["reads_generated"]),
        ("Voice signals only", "detector", lambda r: r["sounds_read"]),
        ("Either side", "detector", lambda r: r["reads_generated"] or r["sounds_read"]),
        ("Both sides agree", "detector", lambda r: r["verdict"] == "ai_assisted"),
    ]
    out = []
    for name, kind, pred in detectors:
        c, f, na, nh = rate(rows, pred)
        out.append({"name": name, "kind": kind, "catch": c, "false_flag": f, "n_ai": na, "n_human": nh})
    return out


def tier_table(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["tier"]].append(r)
    out = []
    for tier in (*AI_TIERS, *HUMAN_TIERS):
        rs = by.get(tier, [])
        if not rs:
            continue
        n = len(rs)
        out.append({"tier": tier, "n": n, "text": sum(r["reads_generated"] for r in rs),
                    "voice": sum(r["sounds_read"] for r in rs), "both": sum(r["verdict"] == "ai_assisted" for r in rs),
                    "median_text_signals": sorted(r["text_signals"] for r in rs)[n // 2]})
    return out


def analyzed_calls(sources: list[pathlib.Path], extra: list[tuple[str, list[dict]]] | None = None) -> list[dict]:
    calls = []
    feeds: list[tuple[str, list[dict] | None, pathlib.Path | None]] = [(p.stem, None, p) for p in sources]
    feeds = [(cid, evs, None) for cid, evs in (extra or [])] + feeds
    for cid, events, p in feeds:
        try:
            if events is None:
                events = load_events(p)
        except Exception as e:  # a half-written dump should not kill the dashboard
            calls.append({"id": cid, "error": str(e), "answers": []})
            continue
        deliveries = {d.question_id: d for d in answer_delivery(events)}
        texts = answer_texts(events)
        obs = [observe(q, texts.get(q, ""), deliveries.get(q)) for q in texts]
        started = next((e.get("occurred_at") for e in events if e.get("type") == "call_started"), None)
        ended = next((e.get("payload", {}) for e in events if e.get("type") == "call_ended"), {})
        answers = []
        for o in obs:
            d = o.to_dict()
            d["confidence"] = answer_confidence(o)
            answers.append(d)
        calls.append({
            "id": cid.replace(".events", ""), "started_at": started, "outcome": ended.get("outcome"),
            "duration_s": ended.get("duration_s"), "turns": sum(1 for e in events if e.get("type") == "turn"),
            "confidence": call_confidence(obs), "problems": problem_areas(obs, events), "answers": answers,
        })
    return calls


def neon_calls(db_url: str, hours: float = 6.0, limit: int = 8) -> list[tuple[str, list[dict]]]:
    """Events for the most recent calls from the reference product's `agent_events` table (BUILD_PLAN §9).
    Read-only; polled on every refresh in --serve mode so a call in progress fills in as its events land."""
    import psycopg

    sql = """
        with recent as (
            select call_id, max(created_at) as last_at from agent_events
            where created_at > now() - (%s * interval '1 hour')
            group by call_id order by last_at desc limit %s)
        select e.call_id, e.seq, e.type, e.payload, e.occurred_at
        from agent_events e join recent r on r.call_id = e.call_id
        order by r.last_at desc, e.seq"""
    out: dict[str, list[dict]] = {}
    with psycopg.connect(db_url, connect_timeout=5) as conn, conn.cursor() as cur:
        try:
            cur.execute(sql, (hours, limit))
        except psycopg.errors.UndefinedColumn:  # no created_at column: fall back to the newest call_ids by seq
            conn.rollback()
            cur.execute("""select e.call_id, e.seq, e.type, e.payload, null from agent_events e
                           where e.call_id in (select call_id from agent_events group by call_id
                                               order by max(seq) desc limit %s) order by e.call_id, e.seq""", (limit,))
        for call_id, seq, type_, payload, occurred_at in cur.fetchall():
            if isinstance(payload, str):
                payload = json.loads(payload)
            out.setdefault(str(call_id), []).append({"call_id": call_id, "seq": seq, "type": type_, "payload": payload,
                                                     "occurred_at": str(occurred_at) if occurred_at else None})
    return list(out.items())


def build_data(call_sources: list[pathlib.Path], db_url: str | None = None) -> dict:
    rows = evaluate(load_fixtures(), "synthetic")
    extra: list[tuple[str, list[dict]]] = []
    db_error = None
    if db_url:
        try:
            extra = neon_calls(db_url)
        except Exception as e:  # the dashboard must render even when the database is unreachable
            db_error = f"{type(e).__name__}: {str(e)[:160]}"
    return {
        "db_feed": bool(db_url), "db_error": db_error,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        "n_fixtures": len(rows),
        "detectors": detector_table(rows),
        "tiers": tier_table(rows),
        "fixtures": [{k: r[k] for k in ("id", "tier", "words", "verdict", "text_signals", "reads_generated", "sounds_read", "evidence")} for r in rows],
        "calls": analyzed_calls(call_sources, extra),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
__LIVE_META__
<title>Interview Integrity</title>
<style>
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --good:#0ca30c; --warn:#fab219; --crit:#d03b3b; --warn-ink:#0b0b0b;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif; --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#d95926; color-scheme:dark; } }
:root[data-theme="dark"]{ --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; color-scheme:dark; }
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:baseline;justify-content:space-between;margin-bottom:18px}
h1{font-size:28px;margin:0;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:4px 0 0;max-width:72ch}
.stamp{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
h2{font-size:18px;margin:28px 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:14px 16px;min-width:0}
.tile .k{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.tile .v{font-size:34px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1;margin:4px 0 2px}
.tile .b{font-size:13px;color:var(--ink2)}
.panel{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:14px 16px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:0 0 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
svg{width:100%;height:auto;display:block}
.bar{transition:opacity .15s}
.row:hover .bar{opacity:.85}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);font-size:12px;padding:6px 8px;border-radius:6px;opacity:0;transform:translate(-50%,-120%);white-space:nowrap;z-index:10}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600}
td.n{font-variant-numeric:tabular-nums;text-align:right}
th.n{text-align:right}
.tbl{overflow-x:auto}
.calls{display:grid;gap:12px}
.call{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:14px 16px}
.call h3{margin:0 0 8px;font-size:15px;display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}
.call h3 small{color:var(--muted);font-weight:400;font-family:var(--mono);font-size:12px}
.ans{display:grid;grid-template-columns:150px 1fr;gap:8px 14px;padding:10px 0;border-top:1px solid var(--grid)}
.ans:first-of-type{border-top:0}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:3px 8px;border-radius:999px;white-space:nowrap}
.badge.crit{background:var(--crit);color:#fff}
.badge.warn{background:var(--warn);color:var(--warn-ink)}
.badge.good{background:var(--good);color:#fff}
.badge.mute{background:var(--grid);color:var(--ink2)}
.qid{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:6px}
.metrics{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--ink2);font-variant-numeric:tabular-nums}
.metrics b{color:var(--ink);font-weight:600}
ul.ev{margin:6px 0 0;padding-left:18px;font-size:13px;color:var(--ink2)}
.empty{color:var(--muted);font-style:italic}
.callhead{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:start;margin-bottom:10px}
.conf{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.conf .score{font-size:30px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1}
.conf .band{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2)}
.meter{width:160px;height:8px;background:var(--grid);border-radius:4px;overflow:hidden}
.meter i{display:block;height:100%;border-radius:4px}
.problems{margin:0 0 10px;padding:0;list-style:none;display:grid;gap:4px}
.problems li{font-size:13px;color:var(--ink2);display:grid;grid-template-columns:70px 1fr;gap:10px}
.sev{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;padding:2px 6px;border-radius:4px;text-align:center;align-self:start}
.sev.high{background:var(--crit);color:#fff}.sev.medium{background:var(--warn);color:var(--warn-ink)}.sev.low{background:var(--grid);color:var(--ink2)}
.decision{display:flex;gap:8px;flex-wrap:wrap;align-items:center;border-top:1px solid var(--grid);padding-top:10px;margin-top:6px}
.decision .lbl{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-right:4px}
.decision button{font:inherit;font-size:13px;padding:6px 12px;border-radius:999px;border:1px solid var(--axis);background:var(--surface);color:var(--ink);cursor:pointer}
.decision button[aria-pressed="true"].go{background:var(--good);color:#fff;border-color:var(--good)}
.decision button[aria-pressed="true"].nogo{background:var(--crit);color:#fff;border-color:var(--crit)}
.decision button[aria-pressed="true"].hold{background:var(--warn);color:var(--warn-ink);border-color:var(--warn)}
.decision input{font:inherit;font-size:13px;padding:6px 10px;border-radius:6px;border:1px solid var(--axis);background:var(--surface);color:var(--ink);flex:1 1 200px;min-width:0}
.decision .saved{font-size:12px;color:var(--muted)}
button:focus-visible,input:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.method{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.method p{margin:0 0 6px}
.method b{display:block;margin-bottom:2px}
.hero{margin:6px 0 10px}
.herogrid{display:grid;grid-template-columns:1fr 260px;gap:12px;margin-bottom:12px}
.verdict{border-radius:10px;padding:18px 20px;border:1px solid var(--grid);background:var(--surface);border-left:8px solid var(--good)}
.verdict.crit{border-left-color:var(--crit)}.verdict.warn{border-left-color:var(--warn)}.verdict.good{border-left-color:var(--good)}
.verdict .k,.scorecard .k,.reasons .k,.waiting .k{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.vtitle{font-size:30px;font-weight:700;letter-spacing:-.01em;margin:6px 0 4px;line-height:1.1}
.vsub{color:var(--ink2);max-width:70ch}
.vmeta{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:12px;font-size:14px;color:var(--ink2);font-variant-numeric:tabular-nums}
.vmeta b{color:var(--ink)}
.scorecard{border-radius:10px;padding:18px 20px;border:1px solid var(--grid);background:var(--surface);display:flex;flex-direction:column;justify-content:center}
.scorecard .big{font-size:56px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums;margin:6px 0 10px}
.scorecard .of{font-size:18px;color:var(--muted);font-weight:500;margin-left:4px}
.scorecard .meter{width:100%;position:relative;overflow:visible}
.scorecard .thr{position:absolute;top:-4px;width:2px;height:16px;background:var(--ink);opacity:.6}
.scorecard .b{font-size:13px;color:var(--ink2);margin-top:8px}
.reasons{margin-bottom:12px}
.reasons ul{margin:8px 0 0;padding-left:18px;display:grid;gap:8px}
.reasons li{font-size:14px;color:var(--ink2)}
.reasons li b{color:var(--ink)}
.hero .decision{border-top:0;padding-top:0}
.waiting p{color:var(--ink2);margin:6px 0 0}
hr.break{border:0;border-top:2px dashed var(--axis);margin:34px 0 10px;page-break-after:always;break-after:page}
h2.pagehead{font-size:24px;margin:0 0 4px}
.earlier{margin-top:12px}
@media (max-width:760px){ .herogrid{grid-template-columns:1fr} }
@media (max-width:640px){ .ans{grid-template-columns:1fr} }
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>Interview integrity</h1>
    <p class="sub">For the recruiter running the screen. Did the candidate's answers read as AI-generated, and did they sound read aloud? Evidence, a score, and your decision. The analysis never speaks to the candidate.</p>
  </div>
  <div class="stamp" id="stamp">__LIVE_BADGE__</div>
</header>

<section id="recruiter" class="hero"></section>

<hr class="break">
<h2 class="pagehead">Interview breakdown</h2>
<p class="sub">Every recorded answer of the current call with the signals behind it, then the call mechanics. Updates as the call runs.</p>
<div id="breakdown" class="calls"></div>
<details class="panel earlier" id="earlier"><summary>Earlier calls</summary><div class="calls" id="earlier-calls"></div></details>

<hr class="break">
<h2 class="pagehead">Detector performance</h2>
<p class="sub">How the detector behaves on labelled answers, beside the do-nothing baselines. This is the evidence behind the score above.</p>
<div class="tiles" id="tiles"></div>
<h2>Catch rate vs false-flag rate, per detector variant</h2>
<div class="panel">
  <div class="legend"><span><i style="background:var(--s1)"></i>Catch rate on AI-written answers</span><span><i style="background:var(--s2)"></i>False-flag rate on human answers</span></div>
  <div id="chart"></div>
</div>
<h2>The four tiers</h2>
<div class="panel tbl"><table id="tiers"></table></div>

<h2>How it works, and what it cannot claim</h2>
<div class="method">
  <div class="panel"><b>Voice side: did it sound read?</b><p>Onset delay after the question, words per minute, fillers and mid-sentence pauses per 100 words, restarts, and the fluency drop when the agent asks an unscripted follow-up about a detail the candidate just said. Three or more signals on an answer of 40+ words = "sounds read".</p></div>
  <div class="panel"><b>Text side: does the transcript read as generated?</b><p>No disfluencies in the transcript, enumerating markers, no concrete details or episode markers, long uniform sentences, formal register, stock competency phrases. Three or more on 25+ words = "reads generated".</p></div>
  <div class="panel"><b>Combined, and the score</b><p>Text alone is a note. Voice alone is a note. Both together is "AI-assisted". The score (0 to 100) comes from the two strongest answers, discounted when fewer than three answers were long enough to judge. Bands: under 35 no indication, 35 to 59 possible, 60 and above likely.</p></div>
  <div class="panel"><b>Caveats we say out loud</b><p>Thresholds are uncalibrated until enough real calls exist. Fluent speakers, prepared honest stories, non-native speakers and anxious candidates all look like readers to a naive detector; that is why it is an observation for a human, with the numbers behind it, and never an auto-reject.</p></div>
</div>
</div>
<div class="tip" id="tip"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const pct = x => x == null ? 'n/a' : Math.round(x * 100) + '%';
document.getElementById('stamp').insertAdjacentText('beforeend', ` Generated ${D.generated_at} · ${D.n_fixtures} labelled answers · ${D.calls.length} call(s)` + (D.db_feed ? (D.db_error ? ` · database feed error: ${D.db_error}` : ' · database feed on') : ' · local files only'));

// ---- Recruiter view: the current (most recent) call ----
const THRESHOLD = 0.6;  // confidence at or above this = "likely AI-assisted" (band high)
const badge = v => ({ ai_assisted: ['crit', '● AI-assisted'], reads_generated: ['warn', '▲ reads generated'], sounds_read: ['warn', '▲ sounds read'], none: ['good', '✓ no flags'], ineligible: ['mute', '– too short'] }[v] || ['mute', v]);
const current = D.calls[0] || null;
const hero = document.getElementById('recruiter');
if (!current) {
  hero.innerHTML = `<div class="panel waiting"><div class="k">Waiting for the first call</div><p>${D.db_feed ? (D.db_error ? 'Database feed error: ' + D.db_error : 'Database feed connected. The call appears here as soon as its first event lands.') : 'No feed connected yet. Add DATABASE_URL_READONLY to .env.local, or drop a call\'s events into calls/.'}</p></div>`;
} else {
  const conf = current.confidence || { score: 0, band: 'low', eligible_answers: 0, answers: 0, strong: [], flagged: [] };
  const score = Math.round(conf.score * 100);
  const verdict = conf.score >= THRESHOLD ? ['crit', 'Likely AI-assisted', 'Above the threshold. Two independent signal sets agree on at least one answer.']
                : conf.band === 'medium' ? ['warn', 'Possible AI assistance', 'Below the threshold. One side flagged, or weak agreement. Read the reasons before deciding.']
                : ['good', 'No indication of AI assistance', 'Below the threshold on every recorded answer so far.'];
  const inProgress = !current.outcome;
  const reasons = [];
  (current.answers || []).forEach(a => {
    if (a.verdict === 'ai_assisted' || a.verdict === 'reads_generated' || a.verdict === 'sounds_read') {
      const [cls, lab] = badge(a.verdict);
      reasons.push(`<li><span class="badge ${cls}">${lab}</span> <b>${a.question_id}</b> (answer score ${Math.round((a.confidence || 0) * 100)}): ${a.evidence.slice(0, 4).join('; ')}</li>`);
    }
  });
  (current.problems || []).filter(p => p.severity === 'high' && !(current.answers || []).some(a => a.question_id === p.area)).forEach(p => reasons.push(`<li><span class="sev high">high</span> <b>${p.area}</b>: ${p.note}</li>`));
  const total = (current.answers || []).length, flagged = (conf.flagged || []).length, strong = (conf.strong || []).length;
  hero.innerHTML = `
    <div class="herogrid">
      <div class="verdict ${verdict[0]}">
        <div class="k">${inProgress ? '● Call in progress' : 'Call complete'} · ${current.id}${current.duration_s ? ` · ${current.duration_s} s` : ''}</div>
        <div class="vtitle">${verdict[1]}</div>
        <div class="vsub">${verdict[2]}</div>
        <div class="vmeta"><span><b>${strong}</b> of ${total} answers AI-assisted (both sides)</span><span><b>${flagged}</b> flagged on one side or both</span><span><b>${conf.eligible_answers}</b> of ${total} long enough to judge</span></div>
      </div>
      <div class="scorecard">
        <div class="k">Overall score</div>
        <div class="big">${score}<span class="of">/100</span></div>
        <div class="meter"><i style="width:${score}%;background:var(--${verdict[0]})"></i><span class="thr" style="left:${THRESHOLD * 100}%" title="threshold ${THRESHOLD * 100}"></span></div>
        <div class="b">threshold ${THRESHOLD * 100} · ${conf.band} confidence</div>
      </div>
    </div>
    <div class="reasons panel">
      <div class="k">Reasons</div>
      ${reasons.length ? `<ul>${reasons.join('')}</ul>` : `<p class="empty">${total ? 'No answer has crossed a signal threshold yet.' : 'No answers recorded yet; the first one appears after the agent records it.'}</p>`}
    </div>
    <div class="decision" data-call="${current.id}"><span class="lbl">Your decision</span><button class="go" data-v="go">Go</button><button class="hold" data-v="hold">Follow up</button><button class="nogo" data-v="nogo">No-go</button><input type="text" id="note-${current.id}" placeholder="Why (saved on this device only)"><span class="saved"></span></div>`;
}

// ---- Interview breakdown: answers of the current call, then earlier calls ----
const answerBlock = a => { const [cls, lab] = badge(a.verdict); const d = a.delivery; const t = a.text;
  return `<div class="ans"><div><span class="badge ${cls}">${lab}</span><div class="qid">${a.question_id} · score ${Math.round((a.confidence || 0) * 100)}</div></div><div>
    <div class="metrics">${d ? `<span>onset <b>${d.onset_ms ?? '?'} ms</b></span><span><b>${d.main.wpm ?? '?'}</b> wpm</span><span>fillers <b>${d.main.fillers_per100}</b>/100</span><span>pauses <b>${d.main.pauses_per100}</b>/100</span><span>restarts <b>${d.main.restarts_per100}</b>/100</span>${d.follow_up ? `<span>follow-up disfluency <b>${d.follow_up.disfluency_per100}</b> vs <b>${d.main.disfluency_per100}</b></span>` : ''}` : '<span>no word timings</span>'}<span><b>${t.words}</b> words</span><span>text signals <b>${t.signals.length}</b>/6</span></div>
    ${a.evidence.length ? `<ul class="ev">${a.evidence.map(e => `<li>${e}</li>`).join('')}</ul>` : '<div class="empty">no signals</div>'}</div></div>`; };
const callCard = (c, withDecision) => {
  const conf = c.confidence || { score: 0, band: 'low', eligible_answers: 0, answers: 0 };
  const color = conf.band === 'high' ? 'var(--crit)' : conf.band === 'medium' ? 'var(--warn)' : 'var(--good)';
  const head = `<div class="callhead"><div><h3>Call <small>${c.id}</small> ${c.outcome ? `<small>${c.outcome} · ${c.duration_s ?? '?'} s · ${c.turns} turns</small>` : '<small>in progress</small>'} ${c.started_at ? `<small>${c.started_at}</small>` : ''}</h3>
      ${c.problems && c.problems.length ? `<ul class="problems">${c.problems.map(p => `<li><span class="sev ${p.severity}">${p.severity}</span><span><b>${p.area}</b>: ${p.note}</span></li>`).join('')}</ul>` : '<div class="empty" style="margin-bottom:10px">No problem areas detected on this call.</div>'}</div>
    <div class="conf"><div><div class="score">${Math.round(conf.score * 100)}</div><div class="band">${conf.band} confidence · ${conf.eligible_answers}/${conf.answers} answers eligible</div></div><div class="meter"><i style="width:${Math.round(conf.score * 100)}%;background:${color}"></i></div></div></div>`;
  const body = c.error ? `<div class="empty">${c.error}</div>` : c.answers.length ? c.answers.map(answerBlock).join('') : '<div class="empty">no recorded answers yet</div>';
  const dec = withDecision ? `<div class="decision" data-call="${c.id}"><span class="lbl">Recruiter decision</span><button class="go" data-v="go">Go</button><button class="hold" data-v="hold">Follow up</button><button class="nogo" data-v="nogo">No-go</button><input type="text" id="note2-${c.id}" placeholder="Why (saved on this device only)"><span class="saved"></span></div>` : '';
  return `<div class="call">${head}${body}${dec}</div>`;
};
document.getElementById('breakdown').innerHTML = current ? callCard(current, false) : '<div class="panel empty">The breakdown fills in once a call is in progress.</div>';
const earlier = D.calls.slice(1);
const earlierEl = document.getElementById('earlier');
if (earlier.length) { document.getElementById('earlier-calls').innerHTML = earlier.map(c => callCard(c, true)).join(''); earlierEl.querySelector('summary').textContent = `Earlier calls (${earlier.length})`; } else { earlierEl.hidden = true; }

// Human in the loop: the decision is the recruiter's. Nothing is pre-selected; saved per device in localStorage.
const store = { get(k) { try { return JSON.parse(localStorage.getItem('cheat-detect:' + k) || 'null'); } catch { return null; } }, set(k, v) { try { localStorage.setItem('cheat-detect:' + k, JSON.stringify(v)); return true; } catch { return false; } } };
document.querySelectorAll('.decision').forEach(d => {
  const id = d.dataset.call, note = d.querySelector('input'), saved = d.querySelector('.saved');
  const render = s => { d.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(!!s && b.dataset.v === s.v))); if (s) { if (document.activeElement !== note) note.value = s.note || ''; saved.textContent = `saved ${s.at}`; } };
  render(store.get(id));
  const save = v => { const s = { v: v ?? (d.querySelector('[aria-pressed="true"]')?.dataset.v || null), note: note.value, at: new Date().toLocaleTimeString() }; if (!store.set(id, s)) saved.textContent = 'could not save on this device'; else render(s); };
  d.querySelectorAll('button').forEach(b => b.addEventListener('click', () => save(b.dataset.v)));
  note.addEventListener('change', () => save(null));
});

// ---- Detector performance (deeper insight) ----
// Tiles
const both = D.detectors.find(d => d.name === 'Both sides agree');
const coin = D.detectors.find(d => d.name === 'Coin flip');
const para = D.tiers.find(t => t.tier === 'ai_paraphrased');
const prep = D.tiers.find(t => t.tier === 'human_prepared');
const tiles = [
  { k: 'Catch rate (both sides agree)', v: pct(both?.catch), b: `coin flip: ${pct(coin?.catch)} · never flag: 0%` },
  { k: 'False-flag rate (both sides agree)', v: pct(both?.false_flag), b: `coin flip: ${pct(coin?.false_flag)} · always flag: 100%` },
  { k: 'Hard case: paraphrased AI answers caught', v: para ? `${para.both}/${para.n}` : 'n/a', b: 'generated text re-spoken in the candidate\'s words' },
  { k: 'Risk case: prepared honest stories flagged', v: prep ? `${prep.both}/${prep.n}` : 'n/a', b: 'rehearsed true story, fluent and structured' },
];
document.getElementById('tiles').innerHTML = tiles.map(t => `<div class="tile"><div class="k">${t.k}</div><div class="v">${t.v}</div><div class="b">${t.b}</div></div>`).join('');

// Grouped horizontal bars (one axis, 0–100%)
(function chart(){
  const rows = D.detectors, rowH = 46, barH = 10, left = 250, right = 60, w = 1100, top = 18, bottom = 30;
  const h = top + rows.length * rowH + bottom, plotW = w - left - right, x = v => left + (v ?? 0) * plotW;
  let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Catch rate and false-flag rate per detector">`;
  for (let g = 0; g <= 100; g += 25) {
    const gx = x(g / 100);
    s += `<line x1="${gx}" y1="${top - 6}" x2="${gx}" y2="${h - bottom + 4}" stroke="var(--grid)" stroke-width="1"/>`;
    s += `<text x="${gx}" y="${h - 8}" fill="var(--muted)" font-size="11" text-anchor="middle">${g}%</text>`;
  }
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const label = r.kind === 'baseline' ? `${r.name}` : r.name;
    s += `<g class="row" data-i="${i}">`;
    s += `<text x="${left - 12}" y="${y + rowH / 2 + 1}" fill="${r.kind === 'baseline' ? 'var(--muted)' : 'var(--ink)'}" font-size="13" text-anchor="end" dominant-baseline="middle">${label}</text>`;
    const bars = [[r.catch, 'var(--s1)', 'catch'], [r.false_flag, 'var(--s2)', 'false flag']];
    bars.forEach(([v, c, name], j) => {
      const by = y + rowH / 2 - barH - 1 + j * (barH + 2);
      const bw = Math.max(0, x(v) - left);
      s += `<rect class="bar" x="${left}" y="${by}" width="${bw}" height="${barH}" rx="0" fill="${c}" data-tip="${r.name}: ${name} ${pct(v)}"/>`;
      if (bw > 0) s += `<rect x="${left + bw - 4}" y="${by}" width="4" height="${barH}" rx="2" fill="${c}"/>`;
      s += `<text x="${left + bw + 6}" y="${by + barH / 2 + 1}" fill="var(--ink2)" font-size="11" dominant-baseline="middle">${pct(v)}</text>`;
    });
    s += `</g>`;
  });
  s += `<line x1="${left}" y1="${top - 6}" x2="${left}" y2="${h - bottom + 4}" stroke="var(--axis)" stroke-width="1"/></svg>`;
  const el = document.getElementById('chart'); el.innerHTML = s;
  const tip = document.getElementById('tip');
  el.querySelectorAll('[data-tip]').forEach(n => {
    n.addEventListener('mousemove', e => { tip.textContent = n.dataset.tip; tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; tip.style.opacity = 1; });
    n.addEventListener('mouseleave', () => tip.style.opacity = 0);
  });
})();

// Tiers
document.getElementById('tiers').innerHTML = `<thead><tr><th>Tier</th><th>What it is</th><th class="n">n</th><th class="n">Text flags</th><th class="n">Voice flags</th><th class="n">Both</th><th class="n">Median text signals</th></tr></thead><tbody>` +
  D.tiers.map(t => `<tr><td><code>${t.tier}</code></td><td>${{ai_verbatim:'generated text read aloud', ai_paraphrased:'generated text re-spoken in own words', human_spontaneous:'a story told live', human_prepared:'a rehearsed true story'}[t.tier] || ''}</td><td class="n">${t.n}</td><td class="n">${t.text}/${t.n}</td><td class="n">${t.voice}/${t.n}</td><td class="n">${t.both}/${t.n}</td><td class="n">${t.median_text_signals}</td></tr>`).join('') + '</tbody>';

</script>
</body>
</html>
"""


def call_sources(calls: pathlib.Path) -> list[pathlib.Path]:
    if calls.is_dir():
        # newest first so the live call is on top
        return sorted(calls.glob("*.events.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) + sorted(calls.glob("*.json"))
    return [calls] if calls.exists() else []


def render(calls: pathlib.Path, live: bool, refresh_s: int = 4, db_url: str | None = None) -> str:
    data = build_data(call_sources(calls), db_url)
    html = HTML.replace("__DATA__", json.dumps(data, default=str).replace("</", "<\\/"))
    html = html.replace("__LIVE_META__", f'<meta http-equiv="refresh" content="{refresh_s}">' if live else "")
    html = html.replace("__LIVE_BADGE__", '<span style="color:var(--crit);font-weight:600">● LIVE</span> refreshing every %d s ·' % refresh_s if live else "")
    return html


def serve(calls: pathlib.Path, port: int, refresh_s: int, db_url: str | None = None) -> None:
    """Demo mode: rebuild the page on every request; the page refreshes itself, so a call in progress shows each
    answer's verdict as the agent records it (the worker appends events to calls/*.events.jsonl live)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = render(calls, live=True, refresh_s=refresh_s, db_url=db_url).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            return

    print(f"live dashboard on http://127.0.0.1:{port}  (watching {calls}, refresh {refresh_s}s; Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=pathlib.Path, default=HERE.parents[1] / "calls", help="dir of *.events.jsonl, or one file")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "dashboard.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--serve", type=int, metavar="PORT", help="live demo mode: serve and auto-refresh")
    ap.add_argument("--refresh", type=int, default=4, help="seconds between refreshes in --serve mode")
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL_READONLY") or None,
                    help="Postgres URL of the reference product's database (agent_events); default $DATABASE_URL_READONLY")
    args = ap.parse_args()
    if args.serve:
        if args.open:
            subprocess.Popen(["open", f"http://127.0.0.1:{args.serve}"])
        serve(args.calls, args.serve, args.refresh, args.db)
        return
    html = render(args.calls, live=False, db_url=args.db)
    args.out.write_text(html)
    print(f"wrote {args.out}")
    if args.open:
        subprocess.run(["open", str(args.out)], check=False)


if __name__ == "__main__":
    main()
