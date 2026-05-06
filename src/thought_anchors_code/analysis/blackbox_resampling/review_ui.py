"""Generate a static HTML review UI for black-box resampling outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from thought_anchors_code.analysis.whitebox_attention.trace_utils import load_rollouts_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML UI for inspecting black-box resampling results."
    )
    parser.add_argument("input", type=Path, help="Input black-box resampling JSONL file.")
    parser.add_argument(
        "--rollout-file",
        type=Path,
        help="Original rollout JSONL file. When provided, the UI shows original reasoning.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/blackbox_resampling_review.html"),
        help="Output HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_review_rows(args.input, args.rollout_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(rows), encoding="utf-8")
    print(f"Wrote black-box resampling review UI to {args.output}")


def load_review_rows(path: str | Path, rollout_path: str | Path | None = None) -> list[dict]:
    rollout_map = _load_rollout_map(rollout_path) if rollout_path else {}
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = (
                str(payload.get("dataset_name")),
                str(payload.get("task_id")),
                int(payload.get("sample_id", 0)),
            )
            rows.append(_build_review_row(payload, rollout_map.get(key)))
    return rows


def _load_rollout_map(path: str | Path) -> dict[tuple[str, str, int], dict[str, str]]:
    return {
        (rollout.dataset_name, rollout.task_id, rollout.sample_id): {
            "reasoning": rollout.reasoning,
            "answer": rollout.answer,
        }
        for rollout in load_rollouts_jsonl(path)
    }


def _build_review_row(payload: dict, rollout: dict[str, str] | None = None) -> dict:
    resamples = payload.get("resamples") or []
    correctness = [
        resample.get("is_correct")
        for resample in resamples
        if resample.get("is_correct") is not None
    ]
    pass_rate = mean(bool(value) for value in correctness) if correctness else None
    original_is_correct = payload.get("original_is_correct")
    pass_rate_delta = (
        None
        if pass_rate is None or original_is_correct is None
        else pass_rate - float(bool(original_is_correct))
    )
    complete_resamples = sum(1 for resample in resamples if resample.get("complete") is True)
    changed_answers = sum(
        1
        for resample in resamples
        if resample.get("answer") and resample.get("answer") != payload.get("original_answer")
    )

    return {
        "model_id": payload.get("model_id"),
        "dataset_name": payload.get("dataset_name"),
        "task_id": payload.get("task_id"),
        "sample_id": payload.get("sample_id"),
        "sentence_index": payload.get("sentence_index"),
        "selection": payload.get("selection"),
        "sentence_text": payload.get("sentence_text"),
        "sentence_score": payload.get("sentence_score"),
        "code_sentence_score": payload.get("code_sentence_score"),
        "original_answer": payload.get("original_answer"),
        "original_reasoning": (rollout or {}).get("reasoning"),
        "original_is_correct": original_is_correct,
        "prefix_sentence_count": payload.get("prefix_sentence_count"),
        "suffix_sentence_count": payload.get("suffix_sentence_count"),
        "num_resamples": len(resamples),
        "evaluated_resamples": len(correctness),
        "complete_resamples": complete_resamples,
        "changed_answers": changed_answers,
        "resample_pass_rate": pass_rate,
        "pass_rate_delta": pass_rate_delta,
        "resamples": resamples,
    }


def render_html(rows: list[dict]) -> str:
    data_json = json.dumps(rows, ensure_ascii=False)
    return (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Black-Box Resampling Review</title>
<style>
:root { color-scheme: dark; --bg:#0c0f12; --ink:#edf1f2; --muted:#9ba7aa; --panel:#161b1f; --panel2:#101417; --line:#2b3439; --blue:#79b8ff; --green:#5fd190; --red:#ff7474; --amber:#f5b95f; --cyan:#55d6d0; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { position:sticky; top:0; z-index:5; display:grid; grid-template-columns:minmax(220px,1fr) 150px 150px 170px; gap:10px; padding:12px 14px; border-bottom:1px solid var(--line); background:rgba(12,15,18,.96); backdrop-filter:blur(8px); }
input, select { min-width:0; border:1px solid #354147; border-radius:8px; background:#07090b; color:var(--ink); padding:9px 10px; outline:none; }
input:focus, select:focus { border-color:var(--blue); }
main { display:grid; grid-template-columns:390px minmax(0,1fr); min-height:calc(100vh - 58px); }
#list { max-height:calc(100vh - 58px); overflow:auto; border-right:1px solid var(--line); background:#090c0e; }
.item { padding:12px 14px; border-bottom:1px solid #20282d; cursor:pointer; }
.item:hover { background:#11171a; }
.item.active { background:#182025; box-shadow:inset 3px 0 0 var(--cyan); }
.itemTop { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
.task { font-size:16px; font-weight:750; overflow-wrap:anywhere; }
.mini { margin-top:7px; color:var(--muted); font-size:12px; line-height:1.35; }
.metrics { display:grid; grid-template-columns:62px 1fr 62px 1fr; gap:6px; align-items:center; margin-top:9px; font-size:11px; color:var(--muted); }
.bar { height:7px; border-radius:999px; overflow:hidden; background:#222b31; }
.fillDelta { display:block; height:100%; background:var(--red); }
.fillScore { display:block; height:100%; background:var(--cyan); }
.badge { display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; white-space:nowrap; background:#30383d; }
.good { color:#c8f4da; background:rgba(95,209,144,.2); }
.bad { color:#ffd0d0; background:rgba(255,116,116,.2); }
.neutral { color:#d7dee0; background:#343c42; }
#detail { max-height:calc(100vh - 58px); overflow:auto; }
.hero { position:sticky; top:0; z-index:3; padding:18px 22px 14px; border-bottom:1px solid var(--line); background:rgba(12,15,18,.96); backdrop-filter:blur(8px); }
.titleRow { display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:14px; }
h1 { margin:0; font-size:26px; line-height:1.15; overflow-wrap:anywhere; }
.grid { display:grid; grid-template-columns:repeat(5, minmax(120px,1fr)); gap:10px; }
.card { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:11px; min-width:0; }
.label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
.value { margin-top:5px; font-size:18px; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }
.content { padding:18px 22px 34px; }
.sentence { display:grid; grid-template-columns:120px minmax(0,1fr); gap:14px; border:1px solid #273137; border-radius:8px; background:var(--panel2); padding:13px; margin-bottom:14px; }
.idx { color:var(--muted); font-variant-numeric:tabular-nums; }
.text { white-space:pre-wrap; line-height:1.5; }
.resample { border:1px solid #273137; border-radius:8px; background:#0f1316; margin:12px 0; overflow:hidden; }
.resampleHead { display:flex; justify-content:space-between; gap:10px; padding:10px 12px; border-bottom:1px solid #273137; background:#141a1e; }
.compare { display:grid; grid-template-columns:1fr 1fr; gap:0; }
.compareHead { display:grid; grid-template-columns:1fr 1fr; border-bottom:1px solid #273137; }
.colTitle { display:flex; align-items:center; justify-content:space-between; gap:10px; min-width:0; padding:9px 12px; border-right:1px solid #273137; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; background:#11171a; }
.colTitle:last-child { border-right:0; }
.compareRow { display:grid; grid-template-columns:1fr 1fr; border-bottom:1px solid #273137; }
.compareRow:last-child { border-bottom:0; }
.cell { min-width:0; border-right:1px solid #273137; background:#090c0e; }
.cell:last-child { border-right:0; }
.cellLabel { padding:9px 12px; color:var(--blue); border-bottom:1px solid #20282d; background:#0f1417; }
.cell.missing pre { color:var(--muted); }
.note { color:var(--muted); font-size:12px; margin-top:6px; }
pre { margin:0; white-space:pre-wrap; overflow:auto; padding:13px; background:#080a0c; line-height:1.45; }
.empty { padding:24px; color:var(--muted); }
@media (max-width: 980px) { header { grid-template-columns:1fr; } main { grid-template-columns:1fr; } #list { max-height:290px; border-right:0; border-bottom:1px solid var(--line); } .grid { grid-template-columns:1fr 1fr; } .sentence { grid-template-columns:1fr; } .compareHead, .compareRow { grid-template-columns:1fr; } .colTitle, .cell { border-right:0; border-bottom:1px solid #273137; } .colTitle:last-child, .cell:last-child { border-bottom:0; } }
</style>
</head>
<body>
<header>
  <input id="search" placeholder="Search task, sentence, answer...">
  <select id="correct"><option value="all">all originals</option><option value="true">original correct</option><option value="false">original incorrect</option><option value="unknown">unknown</option></select>
  <select id="impact"><option value="all">all impacts</option><option value="hurts">hurts pass rate</option><option value="helps">helps pass rate</option><option value="stable">stable</option></select>
  <select id="sort"><option value="task">sort: task</option><option value="impact">sort: impact</option><option value="score">sort: receiver score</option><option value="index">sort: sentence index</option></select>
</header>
<main>
  <aside id="list"></aside>
  <section id="detail"></section>
</main>
<script>
const DATA = __DATA__;
let current = 0;
const list = document.getElementById('list');
const detail = document.getElementById('detail');
const search = document.getElementById('search');
const correct = document.getElementById('correct');
const impact = document.getElementById('impact');
const sort = document.getElementById('sort');
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[c])); }
function num(x) { return typeof x === 'number' && Number.isFinite(x) ? x : null; }
function fmt(x) { x = num(x); return x === null ? 'n/a' : x.toFixed(3); }
function pct(x, max) { x = Math.abs(num(x) ?? 0); return max <= 0 ? 0 : Math.max(4, Math.min(100, 100 * x / max)); }
function correctnessBadge(v) { if (v === true) return '<span class="badge good">correct</span>'; if (v === false) return '<span class="badge bad">incorrect</span>'; return '<span class="badge neutral">unknown</span>'; }
function deltaBadge(x) { x = num(x); if (x === null) return '<span class="badge neutral">not evaluated</span>'; if (x < 0) return `<span class="badge bad">${fmt(x)}</span>`; if (x > 0) return `<span class="badge good">+${fmt(x)}</span>`; return '<span class="badge neutral">0.000</span>'; }
function rowScore(row) { return Math.max(num(row.sentence_score) ?? 0, num(row.code_sentence_score) ?? 0); }
function filtered() {
  const q = search.value.toLowerCase();
  const rows = DATA.filter(row => {
    if (correct.value === 'true' && row.original_is_correct !== true) return false;
    if (correct.value === 'false' && row.original_is_correct !== false) return false;
    if (correct.value === 'unknown' && row.original_is_correct !== null && row.original_is_correct !== undefined) return false;
    const delta = num(row.pass_rate_delta);
    if (impact.value === 'hurts' && !(delta < 0)) return false;
    if (impact.value === 'helps' && !(delta > 0)) return false;
    if (impact.value === 'stable' && delta !== 0) return false;
    return !q || [row.task_id, row.sentence_text, row.original_answer, row.dataset_name].some(v => String(v ?? '').toLowerCase().includes(q));
  });
  rows.sort((a, b) => {
    if (sort.value === 'task') return String(a.task_id).localeCompare(String(b.task_id), undefined, { numeric:true }) || a.sentence_index - b.sentence_index;
    if (sort.value === 'index') return a.sentence_index - b.sentence_index;
    if (sort.value === 'score') return rowScore(b) - rowScore(a);
    return Math.abs(num(b.pass_rate_delta) ?? -Infinity) - Math.abs(num(a.pass_rate_delta) ?? -Infinity);
  });
  return rows;
}
function renderList() {
  const rows = filtered();
  const maxDelta = Math.max(...rows.map(r => Math.abs(num(r.pass_rate_delta) ?? 0)), 0);
  const maxScore = Math.max(...rows.map(rowScore), 0);
  list.innerHTML = rows.map((row, i) => `<div class="item ${i === current ? 'active' : ''}" onclick="current=${i}; render()">
    <div class="itemTop"><div class="task">${esc(row.task_id)} · #${esc(row.sentence_index)}</div><div>${deltaBadge(row.pass_rate_delta)}</div></div>
    <div class="mini">${esc(row.dataset_name)} · sample ${esc(row.sample_id)} · ${row.evaluated_resamples}/${row.num_resamples} evaluated · ${correctnessBadge(row.original_is_correct)}</div>
    <div class="metrics">
      <span>delta</span><span class="bar"><span class="fillDelta" style="width:${pct(row.pass_rate_delta, maxDelta)}%"></span></span>
      <span>score</span><span class="bar"><span class="fillScore" style="width:${pct(rowScore(row), maxScore)}%"></span></span>
    </div>
  </div>`).join('');
}
function renderDetail() {
  const rows = filtered();
  if (!rows.length) { detail.innerHTML = '<div class="empty">No matching interventions.</div>'; return; }
  if (current >= rows.length) current = 0;
  const row = rows[current];
  const originalReasoning = row.original_reasoning || `Original rollout reasoning was not embedded. Rebuild with --rollout-file to show it here.`;
  const originalReasoningClass = row.original_reasoning ? '' : ' missing';
  const resamples = row.resamples.map(resample => `<div class="resample">
    <div class="resampleHead"><span>draw ${Number(resample.resample_id) + 1} of ${row.num_resamples}</span><span>${correctnessBadge(resample.is_correct)} ${resample.complete ? '<span class="badge good">complete</span>' : '<span class="badge neutral">incomplete</span>'}</span></div>
    <div class="compareHead">
      <div class="colTitle"><span>Original rollout</span><span>${correctnessBadge(row.original_is_correct)}</span></div>
      <div class="colTitle"><span>Generated draw</span><span>${correctnessBadge(resample.is_correct)}</span></div>
    </div>
    <div class="compareRow">
      <div class="cell${originalReasoningClass}">
        <div class="cellLabel">Reasoning</div>
        <pre>${esc(originalReasoning)}</pre>
      </div>
      <div class="cell">
        <div class="cellLabel">Reasoning continuation</div>
        <pre>${esc(resample.reasoning || '')}</pre>
      </div>
    </div>
    <div class="compareRow">
      <div class="cell">
        <div class="cellLabel">Answer</div>
        <pre>${esc(row.original_answer || '')}</pre>
      </div>
      <div class="cell">
        <div class="cellLabel">Answer</div>
        <pre>${esc(resample.answer || '')}</pre>
      </div>
    </div>
  </div>`).join('');
  detail.innerHTML = `<div class="hero">
    <div class="titleRow"><h1>${esc(row.task_id)} · sentence #${esc(row.sentence_index)}</h1><div>${correctnessBadge(row.original_is_correct)} ${deltaBadge(row.pass_rate_delta)}</div></div>
    <div class="grid">
      <div class="card"><div class="label">Pass Rate</div><div class="value">${fmt(row.resample_pass_rate)}</div></div>
      <div class="card"><div class="label">Delta</div><div class="value">${fmt(row.pass_rate_delta)}</div></div>
      <div class="card"><div class="label">Changed Answers</div><div class="value">${row.changed_answers} / ${row.num_resamples}</div></div>
      <div class="card"><div class="label">Sentence Score</div><div class="value">${fmt(row.sentence_score)}</div></div>
      <div class="card"><div class="label">Code Score</div><div class="value">${fmt(row.code_sentence_score)}</div></div>
    </div>
  </div>
  <div class="content">
    <div class="sentence"><div class="idx">omitted sentence<br>prefix ${esc(row.prefix_sentence_count)} · suffix ${esc(row.suffix_sentence_count)}</div><div class="text">${esc(row.sentence_text)}</div></div>
    <h2>Generated Draws</h2>
    <div class="note">Each draw is one independent continuation sampled after omitting this sentence. Runs with --num-resamples 1 show draw 1 of 1; larger runs show multiple draws for the same intervention.</div>
    ${resamples || '<div class="empty">No resamples stored for this intervention.</div>'}
  </div>`;
}
function render() { renderList(); renderDetail(); }
[search, correct, impact, sort].forEach(el => el.addEventListener('input', () => { current = 0; render(); }));
render();
</script>
</body>
</html>
"""
    ).replace("__DATA__", data_json)


if __name__ == "__main__":
    main()
