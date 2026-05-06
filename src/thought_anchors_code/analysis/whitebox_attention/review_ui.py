"""Generate a lightweight HTML review UI for receiver-head analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
    split_reasoning_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML UI for inspecting rollouts and attention scores."
    )
    parser.add_argument("rollout_file", type=Path, help="Input rollout JSONL file.")
    parser.add_argument("summary_file", type=Path, help="Receiver-head summary JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/attention_review.html"),
        help="Output HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts = load_rollouts_jsonl(args.rollout_file)
    summaries = _load_summaries(args.summary_file)
    rows = []
    for rollout in rollouts:
        key = (rollout.model_id, rollout.dataset_name, rollout.task_id, rollout.sample_id)
        summary = summaries.get(key)
        if summary is None:
            continue
        rows.append(
            {
                "model_id": rollout.model_id,
                "dataset_name": rollout.dataset_name,
                "task_id": rollout.task_id,
                "sample_id": rollout.sample_id,
                "complete": rollout.complete,
                "is_correct": rollout.is_correct,
                "prompt": rollout.prompt,
                "reasoning_sentences": split_reasoning_steps(rollout.reasoning),
                "answer": rollout.answer,
                "sentence_scores": summary.get("sentence_scores") or [],
                "code_sentence_scores": summary.get("code_sentence_scores") or [],
                "receiver_head_scores": summary.get("receiver_head_scores") or [],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_html(rows), encoding="utf-8")
    print(f"Wrote attention review UI to {args.output}")


def _load_summaries(path: Path) -> dict[tuple[str, str, str, int], dict]:
    summaries = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = (
                str(payload.get("model_id")),
                str(payload.get("dataset_name")),
                str(payload.get("task_id")),
                int(payload.get("sample_id", 0)),
            )
            summaries[key] = payload
    return summaries


def _render_html(rows: list[dict]) -> str:
    data_json = json.dumps(rows, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Receiver-Head Review</title>
<style>
:root {{ color-scheme: dark; --bg: #0d1016; --panel: #151a22; --panel2: #10141b; --line: #293241; --muted: #93a1b5; --text: #edf4ff; --blue: #75a7ff; --amber: #ffb75f; --green: #47c784; --red: #ff6b75; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
header {{ position: sticky; top: 0; z-index: 5; display: grid; grid-template-columns: 1fr 150px 150px 165px; gap: 12px; align-items: center; padding: 12px 16px; background: rgba(13,16,22,.97); border-bottom: 1px solid var(--line); backdrop-filter: blur(8px); }}
input, select {{ background: #080b10; color: var(--text); border: 1px solid #344153; border-radius: 9px; padding: 9px 11px; outline: none; }}
input:focus, select:focus {{ border-color: var(--blue); }}
main {{ display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: calc(100vh - 58px); }}
#list {{ border-right: 1px solid var(--line); overflow: auto; max-height: calc(100vh - 58px); background: #0a0d12; }}
.item {{ padding: 11px 14px; border-bottom: 1px solid #202835; cursor: pointer; }}
.item:hover {{ background: #131923; }}
.item.active {{ background: #1b2330; box-shadow: inset 3px 0 0 var(--blue); }}
.itemTop {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; }}
.task {{ font-size: 17px; font-weight: 750; letter-spacing: .01em; }}
.mini {{ color: var(--muted); font-size: 11px; line-height: 1.35; margin-top: 6px; }}
.spark {{ display: grid; grid-template-columns: 46px 1fr 46px 1fr; gap: 5px; align-items: center; margin-top: 8px; font-size: 11px; color: var(--muted); }}
.bar {{ height: 6px; border-radius: 999px; background: #202835; overflow: hidden; }}
.fillR {{ height: 100%; background: linear-gradient(90deg, #7d4d20, var(--amber)); }}
.fillC {{ height: 100%; background: linear-gradient(90deg, #24456d, var(--blue)); }}
.meta {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
.badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; background: #303743; white-space: nowrap; }}
.good {{ background: rgba(71,199,132,.22); color: #a9f0c8; }} .bad {{ background: rgba(255,107,117,.22); color: #ffc1c5; }} .unknown {{ background: #414752; }}
.complete {{ background: rgba(117,167,255,.18); color: #c5d9ff; }} .incomplete {{ background: rgba(255,183,95,.18); color: #ffe0b7; }}
#detail {{ overflow: auto; max-height: calc(100vh - 58px); }}
.hero {{ position: sticky; top: 0; z-index: 3; padding: 18px 22px 14px; background: rgba(13,16,22,.96); border-bottom: 1px solid var(--line); backdrop-filter: blur(8px); }}
.titleRow {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }}
h2 {{ margin: 0; font-size: 28px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
.label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }}
.value {{ font-size: 18px; margin-top: 5px; font-variant-numeric: tabular-nums; }}
.content {{ padding: 18px 22px 32px; }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; color: var(--muted); font-size: 13px; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 99px; margin-right: 5px; }}
.sentence {{ display: grid; grid-template-columns: 74px 120px minmax(0, 1fr); gap: 13px; align-items: start; margin: 8px 0; padding: 11px; border: 1px solid #253040; border-radius: 12px; background: var(--panel2); }}
.sentence:hover {{ border-color: #3e4d62; background: #151b25; }}
.idx {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
.scoreBox {{ display: grid; gap: 7px; font-size: 12px; font-variant-numeric: tabular-nums; }}
.scoreLine {{ display: grid; grid-template-columns: 16px 1fr; align-items: center; gap: 6px; }}
.text {{ white-space: pre-wrap; line-height: 1.5; font-size: 15px; }}
.notAnalyzed {{ opacity: .38; }}
details {{ margin-top: 18px; }}
summary {{ cursor: pointer; color: var(--blue); }}
pre {{ white-space: pre-wrap; background: #080b10; border: 1px solid #252f3e; border-radius: 10px; padding: 14px; overflow: auto; }}
@media (max-width: 950px) {{ main {{ grid-template-columns: 1fr; }} #list {{ max-height: 280px; border-right: 0; border-bottom: 1px solid var(--line); }} header {{ grid-template-columns: 1fr; }} .grid {{ grid-template-columns: 1fr 1fr; }} .sentence {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <input id="search" placeholder="Search task, prompt, answer...">
  <select id="correct"><option value="all">all correctness</option><option value="true">correct</option><option value="false">incorrect</option><option value="unknown">unknown</option></select>
  <select id="complete"><option value="all">all completion</option><option value="true">complete</option><option value="false">incomplete</option></select>
  <select id="sort"><option value="task">sort: task</option><option value="max_r">sort: max R score</option><option value="max_c">sort: max C score</option><option value="sentences">sort: sentence count</option></select>
</header>
<main>
  <aside id="list"></aside>
  <section id="detail"></section>
</main>
<script>
const DATA = {data_json};
let current = 0;
const list = document.getElementById('list');
const detail = document.getElementById('detail');
const search = document.getElementById('search');
const correct = document.getElementById('correct');
const complete = document.getElementById('complete');
const sort = document.getElementById('sort');

function cleanScore(x) {{ return typeof x === 'number' && Number.isFinite(x) ? x : null; }}
function stats(row) {{
  const scores = row.sentence_scores.map(cleanScore).filter(x => x !== null);
  const codeScores = row.code_sentence_scores.map(cleanScore).filter(x => x !== null);
  return {{
    max: scores.length ? Math.max(...scores) : null,
    mean: scores.length ? scores.reduce((a,b)=>a+b,0)/scores.length : null,
    codeMax: codeScores.length ? Math.max(...codeScores) : null,
  }};
}}
function badge(v) {{
  if (v === true) return '<span class="badge good">correct</span>';
  if (v === false) return '<span class="badge bad">incorrect</span>';
  return '<span class="badge unknown">unknown</span>';
}}
function completeBadge(v) {{ return v ? '<span class="badge complete">complete</span>' : '<span class="badge incomplete">incomplete</span>'; }}
function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function fmt(x) {{ return x === null ? 'n/a' : x.toExponential(2); }}
function pct(x, max) {{ return x === null || max <= 0 ? 0 : Math.max(3, Math.min(100, 100 * x / max)); }}
function filtered() {{
  const q = search.value.toLowerCase();
  let rows = DATA.filter(row => {{
    const c = correct.value;
    if (c === 'true' && row.is_correct !== true) return false;
    if (c === 'false' && row.is_correct !== false) return false;
    if (c === 'unknown' && row.is_correct !== null && row.is_correct !== undefined) return false;
    const done = complete.value;
    if (done === 'true' && row.complete !== true) return false;
    if (done === 'false' && row.complete !== false) return false;
    return !q || [row.task_id, row.prompt, row.answer].some(x => String(x || '').toLowerCase().includes(q));
  }});
  rows.sort((a, b) => {{
    if (sort.value === 'task') return a.task_id.localeCompare(b.task_id, undefined, {{ numeric: true }});
    if (sort.value === 'sentences') return b.reasoning_sentences.length - a.reasoning_sentences.length;
    if (sort.value === 'max_c') return (stats(b).codeMax ?? -Infinity) - (stats(a).codeMax ?? -Infinity);
    return (stats(b).max ?? -Infinity) - (stats(a).max ?? -Infinity);
  }});
  return rows;
}}
function renderList() {{
  const rows = filtered();
  const globalMax = Math.max(...rows.map(r => Math.max(stats(r).max ?? 0, stats(r).codeMax ?? 0)), 0);
  list.innerHTML = rows.map((row, i) => {{
    const st = stats(row);
    return `<div class="item ${{i === current ? 'active' : ''}}" onclick="current=${{i}}; render()">
      <div class="itemTop"><span class="task">${{esc(row.task_id)}}</span><span>${{badge(row.is_correct)}} ${{completeBadge(row.complete)}}</span></div>
      <div class="mini">analyzed ${{row.sentence_scores.length}} / full ${{row.reasoning_sentences.length}} · sample ${{row.sample_id}}</div>
      <div class="spark">
        <span>R ${{fmt(st.max)}}</span><span class="bar"><span class="fillR" style="display:block;width:${{pct(st.max, globalMax)}}%"></span></span>
        <span>C ${{fmt(st.codeMax)}}</span><span class="bar"><span class="fillC" style="display:block;width:${{pct(st.codeMax, globalMax)}}%"></span></span>
      </div>
    </div>`;
  }}).join('');
}}
function renderDetail() {{
  const rows = filtered();
  if (!rows.length) {{ detail.innerHTML = '<p>No matching rows.</p>'; return; }}
  if (current >= rows.length) current = 0;
  const row = rows[current];
  const st = stats(row);
  const max = Math.max(st.max ?? 0, st.codeMax ?? 0);
  const sentences = row.reasoning_sentences.map((sentence, i) => {{
    const score = cleanScore(row.sentence_scores[i]);
    const codeScore = cleanScore(row.code_sentence_scores[i]);
    const analyzed = i < row.sentence_scores.length;
    return `<div class="sentence ${{analyzed ? '' : 'notAnalyzed'}}">
      <div class="idx">#${{i}}${{analyzed ? '' : '<br>not analyzed'}}</div>
      <div class="scoreBox">
        <div class="scoreLine"><b style="color:var(--amber)">R</b><span>${{fmt(score)}}</span></div>
        <div class="bar"><span class="fillR" style="display:block;width:${{pct(score, max)}}%"></span></div>
        <div class="scoreLine"><b style="color:var(--blue)">C</b><span>${{fmt(codeScore)}}</span></div>
        <div class="bar"><span class="fillC" style="display:block;width:${{pct(codeScore, max)}}%"></span></div>
      </div>
      <div class="text">${{esc(sentence)}}</div>
    </div>`;
  }}).join('');
  detail.innerHTML = `<div class="hero"><div class="titleRow"><h2>${{esc(row.task_id)}}</h2><span>${{badge(row.is_correct)}} ${{completeBadge(row.complete)}}</span></div>
    <div class="grid">
      <div class="card"><div class="label">Dataset</div><div class="value">${{esc(row.dataset_name)}}</div></div>
      <div class="card"><div class="label">Sentences</div><div class="value">${{row.sentence_scores.length}} / ${{row.reasoning_sentences.length}}</div></div>
      <div class="card"><div class="label">Reasoning Max</div><div class="value">${{fmt(st.max)}}</div></div>
      <div class="card"><div class="label">Code Max</div><div class="value">${{fmt(st.codeMax)}}</div></div>
    </div></div>
    <div class="content"><div class="legend"><span><span class="dot" style="background:var(--amber)"></span>R: later reasoning -> sentence</span><span><span class="dot" style="background:var(--blue)"></span>C: final code -> sentence</span><span>Dimmed rows were outside the analyzed prefix.</span></div>
    <h3>Reasoning Sentences</h3>${{sentences}}
    <details><summary>Prompt</summary><pre>${{esc(row.prompt)}}</pre></details>
    <details open><summary>Answer / Code</summary><pre>${{esc(row.answer)}}</pre></details>
    <details><summary>Receiver Head Scores</summary><pre>${{esc(JSON.stringify(row.receiver_head_scores, null, 2))}}</pre></details></div>`;
}}
function render() {{ renderList(); renderDetail(); }}
[search, correct, complete, sort].forEach(el => el.addEventListener('input', () => {{ current = 0; render(); }}));
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
