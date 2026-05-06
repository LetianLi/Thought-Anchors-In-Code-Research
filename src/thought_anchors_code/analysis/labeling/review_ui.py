"""Generate a static HTML review UI for labeled rollout outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

UNLABELED_TAG = "unlabeled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML UI for inspecting labeled reasoning rollouts."
    )
    parser.add_argument("labeled_file", type=Path, help="Input labeled rollout JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/label_review.html"),
        help="Output HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_labeled_rollouts_jsonl(args.labeled_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(rows), encoding="utf-8")
    print(f"Wrote label review UI for {len(rows)} rows to {args.output}")


def load_labeled_rollouts_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    fallback_rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            row = normalize_labeled_row(payload, line_number=line_number)
            if (
                row["dataset_name"] is None
                or row["task_id"] is None
                or row["sample_id"] is None
            ):
                fallback_rows.append(row)
                continue
            key = (str(row["dataset_name"]), str(row["task_id"]), int(row["sample_id"]))
            rows_by_key[key] = row
    return [*fallback_rows, *rows_by_key.values()]


def normalize_labeled_row(payload: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    sentences = payload.get("sentences") or []
    normalized_sentences = []
    if isinstance(sentences, list):
        for fallback_index, sentence in enumerate(sentences, start=1):
            if isinstance(sentence, dict):
                index = str(sentence.get("index") or fallback_index)
                text = str(sentence.get("text") or "")
            else:
                index = str(fallback_index)
                text = str(sentence)
            normalized_sentences.append({"index": index, "text": text})

    labels = payload.get("labels") or {}
    if not isinstance(labels, dict):
        labels = {}

    tag_counts: dict[str, int] = {}
    dependency_count = 0
    labeled_sentence_count = 0
    unlabeled_sentence_count = 0
    for sentence in normalized_sentences:
        label = labels.get(sentence["index"]) or {}
        if not isinstance(label, dict):
            unlabeled_sentence_count += 1
            continue
        tags = label.get("function_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        concrete_tags = [
            str(tag).strip()
            for tag in tags
            if str(tag).strip() and str(tag).strip() != "unknown"
        ] if isinstance(tags, list) else []
        if concrete_tags:
            labeled_sentence_count += 1
            for tag in concrete_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        else:
            unlabeled_sentence_count += 1
        dependencies = label.get("depends_on") or []
        if isinstance(dependencies, list):
            dependency_count += len([dependency for dependency in dependencies if dependency])

    return {
        "line_number": line_number,
        "model_id": payload.get("model_id"),
        "dataset_name": payload.get("dataset_name"),
        "task_id": payload.get("task_id"),
        "sample_id": payload.get("sample_id"),
        "complete": payload.get("complete"),
        "is_correct": payload.get("is_correct"),
        "prompt": payload.get("prompt"),
        "reasoning": payload.get("reasoning"),
        "answer": payload.get("answer"),
        "sentences": normalized_sentences,
        "labels": labels,
        "label_provider": payload.get("label_provider"),
        "label_model": payload.get("label_model"),
        "validation_warnings": payload.get("validation_warnings") or [],
        "labeled_at": payload.get("labeled_at"),
        "tag_counts": tag_counts,
        "dependency_count": dependency_count,
        "labeled_sentence_count": labeled_sentence_count,
        "unlabeled_sentence_count": unlabeled_sentence_count,
    }


def render_html(rows: list[dict[str, Any]]) -> str:
    data_json = json.dumps(rows, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reasoning Label Review</title>
<style>
:root {{ color-scheme: dark; --bg:#0d1016; --panel:#151a22; --panel2:#10141b; --line:#293241; --muted:#93a1b5; --text:#edf4ff; --blue:#75a7ff; --green:#47c784; --red:#ff6b75; --amber:#ffb75f; --violet:#c18cff; --cyan:#5bd5d5; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }}
header {{ position:sticky; top:0; z-index:5; display:grid; grid-template-columns:1fr 150px 155px 170px 170px; gap:12px; align-items:center; padding:12px 16px; background:rgba(13,16,22,.97); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }}
input, select {{ background:#080b10; color:var(--text); border:1px solid #344153; border-radius:9px; padding:9px 11px; outline:none; }}
input:focus, select:focus {{ border-color:var(--blue); }}
main {{ display:grid; grid-template-columns:380px minmax(0,1fr); min-height:calc(100vh - 58px); }}
#list {{ border-right:1px solid var(--line); overflow:auto; max-height:calc(100vh - 58px); background:#0a0d12; }}
.item {{ padding:12px 14px; border-bottom:1px solid #202835; cursor:pointer; }}
.item:hover {{ background:#131923; }}
.item.active {{ background:#1b2330; box-shadow:inset 3px 0 0 var(--blue); }}
.itemTop {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
.task {{ font-size:16px; font-weight:750; letter-spacing:.01em; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.mini {{ color:var(--muted); font-size:11px; line-height:1.35; margin-top:6px; }}
.tagCloud {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }}
.badge, .tag {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; white-space:nowrap; background:#303743; }}
.tag {{ border:1px solid rgba(255,255,255,.08); }}
.good {{ background:rgba(71,199,132,.22); color:#a9f0c8; }} .bad {{ background:rgba(255,107,117,.22); color:#ffc1c5; }} .unknown {{ background:#414752; }}
.complete {{ background:rgba(117,167,255,.18); color:#c5d9ff; }} .incomplete {{ background:rgba(255,183,95,.18); color:#ffe0b7; }}
#detail {{ overflow:auto; max-height:calc(100vh - 58px); }}
.hero {{ position:sticky; top:0; z-index:3; padding:18px 22px 14px; background:rgba(13,16,22,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }}
.titleRow {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }}
h2 {{ margin:0; font-size:26px; }}
.grid {{ display:grid; grid-template-columns:repeat(6, minmax(110px,1fr)); gap:10px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }}
.label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }}
.value {{ font-size:18px; margin-top:5px; font-variant-numeric:tabular-nums; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.content {{ padding:18px 22px 32px; }}
.sentence {{ display:grid; grid-template-columns:58px minmax(170px,240px) minmax(0,1fr) minmax(120px,180px); gap:12px; align-items:start; margin:8px 0; padding:12px; border:1px solid #253040; border-radius:12px; background:var(--panel2); }}
.sentence:hover {{ border-color:#3e4d62; background:#151b25; }}
.idx {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
.text {{ white-space:pre-wrap; line-height:1.5; font-size:15px; }}
.deps {{ color:var(--muted); font-size:13px; line-height:1.45; }}
.depLink {{ color:var(--blue); cursor:pointer; text-decoration:none; margin-right:6px; }}
.depLink:hover {{ text-decoration:underline; }}
.warn {{ border:1px solid rgba(255,183,95,.45); background:rgba(255,183,95,.11); color:#ffe0b7; border-radius:12px; padding:10px 12px; margin-bottom:14px; }}
details {{ margin-top:18px; }}
summary {{ cursor:pointer; color:var(--blue); }}
pre {{ white-space:pre-wrap; background:#080b10; border:1px solid #252f3e; border-radius:10px; padding:14px; overflow:auto; }}
.problem_setup {{ background:rgba(117,167,255,.18); color:#cfe0ff; }}
.plan_generation {{ background:rgba(193,140,255,.18); color:#e4cfff; }}
.fact_retrieval {{ background:rgba(91,213,213,.18); color:#c7f8f8; }}
.active_computation {{ background:rgba(71,199,132,.18); color:#bdf1d2; }}
.result_consolidation {{ background:rgba(255,183,95,.18); color:#ffe0b7; }}
.uncertainty_management {{ background:rgba(255,107,117,.18); color:#ffc1c5; }}
.final_answer_emission {{ background:rgba(255,230,92,.18); color:#fff3a8; }}
.self_checking {{ background:rgba(136,221,120,.18); color:#dcffd6; }}
.unknown_tag {{ background:#414752; color:#d8dee8; }}
.unlabeled {{ background:rgba(255,183,95,.12); color:#ffe0b7; border-color:rgba(255,183,95,.35); }}
@media (max-width:1050px) {{ main {{ grid-template-columns:1fr; }} #list {{ max-height:300px; border-right:0; border-bottom:1px solid var(--line); }} header {{ grid-template-columns:1fr; }} .grid {{ grid-template-columns:1fr 1fr; }} .sentence {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <input id="search" placeholder="Search task, sentence, prompt, answer...">
  <select id="tagFilter"><option value="all">all tags</option></select>
  <select id="correct"><option value="all">all correctness</option><option value="true">correct</option><option value="false">incorrect</option><option value="unknown">unknown</option></select>
  <select id="warnings"><option value="all">all warnings</option><option value="with">with warnings</option><option value="without">without warnings</option></select>
  <select id="sort"><option value="task">sort: task</option><option value="sentences">sort: sentences</option><option value="deps">sort: dependencies</option><option value="warnings">sort: warnings</option></select>
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
const tagFilter = document.getElementById('tagFilter');
const correct = document.getElementById('correct');
const warnings = document.getElementById('warnings');
const sort = document.getElementById('sort');
const TAGS = [...new Set(DATA.flatMap(row => Object.keys(row.tag_counts || {{}})))].sort();
tagFilter.innerHTML += `<option value="{UNLABELED_TAG}">unlabeled</option>`;
tagFilter.innerHTML += TAGS.map(tag => `<option value="${{escAttr(tag)}}">${{esc(tag)}}</option>`).join('');

function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function escAttr(s) {{ return esc(s).replace(/`/g, '&#96;'); }}
function badge(v) {{
  if (v === true) return '<span class="badge good">correct</span>';
  if (v === false) return '<span class="badge bad">incorrect</span>';
  return '<span class="badge unknown">unknown</span>';
}}
function completeBadge(v) {{ return v ? '<span class="badge complete">complete</span>' : '<span class="badge incomplete">incomplete</span>'; }}
function tagClass(tag) {{ return tag === '{UNLABELED_TAG}' ? '{UNLABELED_TAG}' : (TAGS.includes(tag) ? tag : 'unknown_tag'); }}
function tagPill(tag, count=null) {{ return `<span class="tag ${{tagClass(tag)}}">${{esc(tag)}}${{count === null ? '' : ` ×${{count}}`}}</span>`; }}
function labelFor(row, index) {{
  const label = (row.labels || {{}})[String(index)] || {{}};
  let tags = label.function_tags || [];
  if (!Array.isArray(tags)) tags = [String(tags)];
  tags = tags.map(tag => String(tag).trim()).filter(tag => tag && tag !== 'unknown');
  if (!tags.length) tags = ['{UNLABELED_TAG}'];
  let deps = label.depends_on || [];
  if (!Array.isArray(deps)) deps = [String(deps)];
  return {{ tags, deps }};
}}
function filtered() {{
  const q = search.value.toLowerCase();
  let rows = DATA.filter(row => {{
    const c = correct.value;
    if (c === 'true' && row.is_correct !== true) return false;
    if (c === 'false' && row.is_correct !== false) return false;
    if (c === 'unknown' && row.is_correct !== null && row.is_correct !== undefined) return false;
    if (warnings.value === 'with' && !(row.validation_warnings || []).length) return false;
    if (warnings.value === 'without' && (row.validation_warnings || []).length) return false;
    if (tagFilter.value === '{UNLABELED_TAG}' && !(row.unlabeled_sentence_count || 0)) return false;
    if (tagFilter.value !== 'all' && tagFilter.value !== '{UNLABELED_TAG}' && !(row.tag_counts || {{}})[tagFilter.value]) return false;
    const haystack = [
      row.task_id, row.dataset_name, row.label_model, row.prompt, row.answer,
      ...(row.sentences || []).map(s => s.text),
    ].join('\\n').toLowerCase();
    return !q || haystack.includes(q);
  }});
  rows.sort((a, b) => {{
    if (sort.value === 'sentences') return (b.sentences || []).length - (a.sentences || []).length;
    if (sort.value === 'deps') return (b.dependency_count || 0) - (a.dependency_count || 0);
    if (sort.value === 'warnings') return (b.validation_warnings || []).length - (a.validation_warnings || []).length;
    return String(a.task_id).localeCompare(String(b.task_id), undefined, {{ numeric:true }});
  }});
  return rows;
}}
function renderList() {{
  const rows = filtered();
  list.innerHTML = rows.map((row, i) => {{
    const tags = Object.entries(row.tag_counts || {{}}).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([tag,count]) => tagPill(tag, count)).join('');
    const unlabeled = row.unlabeled_sentence_count ? tagPill('{UNLABELED_TAG}', row.unlabeled_sentence_count) : '';
    const warn = (row.validation_warnings || []).length ? `<span class="badge incomplete">${{row.validation_warnings.length}} warn</span>` : '';
    return `<div class="item ${{i === current ? 'active' : ''}}" onclick="current=${{i}}; render()">
      <div class="itemTop"><span class="task">${{esc(row.task_id)}}</span><span>${{badge(row.is_correct)}} ${{warn}}</span></div>
      <div class="mini">${{esc(row.dataset_name)}} · sample ${{row.sample_id}} · ${{row.labeled_sentence_count}}/${{(row.sentences || []).length}} labels · ${{row.unlabeled_sentence_count || 0}} unlabeled · ${{row.dependency_count}} deps</div>
      <div class="mini">${{esc(row.label_provider)}} / ${{esc(row.label_model)}} · line ${{row.line_number}}</div>
      <div class="tagCloud">${{tags}}${{unlabeled}}</div>
    </div>`;
  }}).join('');
}}
function renderDetail() {{
  const rows = filtered();
  if (!rows.length) {{ detail.innerHTML = '<div class="content"><p>No matching rows.</p></div>'; return; }}
  if (current >= rows.length) current = 0;
  const row = rows[current];
  const warningsHtml = (row.validation_warnings || []).length
    ? `<div class="warn"><b>Validation warnings</b><br>${{(row.validation_warnings || []).map(esc).join('<br>')}}</div>`
    : '';
  const allTags = Object.entries(row.tag_counts || {{}}).sort((a,b)=>b[1]-a[1]).map(([tag,count]) => tagPill(tag, count)).join('');
  const unlabeledTag = row.unlabeled_sentence_count ? tagPill('{UNLABELED_TAG}', row.unlabeled_sentence_count) : '';
  const sentences = (row.sentences || []).map((sentence) => {{
    const label = labelFor(row, sentence.index);
    const deps = label.deps.length
      ? label.deps.map(dep => `<a class="depLink" onclick="scrollToSentence('${{escAttr(dep)}}')">#${{esc(dep)}}</a>`).join('')
      : '<span>none</span>';
    return `<div class="sentence" id="sentence-${{escAttr(sentence.index)}}">
      <div class="idx">#${{esc(sentence.index)}}</div>
      <div class="tagCloud">${{label.tags.map(tag => tagPill(tag)).join('')}}</div>
      <div class="text">${{esc(sentence.text)}}</div>
      <div class="deps"><b>depends on</b><br>${{deps}}</div>
    </div>`;
  }}).join('');
  detail.innerHTML = `<div class="hero"><div class="titleRow"><h2>${{esc(row.task_id)}}</h2><span>${{badge(row.is_correct)}} ${{completeBadge(row.complete)}}</span></div>
    <div class="grid">
      <div class="card"><div class="label">Dataset</div><div class="value">${{esc(row.dataset_name)}}</div></div>
      <div class="card"><div class="label">Sentences</div><div class="value">${{row.labeled_sentence_count}} / ${{(row.sentences || []).length}}</div></div>
      <div class="card"><div class="label">Unlabeled</div><div class="value">${{row.unlabeled_sentence_count || 0}}</div></div>
      <div class="card"><div class="label">Dependencies</div><div class="value">${{row.dependency_count}}</div></div>
      <div class="card"><div class="label">Provider</div><div class="value">${{esc(row.label_provider)}}</div></div>
      <div class="card"><div class="label">Model</div><div class="value" title="${{escAttr(row.label_model)}}">${{esc(row.label_model)}}</div></div>
    </div></div>
    <div class="content">
      ${{warningsHtml}}
      <div class="tagCloud" style="margin-bottom:16px">${{allTags}}${{unlabeledTag}}</div>
      <h3>Sentence Labels</h3>${{sentences}}
      <details><summary>Prompt</summary><pre>${{esc(row.prompt)}}</pre></details>
      <details open><summary>Answer / Code</summary><pre>${{esc(row.answer)}}</pre></details>
      <details><summary>Raw labels</summary><pre>${{esc(JSON.stringify(row.labels, null, 2))}}</pre></details>
    </div>`;
}}
function scrollToSentence(index) {{
  const node = document.getElementById(`sentence-${{index}}`);
  if (node) node.scrollIntoView({{ behavior:'smooth', block:'center' }});
}}
function render() {{ renderList(); renderDetail(); }}
[search, tagFilter, correct, warnings, sort].forEach(el => el.addEventListener('input', () => {{ current = 0; render(); }}));
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

