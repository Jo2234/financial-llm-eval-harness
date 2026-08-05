"""Dependency-free, offline HTML reports with inspectable case evidence."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

REPORT_CSS = Path(__file__).with_name("report.css").read_text()

FILTER_SCRIPT = """
<script>
(() => {
  const form = document.querySelector('#case-filters');
  const cases = [...document.querySelectorAll('.case')];
  const query = document.querySelector('#case-search');
  const category = document.querySelector('#case-category');
  const status = document.querySelector('#case-status');
  function filter() {
    const term = query.value.trim().toLocaleLowerCase();
    let visible = 0;
    for (const item of cases) {
      const matches = (!term || item.dataset.search.includes(term)) &&
        (!category.value || item.dataset.category === category.value) &&
        (!status.value || item.dataset.status === status.value ||
          (status.value === 'refusal' && item.dataset.refusal === 'true'));
      item.hidden = !matches;
      if (matches) visible++;
    }
    document.querySelector('#case-count').textContent = `${visible} of ${cases.length} cases`;
    document.querySelector('#no-results').hidden = visible !== 0;
  }
  form.addEventListener('submit', event => event.preventDefault());
  form.addEventListener('input', filter);
  form.addEventListener('change', filter);
  form.addEventListener('reset', () => { requestAnimationFrame(filter); query.focus(); });
  document.querySelector('#empty-reset').addEventListener('click', () => form.reset());
  form.hidden = false;
})();
</script>
"""


def _json(value: Any) -> str:
    return escape(json.dumps(value, indent=2, ensure_ascii=False))


def _list(values: list[Any], empty: str) -> str:
    if not values:
        return f'<p class="muted">{escape(empty)}</p>'
    return "<ul>" + "".join(f"<li>{escape(str(v))}</li>" for v in values) + "</ul>"


def render_report(payload: dict[str, Any], recommendations: list[str]) -> str:
    s, metadata = payload["summary"], payload.get("metadata", {})
    rows = []
    for row in payload["results"]:
        definition = row.get("case_definition", {})
        question = row.get("question") or definition.get("question") or row["case_id"]
        category = row["category"]
        expected = definition.get("refusal_expected", False)
        state = "pass" if row["passed"] else "fail"
        execution = row.get("execution_status", "error" if row.get("error") else "not recorded")
        search = " ".join(
            [row["case_id"], category, question, *definition.get("company_ids", []), *row.get("tags", [])]
        ).lower()
        mini = "".join(
            f"<div>{label}<b>{row[key]:.1%}</b></div>"
            for key, label in [
                ("answer_point_recall", "Answer-point recall"),
                ("citation_precision", "Citation precision"),
                ("citation_recall", "Required-document recall"),
            ]
            if key in row
        )
        missing = _list(row.get("missing_points", []), "No missing points recorded.")
        citations = _list(
            [
                f"{c.get('document_id', 'Unknown document')} — {c.get('reason', 'Citation mismatch')}"
                for c in row.get("bad_citations", [])
            ],
            "No citation mismatches recorded.",
        )
        missing_docs = _list(row.get("missing_required_citations", []), "No missing required citations recorded.")
        refused = "Yes" if row.get("refused") else "No"
        behavior = "Correct" if row.get("refusal_correct") else "Incorrect"
        if row.get("behavior_evaluated") is False:
            behavior = "Unavailable; response excluded from behavior accuracy"
        error = (
            f'<p class="fail"><strong>Execution error:</strong> {escape(row["error"])}</p>' if row.get("error") else ""
        )
        response = escape(str(row.get("answer") or "No response text recorded."))
        rows.append(f'''<details class="case" data-case-id="{escape(row["case_id"])}" data-category="{escape(category)}" data-status="{state}" data-refusal="{str(expected).lower()}" data-search="{escape(search)}">
<summary>
<span>
<span class="case-meta">{escape(category.replace("_", " ").title())} · {escape(row["case_id"])}</span>
<span class="case-title">{escape(question)}</span>
</span>
<span class="case-score">
<span class="status {state}">{"Passed" if row["passed"] else "Failed"}</span>
<span>{row["overall_score"]:.3f} score</span>
<span class="expand-icon" aria-hidden="true">+</span>
</span>
</summary>
<div class="case-body">{error}<div class="mini-metrics">{mini}</div>
<div class="evidence-grid">
<div>
<h4>Missing factual points</h4>{missing}<h4>Citation mismatches</h4>{citations}<h4>Missing required citations</h4>{missing_docs}</div>
<div>
<h4>Refusal behavior</h4>
<p>Expected refusal: {"Yes" if expected else "No"} · Detected refusal: {refused}<br>Behavior check: {behavior}</p>
<h4>Execution</h4>
<p>{escape(execution.title())} · {escape(str(row.get("latency_ms", "Not recorded")))} ms</p>
<h4>Recorded response</h4>
<p>{response}</p>
</div>
</div>
<details class="raw">
<summary>Full case record, citations and scoring evidence</summary>
<pre>{_json(row)}</pre>
</details>
</div>
</details>''')
    options = "".join(
        f'<option value="{escape(category)}">{escape(category.replace("_", " ").title())}</option>'
        for category in sorted({r["category"] for r in payload["results"]})
    )
    categories = "".join(
        f'<tr><th scope="row">{escape(category.replace("_", " ").title())}</th><td>{metrics["total_cases"]}</td><td>{metrics["overall_score"]:.3f}</td><td>{metrics["passed_cases"]}</td></tr>'
        for category, metrics in payload.get("category_breakdown", {}).items()
    )
    history = (
        '<div class="note"><strong>Historical response rescore:</strong> saved answers were rescored offline; no new target requests were made.</div>'
        if metadata.get("rescored_at")
        else ""
    )
    redacted = (
        '<div class="note">Public export: response prose is omitted. Original response hashes, citation metadata and scores are preserved. Text cannot be independently rescored without the original local or newly reproduced captures.</div>'
        if metadata.get("public_export", {}).get("redacted")
        else ""
    )
    target = escape(str(payload["target"]))
    mode = escape(
        str(
            metadata.get(
                "evaluation_mode", "historical-capture-rescore" if metadata.get("rescored_at") else payload["target"]
            )
        )
    )
    total = s["total_cases"]
    gate = "Passed" if payload["passed"] else "Not passed"
    recommendation_list = _list(recommendations, "No recommendations recorded.")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Financial QA Eval Report</title><style>{REPORT_CSS}</style></head><body>
<a class="skip" href="#cases">Skip to cases</a>
<main>
<header class="masthead">
<a class="brand" href="#">
<span class="brand-mark" aria-hidden="true">ƒ</span>Financial QA / Eval report</a>
<nav class="nav-links" aria-label="Report links">
<a href="results.json">Results JSON ↗</a>
<a href="failures.csv">Failures CSV ↓</a>
</nav>
</header>
<header class="hero">
<p class="eyebrow">
<span class="pill">Saved evaluation</span>{escape(str(metadata.get("started_at", "Capture date not recorded")))}</p>
<h1>Every answer.<br>Every check.</h1>
<p class="lede">Inspect the deterministic checks behind <strong>{target}</strong>. Find failed cases, compare categories and trace each result to its scoring evidence.</p>
<p class="muted">Evaluation mode: <code>{mode}</code> · Run gate: <strong>{gate}</strong>
</p>{history}
<dl class="stats">
<div class="stat">
<dt>Strict case passes</dt>
<dd>{s["passed_cases"]} <small>/ {total}</small>
</dd>
</div>
<div class="stat">
<dt>Composite score</dt>
<dd>{s["overall_score"]:.3f}<small> / 1</small>
</dd>
</div>
<div class="stat">
<dt>Citation precision</dt>
<dd>{s.get("citation_precision", 0):.1%}</dd>
</div>
<div class="stat">
<dt>Failed cases</dt>
<dd class="fail">{s["failed_cases"]}</dd>
</div>
</dl>
<p class="hint">Composite score is not accuracy. The pass gate covers deterministic checks only; contextual judge rubrics are not automatically evaluated. Refusal accuracy uses only nonempty, error-free responses. Unavailable responses remain failed cases.</p><nav class="anchor-nav" aria-label="Report sections"><a href="#cases">Case explorer ↓</a><a href="#categories">Category breakdown</a><a href="#recommendations">Recommendations</a><a href="#provenance">Metrics & provenance</a></nav></header>
<section id="cases"><div class="section-heading"><h2>Case explorer</h2><p class="muted">Open a case to inspect the evidence</p></div>{redacted}
<form id="case-filters" class="filters" hidden>
<div class="search-field">
<label for="case-search">Search cases</label>
<input id="case-search" type="search" placeholder="Question, company or case ID…" autocomplete="off">
</div>
<div>
<label for="case-category">Category</label>
<select id="case-category">
<option value="">All categories</option>{options}</select>
</div>
<div>
<label for="case-status">Outcome</label>
<select id="case-status">
<option value="">All outcomes</option>
<option value="fail">Failed</option>
<option value="pass">Passed</option>
<option value="refusal">Expected refusal</option>
</select>
</div>
<button type="reset" class="secondary">Reset filters</button>
</form>
<noscript>
<p class="noscript">All cases are shown below. Browser search and case details work without JavaScript.</p>
</noscript>
<div class="results-count">
<span id="case-count" role="status" aria-live="polite">{len(rows)} of {len(rows)} cases</span>
<span>Score shown on a 0–1 scale</span>
</div>
<div id="no-results" class="empty" hidden>
<h3>No matching cases</h3>
<p>Try a different question, company or outcome.</p>
<button id="empty-reset" class="secondary" type="button">Clear all filters</button>
</div>{"".join(rows)}</section>
<section id="categories" class="section">
<h2>Category breakdown</h2>
<p class="muted">Composite score and strict passes, grouped by the suite’s categories.</p>
<div class="table category-table" role="region" aria-label="Category metrics" tabindex="0">
<table>
<thead>
<tr>
<th>Category</th>
<th>Cases</th>
<th>Composite score</th>
<th>Passed</th>
</tr>
</thead>
<tbody>{categories}</tbody>
</table>
</div>
</section>
<section id="recommendations" class="section"><h2>Recommendations</h2>{recommendation_list}</section>
<section id="provenance" class="section">
<h2>Metrics & provenance</h2>
<p class="muted">Original recorded values, including scorer version and execution context.</p>
<details class="raw">
<summary>All aggregate metrics</summary>
<pre>{_json(s)}</pre>
</details>
<details class="raw">
<summary>Run provenance</summary>
<pre>{_json(metadata)}</pre>
</details>
<details class="raw">
<summary>Run gate and violations</summary>
<pre>{_json(payload.get("gate", {}))}</pre>
</details>
</section>
<footer>Financial LLM Evaluation Harness · Saved results, rendered offline. <a href="https://github.com/Jo2234/financial-llm-eval-harness">Harness source ↗</a></footer></main>{FILTER_SCRIPT}</body></html>"""
