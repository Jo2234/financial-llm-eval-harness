# Financial LLM Evaluation Harness

Library-first evaluation harness for financial QA systems. It validates curated financial QA cases, runs them against a target adapter, scores answer quality and citation behavior, and writes machine-readable plus human-readable reports.

## Eval Suite

The core suite lives at `evals/core.yaml` and currently contains 50 cases:

| Category | Cases | Purpose |
|---|---:|---|
| `factual_extraction` | 15 | Pull specific metrics, drivers, risks, or management statements from a filing. |
| `cited_summary` | 10 | Summarize a filing section with source-grounded citations. |
| `multi_document_synthesis` | 10 | Compare annual, quarterly, and transcript-style sources. |
| `company_comparison` | 5 | Compare companies across growth, margin, macro, and supply-chain exposure. |
| `refusal` | 5 | Check that the system refuses unsupported or non-public questions. |
| `adversarial` | 5 | Catch investment advice, stale knowledge, fake citation, prompt injection, and unsupported precision traps. |

Each case includes an `id`, `category`, `difficulty`, `question`, `expected_answer_points`, `required_citation_rules`, `must_not_include`, `refusal_expected`, and tags. Multi-document cases list every source document in `documents`; the deterministic citation rule is kept to the primary document so the built-in mock adapter remains useful as a smoke test.

## Commands

Run commands from the project root:

```bash
cd projects/financial-llm-eval-harness
```

Validate the suite:

```bash
PYTHONPATH=. python -m fin_eval.cli validate-suite --suite evals/core.yaml
```

Run the suite with the built-in mock target:

```bash
PYTHONPATH=. python -m fin_eval.cli run --suite evals/core.yaml --target mock --out runs/latest
```

Run selected cases:

```bash
PYTHONPATH=. python -m fin_eval.cli run --suite evals/core.yaml --target mock --category refusal,adversarial --out runs/traps
PYTHONPATH=. python -m fin_eval.cli run --suite evals/core.yaml --target mock --tag nvda --limit 5 --out runs/nvda_sample
PYTHONPATH=. python -m fin_eval.cli run --suite evals/core.yaml --target mock --case-id nvda_10k_datacenter_revenue_001 --out runs/single
```

Evaluate the copilot API:

```bash
PYTHONPATH=. python -m fin_eval.cli run \
  --suite evals/core.yaml \
  --target copilot-api \
  --base-url http://localhost:8001 \
  --endpoint /research/chat \
  --timeout 20 \
  --out runs/copilot
```

Override CI gate thresholds:

```bash
PYTHONPATH=. python -m fin_eval.cli run \
  --suite evals/core.yaml \
  --target mock \
  --threshold-overall 0.85 \
  --threshold-citation-precision 0.90 \
  --threshold-citation-recall 0.80 \
  --threshold-refusal-accuracy 0.95 \
  --max-error-rate 0.02 \
  --max-median-latency-ms 8000 \
  --out runs/strict
```

Print generated reports:

```bash
PYTHONPATH=. python -m fin_eval.cli report --run runs/latest --format markdown
PYTHONPATH=. python -m fin_eval.cli report --run runs/latest --format html
PYTHONPATH=. python -m fin_eval.cli report --run runs/latest --format json
```

List failures:

```bash
PYTHONPATH=. python -m fin_eval.cli list-failures --run runs/latest
PYTHONPATH=. python -m fin_eval.cli list-failures --run runs/latest --json
```

Compare two runs:

```bash
PYTHONPATH=. python -m fin_eval.cli compare --baseline runs/baseline --candidate runs/latest
PYTHONPATH=. python -m fin_eval.cli compare --baseline runs/baseline --candidate runs/latest --gate
```

Run tests:

```bash
PYTHONPATH=. pytest
```

## Run Outputs

`fin_eval.cli run` writes these artifacts to the `--out` directory:

| File | Format | Contents |
|---|---|---|
| `results.json` | JSON | Full run payload with metadata, summary metrics, category breakdown, gate result, and per-case results. |
| `summary.md` | Markdown | Human-readable run report with aggregate metrics, gate violations, category metrics, failures, severe hallucination cases, slowest cases, costliest cases, and recommendations. |
| `report.html` | HTML | Browser-friendly version of the run report. |
| `failures.csv` | CSV | Failed case rows with case ID, category, difficulty, score, error, and answer. |
| `config.json` | JSON | Suite path, target, endpoint, fixture, filters, and thresholds used for the run. |

The CLI prints a compact JSON summary to stdout and exits with code `0` when the aggregate gate passes. It exits with code `1` when the gate fails.

Important top-level `results.json` fields:

- `metadata`: suite, target, endpoint, run size, start time, and duration.
- `summary`: aggregate quality, citation, refusal, format, latency, token, cost, and error metrics.
- `category_breakdown`: aggregate metrics grouped by case category.
- `gate`: thresholds and any violations.
- `results`: per-case score details, answer text, citations, token usage, and raw response.

## Report Template

Use `report_templates/eval_report_template.md` when drafting a manual report or PR summary around generated run artifacts. The runtime reporter is currently implemented in `fin_eval/runner.py`; the template is documentation-only.
