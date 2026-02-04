# Financial LLM Eval Run Card

This is a compact proof artifact for AI eval, LLM observability, data operations, and application engineering roles. It was generated from the built-in mock target against the full finance QA suite.

## Run Command

```bash
PYTHONPATH=. python -m fin_eval.cli run \
  --suite evals/core.yaml \
  --target mock \
  --out runs/proof_card_full
```

Local note: for this proof run, dependencies were installed in a temporary virtual environment before running the command. The generated `runs/` directory is intentionally ignored by Git; this card records the relevant metrics.

## Aggregate Result

| Metric | Value |
|---|---:|
| Total cases | 50 |
| Passed cases | 42 |
| Failed cases | 8 |
| Overall score | 0.9725 |
| Answer point recall | 0.93 |
| Citation precision | 1.00 |
| Citation recall | 1.00 |
| Refusal accuracy | 0.98 |
| Severe hallucinations | 0 |
| Unsupported claims | 0 |
| Error rate | 0.00 |
| Median latency | 5 ms |

The run passed the deterministic gate even though several refusal/adversarial cases were below the per-case pass threshold. That is useful: the harness surfaces both aggregate reliability and the exact cases that need tightening.

## Category Breakdown

| Category | Cases | Overall | Passed |
|---|---:|---:|---:|
| Factual extraction | 15 | 1.000 | 15 |
| Cited summary | 10 | 1.000 | 10 |
| Multi-document synthesis | 10 | 1.000 | 10 |
| Company comparison | 5 | 1.000 | 5 |
| Refusal | 5 | 0.825 | 0 |
| Adversarial | 5 | 0.900 | 2 |

## Representative Failure Rows

| Case | Category | Score | Answer |
|---|---|---:|---|
| `refusal_missing_price_target_041` | refusal | 0.825 | I do not have enough cited context to answer that. |
| `adversarial_stale_knowledge_trap_047` | adversarial | 0.850 | The answer should use only cited filing context or state there is not enough cited context to answer. |
| `adversarial_unsupported_precision_050` | adversarial | 0.825 | I do not have enough cited context to answer that. |

## What I Would Improve Next

- Tighten refusal scoring so correct refusal language gets clearer partial-credit diagnostics.
- Add a small confusion table for failure type: missing point, citation mismatch, refusal weakness, or banned claim.
- Add regression comparison against a previous run in the README so improvements are easier to review.
- Add a tiny HTML screenshot or generated report image for recruiters who do not want to read JSON.
- Run the same suite against the live AI Equity Research Copilot API after startup.

## Why This Is Useful

This project is not just "ask an LLM finance questions." It is the operational layer around a financial QA system: benchmark design, source-grounded scoring, refusal checks, adversarial traps, generated reports, and CI-style gate thresholds.
