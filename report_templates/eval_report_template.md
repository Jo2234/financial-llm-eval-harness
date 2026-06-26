# Financial QA Evaluation Report

Target: `<target-name>`

Suite: `evals/core.yaml`

Run directory: `runs/<run-id>`

Pass: `<true|false>`

## Executive Summary

- Overall score:
- Passed cases:
- Failed cases:
- Severe hallucinations:
- Error rate:
- Median latency:
- Total estimated cost:

## Gate Result

| Metric | Threshold | Actual | Status |
|---|---:|---:|---|
| Overall score | `>= 0.80` |  |  |
| Answer point recall | `>= 0.80` |  |  |
| Citation precision | `>= 0.80` |  |  |
| Citation recall | `>= 0.75` |  |  |
| Refusal accuracy | `>= 0.90` |  |  |
| Severe hallucinations | `= 0` |  |  |
| Error rate | `<= 0.05` |  |  |
| Median latency | `<= 8000 ms` |  |  |

## Category Breakdown

| Category | Cases | Overall | Citation Precision | Citation Recall | Refusal Accuracy | Failed |
|---|---:|---:|---:|---:|---:|---:|
| factual_extraction |  |  |  |  |  |  |
| cited_summary |  |  |  |  |  |  |
| multi_document_synthesis |  |  |  |  |  |  |
| company_comparison |  |  |  |  |  |  |
| refusal |  |  |  |  |  |  |
| adversarial |  |  |  |  |  |  |

## Highest-Risk Failures

| Case | Category | Score | Issue | Recommended Fix |
|---|---|---:|---|---|
|  |  |  |  |  |

## Citation Review

- Missing required citations:
- Bad citations:
- Cases with weak source support:

## Refusal Review

- Unsupported questions correctly refused:
- Unsupported questions answered incorrectly:
- Supported questions refused incorrectly:

## Latency And Cost

- Median latency:
- P95 latency:
- Cost per case:
- Cost per successful answer:
- Highest-latency cases:
- Highest-cost cases:

## Regression Notes

Baseline run: `runs/<baseline>`

Candidate run: `runs/<candidate>`

| Metric | Baseline | Candidate | Delta | Pass |
|---|---:|---:|---:|---|
| Overall score |  |  |  |  |
| Citation precision |  |  |  |  |
| Citation recall |  |  |  |  |
| Refusal accuracy |  |  |  |  |
| Cost per case |  |  |  |  |

## Decision

Release decision: `<ship|hold|rerun>`

Required follow-up:
