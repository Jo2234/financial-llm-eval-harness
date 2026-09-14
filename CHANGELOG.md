# Release notes

## Unreleased — scorer v3

- Normalize supported currency/scale forms and retain numeric commas when checking quantity directions.
- Add local metric/topic binding, including a capitalization heuristic for a new named subject despite a shared company label, and simple direct amount negation checks, plus finite affirmative advice, non-year price-target, guarantee and refusal-forecast rules. These are deterministic flags with documented misses/false flags, not semantic entailment or authenticated hallucination measures.
- Accept explicit document maps with duplicate/unknown/missing identity validation, scorer-facing canonical citations and preserved raw UUIDs. API requests contain only question/company scope/top_k.
- Include the complete public fetch/serve/ingest/run/export workflow, pinned dependencies and methodology in `benchmark/` without depending on an external report repository.
- Verify fixed workflow inventories and validate all retained non-gold records and cross-file links before writing public export files. Typed config/metadata/source/result/ingestion schemas, credential-free source URLs and target free-text redaction preserve measured scores/gates without rescoring responses. Curated gold remains an explicit trust boundary.
- CI installs pinned source normalizers and checks out the public Copilot parser at a fixed revision for its required offline PDF integration test; no target/model calls are made.

Scorer v3 requires a new baseline. Existing historical captures/examples and suite facts remain unchanged.

## 0.2.0 — 2026-09-11

The first packaged release of the source-grounded suite and scorer v2. The repository previously declared version 0.1.0; this release includes the subsequent evaluation corrections.

- 50 curated financial QA cases with pinned primary-source evidence; a separate two-case plumbing suite.
- Execution errors and empty responses are distinct from refusal behavior, with explicit behavior denominators.
- Case fingerprints and scorer versions prevent misleading regression comparisons across changed cohorts.
- CLI reports include JSON, Markdown, HTML, CSV, and JSONL artifacts.
- MIT license text and source distribution containing suites, evidence metadata, examples, and source tests. The wheel contains the library and CLI; suites are supplied separately from the matching source release.

### Install and smoke test

Download the wheel and source archive from the matching GitHub release, verify their published SHA-256 checksums, then:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install financial_llm_eval_harness-0.2.0-py3-none-any.whl
tar -xzf financial_llm_eval_harness-0.2.0.tar.gz
cd financial_llm_eval_harness-0.2.0
fin-eval validate-suite --suite evals/core.yaml
fin-eval run --suite evals/plumbing.yaml --target mock --out runs/smoke
fin-eval compare --baseline runs/smoke --candidate runs/smoke --gate
```

The smoke run demonstrates installation and reporting, not model quality. For quality measurements, use the current source corpus and a real target; reference fixtures only exercise the scorer. The deterministic scorer measures lexical/numeric coverage and citation metadata, not semantic entailment or verified hallucination rates.

### Build from source

```sh
python -m pip install build
python -m build
```

Do not compare v1 historical reports with v2 quality scores. Original historical artifacts remain separately preserved; a fresh execution is different from offline rescoring of saved responses.
