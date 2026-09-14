# Reproduce the deterministic comparison

This public directory contains the complete fetch → serve → ingest → capture → public-export workflow. It requires only this public harness, the public [Equity Copilot](https://github.com/Jo2234/ai-equity-research-copilot), and the primary-source URLs in [the evidence manifest](../evals/evidence/manifest.yaml). No other repository or hosted private report is required. See [methodology and limitations](METHODOLOGY.md) before interpreting scores.

A new run measures the current 50-case factual suite with scorer v3. Both targets are deterministic: Copilot uses its explicit `local` extractive provider and the comparison baseline uses lexical token overlap. Neither target uses an LLM. This workflow establishes a **new baseline**, not an updated score for an earlier capture. The checked-in historical examples retain their original metrics.

## Install

Clone both public repositories adjacent to one another and record the revisions you use. These scripts reproduce the **current workflow**, producing a new scorer-v3 baseline. They do not reproduce the September 18 historical 34/50 score or any other v2 result, and this repository does not publish the original full response captures for rescoring.

Keep this current harness checkout to use `benchmark/`; older harness revisions did not contain this directory. Do not check out a legacy harness and then run these commands. Exact reproduction of a historical result needs its original scorer, suite, target and runner revisions plus matching full source snapshots; the current commands are not a substitute. Historical figures must retain their original scorer/version labels. For a new v3 comparison against a pinned older Copilot target, keep the current harness and check out only the desired public Copilot revision, then record that revision in the new capture.

```sh
git clone https://github.com/Jo2234/financial-llm-eval-harness.git
git clone https://github.com/Jo2234/ai-equity-research-copilot.git
cd financial-llm-eval-harness
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r benchmark/requirements.txt
python -m pip install --no-deps -e .
fin-eval validate-suite --suite evals/core.yaml
python -m pytest
```

The exact package pins include the Copilot backend and the source normalizers: `beautifulsoup4==4.13.4` and `PyMuPDF==1.26.7`. They prevent extraction-library drift from being silently mistaken for the pinned source text. These pins were installation-tested with Python 3.12; package/source hashes are recorded separately from interpreter version. The wheel installs the library/CLI; this workflow runs the scripts and suites from the matching source checkout.

## Fetch exact sources

Keep raw documents and full captures **outside the public checkout**. Every destination below must be new. Replace the example User-Agent with your own name/contact.

```sh
python benchmark/fetch_sources.py --out ../financial-sources-new \
  --user-agent 'Your Name your-contact@example.com'
```

The helper fetches the 16 manifest URLs, checks raw SHA-256 hashes, normalizes each document, and checks the pinned normalized-text hash. The runner additionally verifies 40 evidence-context hashes. SEC access restrictions, changed issuer markup, unavailable source URLs, or changed extraction output can block byte-identical reproduction. A failed check stops the workflow. Obtain the pinned snapshot or deliberately curate/version a replacement suite; never substitute reference answers or silently accept a changed file.

## Start a fresh local API

In terminal 1, from this harness directory with the virtual environment active:

```sh
python benchmark/serve.py --copilot ../ai-equity-research-copilot \
  --state ../financial-api-state-new --port 8766
```

The server binds loopback, explicitly sets `AIERC_LLM_PROVIDER=local`, removes provider-key environment variables in its own process, and creates an unseeded measured corpus. The import-time demo app is isolated in a separate directory. Leave this process running while using terminal 2. Stop it with Ctrl-C after capture.

In terminal 2, activate the same environment and run from this harness directory:

```sh
. .venv/bin/activate
python benchmark/ingest.py --copilot ../ai-equity-research-copilot \
  --sources ../financial-sources-new --work ../financial-ingestion-new
python benchmark/run.py --copilot ../ai-equity-research-copilot \
  --sources ../financial-sources-new \
  --ingestion ../financial-ingestion-new/equity-ingestion.json \
  --out ../financial-capture-new
```

Ingestion hash-checks all raw inputs and prepares production extraction before touching the API. The API must contain zero companies/documents/chunks. Complete HTML passes through Copilot `sec.html_to_text`; PDFs use its production parser. The upload responses establish `document-map.json`. The runner validates this map against the suite/corpus, checks production-derived text hashes, and gives the baseline those same complete extracted texts.

The only target request fields are `question`, `company_ids`, and `top_k`. Expected documents, answers, evidence facts, rubrics, case IDs and fixtures remain in the evaluator. Response document UUIDs are mapped to canonical IDs only for scoring; original raw responses keep their UUIDs. Missing/unmapped citations score as unverifiable, and an explicit wrong ID cannot borrow identity from a label or excerpt.

Requests run once per case in suite order: API then baseline, no retries or warmup. Every successful API response must report the declared local provider/model. A different provider aborts the run. Error responses remain execution failures, including in the public export. This validates the response contract; it does not remotely authenticate arbitrary servers. Use the supplied local server and inspect its recorded `serve-config.json` when assessing settings provenance.

## Export without target prose

```sh
python benchmark/export_public.py --captured ../financial-capture-new \
  --out ../financial-public-new
```

This verifies the fixed workflow artifact inventory, validates all retained non-gold inputs and their cross-file links before writing export files, creates a new staging/export directory, and leaves original captures unchanged. Target answers, excerpts, labels/titles, limitations, key points, debug passages and error messages are omitted or fingerprinted. Reviewed citation identifiers, typed numeric metadata, curated evaluator definitions, provenance, metrics and gates remain. Configuration, metadata, sources, captures, results and ingestion reject unclassified fields, arbitrary prose in typed fields and inconsistent recorded identities/counts/hashes/timings. Source URLs must be credential-free HTTPS URLs with named hosts and no port, query or fragment; all pinned URLs meet this policy. Saved metrics/gates are checked for consistency, not recomputed from answer text. A local-path scan is a secondary check, not a substitute for field classification. Keep free-form review notes outside the capture directory: unreviewed artifact filenames stop export. Raw filings are never copied. See the method for the trusted curated-gold and identifier/URL-policy limits.

Public metrics/gates remain identical to their measured originals. `ORIGINAL_SHA256SUMS.json` commits to originals; `SHA256SUMS.json` describes public files. Those inventories intentionally differ. Public redacted records cannot independently rescore removed answer text. Keep originals locally or make a new capture for rescoring, with a separate rescoring timestamp and preserved execution timestamps.

## Use the generic CLI map

For an already ingested corpus, supply the recorded map explicitly:

```sh
fin-eval run --suite evals/core.yaml --target copilot-api \
  --base-url http://127.0.0.1:8766 \
  --document-map ../financial-ingestion-new/document-map.json \
  --out runs/current-mapped
```

The CLI accepts either the plain JSON map or the ingestion manifest. It rejects duplicate keys, duplicate canonical assignments, blanks, unknown canonical IDs and missing selected-case document coverage before any target request. A map records caller-supplied identity consistency, not independent source authenticity. `ingest.py` supplies stronger provenance by recording actual upload IDs after raw-source verification. The CLI records map file/content hashes; the benchmark records the upload-map and ingestion hashes.
