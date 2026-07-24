# Source provenance and curation

`../core.yaml` is the 50-case **quality v2** suite. It keeps the original case IDs, while `../legacy_core_v1.yaml` preserves the original file byte for byte for historical rescoring. Changes to facts, questions, required citations and evidence make v1 and v2 different cohorts; their scores must not be gated against each other.

`manifest.yaml` pins 16 primary documents and 40 evidence entries, verified on September 6, 2026. SEC accession URLs identify the exact filing. Microsoft's annual report and earnings-call pages are hosted by Microsoft; NVIDIA's transcript is issuer-hosted on its investor-relations CDN, and JPMorgan's transcript is on its own website. NVIDIA's issuer-hosted transcript is a FactSet corrected transcript; its provenance is not an assertion that NVIDIA authored the transcription.

Each `excerpt` is a **short verbatim locator**, not a complete paragraph. The separate `facts` are curator-written paraphrases of the supporting passages, including relevant numerical facts. They are never represented as quotations. The source URL, section, fiscal period, raw-file SHA256, normalized-text SHA256 and a 1,200-character context SHA256 let a reviewer recover and verify the surrounding primary-source passage without copying whole copyrighted reports into this repository. The excerpt and fact entries are shared through YAML aliases when multiple cases use the same evidence. Source hashes are provenance checks, not proof that an arbitrary model answer is entailed by a filing.

The normalized context starts at `context_start_character`; offsets are zero based. HTML normalization uses BeautifulSoup's `html.parser`, removes script/style elements, joins visible text with spaces and collapses whitespace. PDF normalization uses PyMuPDF text extraction in page order, joins text with spaces and collapses whitespace. Extraction/library changes can alter hashes. Re-fetching an issuer website can also change a raw hash when navigation or markup changes; do not silently replace a pinned source or expected target when this happens. Verify the document and relevant passage, then deliberately version the suite.

The fiscal-year labels matter:

| Company | Fiscal 2025 annual period end | Fiscal 2026 Q1 period end |
|---|---|---|
| NVIDIA | 2025-01-26 | 2025-04-27 |
| Apple | 2025-09-27 | 2025-12-27 |
| Microsoft | 2025-06-30 | 2025-09-30 |
| JPMorgan | 2025-12-31 | 2026-03-31 |
| Exxon Mobil | 2025-12-31 | 2026-03-31 |
| Tesla | 2025-12-31 | 2026-03-31 |

All 43 answerable cases have factual targets and evidence from every required document. All seven refusal cases have empty factual target lists; their behavioral requirements belong in `judge_rubric`. Refusal is relative to the supplied pinned context, not a claim that the information does not exist anywhere. Broad forbidden substrings such as `$`, `%` and `guaranteed` were moved to contextual rubric instructions because their presence can be legitimate in a negation, quotation or factual explanation. The rubric distinguishes making an unsupported assertion from discussing it. A deterministic lexical score alone cannot fully evaluate these behavioral instructions.

## Deliberate source replacement

`aapl_10q_transcript_geography_bridge_033` retains its historical case ID but now compares the quarterly geographic sales disclosure with Tim Cook's demand statement in Apple's **earnings release**, furnished as Exhibit 99.1 to its January 29, 2026 8-K. Apple's official release said its call replay would be available for approximately two weeks; no official archived transcript was verified during curation. The question, document ID, citations and tag were changed to identify the actual release. This is a disclosed case-definition change, never a release mislabeled as a transcript. Apple's total regional sales and Cook's regional iPhone records are different measures and should be described as such.

## Offline fixtures

- `../fixtures/core_factual.json` contains 50 curated reference answers, including every required document citation. It tests scorer behavior and file plumbing. It is **not an independent model run**: reference wording was curated from these expected facts.
- `../fixtures/contrasts.json` contrasts a correct paraphrase, a rubric echo, a compliant refusal, a missing second document and a reversed factual direction. Expected outcomes are test assertions, outside the target response payload.
- `../plumbing.yaml` is a separate small synthetic smoke suite. Its invented company and 10-unit revenue are explicitly synthetic and do not enter the financial quality suite.

Source metadata is evaluation provenance. It must not be passed to a target as an answer key while advertising the result as independent quality. Supply the intended source corpus to the target, retain these curated targets for scoring, and report any source-loading or execution failure separately from model behavior.
