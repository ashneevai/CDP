# External 1,000-Page Production Qualification

This qualification is intentionally two-phase so ground truth cannot influence inference.
Raw images, raw predictions, and truth must remain outside Git.

## 1. Prepare the external corpus manifest

Create a JSONL manifest on the controlled execution host with exactly 1,000 records.
Each record must contain a stable document ID, absolute/local mounted path, and SHA-256.
Optional `group` and `package_id` values may be included.

```json
{"document_id":"DOC-0001","path":"/secure/corpus/page-0001.png","sha256":"...","group":"A","package_id":"CLAIM-0001"}
```

The runner independently hashes every source file and fails if any file is missing,
duplicated, changed, or if the manifest is not exactly 1,000 pages.

## 2. Run production inference and freeze predictions

The inference command must be the real CDP production inference entrypoint available in
the controlled runtime. It receives three placeholders:

- `{input}`: source image path
- `{document_id}`: stable external document ID
- `{output}`: per-page JSON output path

The command must write one JSON object to `{output}`. That object must contain the
production prediction shape consumed by `evaluation.promotion_scorer`.

Example invocation pattern:

```bash
python -m evaluation.external_qualification run \
  --manifest /secure/qualification/corpus_manifest.jsonl \
  --predictions /secure/qualification/predictions.jsonl \
  --freeze /secure/qualification/prediction_freeze.json \
  --runtime-manifest-id "$CDP_RUNTIME_MANIFEST_ID" \
  --corpus-id external-1000-v1 \
  --inference-command 'python -m <production-inference-entrypoint> --input {input} --document-id {document_id} --output {output}'
```

The run stops on the first failed page. A successful run creates an immutable identity
record containing the corpus hash, runtime manifest ID, prediction hash and page count.
The freeze explicitly records `truth_present=false`.

Do not reveal or mount truth to the inference process before this step completes.

## 3. Verify the freeze before truth handoff

```bash
python -m evaluation.external_qualification verify \
  --predictions /secure/qualification/predictions.jsonl \
  --freeze /secure/qualification/prediction_freeze.json
```

Any change to predictions after the freeze is a hard failure.

## 4. Independently create/reveal truth

Truth must be produced independently of the frozen predictions and follow
`evaluation.truth_contract`. Keep the raw truth JSONL outside Git.

Each truth record contains:

```json
{
  "document_id": "DOC-0001",
  "package_id": "CLAIM-0001",
  "document_type": "CMS1500",
  "fields": {
    "member_id": {"value": "...", "critical": true}
  }
}
```

The scorer requires exact document coverage between predictions and truth and rejects
duplicates or malformed truth records.

## 5. Score the frozen run

```bash
python -m evaluation.external_qualification score \
  --predictions /secure/qualification/predictions.jsonl \
  --freeze /secure/qualification/prediction_freeze.json \
  --truth /secure/qualification/truth.jsonl \
  --output /secure/qualification/aggregate_report.json \
  --fully-loaded-cost-usd <measured-total-cost>
```

Only the aggregate report is considered Git-safe. Never commit source images, raw
predictions, or raw truth.

The report includes overall/critical accuracy, accepted precision, critical false
accepts, routing accuracy, field and claim HITL, claim STP, latency, cost, failure
taxonomy, cohort metrics, and the promotion-gate decision.

## Qualification invariant

No independent truth means no accuracy result. Missing data must never be replaced by
synthetic truth, inferred labels, or model self-grading.
