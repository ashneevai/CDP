# CDP 1,000-Page Promotion Runbook

This runbook defines the controlled benchmark sequence for the frozen external
`Hackathon - 1000 Claims.zip` corpus.

## Safety rules

- Do not commit the raw ZIP, raw predictions, truth JSONL, or source filenames.
- Do not inspect or create independent truth before predictions are frozen.
- Do not change runtime thresholds, models, routing, or policies after freeze.
- Do not report accuracy until prediction/truth coverage is complete.
- Critical false accepts are a hard promotion blocker.

## 1. Freeze the candidate runtime

Create an explicit production-equivalent `RuntimeManifest` JSON and record the
candidate Git commit SHA. Evaluation and production must use the same manifest.

## 2. Run the frozen 1,000-page corpus

Run the external corpus runner in a private working directory:

```bash
python -m evaluation.external_corpus_runner \
  --corpus-zip "/secure/Hackathon - 1000 Claims.zip" \
  --corpus-manifest evaluation/corpora/hackathon_1000_claims_v1.json \
  --runtime-manifest /secure/runtime_manifest.json \
  --output /secure/cdp_1000_run
```

Expected private outputs:

- `raw_predictions.jsonl`
- `prediction_freeze.json`
- `aggregate_report.json`

`raw_predictions.jsonl` may contain PHI and is never Git-safe.

## 3. Verify the freeze

Before truth is revealed, archive:

- corpus SHA-256
- runtime manifest ID
- prediction SHA-256
- Git commit SHA
- prediction record count

Do not overwrite an existing freeze artifact.

## 4. Create or reveal independent truth

Truth must use hashed `document_id` and `package_id` values from the frozen run.
The truth JSONL record shape is:

```json
{
  "document_id": "hashed-page-id",
  "package_id": "hashed-package-id",
  "document_type": "CMS_1500",
  "fields": {
    "member_id": {"value": "...", "critical": true},
    "provider_npi": {"value": "...", "critical": true}
  }
}
```

Truth values are private and must not be committed.

## 5. Calculate fully loaded cost

Use measured CPU/GPU seconds and explicit hourly rates. Never invent rates.
`evaluation.cost_model` supports CPU, GPU, cloud/API, and fixed run cost.

Record the resulting total run cost for the scorer.

## 6. Score the frozen predictions

```bash
python -m evaluation.run_promotion_score \
  --predictions /secure/cdp_1000_run/raw_predictions.jsonl \
  --freeze /secure/cdp_1000_run/prediction_freeze.json \
  --truth /secure/cdp_1000_truth.jsonl \
  --output /secure/cdp_1000_promotion_report.json \
  --fully-loaded-cost-usd <MEASURED_TOTAL_RUN_COST>
```

The scorer fails closed when prediction identity or truth coverage does not
match the pre-truth freeze.

## 7. Required benchmark metrics

The promotion report contains:

- overall field accuracy
- critical-field accuracy
- accepted precision
- critical accepted precision
- critical false accepts
- routing accuracy when document-type truth is present
- field HITL rate
- claim HITL rate
- true claim STP rate
- mean/P50/P95/P99 page latency
- cloud/API cost
- fully loaded cost and cost/page
- source-group field accuracy
- failure taxonomy

## 8. Promotion gates

Default Phase 9 gates are:

- accepted precision >= 99.5%
- critical accepted precision >= 99.5%
- critical false accepts = 0
- overall field accuracy >= 98%
- critical-field accuracy >= 99.5%
- claim STP >= 80%
- field HITL <= 15%
- claim HITL <= 20%
- P95 latency <= 5 seconds/page
- fully loaded cost <= $0.03/page
- minimum sample size >= 500 pages

A failed gate means `REJECT` or `NEEDS_MORE_DATA`; it must not be bypassed by
lowering thresholds on this frozen benchmark.

## 9. Diagnose failures before changing code

Rank failure cohorts by business impact. At minimum inspect:

1. routing/document-type errors
2. zero-field or missing extraction pages
3. wrong field values
4. critical false accepts
5. correct-but-reviewed fields
6. high-latency pages
7. high-cost provider/fallback paths
8. source-group regressions

Optimize the highest-value cohort first, then evaluate the change on a new
holdout or an approved development set. Do not tune repeatedly on the frozen
1,000-page benchmark.
