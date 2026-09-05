# Governed HITL reduction runbook

This workflow measures and reduces claim-level human review without treating OCR output,
model predictions, or unbound historical field records as truth.

## 1. Prepare and seal decisions

Build an input JSON that conforms to `HITLReductionInput`. Every field record must carry the
exact document, page, and crop identity. Every OCR candidate must carry matching page lineage
and a crop hash. Run:

```bash
idp-hitl-reduction prepare --input cohort.json --output hitl_run
```

The command uses the canonical runtime profile and emits:

- `sealed_predictions.json`: immutable runtime and evaluation-only decisions.
- `blind_review_queue.json`: crop tasks with no selected value, candidate, confidence, reason,
  route status, or blocker rank.
- `blocker_pareto.json`: internal priority list ranked first by claims unlocked by one field.
- `current_metrics.json`: claim HITL, field HITL, machine-recovery backlog, and segment metrics.

Send only the blind queue and its crop references to reviewers. Do not send predictions or the
blocker report.

Create isolated reviewer packs with at least three independent people whenever the queue contains
C2/C3 fields. Two people receive each critical task and a third remains eligible to adjudicate a
disagreement:

```bash
idp-hitl-reduction assign \
  --queue hitl_run/blind_review_queue.json \
  --reviewer reviewer-a \
  --reviewer reviewer-b \
  --reviewer adjudicator-c \
  --output hitl_assignments
```

Give each person only their numbered `reviewer_*.json` pack. Keep
`review_assignment_manifest.json` with the coordinator. Reviewer packs contain only allowlisted
blind-task fields and never include predictions, candidates, confidence, decision reasons,
routes, blocker ranks, co-reviewer identities, or eligible-adjudicator identities. Assignments are
deterministic and bound to the prediction seal by a separate assignment seal. Reviewer identities
are compared using Unicode-normalized, whitespace-normalized, case-insensitive values, so aliases
such as `Reviewer-A` and ` reviewer-a ` cannot satisfy independence.

Before distributing packs, verify the coordinator manifest has not changed:

```bash
idp-hitl-reduction verify-assignments \
  --manifest hitl_assignments/review_assignment_manifest.json \
  --output hitl_assignment_verification
```

## 2. Collect admissible truth

Each reviewer writes one `BlindReviewSubmission` JSON object per assigned task to a separate JSONL
file. Compile the isolated files without opening the sealed predictions:

```bash
idp-hitl-reduction compile-reviews \
  --queue hitl_run/blind_review_queue.json \
  --manifest hitl_assignments/review_assignment_manifest.json \
  --reviews reviewer_001_submissions.jsonl \
  --reviews reviewer_002_submissions.jsonl \
  --reviews reviewer_003_submissions.jsonl \
  --adjudications adjudications.jsonl \
  --output hitl_compiled
```

The command rejects unassigned reviewers, reviewer aliases, duplicate submissions, mismatched
prediction or assignment seals, and ineligible adjudicators. It emits `review_progress.json`, a
prediction-free `adjudication_queue.json`, and score-ready `governed_labels.jsonl`. When reviews
disagree, only a reviewer reserved outside that task's original pair may adjudicate.

Write one `GovernedFieldLabel` JSON object per line. Each label must bind to the prediction seal,
blind task, field, document, page SHA-256, and crop SHA-256.

Admissible authorities are:

- `SOURCE_SYSTEM_GROUND_TRUTH`: requires system, version, snapshot SHA-256, record ID, and explicit
  independent/non-circular assertions.
- `HUMAN_ADJUDICATED`: C2/C3 fields require two different reviewers. A disagreement requires a
  third independent adjudicator. Review timestamps must be after the prediction seal.

`UNREADABLE` and `NOT_APPLICABLE` records remain auditable but do not count as scored truth.

## 3. Score after labels are frozen

```bash
idp-hitl-reduction score \
  --sealed hitl_run/sealed_predictions.json \
  --labels governed_labels.jsonl \
  --output hitl_score
```

The scorer verifies the seal and rejects any page, crop, task, field, or cohort mismatch. It emits
accuracy and accepted-precision metrics separately for runtime and evaluation-only decisions,
then applies the existing production-readiness and route-promotion gates.

Cost gates require explicit measured `cost_per_document_usd` and per-route
`route_cost_per_call_usd` operational evidence. Default zero cost is never inferred from an
unmeasured local run.

Route output is evidence only. The workflow never edits `config/ocr_field_routes.yaml`, and an
evaluation-only route never gains runtime authority automatically.

## Operating rule

Work down `single_blocker_claim_unlocks` first, but add an automated evidence path only when a
frozen independent holdout preserves at least 99.5% accepted precision, produces zero critical
false accepts, and passes the existing claim-HITL and 95% upper-confidence gates. Handwriting and
uncorroborated ambiguous values remain in HITL.
