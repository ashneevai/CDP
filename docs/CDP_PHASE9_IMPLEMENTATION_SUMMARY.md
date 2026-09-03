# CDP Phase 9 — Evidence-Driven STP Modernization

## Objective
Increase safe claim STP by expanding independent evidence, improving routing/assembly, introducing vendor-neutral extraction and promotion governance, and reducing correct-but-reviewed HITL without weakening false-accept protections.

## Safety invariants
- `EvidenceDecisionService` remains the sole machine field-disposition authority.
- `ClaimDecisionService` remains the sole claim STP/block authority.
- OCR, classifiers, candidate fusion, VLMs, document assembly and external providers produce observations/candidates/context only.
- No threshold is relaxed merely to increase STP.
- Critical false accepts remain a hard promotion blocker.
- Raw PHI-bearing reference-corpus data and predictions remain outside Git.

## Implemented waves

### Wave 1 — Runtime reproducibility and frozen-corpus governance
Implemented:
- immutable `RuntimeManifest`
- deterministic manifest identity and parity checks
- frozen corpus manifest and image-quality profile
- prediction-before-truth freeze
- PHI-safe external-corpus runner
- aggregate routing/extraction/decision/latency/cost reporting

### Wave 2 — Independent evidence expansion
Implemented:
- independent-evidence contracts and service
- evidence-lineage de-duplication
- deterministic NPI/date/amount evidence foundations
- adapters for decision-context enrichment
- shadow unlock analysis

### Wave 3 — Document classification and assembly V5 foundation
Implemented:
- hierarchical page/document-family contracts
- conservative ensemble classifier
- correlated-lineage vote de-duplication
- confidence + margin gates
- `SAFE_GENERIC` fail-closed routing
- conservative document assembly for uncertain/blank/duplicate/boundary pages

### Wave 4 — Candidate fusion
Implemented:
- candidate ranking across OCR/spatial/structural/reference/cross-field evidence
- independent-lineage de-duplication
- candidate-only semantics; no acceptance API

### Wave 5 — Vendor-neutral extraction provider layer
Implemented:
- `ExtractionProvider` protocol
- provider capability metadata
- local/cloud provider modes
- privacy, latency and cost-aware routing policy
- provider candidate provenance, model version, cost and latency fields

### Wave 6 — HITL exception model
Implemented:
- exception-oriented field/claim contracts
- source crop/bbox support
- alternatives, available/missing evidence, reason codes and reference status

### Wave 7 — Active learning and model governance
Implemented:
- HITL correction taxonomy
- training-candidate contracts
- model/policy artifact lifecycle: EXPERIMENTAL, CANDIDATE, SHADOW, PRODUCTION, RETIRED
- production artifact registry foundation

### Wave 8 — Confidence calibration
Implemented:
- calibration bins
- Expected Calibration Error
- Brier score
- outcome-based confidence evaluation foundation

### Wave 9 — Production promotion gates
Implemented fail-closed gates for:
- accepted precision
- critical-field precision
- critical false accepts
- P95 latency
- cost/page
- minimum evaluation sample size

### Wave 10 — Regression coverage
Implemented tests for:
- runtime-manifest parity
- frozen-corpus identity
- prediction freeze
- PHI-safe aggregate reporting
- independent evidence
- document assembly
- classification fail-closed behavior
- candidate-lineage de-duplication
- privacy-aware provider routing
- critical false-accept promotion blocking
- absence of field-decision APIs from classifier/fusion components

## External reference corpus
Frozen external corpus:
- 1,000 TIFF pages
- 110 claim packages
- four source groups
- corpus identity is enforced by SHA-256 in the repository manifest

The raw ZIP and raw predictions are intentionally not stored in Git because they may contain PHI/PII.

## Activation policy
The new classification, candidate-fusion, evidence-expansion and provider-routing capabilities are foundations/shadow capabilities until benchmark truth is independently available and promotion gates pass.

Do not enable new machine acceptance solely from truth-blind prediction runs.

## Required measured promotion sequence
1. Run the canonical production-equivalent runtime against the frozen corpus.
2. Freeze raw predictions before truth creation or access.
3. Create/ingest independent truth.
4. Score routing, extraction, critical fields, accepted precision, false accepts, HITL, claim STP, latency and cost.
5. Run shadow evidence-unlock analysis.
6. Promote only field/document cohorts that pass all safety gates.
7. Record the exact `RuntimeManifest` and artifact versions used for every promoted cohort.

## Known external gates
The following cannot be honestly claimed complete without external evidence or environment-specific integrations:
- true accuracy/false-accept/STP metrics for the 1,000-page corpus because the uploaded package contains no independent truth labels
- vendor-specific Google/Azure/AWS/ABBYY execution because credentials/contracts are environment-specific
- production promotion of new evidence policies before benchmark scoring

These are release gates, not reasons to weaken policy or fabricate results.
