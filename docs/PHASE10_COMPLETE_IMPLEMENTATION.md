# Phase 10 — Complete Specialist Accuracy and Qualification Program

## Objective

Raise CDP extraction quality and safe automation without weakening decision policy. The program is intentionally fail-closed: specialist logic may improve candidates and evidence, but production acceptance continues to be controlled by the canonical EvidenceDecisionService / ClaimDecisionService path.

## Implemented phases

### 1. Frozen external qualification
- 1,000-page corpus manifest and SHA validation
- prediction-before-truth freeze
- runtime/corpus/prediction identity binding
- truth contract and promotion scorer
- PHI-safe aggregate reporting

### 2. CMS specialist field policies
Priority fields:
- member ID
- patient name
- DOB
- provider NPI
- federal tax ID
- diagnosis
- procedure / CPT / HCPCS
- service date
- total charge
- account number

Specialist validation adds field-specific normalization and deterministic validation. It does not issue field dispositions.

### 3. Candidate fusion
Candidate fusion considers:
- OCR confidence
- source priority
- spatial/ownership confidence
- agreement
- deterministic validity
- independent lineage

The fusion layer proposes candidates only. It has no autonomous acceptance authority.

### 4. Handwriting escalation
A dedicated uncertainty policy identifies regions where handwriting/poor recovery/low confidence/disagreement justify escalation to a handwriting or multimodal candidate provider. Escalated output remains candidate evidence only.

### 5. Shadow specialist orchestration
Canonical production-representative inference remains unchanged. Specialist outputs attach under `specialist_shadow` so baseline vs specialist can be compared without modifying canonical decisions.

### 6. Deterministic 50-page pilot
The pilot uses a fixed truth-blind stratified allocation:
- Group A: 25
- Group B: 10
- Group C: 8
- Group D: 7

Selection prioritizes package diversity and uses a fixed seed.

### 7. Specialist shadow scorer
Measures:
- canonical baseline field accuracy
- specialist field accuracy
- improved fields
- regressed fields
- per-field baseline/specialist accuracy

No production behavior is changed by this scorer.

### 8. Production activation gate
Specialist promotion is blocked unless all configured gates pass:
- sample size >= 50
- measurable accuracy gain >= 1 percentage point
- accepted precision >= 99.5%
- critical accepted precision >= 99.5%
- critical false accepts = 0
- field HITL <= 15%
- P95 <= 5 seconds/page
- fully loaded cost <= $0.03/page

### 9. 50-page pilot orchestrator
`evaluation.run_50_page_pilot` performs:
1. deterministic pilot selection
2. canonical CDP inference
3. immutable prediction freeze
4. optional independent-truth scoring
5. specialist shadow comparison
6. activation decision

### 10. Full 1,000-page promotion
After the pilot passes, the same frozen qualification framework is used on the full 1,000-page corpus. Truth must remain independent and unavailable until predictions are frozen.

## Safety invariants

- Specialist extraction never directly accepts fields.
- Handwriting/VLM output never directly accepts fields.
- Raw model confidence is not authorization.
- No threshold reduction may be used merely to increase STP.
- Critical false accepts remain a hard stop.
- Prediction/truth hash mismatch invalidates the benchmark.
- Raw external images, raw predictions and raw truth remain outside Git.
- Only aggregate PHI-safe benchmark reports may be committed.

## Promotion sequence

1. Run fixed CMS mini-set for diagnosis.
2. Run 50-page stratified pilot in shadow.
3. Obtain independent truth after prediction freeze.
4. Compare baseline vs specialist.
5. Promote only field-specific changes that pass the activation gate.
6. Freeze the promoted runtime manifest.
7. Run the full 1,000-page qualification.
8. Produce PROMOTE / REJECT / NEEDS_MORE_DATA.
9. If rejected, preserve the result and evaluate the next version on a new untouched holdout.

## Definition of done

Phase 10 code is complete when all implementation and safety tests are green. Production qualification is complete only after the real corpus is executed with independent truth and the promotion gates pass. Architecture/code completion must not be confused with measured production qualification.
