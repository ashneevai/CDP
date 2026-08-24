# Phase 10 — CMS Specialist Extraction and Stratified Pilot

## Objective

Improve high-value CMS claim fields without weakening the production decision policy. The phase is driven by the diagnostic baseline where full-page OCR underperformed on diagnosis, DOB, and procedure fields.

## Implemented together

1. CMS specialist field policies for member ID, patient name, DOB, NPI, tax ID, diagnosis, procedure, service date, charge, and account number.
2. Deterministic field normalizers and validators, including NPI Luhn validation, ICD-style diagnosis validation, CPT/HCPCS-style procedure validation, date validation, and amount normalization.
3. Field-specific candidate fusion using source preference, confidence, spatial support, deterministic validation, agreement, and independent lineage.
4. Handwriting uncertainty assessment that can request a stronger handwriting/multimodal candidate provider but cannot accept a field.
5. Shadow enrichment attached to the external production-equivalent inference adapter. Shadow analysis never modifies canonical values or dispositions.
6. Deterministic, prediction-blind and truth-blind 50-page stratified pilot selection: Group A 25, Group B 10, Group C 8, Group D 7.
7. Unit tests proving specialist modules have no acceptance/disposition authority and pilot selection is reproducible.

## Safety architecture

The phase preserves these invariants:

- OCR and specialist components produce candidates/evidence only.
- Candidate fusion proposes a candidate only.
- Handwriting fallback requests another candidate provider only.
- `EvidenceDecisionService` remains the sole final field decision authority.
- The external qualification run remains immutable after prediction freeze.
- The pilot is selected without using predictions or truth.

## Activation sequence

1. Run the existing fixed 5-page CMS diagnostic set through the external qualification adapter.
2. Compare canonical CDP output and `specialist_shadow` output against independent truth.
3. Run the deterministic 50-page stratified pilot.
4. Score overall field accuracy, critical-field accuracy, accepted precision, critical false accepts, field HITL, claim STP, P50/P95 latency, and fully loaded cost.
5. Promote individual specialist policies only when they improve accuracy/coverage without violating accepted-precision and zero-critical-false-accept gates.
6. Freeze the resulting candidate runtime and proceed to the full 1,000-page qualification.

## Target gates

- Overall validated field accuracy: >= 98%
- Critical accepted precision: >= 99.5%
- Critical false accepts: 0
- Initial claim STP: >= 80%
- Field HITL: <= 15%
- P95 latency: <= 5 seconds/page
- Fully loaded cost: <= $0.03/page

The 69.2% diagnostic Tesseract baseline is not a CDP production score; it is only a sample-difficulty benchmark to beat.
