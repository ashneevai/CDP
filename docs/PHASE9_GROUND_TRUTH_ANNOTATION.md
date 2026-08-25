# Phase 9 independent ground-truth annotation

The annotation team must not receive CDP predictions, confidence scores, routes,
or performance-harness output. Pages are referenced only by hashed `document_id`
and `package_id` from the frozen private manifest.

Annotators record document type, package membership, canonical fields, critical
flags, and claim/package validity when applicable. Critical fields require two
independent values under `annotator_a` and `annotator_b`. Any disagreement requires
an independent adjudicator, an `adjudicated_value`, and `adjudicator_id`. Blank and
not-applicable are distinct values and may not be inferred from model output.

Before scoring, run `python -m evaluation.ground_truth_workstream --manifest ...
--annotations ...`. Incomplete coverage, duplicate identities, package mismatch,
invalid document types, missing dual annotations, and unresolved disagreements all
fail closed. Truth files remain private and physically separate from predictions.
