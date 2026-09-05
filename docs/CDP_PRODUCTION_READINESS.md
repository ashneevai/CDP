# CDP production readiness

Status date: 2026-08-22. Decision: **BLOCKED — DO NOT PROMOTE**.

## Verified in this workspace

- The complete unit suite passes: 632 tests.
- The React/TypeScript HITL production build succeeds.
- Local-first OCR, adaptive registration, reference and deterministic evidence, fail-closed reconciliation, safe-STP policy, selective AI gateway controls, field-level HITL, versioned observability, Kubernetes/KEDA definitions, and scalability preflight are implemented.
- Current development evidence contains zero measured false accepts; no threshold was weakened to increase STP.
- Runtime/evaluation convergence across validation, retry, evaluation, and output passes the complete 647-test unit suite plus the focused parity integration test. OCR candidates persist across the database boundary and retry evidence is append-only.

## Required before promotion

1. Freeze and run a separately sourced untouched holdout with independent truth.
2. Add a broker/database/object-store integration run spanning extraction through output; decision-adapter parity is implemented, but full infrastructure orchestration remains unproven.
3. Run incremental candidate experiments and demonstrate `<30%` HITL without critical-false-accept or critical-accuracy regression.
4. Qualify live OCR/cloud providers under approved PHI, region, cost, timeout, and audit policies.
5. Pass representative 1k/10k/50k-page cluster, 10x burst, soak, backpressure, autoscaling, and dependency-failure tests.
6. Complete security, identity, network-egress, retention, backup/restore, rollback, and disaster-recovery reviews and drills.
7. Obtain named operations, security, compliance, data-governance, and release approvals.

Implementation readiness is materially ahead of evidence readiness. Production promotion remains blocked because external data, infrastructure, and organizational approvals are unavailable in this workspace—not because a safety gate was bypassed.

## Machine-readable external-evidence preflight

CI now runs `scripts/production_evidence_preflight.py` against
`config/production_evidence_requirements.yaml`. Sensitive evidence remains outside Git. Each
required file must be mounted and its independently approved SHA-256 bound in the configuration;
missing, unbound, changed, or path-escaping artifacts fail closed and produce a PHI-free JSON
blocker report. This does not convert absent evidence into a skipped or passing product test.

The production policy also enforces the agreed ceilings directly: P95 latency must be at most
5 seconds and measured fully loaded cost must be at most $0.03 per document. Recording a cost
without meeting the ceiling is not a passing production gate.

Production promotion additionally requires five distinct, named approvals—operations, security,
compliance, data governance, and release. Every approval must carry a verified signature and bind
to the exact 40-character release commit and the approved evidence-bundle SHA-256. Missing,
reused, unverified, or stale approvals can permit shadow qualification but never production.
