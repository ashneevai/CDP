# Strict claim-form identity gate

Status: implemented by policy `strict-form-identity-v1`.

## Architecture

The authoritative route is now:

`OCR signals -> claim/nonclaim screening -> canonical form identity -> safe processing route`

Form-specific CMS-1500 or UB-04 localization is authorized only when the router records `CONFIRMED` identity and the standard-form verifier independently returns `VERIFIED`. The processing-route resolver is the downstream firewall and also requires the classification subtype to match the verified family.

Layout similarity is supporting evidence, not proof of canonical form identity.

A canonical identity requires either a strong family identity anchor combined with family-specific text and structural evidence, or the configured complete field-topology signature with qualifying anchor geometry and structure. Opposing-family identity or explicit noncanonical wording vetoes canonical eligibility.

Structured healthcare claims that lack canonical identity route as `OTHER_CLAIM_FORM` to the generic structured/review path. Insufficiently identified pages remain `UNKNOWN`; neither route may invoke CMS/UB fixed extraction. Diagnostics expose identity state, matched and missing anchors, conflicts, geometry, topology, localization authorization, and reason codes without logging OCR text.

## Historical results

Historical evaluation artifacts remain immutable. Results produced by router policy `3.0`, including earlier 140-page cohorts, are retained for audit but are marked conceptually as `SUPERSEDED_BY_STRICT_FORM_IDENTITY_GATE` for canonical-form classification claims. They must not be compared as if they used the new identity policy.

## Safety and provenance

Regression inputs are synthetic text/layout fixtures derived from observed false-positive categories. They contain no source documents or PHI. No LLM participates in form identity, and the existing numeric routing thresholds were not lowered.