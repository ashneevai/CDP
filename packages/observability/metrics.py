"""Prometheus metrics -- the exact names from the platform spec, defined
once here so every app/worker imports the same `CollectorRegistry` instead
of each inventing its own metric names. `packages/security/redaction.py`
is what keeps PHI out of these (labels are field *names*/document IDs/
route names, never field *values*).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

documents_received_total = Counter(
    "documents_received_total",
    "Documents accepted by the ingestion API",
    ["tenant_id", "detected_format"],
    registry=REGISTRY,
)

pages_processed_total = Counter(
    "pages_processed_total",
    "Pages decoded and preprocessed",
    ["bundle_type"],
    registry=REGISTRY,
)

attachments_skipped_total = Counter(
    "attachments_skipped_total",
    "Bundle B/D pages classified as attachments and preserved without extraction",
    registry=REGISTRY,
)

cache_hits_total = Counter(
    "cache_hits_total",
    "Idempotent re-ingestions served from an existing Document/Claim",
    registry=REGISTRY,
)

ocr_latency_seconds = Histogram(
    "ocr_latency_seconds",
    "Regional OCR call latency",
    ["extraction_method"],
    registry=REGISTRY,
)

classification_latency_seconds = Histogram(
    "classification_latency_seconds",
    "Page routing/classification latency",
    ["method"],
    registry=REGISTRY,
)

validation_failure_total = Counter(
    "validation_failure_total",
    "Deterministic validation failures",
    ["rule_name", "criticality"],
    registry=REGISTRY,
)

retry_total = Counter(
    "retry_total",
    "Alternate-preprocessing OCR retries",
    ["improved"],
    registry=REGISTRY,
)

preprocessing_strategy_cpu_seconds = Histogram(
    "preprocessing_strategy_cpu_seconds",
    "CPU time consumed by a bounded preprocessing retry strategy",
    ["strategy", "outcome"],
    registry=REGISTRY,
)

vlm_invocation_total = Counter(
    "vlm_invocation_total",
    "VLM fallback calls",
    ["insufficient_evidence"],
    registry=REGISTRY,
)

human_review_total = Counter(
    "human_review_total",
    "Fields routed to human review",
    ["reason"],
    registry=REGISTRY,
)

straight_through_rate = Gauge(
    "straight_through_rate",
    "Fraction of claims completed with no human review, over the current window",
    registry=REGISTRY,
)

estimated_cost_usd_total = Counter(
    "estimated_cost_usd_total",
    "Estimated inference/processing cost",
    ["extraction_method"],
    registry=REGISTRY,
)

processing_errors_total = Counter(
    "processing_errors_total",
    "Unhandled errors per worker/stage",
    ["worker", "stage"],
    registry=REGISTRY,
)

kafka_consumer_lag = Gauge(
    "kafka_consumer_lag",
    "Consumer group lag per topic (source: broker admin API, updated by a poller)",
    ["topic", "consumer_group"],
    registry=REGISTRY,
)

field_reconciliation_total = Counter(
    "field_reconciliation_total",
    "Field reconciliation decisions without field values",
    ["field_name", "criticality", "decision"],
    registry=REGISTRY,
)

critical_false_accepts_total = Counter(
    "critical_false_accepts_total",
    "Confirmed critical-field false acceptances; release-blocking when nonzero",
    ["field_name", "model_version"],
    registry=REGISTRY,
)

ai_gateway_requests_total = Counter(
    "ai_gateway_requests_total",
    "External AI gateway requests",
    ["provider", "model", "outcome"],
    registry=REGISTRY,
)

ai_gateway_cost_usd_total = Counter(
    "ai_gateway_cost_usd_total",
    "Actual external AI cost reported by the gateway",
    ["tenant_id", "provider", "model"],
    registry=REGISTRY,
)

ai_gateway_tokens_total = Counter(
    "ai_gateway_tokens_total",
    "External AI tokens by direction",
    ["provider", "model", "direction"],
    registry=REGISTRY,
)

registration_confidence = Histogram(
    "registration_confidence",
    "Accepted and rejected page registration confidence",
    ["algorithm", "accepted"],
    registry=REGISTRY,
)

image_quality_score = Histogram(
    "image_quality_score",
    "Page image quality score distribution",
    ["document_family"],
    registry=REGISTRY,
)

human_review_queue_depth = Gauge(
    "human_review_queue_depth",
    "Review tasks by workflow status",
    ["status", "criticality"],
    registry=REGISTRY,
)

human_review_turnaround_seconds = Histogram(
    "human_review_turnaround_seconds",
    "Time from review task creation to terminal decision",
    ["criticality", "decision"],
    registry=REGISTRY,
)

database_connections = Gauge(
    "database_connections", "Open database connections by state", ["state"], registry=REGISTRY
)
postgres_transactions_total = Counter(
    "postgres_transactions_total",
    "PostgreSQL transactions by outcome",
    ["outcome"],
    registry=REGISTRY,
)
redis_cache_operations_total = Counter(
    "redis_cache_operations_total",
    "Redis cache operations by hit or miss",
    ["result"],
    registry=REGISTRY,
)
object_store_bytes_total = Counter(
    "object_store_bytes_total",
    "S3-compatible object bytes transferred",
    ["operation"],
    registry=REGISTRY,
)
worker_resource_utilization = Gauge(
    "worker_resource_utilization",
    "Worker resource utilization ratio",
    ["worker", "resource"],
    registry=REGISTRY,
)

# Phase 4 production-readiness SLOs. Labels are allow-listed operational
# dimensions; candidate values, document IDs, claim IDs, and patient data are
# deliberately absent.
cdp_field_safe_coverage = Gauge(
    "cdp_field_safe_coverage",
    "Safely auto-resolved field fraction",
    registry=REGISTRY,
)
cdp_raw_accuracy = Gauge(
    "cdp_raw_accuracy",
    "Truth-qualified raw field accuracy",
    registry=REGISTRY,
)
cdp_critical_accuracy = Gauge(
    "cdp_critical_accuracy",
    "Truth-qualified C2/C3 field accuracy",
    registry=REGISTRY,
)
cdp_field_hitl_rate = Gauge(
    "cdp_field_hitl_rate",
    "Field fraction requiring human review",
    registry=REGISTRY,
)
cdp_claim_stp_rate = Gauge(
    "cdp_claim_stp_rate",
    "Claim straight-through-processing fraction",
    registry=REGISTRY,
)
cdp_claim_hitl_rate = Gauge(
    "cdp_claim_hitl_rate",
    "Claim fraction requiring human review",
    registry=REGISTRY,
)
cdp_false_accept_total = Counter(
    "cdp_false_accept_total",
    "Confirmed false field acceptances",
    ["field_name", "criticality"],
    registry=REGISTRY,
)
cdp_critical_false_accept_total = Counter(
    "cdp_critical_false_accept_total",
    "Confirmed C2/C3 false acceptances",
    ["field_name", "criticality"],
    registry=REGISTRY,
)
cdp_route_invocation_total = Counter(
    "cdp_route_invocation_total",
    "Governed OCR route invocations",
    ["route_id", "route_status", "outcome"],
    registry=REGISTRY,
)
cdp_route_shadow_total = Counter(
    "cdp_route_shadow_total",
    "Shadow route observations",
    ["route_id", "outcome"],
    registry=REGISTRY,
)
cdp_route_agreement_total = Counter(
    "cdp_route_agreement_total",
    "Production/shadow agreement observations",
    ["route_id", "agreement"],
    registry=REGISTRY,
)
cdp_route_false_agreement_total = Counter(
    "cdp_route_false_agreement_total",
    "Truth-confirmed false route agreements",
    ["route_id", "criticality"],
    registry=REGISTRY,
)
cdp_router_ml_inference_total = Counter(
    "cdp_router_ml_inference_total",
    "ML eligibility inference attempts",
    ["model_version", "outcome"],
    registry=REGISTRY,
)
cdp_router_ml_inference_latency_seconds = Histogram(
    "cdp_router_ml_inference_latency_seconds",
    "ML eligibility inference latency",
    ["model_version"],
    registry=REGISTRY,
)
cdp_router_ml_proposed_eligibility_total = Counter(
    "cdp_router_ml_proposed_eligibility_total",
    "ML-proposed eligibility by safe family",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_router_ml_fused_eligibility_total = Counter(
    "cdp_router_ml_fused_eligibility_total",
    "Fused eligibility by safe family",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_router_ml_false_eligibility_total = Counter(
    "cdp_router_ml_false_eligibility_total",
    "Truth-confirmed false ML eligibility",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_router_ml_model_version_info = Gauge(
    "cdp_router_ml_model_version_info",
    "Loaded ML eligibility model version",
    ["model_version", "feature_version"],
    registry=REGISTRY,
)
cdp_visual_route_prediction_total = Counter(
    "cdp_visual_route_prediction_total",
    "Visual evidence predictions",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_visual_route_latency_seconds = Histogram(
    "cdp_visual_route_latency_seconds",
    "Visual evidence inference latency",
    ["model_version"],
    registry=REGISTRY,
)
cdp_visual_standard_proposal_total = Counter(
    "cdp_visual_standard_proposal_total",
    "Visual standard-family proposals",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_visual_standard_veto_total = Counter(
    "cdp_visual_standard_veto_total",
    "Visual standard proposals vetoed by existing evidence",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_visual_standard_contradiction_total = Counter(
    "cdp_visual_standard_contradiction_total",
    "Contradiction classes observed for visual proposals",
    ["family", "contradiction_class", "model_version"],
    registry=REGISTRY,
)
cdp_visual_standard_ambiguity_total = Counter(
    "cdp_visual_standard_ambiguity_total",
    "Visual standard proposals made ambiguous",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_visual_false_standard_total = Counter(
    "cdp_visual_false_standard_total",
    "Truth-confirmed false visual standard proposals",
    ["family", "model_version"],
    registry=REGISTRY,
)
cdp_visual_model_version_info = Gauge(
    "cdp_visual_model_version_info",
    "Loaded visual evidence model version",
    ["model_version", "feature_version"],
    registry=REGISTRY,
)
cdp_document_taxonomy_total = Counter(
    "cdp_document_taxonomy_total",
    "Canonical document taxonomy outcomes",
    ["top_level_class", "document_family", "taxonomy_version"],
    registry=REGISTRY,
)
cdp_processing_route_total = Counter(
    "cdp_processing_route_total",
    "Canonical processing route outcomes",
    ["processing_route", "policy_version"],
    registry=REGISTRY,
)
cdp_standard_nomination_total = Counter(
    "cdp_standard_nomination_total",
    "Standard-family nominations",
    ["family", "classifier_version"],
    registry=REGISTRY,
)
cdp_standard_verification_total = Counter(
    "cdp_standard_verification_total",
    "Standard-form verification outcomes",
    ["family", "status", "policy_version"],
    registry=REGISTRY,
)
cdp_standard_verification_failure_total = Counter(
    "cdp_standard_verification_failure_total",
    "Failed or ambiguous standard verification",
    ["family", "status", "reason_code"],
    registry=REGISTRY,
)
cdp_false_standard_authorization_total = Counter(
    "cdp_false_standard_authorization_total",
    "Truth-confirmed non-standard fixed-extractor authorization",
    ["authorized_family", "taxonomy_version"],
    registry=REGISTRY,
)
cdp_safe_standard_fallback_total = Counter(
    "cdp_safe_standard_fallback_total",
    "Correct standard nominations safely sent to layout extraction",
    ["family", "verification_status", "reason_code"],
    registry=REGISTRY,
)
cdp_processing_route_accuracy = Gauge(
    "cdp_processing_route_accuracy",
    "Processing-route accuracy in governed evaluation",
    ["partition", "source_family"],
    registry=REGISTRY,
)
cdp_routing_abstention_total = Counter(
    "cdp_routing_abstention_total",
    "Safe routing abstentions",
    ["stage", "reason_code"],
    registry=REGISTRY,
)
cdp_routing_latency_seconds = Histogram(
    "cdp_routing_latency_seconds",
    "Canonical hierarchical routing latency",
    ["stage", "context"],
    registry=REGISTRY,
)
cdp_corpus_intake_total = Counter(
    "cdp_corpus_intake_total",
    "Governed corpus assets entering intake",
    ["outcome"],
    registry=REGISTRY,
)
cdp_corpus_qualified_total = Counter(
    "cdp_corpus_qualified_total",
    "Governed corpus assets qualified",
    ["corpus_version"],
    registry=REGISTRY,
)
cdp_corpus_excluded_total = Counter(
    "cdp_corpus_excluded_total",
    "Governed corpus assets excluded by controlled reason",
    ["reason_code"],
    registry=REGISTRY,
)
cdp_corpus_source_total = Counter(
    "cdp_corpus_source_total",
    "Governed source attestations evaluated",
    ["status"],
    registry=REGISTRY,
)
cdp_review_agreement = Gauge(
    "cdp_review_agreement",
    "Blind reviewer agreement by non-PHI label dimension",
    ["dimension"],
    registry=REGISTRY,
)
cdp_loso_source_run_total = Counter(
    "cdp_loso_source_run_total",
    "LOSO source rotations attempted",
    ["outcome"],
    registry=REGISTRY,
)
cdp_unverified_fixed_authorization_total = Counter(
    "cdp_unverified_fixed_authorization_total",
    "Fixed routes lacking matching VERIFIED evidence",
    ["authorized_family"],
    registry=REGISTRY,
)
cdp_route_firewall_violation_total = Counter(
    "cdp_route_firewall_violation_total",
    "Route-to-extractor firewall violations",
    ["authorized_route"],
    registry=REGISTRY,
)
cdp_policy_decision_total = Counter(
    "cdp_policy_decision_total",
    "Canonical field/claim policy decisions",
    ["policy_id", "decision"],
    registry=REGISTRY,
)
cdp_claim_blocker_total = Counter(
    "cdp_claim_blocker_total",
    "Blocking field decisions by safe dimensions",
    ["document_family", "field_name", "criticality"],
    registry=REGISTRY,
)
cdp_perfect_but_not_stp_total = Counter(
    "cdp_perfect_but_not_stp_total",
    "Perfectly extracted claims that remain blocked by canonical policy",
    ["document_family", "blocker_class"],
    registry=REGISTRY,
)
cdp_single_blocker_claim_total = Counter(
    "cdp_single_blocker_claim_total",
    "Claims with exactly one canonical blocking field",
    ["document_family", "field_name", "criticality"],
    registry=REGISTRY,
)
cdp_claim_unlock_total = Counter(
    "cdp_claim_unlock_total",
    "Claims transitioning from HITL to canonical STP",
    ["document_family", "profile"],
    registry=REGISTRY,
)
cdp_claim_unlock_by_field = Counter(
    "cdp_claim_unlock_by_field",
    "Canonical claim unlocks attributable to a blocking field",
    ["document_family", "field_name", "criticality"],
    registry=REGISTRY,
)
cdp_federal_tax_no_safe_coverage = Gauge(
    "cdp_federal_tax_no_safe_coverage",
    "Safely accepted federal tax number fraction over observable truth rows",
    registry=REGISTRY,
)
cdp_cost_per_document = Gauge(
    "cdp_cost_per_document",
    "Measured processing cost per document in USD",
    registry=REGISTRY,
)
cdp_cost_per_stp_claim = Gauge(
    "cdp_cost_per_stp_claim",
    "Measured processing cost per STP claim in USD",
    registry=REGISTRY,
)
cdp_cost_per_review_avoided = Gauge(
    "cdp_cost_per_review_avoided",
    "Measured processing cost per review avoided in USD",
    registry=REGISTRY,
)

# Phase 8.2 canonical standard-form processing telemetry. Labels are limited
# to bounded operational dimensions and must never contain OCR text, IDs, or
# any other claim/patient values.
cdp_page_processing_seconds = Histogram(
    "cdp_page_processing_seconds",
    "Canonical standard-form page processing time",
    ["document_family", "stage"],
    registry=REGISTRY,
)
cdp_full_page_ocr_seconds = Histogram(
    "cdp_full_page_ocr_seconds",
    "Full-page OCR wall time",
    ["engine"],
    registry=REGISTRY,
)
cdp_field_candidate_seconds = Histogram(
    "cdp_field_candidate_seconds",
    "Field candidate generation wall time",
    ["document_family"],
    registry=REGISTRY,
)
cdp_pages_processed_total = Counter(
    "cdp_pages_processed_total",
    "Canonical standard-form pages processed",
    ["document_family", "outcome"],
    registry=REGISTRY,
)
cdp_pages_per_minute = Gauge(
    "cdp_pages_per_minute",
    "Observed canonical standard-form throughput",
    ["worker_count"],
    registry=REGISTRY,
)
cdp_full_page_ocr_calls_total = Counter(
    "cdp_full_page_ocr_calls_total",
    "Full-page OCR executions",
    ["engine"],
    registry=REGISTRY,
)
cdp_regional_ocr_calls_total = Counter(
    "cdp_regional_ocr_calls_total",
    "Regional OCR executions",
    ["engine"],
    registry=REGISTRY,
)
cdp_secondary_ocr_rate = Gauge(
    "cdp_secondary_ocr_rate",
    "Fraction of eligible fields invoking secondary OCR",
    registry=REGISTRY,
)
cdp_secondary_ocr_resolution_rate = Gauge(
    "cdp_secondary_ocr_resolution_rate",
    "Fraction of secondary OCR calls resolving a field",
    registry=REGISTRY,
)
# Phase 8.3 canonical spelling. Keep the older metric above for dashboard
# compatibility while consumers migrate.
cdp_secondary_resolution_rate = Gauge(
    "cdp_secondary_resolution_rate",
    "Fraction of secondary OCR calls resolving a field",
    registry=REGISTRY,
)
cdp_ub_row_recall = Gauge(
    "cdp_ub_row_recall",
    "Truth-qualified UB service-line row recall",
    registry=REGISTRY,
)
cdp_ub_exact_row_accuracy = Gauge(
    "cdp_ub_exact_row_accuracy",
    "Truth-qualified UB exact service-line accuracy",
    registry=REGISTRY,
)
cdp_safe_field_coverage = Gauge(
    "cdp_safe_field_coverage",
    "Correct canonically accepted field fraction",
    registry=REGISTRY,
)
cdp_review_fields_per_page = Gauge(
    "cdp_review_fields_per_page",
    "Canonical review fields per processed page",
    registry=REGISTRY,
)
cdp_machine_cost_per_page = Gauge(
    "cdp_machine_cost_per_page",
    "Measured machine compute cost per page in USD",
    registry=REGISTRY,
)
cdp_hitl_cost_per_page = Gauge(
    "cdp_hitl_cost_per_page",
    "Modeled human review cost per page in USD",
    registry=REGISTRY,
)
cdp_total_cost_per_page = Gauge(
    "cdp_total_cost_per_page",
    "Fully loaded processing cost per page in USD",
    registry=REGISTRY,
)
cdp_queue_lag = Gauge(
    "cdp_queue_lag",
    "Consumer lag for a governed pipeline stage",
    ["topic", "consumer_group"],
    registry=REGISTRY,
)
cdp_p95_document_latency = Gauge(
    "cdp_p95_document_latency",
    "P95 end-to-end document latency in seconds",
    ["document_family"],
    registry=REGISTRY,
)

# Phase 8.21 selective PP-OCRv5 challenger telemetry. Dimensions are bounded
# operational labels; candidate text and claim identifiers are prohibited.
ppocr_challenge_rate = Gauge(
    "ppocr_challenge_rate", "Eligible fields challenged by PP-OCRv5", registry=REGISTRY
)
ppocr_win_rate = Gauge(
    "ppocr_win_rate", "PP-OCRv5 adjudicated wins per challenge", registry=REGISTRY
)
ppocr_agreement_rate = Gauge(
    "ppocr_agreement_rate", "Primary/challenger agreement rate", registry=REGISTRY
)
ppocr_disagreement_rate = Gauge(
    "ppocr_disagreement_rate", "Primary/challenger disagreement rate", registry=REGISTRY
)
challenger_blockers_removed = Counter(
    "challenger_blockers_removed",
    "Canonical blockers removed by challenger evidence",
    registry=REGISTRY,
)
challenger_claims_unlocked = Counter(
    "challenger_claims_unlocked", "Claims unlocked by challenger evidence", registry=REGISTRY
)
accuracy_by_source_quality_band = Gauge(
    "accuracy_by_source_quality_band",
    "Accepted accuracy by source quality band",
    ["source", "quality_band"],
    registry=REGISTRY,
)
hitl_by_source_quality_band = Gauge(
    "hitl_by_source_quality_band",
    "HITL rate by source quality band",
    ["source", "quality_band"],
    registry=REGISTRY,
)
latency_by_engine = Histogram(
    "latency_by_engine", "OCR execution latency in seconds", ["engine"], registry=REGISTRY
)
ocr_calls_per_claim = Histogram(
    "ocr_calls_per_claim", "OCR calls issued per claim", registry=REGISTRY
)

# Closed-world Azure OpenAI adjudication. Labels are bounded operational
# dimensions only; candidate values and claim/page identifiers are prohibited.
azure_openai_requests_total = Counter(
    "azure_openai_requests_total",
    "Azure adjudication requests",
    ["deployment", "outcome"],
    registry=REGISTRY,
)
azure_openai_requests_tier1 = Counter(
    "azure_openai_requests_tier1",
    "Tier-1 text adjudication requests",
    ["deployment"],
    registry=REGISTRY,
)
azure_openai_requests_tier2 = Counter(
    "azure_openai_requests_tier2",
    "Tier-2 field-crop adjudication requests",
    ["deployment"],
    registry=REGISTRY,
)
azure_openai_input_tokens = Counter(
    "azure_openai_input_tokens",
    "Azure adjudication input tokens",
    ["deployment"],
    registry=REGISTRY,
)
azure_openai_output_tokens = Counter(
    "azure_openai_output_tokens",
    "Azure adjudication output tokens",
    ["deployment"],
    registry=REGISTRY,
)
azure_openai_latency_ms = Histogram(
    "azure_openai_latency_ms",
    "Azure adjudication latency milliseconds",
    ["deployment", "tier"],
    registry=REGISTRY,
)
azure_openai_cost_usd = Counter(
    "azure_openai_cost_usd", "Configured Azure adjudication cost", ["deployment"], registry=REGISTRY
)
llm_cost_per_page = Gauge(
    "llm_cost_per_page", "Configured mean LLM cost per page", registry=REGISTRY
)
llm_routing_rate = Gauge(
    "llm_routing_rate", "Fraction routed to paid adjudication", ["tier"], registry=REGISTRY
)
llm_cache_hit_rate = Gauge(
    "llm_cache_hit_rate", "LLM adjudication cache hit rate", registry=REGISTRY
)
llm_blockers_removed = Counter(
    "llm_blockers_removed", "Shadow or authoritative blockers removed", ["mode"], registry=REGISTRY
)
llm_claims_unlocked = Counter(
    "llm_claims_unlocked", "Shadow or authoritative claims unlocked", ["mode"], registry=REGISTRY
)
llm_hitl_fallback = Counter(
    "llm_hitl_fallback", "Fail-closed LLM HITL fallbacks", ["reason"], registry=REGISTRY
)
llm_budget_rejected = Counter(
    "llm_budget_rejected", "LLM requests rejected by cost policy", ["reason"], registry=REGISTRY
)
llm_provider_errors = Counter(
    "llm_provider_errors", "Bounded Azure provider failures", ["category"], registry=REGISTRY
)
