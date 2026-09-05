"""Shared runtime settings, loaded from environment variables (see
`.env.example` for the full list). One source of truth for env var names so
every app/worker agrees on them."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Versioning â€” part of the idempotency key
    pipeline_version: str = "0.1.0"
    schema_version: str = "1.0"

    # Database
    database_url: str = "sqlite:///:memory:"

    # Object storage (MinIO locally / any S3-compatible endpoint in prod)
    object_store_endpoint: str = "http://localhost:9000"
    object_store_access_key: str = "minioadmin"
    object_store_secret_key: str = "minioadmin"
    object_store_bucket: str = "idp-documents"
    object_store_use_ssl: bool = False

    # Kafka-compatible bus
    kafka_bootstrap_servers: str = "localhost:19092"
    use_in_memory_bus: bool = False  # true for local dev/tests without Docker

    # Redis (caching layer for the hybrid router's cache stage)
    redis_url: str = "redis://localhost:6379/0"

    # Ingestion limits
    max_upload_size_bytes: int = 50 * 1024 * 1024  # 50 MB

    # VLM â€” disabled by default; pipeline must run fully without it
    vlm_enabled: bool = False
    vlm_endpoint: str = "http://localhost:8001/v1"
    vlm_model_name: str = "qwen2.5-vl-3b-instruct"

    # Azure OpenAI crop-only production fallback. Outputs remain review-only
    # until an untouched holdout authorizes individual routes.
    azure_ai_evaluation_enabled: bool = False
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_ai_evaluation_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_review_only: bool = True

    # Cost-capped closed-world LLM adjudication. Local-only by default.
    llm_enabled: bool = False
    llm_mode: str = "SHADOW_ONLY"
    llm_provider: str = "azure_openai"
    llm_target_avg_cost_per_page_usd: float = 0.0005
    llm_max_avg_cost_per_page_usd: float = 0.001
    llm_max_cost_per_claim_usd: float = 0.005
    llm_max_tier1_calls_per_page: int = 1
    llm_max_tier2_calls_per_page: int = 1
    llm_tier1_max_input_tokens: int = 500
    llm_tier1_max_output_tokens: int = 40
    llm_timeout_seconds: float = 15.0
    llm_max_retries: int = 1
    azure_openai_input_cost_per_million_tokens: float | None = None
    azure_openai_output_cost_per_million_tokens: float | None = None
    azure_openai_cached_input_cost_per_million_tokens: float | None = None

    # Central external-AI gateway. Disabled and budget-zero by default.
    ai_gateway_enabled: bool = False
    ai_phi_external_processing_approved: bool = False
    ai_approved_regions: str = ""
    ai_allowed_models: str = ""
    ai_daily_budget_usd: float = 0.0
    ai_max_requests_per_minute: int = 0
    ai_max_crop_bytes: int = 2_000_000
    ai_timeout_seconds: float = 15.0
    ai_max_retries: int = 1
    ocr_audit_path: str = "/data/audit/ocr_calls.jsonl"
    vertex_region: str = "us-central1"
    aws_textract_region: str = "us-east-1"

    # Append-only reviewer correction memory. Exemplars are tenant/field scoped
    # and may guide VLM extraction, but never bypass deterministic validation.
    correction_memory_path: str = "/data/feedback/corrections.jsonl"
    correction_exemplar_limit: int = 3

    # Handwriting OCR -- opt-in because the model is large and is downloaded
    # separately from the lightweight application image.
    handwriting_enabled: bool = False
    trocr_model_name: str = "microsoft/trocr-base-handwritten"
    trocr_device: str = "auto"
    trocr_min_confidence: float = 0.55

    cloud_handwriting_enabled: bool = False
    cloud_handwriting_provider: str = "azure"
    cloud_handwriting_endpoint: str | None = None
    cloud_handwriting_credential: str | None = None
    cloud_handwriting_timeout_seconds: float = 10.0

    # Multi-tenancy default (overridden per-request where applicable)
    default_tenant_id: str = "default"
    enable_router_v3: bool = False
    # Router V4 remains evaluation-only until an independent holdout passes.
    enable_router_v4: bool = False
    enable_rem03a_eligibility: bool = False
    enable_ml_eligibility: bool = False
    enable_ml_eligibility_shadow: bool = False
    enable_visual_evidence: bool = False
    enable_visual_evidence_shadow: bool = False
    enable_router_v2: bool = True


def get_settings() -> Settings:
    return Settings()
