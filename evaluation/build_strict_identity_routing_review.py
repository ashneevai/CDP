"""Build and score prediction-blind review queues for frozen routing replays.

The annotator-facing queues contain immutable page lineage and review controls,
but no candidate prediction or risk ranking. Accuracy remains blocked until the
required independent human reviews and adjudications are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ALLOWED_LABELS = {
    "CMS1500",
    "UB04",
    "OTHER_CLAIM_FORM",
    "SUPPORTING_DOCUMENT",
    "NON_CLAIM",
    "UNKNOWN",
}
STANDARD_CLASSES = {"CMS1500", "UB04"}
SCHEMA_VERSION = "strict-identity-routing-review-v2"
ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "evaluation_data/strict_identity_replay_v3/pages"
PRECEDING = ROOT / "evaluation_data/strict_identity_replay_v2/pages"
OUTPUT = ROOT / "evaluation_data/strict_identity_routing_review"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay_source_manifest_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text("utf-8"))
    value = payload.get("input_manifest_sha256")
    return value if isinstance(value, str) else None


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _load(directory: Path) -> tuple[dict[str, dict[str, Any]], str, str]:
    records: dict[str, dict[str, Any]] = {}
    policy_hashes: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text("utf-8"))
        page_id = record["source_page_id"]
        if page_id in records:
            raise ValueError(f"DUPLICATE_PAGE_ID:{page_id}")
        page_sha = record.get("source_page_sha256")
        if not _valid_sha(page_sha) or page_sha != record.get("ocr_provenance", {}).get(
            "rendered_page_sha256"
        ):
            raise ValueError(f"PAGE_SHA_MISMATCH:{page_id}")
        policy_sha = record.get("decision_provenance", {}).get("decision_policy_sha256")
        if not _valid_sha(policy_sha):
            raise ValueError(f"MISSING_OR_INVALID_DECISION_POLICY_SHA:{page_id}")
        policy_hashes.add(policy_sha)
        records[page_id] = record
    if not records:
        raise ValueError("EMPTY_REPLAY_SNAPSHOT")
    if len(policy_hashes) != 1:
        raise ValueError("MIXED_DECISION_POLICY_SHA")
    page_set_sha = _digest(
        [
            {"source_page_id": page_id, "source_page_sha256": record["source_page_sha256"]}
            for page_id, record in sorted(records.items())
        ]
    )
    return records, next(iter(policy_hashes)), page_set_sha


def _conflict(record: dict[str, Any]) -> bool:
    return any(
        bool(values)
        for values in record.get("form_identity", {}).get("conflicting_anchors", {}).values()
    )


def _near_miss(record: dict[str, Any]) -> bool:
    eligibility = record.get("form_identity", {}).get("family_eligibility", {})
    return any(
        not values.get("eligible")
        and values.get("high_value_anchor_count", 0) >= 2
        and sum(
            not bool(passed) for passed in values.get("authorization_gates", {}).values()
        )
        <= 2
        for values in eligibility.values()
    )


def _risk(record: dict[str, Any], preceding: dict[str, Any]) -> tuple[int, list[str]]:
    chain = record["production_chain"]
    nomination = record["routing_result"]["router_nomination"]
    reasons: list[str] = []
    if chain["fixed_extractor_authorized"]:
        reasons.append("FIXED_EXTRACTOR_AUTHORIZED")
    if nomination in STANDARD_CLASSES:
        reasons.append("STANDARD_ROUTER_NOMINATION")
    if "STANDARD_IDENTITY_CLASSIFICATION_MISMATCH" in chain["decision_reason_codes"]:
        reasons.append("FAMILY_MISMATCH")
    if _conflict(record):
        reasons.append("CONFLICTING_IDENTITY")
    if _near_miss(record):
        reasons.append("THRESHOLD_NEAR_MISS")
    if (
        preceding.get("form_identity", {}).get("localization_allowed", False)
        and not chain["fixed_extractor_authorized"]
    ):
        reasons.append("PRECEDING_POLICY_AUTHORIZED_CURRENT_REJECTED")
    priority = (
        0
        if "FIXED_EXTRACTOR_AUTHORIZED" in reasons
        else 1
        if "STANDARD_ROUTER_NOMINATION" in reasons
        else 2
        if any(
            reason in reasons
            for reason in ("FAMILY_MISMATCH", "CONFLICTING_IDENTITY", "THRESHOLD_NEAR_MISS")
        )
        else 3
        if "PRECEDING_POLICY_AUTHORIZED_CURRENT_REJECTED" in reasons
        else 4
    )
    return priority, reasons


def _round_robin(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave candidate-class/package strata for rapid-cohort coverage."""
    grouped: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for record in sorted(
        records,
        key=lambda item: (
            str(item.get("candidate_class")),
            str(item.get("package_id")),
            item["source_page_id"],
        ),
    ):
        grouped[(str(record.get("candidate_class")), str(record.get("package_id")))].append(
            record
        )
    ordered: list[dict[str, Any]] = []
    while grouped:
        for key in sorted(grouped):
            ordered.append(grouped[key].popleft())
            if not grouped[key]:
                del grouped[key]
    return ordered


def _blind_order(records: list[dict[str, Any]], salt: str, mode: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{salt}|{mode}|{record['source_page_id']}|{record['source_page_sha256']}".encode()
        ).hexdigest(),
    )


def _provenance_summary(path: Path | None, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "status": "BLOCKED_LABEL_PROVENANCE_UNAVAILABLE",
            "provenance_manifest_sha256": None,
            "records_declared": None,
            "admissible_exact_bindings": None,
            "accuracy_scoring_authorized": False,
        }
    payload = json.loads(path.read_text("utf-8"))
    items = payload.get("records")
    if not isinstance(items, list):
        raise ValueError("INVALID_PROVENANCE_RECORDS")
    seen: set[str] = set()
    admissible = 0
    for item in items:
        page_id = item.get("source_page_id")
        if page_id in seen:
            raise ValueError("DUPLICATE_PROVENANCE_PAGE")
        seen.add(page_id)
        replay = records.get(page_id)
        if replay is None or item.get("source_page_sha256") != replay["source_page_sha256"]:
            raise ValueError("PROVENANCE_PAGE_LINEAGE_MISMATCH")
        if item.get("label_authority") in {
            "SOURCE_SYSTEM_GROUND_TRUTH",
            "INDEPENDENT_HUMAN_ADJUDICATION",
        }:
            admissible += 1
    return {
        "status": "PROVENANCE_VERIFIED" if admissible else "BLOCKED_NO_ADMISSIBLE_LABELS",
        "provenance_manifest_sha256": _file_digest(path),
        "records_declared": len(items),
        "admissible_exact_bindings": admissible,
        "accuracy_scoring_authorized": admissible > 0,
    }


def _queue(
    *,
    mode: str,
    records: list[dict[str, Any]],
    audit: dict[str, dict[str, Any]],
    common: dict[str, Any],
    blind_salt: str,
) -> dict[str, Any]:
    blind_records: list[dict[str, Any]] = []
    for position, record in enumerate(_blind_order(records, blind_salt, mode), 1):
        page_id = record["source_page_id"]
        hard_confuser = audit[page_id]["priority"] < 4
        required_reviews = 2 if mode == "full" or hard_confuser else 1
        blind_records.append(
            {
                "queue_position": position,
                "source_page_id": page_id,
                "source_page_sha256": record["source_page_sha256"],
                "package_id": record["package_id"],
                "blind_task_id": hashlib.sha256(
                    (
                        f"routing-review-v2|{common['page_set_sha256']}|"
                        f"{common['current_decision_policy_sha256']}|{mode}|{page_id}|"
                        f"{record['source_page_sha256']}"
                    ).encode()
                ).hexdigest(),
                "local_image_reference": f"source-page://{page_id}",
                "allowed_labels": sorted(ALLOWED_LABELS),
                "required_reviewer_roles": ["REVIEWER_A", "REVIEWER_B"],
                "required_independent_reviews": required_reviews,
                "dynamic_second_review_triggers": [
                    "FIRST_LABEL_IS_CMS1500_OR_UB04",
                    "FIRST_LABEL_IS_UNKNOWN",
                    "IDENTITY_AMBIGUITY_DECLARED",
                ],
                "disagreement_or_ambiguity_requires_independent_adjudicator": True,
                "review_state": "REVIEW_REQUIRED",
            }
        )
    queue: dict[str, Any] = {
        **common,
        "mode": mode,
        "status": "PRELIMINARY_RISK_COHORT" if mode == "rapid" else "CORPUS_WIDE",
        "records": blind_records,
    }
    queue["queue_content_sha256"] = _digest(queue)
    return queue


def build(
    current_dir: Path = CURRENT,
    preceding_dir: Path = PRECEDING,
    output: Path = OUTPUT,
    rapid_size: int = 300,
    *,
    expected_page_count: int = 2173,
    source_manifest_sha256: str | None = None,
    blind_salt: str | None = None,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    current, current_policy_sha, current_page_set_sha = _load(current_dir)
    preceding, preceding_policy_sha, preceding_page_set_sha = _load(preceding_dir)
    if set(current) != set(preceding) or any(
        current[page_id]["source_page_sha256"]
        != preceding[page_id]["source_page_sha256"]
        for page_id in current
    ):
        raise ValueError("CURRENT_PRECEDING_PAGE_SET_MISMATCH")
    if current_page_set_sha != preceding_page_set_sha:
        raise ValueError("CURRENT_PRECEDING_PAGE_SET_MISMATCH")
    if len(current) != expected_page_count:
        raise ValueError(f"PARTIAL_REPLAY:{len(current)}:{expected_page_count}")
    if rapid_size <= 0:
        raise ValueError("INVALID_RAPID_SIZE")
    manifest_path = current_dir.parent / "manifest.json"
    resolved_source_sha = source_manifest_sha256 or _replay_source_manifest_sha(manifest_path)
    if not _valid_sha(resolved_source_sha):
        raise ValueError("MISSING_OR_INVALID_SOURCE_MANIFEST_SHA")
    salt = blind_salt or secrets.token_hex(32)

    audit: dict[str, dict[str, Any]] = {}
    mandatory: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    for page_id, record in current.items():
        priority, reasons = _risk(record, preceding[page_id])
        audit[page_id] = {
            "priority": priority,
            "risk_reasons": reasons,
            "current_candidate_class": record["candidate_class"],
            "current_router_nomination": record["routing_result"]["router_nomination"],
            "current_fixed_authorized": record["production_chain"][
                "fixed_extractor_authorized"
            ],
            "preceding_localization_allowed": preceding[page_id]
            .get("form_identity", {})
            .get("localization_allowed", False),
        }
        (mandatory if priority < 4 else remainder).append(record)
    rapid_target = min(rapid_size, len(current))
    rapid_records = [
        *sorted(mandatory, key=lambda record: record["source_page_id"]),
        *_round_robin(remainder)[: max(0, rapid_target - len(mandatory))],
    ]
    common = {
        "schema_version": SCHEMA_VERSION,
        "blind_initial_annotation": True,
        "predictions_in_queue": False,
        "trusted_ground_truth": False,
        "label_authority_required": "INDEPENDENT_HUMAN_REVIEW_AND_ADJUDICATION",
        "expected_replay_pages": expected_page_count,
        "total_replay_pages": len(current),
        "page_set_sha256": current_page_set_sha,
        "source_manifest_sha256": resolved_source_sha,
        "current_decision_policy_sha256": current_policy_sha,
        "preceding_decision_policy_sha256": preceding_policy_sha,
    }
    rapid = _queue(
        mode="rapid", records=rapid_records, audit=audit, common=common, blind_salt=salt
    )
    full = _queue(
        mode="full",
        records=list(current.values()),
        audit=audit,
        common=common,
        blind_salt=salt,
    )
    coordinator: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "access": "COORDINATOR_ONLY_NOT_FOR_INITIAL_ANNOTATORS",
        "blind_order_salt": salt,
        "page_set_sha256": current_page_set_sha,
        "source_manifest_sha256": resolved_source_sha,
        "current_decision_policy_sha256": current_policy_sha,
        "preceding_decision_policy_sha256": preceding_policy_sha,
        "rapid_target_pages": rapid_target,
        "rapid_actual_pages": len(rapid_records),
        "rapid_target_expanded_for_mandatory_risk": len(rapid_records) > rapid_target,
        "rapid_queue_content_sha256": rapid["queue_content_sha256"],
        "full_queue_content_sha256": full["queue_content_sha256"],
        "risk_counts": {
            reason: sum(reason in values["risk_reasons"] for values in audit.values())
            for reason in sorted(
                {reason for values in audit.values() for reason in values["risk_reasons"]}
            )
        },
        "records": audit,
    }
    coordinator["coordinator_content_sha256"] = _digest(coordinator)
    provenance = _provenance_summary(provenance_path, current)
    provenance.update(
        {
            "schema_version": SCHEMA_VERSION,
            "page_set_sha256": current_page_set_sha,
            "source_manifest_sha256": resolved_source_sha,
            "current_decision_policy_sha256": current_policy_sha,
            "replay_pages": len(current),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("rapid_blind_queue.json", rapid),
        ("full_blind_queue.json", full),
        ("coordinator_risk_manifest.json", coordinator),
        ("trusted_label_provenance.json", provenance),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {
        "rapid_target_pages": rapid_target,
        "rapid_pages": len(rapid["records"]),
        "rapid_double_review_pages": sum(
            record["required_independent_reviews"] == 2 for record in rapid["records"]
        ),
        "full_pages": len(full["records"]),
        "full_double_review_pages": sum(
            record["required_independent_reviews"] == 2 for record in full["records"]
        ),
        "status": provenance["status"],
        "risk_counts": coordinator["risk_counts"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _wilson(successes: int, total: int) -> dict[str, float | int] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "numerator": successes,
        "denominator": total,
        "estimate": proportion,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _validate_queue(queue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if queue.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("UNSUPPORTED_QUEUE_SCHEMA")
    declared_hash = queue.get("queue_content_sha256")
    unsigned = {key: value for key, value in queue.items() if key != "queue_content_sha256"}
    if not _valid_sha(declared_hash) or _digest(unsigned) != declared_hash:
        raise ValueError("QUEUE_CONTENT_SHA_MISMATCH")
    if queue.get("mode") not in {"rapid", "full"}:
        raise ValueError("INVALID_QUEUE_MODE")
    if int(queue.get("expected_replay_pages", 0)) != int(
        queue.get("total_replay_pages", -1)
    ):
        raise ValueError("QUEUE_REPLAY_COUNT_MISMATCH")
    if not all(
        _valid_sha(queue.get(field))
        for field in (
            "page_set_sha256",
            "source_manifest_sha256",
            "current_decision_policy_sha256",
            "preceding_decision_policy_sha256",
        )
    ):
        raise ValueError("INVALID_QUEUE_PROVENANCE_SHA")
    tasks: dict[str, dict[str, Any]] = {}
    page_ids: set[str] = set()
    positions: set[int] = set()
    for task in queue.get("records", []):
        task_id = task.get("blind_task_id")
        page_id = task.get("source_page_id")
        position = task.get("queue_position")
        if (
            not _valid_sha(task_id)
            or not isinstance(page_id, str)
            or not _valid_sha(task.get("source_page_sha256"))
            or not isinstance(position, int)
            or position < 1
            or task.get("required_independent_reviews") not in {1, 2}
        ):
            raise ValueError("INVALID_QUEUE_TASK")
        if task_id in tasks:
            raise ValueError("DUPLICATE_BLIND_TASK_ID")
        if page_id in page_ids:
            raise ValueError("DUPLICATE_QUEUE_PAGE_ID")
        if position in positions:
            raise ValueError("DUPLICATE_QUEUE_POSITION")
        tasks[task_id] = task
        page_ids.add(page_id)
        positions.add(position)
    return tasks


def _null_metrics(mode: str) -> dict[str, Any]:
    accuracy_name = (
        "rapid_risk_cohort_exact_accuracy"
        if mode == "rapid"
        else "corpus_overall_exact_accuracy"
    )
    return {
        accuracy_name: None,
        "macro_f1_supported_classes": None,
        "macro_f1_all_defined_classes": None,
        "per_class": None,
        "confusion_matrix": None,
        "cms1500_precision": None,
        "cms1500_recall": None,
        "ub04_precision": None,
        "ub04_recall": None,
        "fixed_authorization_precision": None,
        "fixed_authorization_recall": None,
        "false_standard_authorization_rate": None,
        "false_standard_authorization_count": None,
        "wrong_family_authorization_count": None,
        "authorization_coverage": None,
        "page_review_abstention_rate": None,
        "reviewer_agreement": None,
        "disagreement_count": None,
        "adjudication_count": None,
    }


def score(
    queue_path: Path,
    current_dir: Path,
    reviews_path: Path,
    adjudications_path: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    queue = json.loads(queue_path.read_text("utf-8"))
    tasks = _validate_queue(queue)
    current, current_policy_sha, page_set_sha = _load(current_dir)
    if len(current) != int(queue["expected_replay_pages"]):
        raise ValueError("CURRENT_REPLAY_PAGE_COUNT_MISMATCH")
    if page_set_sha != queue["page_set_sha256"]:
        raise ValueError("CURRENT_REPLAY_PAGE_SET_SHA_MISMATCH")
    if current_policy_sha != queue["current_decision_policy_sha256"]:
        raise ValueError("CURRENT_REPLAY_POLICY_SHA_MISMATCH")
    queue_page_ids = {task["source_page_id"] for task in tasks.values()}
    if queue["mode"] == "full" and queue_page_ids != set(current):
        raise ValueError("FULL_QUEUE_NOT_CORPUS_COMPLETE")
    for task in tasks.values():
        page = current.get(task["source_page_id"])
        if page is None or page["source_page_sha256"] != task["source_page_sha256"]:
            raise ValueError("QUEUE_PAGE_LINEAGE_MISMATCH")

    reviews_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    review_events: set[tuple[str, str, str]] = set()
    for review in _read_jsonl(reviews_path):
        review_task_id = review.get("blind_task_id")
        if not isinstance(review_task_id, str):
            raise ValueError("UNKNOWN_BLIND_TASK")
        review_task = tasks.get(review_task_id)
        if review_task is None:
            raise ValueError("UNKNOWN_BLIND_TASK")
        if (
            review.get("source_page_id") != review_task["source_page_id"]
            or review.get("source_page_sha256") != review_task["source_page_sha256"]
        ):
            raise ValueError("REVIEW_PAGE_LINEAGE_MISMATCH")
        if review.get("reviewed_label") not in ALLOWED_LABELS:
            raise ValueError("INVALID_REVIEW_LABEL")
        if (
            review.get("review_status") != "COMPLETED"
            or not review.get("reviewer_id")
            or not review.get("review_session_id")
            or review.get("reviewer_role") not in {"REVIEWER_A", "REVIEWER_B"}
            or not isinstance(review.get("identity_ambiguity", False), bool)
        ):
            raise ValueError("INCOMPLETE_REVIEW_AUTHORITY")
        event_key = (
            review["blind_task_id"],
            review["reviewer_id"],
            review["review_session_id"],
        )
        if event_key in review_events:
            raise ValueError("DUPLICATE_REVIEW_EVENT")
        review_events.add(event_key)
        reviews_by_task[review["blind_task_id"]].append(review)

    adjudications: dict[str, dict[str, Any]] = {}
    for adjudication in _read_jsonl(adjudications_path):
        task_id = adjudication.get("blind_task_id")
        if not isinstance(task_id, str):
            raise ValueError("UNKNOWN_ADJUDICATION_TASK")
        adjudication_task = tasks.get(task_id)
        if adjudication_task is None:
            raise ValueError("UNKNOWN_ADJUDICATION_TASK")
        if task_id in adjudications:
            raise ValueError("DUPLICATE_ADJUDICATION_EVENT")
        if (
            adjudication.get("source_page_id") != adjudication_task["source_page_id"]
            or adjudication.get("source_page_sha256")
            != adjudication_task["source_page_sha256"]
        ):
            raise ValueError("ADJUDICATION_PAGE_LINEAGE_MISMATCH")
        adjudications[task_id] = adjudication

    labels: dict[str, str] = {}
    pending_reviews = 0
    pending_adjudications = 0
    agreements = disagreements = adjudication_count = 0
    completed_reviews = 0
    for task_id, task in tasks.items():
        reviews = reviews_by_task.get(task_id, [])
        if len(reviews) > 2:
            raise ValueError("TOO_MANY_REVIEW_EVENTS")
        if reviews and reviews[0]["reviewer_role"] != "REVIEWER_A":
            raise ValueError("FIRST_REVIEW_MUST_BE_REVIEWER_A")
        first = reviews[0] if reviews else None
        dynamic_second = bool(
            first
            and (
                first["reviewed_label"] in STANDARD_CLASSES | {"UNKNOWN"}
                or first.get("identity_ambiguity", False)
            )
        )
        required = max(int(task["required_independent_reviews"]), 2 if dynamic_second else 1)
        completed_reviews += len(reviews)
        if len(reviews) < required:
            pending_reviews += required - len(reviews)
            continue
        selected = reviews[:required]
        ambiguity = any(review.get("identity_ambiguity", False) for review in selected)
        if required == 2:
            if (
                selected[1]["reviewer_role"] != "REVIEWER_B"
                or selected[0]["reviewer_id"] == selected[1]["reviewer_id"]
                or selected[0]["review_session_id"] == selected[1]["review_session_id"]
            ):
                raise ValueError("REVIEWS_NOT_INDEPENDENT")
            agreed = selected[0]["reviewed_label"] == selected[1]["reviewed_label"]
            agreements += int(agreed)
            disagreements += int(not agreed)
            if agreed and not ambiguity:
                labels[task_id] = selected[0]["reviewed_label"]
                continue
            resolved_adjudication = adjudications.get(task_id)
            if resolved_adjudication is None:
                pending_adjudications += 1
                continue
            if (
                resolved_adjudication.get("adjudicator_id")
                in {review["reviewer_id"] for review in selected}
                or resolved_adjudication.get("adjudication_session_id")
                in {review["review_session_id"] for review in selected}
                or resolved_adjudication.get("final_label") not in ALLOWED_LABELS
                or resolved_adjudication.get("adjudication_status") != "COMPLETED"
                or not resolved_adjudication.get("adjudicator_id")
                or not resolved_adjudication.get("adjudication_session_id")
            ):
                raise ValueError("ADJUDICATION_NOT_INDEPENDENT_OR_COMPLETE")
            labels[task_id] = resolved_adjudication["final_label"]
            adjudication_count += 1
        else:
            labels[task_id] = selected[0]["reviewed_label"]

    progress = {
        "queue_tasks": len(tasks),
        "total_replay_pages": int(queue["total_replay_pages"]),
        "completed_reviews": completed_reviews,
        "pending_reviews": pending_reviews,
        "pending_adjudications": pending_adjudications,
        "scorable_pages": len(labels),
        "unscorable_pages": len(tasks) - len(labels),
    }
    scope = {
        "routing_accuracy": (
            "RISK_ENRICHED_COHORT_NOT_A_CORPUS_ESTIMATE"
            if queue["mode"] == "rapid"
            else "FULL_FROZEN_CORPUS"
        ),
        "field_extraction_accuracy": "NOT_EVALUATED",
        "accepted_field_precision": "NOT_EVALUATED",
        "page_review_rate": "ROUTING_ONLY_PRODUCTION_DECISION_STATE",
        "field_hitl": "NOT_EVALUATED",
        "claim_hitl_stp": "NOT_EVALUATED",
        "cached_replay_timing": "SEPARATE_EXISTING_ARTIFACT",
        "fresh_latency_and_cost": "NOT_EVALUATED",
    }
    evidence = {
        "queue_content_sha256": queue["queue_content_sha256"],
        "page_set_sha256": page_set_sha,
        "source_manifest_sha256": queue["source_manifest_sha256"],
        "current_decision_policy_sha256": current_policy_sha,
        "reviews_sha256": _file_digest(reviews_path),
        "adjudications_sha256": _file_digest(adjudications_path),
    }
    if len(labels) != len(tasks) or pending_reviews or pending_adjudications:
        result: dict[str, Any] = {
            "status": "BLOCKED_HUMAN_LABELS",
            "reason": "BLOCKED_HUMAN_LABELS",
            "progress": progress,
            "metrics": _null_metrics(queue["mode"]),
            "metric_scope": scope,
            "evidence": evidence,
        }
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
        return result

    truth = {tasks[task_id]["source_page_id"]: label for task_id, label in labels.items()}
    classes = sorted(ALLOWED_LABELS)
    confusion = {actual: {predicted: 0 for predicted in classes} for actual in classes}
    correct = 0
    for page_id, actual in truth.items():
        predicted = current[page_id]["candidate_class"]
        if predicted not in ALLOWED_LABELS:
            predicted = "UNKNOWN"
        confusion[actual][predicted] += 1
        correct += int(actual == predicted)

    per_class: dict[str, Any] = {}
    supported_f1: list[float] = []
    all_f1: list[float] = []
    for label in classes:
        tp = confusion[label][label]
        predicted_total = sum(confusion[actual][label] for actual in classes)
        actual_total = sum(confusion[label].values())
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / actual_total if actual_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        all_f1.append(f1)
        if actual_total:
            supported_f1.append(f1)
        per_class[label] = {
            "support": actual_total,
            "precision": _wilson(tp, predicted_total),
            "recall": _wilson(tp, actual_total),
            "f1": f1,
        }

    authorized = [
        (page_id, current[page_id])
        for page_id in truth
        if current[page_id]["production_chain"]["fixed_extractor_authorized"]
    ]
    correct_authorized = sum(
        truth[page_id] == record["production_chain"]["verified_identity_family"]
        for page_id, record in authorized
    )
    true_standard = sum(label in STANDARD_CLASSES for label in truth.values())
    nonstandard_truth = sum(label not in STANDARD_CLASSES for label in truth.values())
    false_standard_authorizations = sum(
        truth[page_id] not in STANDARD_CLASSES for page_id, _ in authorized
    )
    wrong_family_authorizations = sum(
        truth[page_id] in STANDARD_CLASSES
        and truth[page_id] != record["production_chain"]["verified_identity_family"]
        for page_id, record in authorized
    )
    review_required = sum(
        current[page_id].get("confidence_band") in {"REVIEW_REQUIRED", "UNKNOWN"}
        or current[page_id]["production_chain"].get("processing_route") == "SAFE_UNKNOWN"
        for page_id in truth
    )
    double_reviewed = agreements + disagreements
    accuracy_name = (
        "rapid_risk_cohort_exact_accuracy"
        if queue["mode"] == "rapid"
        else "corpus_overall_exact_accuracy"
    )
    result = {
        "status": (
            "ROUTING_PRELIMINARY_RISK_COHORT"
            if queue["mode"] == "rapid"
            else "ROUTING_CORPUS_EVALUATED"
        ),
        "reason": None,
        "progress": progress,
        "metric_scope": scope,
        "evidence": evidence,
        "metrics": {
            accuracy_name: _wilson(correct, len(truth)),
            "macro_f1_supported_classes": (
                sum(supported_f1) / len(supported_f1) if supported_f1 else None
            ),
            "macro_f1_all_defined_classes": sum(all_f1) / len(all_f1),
            "per_class": per_class,
            "confusion_matrix": confusion,
            "cms1500_precision": per_class["CMS1500"]["precision"],
            "cms1500_recall": per_class["CMS1500"]["recall"],
            "ub04_precision": per_class["UB04"]["precision"],
            "ub04_recall": per_class["UB04"]["recall"],
            "fixed_authorization_precision": _wilson(correct_authorized, len(authorized)),
            "fixed_authorization_recall": _wilson(correct_authorized, true_standard),
            "false_standard_authorization_rate": _wilson(
                false_standard_authorizations, nonstandard_truth
            ),
            "false_standard_authorization_count": false_standard_authorizations,
            "wrong_family_authorization_count": wrong_family_authorizations,
            "authorization_coverage": _wilson(len(authorized), len(truth)),
            "page_review_abstention_rate": _wilson(review_required, len(truth)),
            "reviewer_agreement": _wilson(agreements, double_reviewed),
            "disagreement_count": disagreements,
            "adjudication_count": adjudication_count,
        },
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-dir", type=Path, default=CURRENT)
    parser.add_argument("--preceding-dir", type=Path, default=PRECEDING)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rapid-size", type=int, default=300)
    parser.add_argument("--expected-pages", type=int, default=2173)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--blind-salt")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--score-queue", type=Path)
    parser.add_argument("--reviews", type=Path, default=OUTPUT / "reviews.jsonl")
    parser.add_argument("--adjudications", type=Path, default=OUTPUT / "adjudications.jsonl")
    parser.add_argument("--score-output", type=Path, default=OUTPUT / "routing_metrics.json")
    args = parser.parse_args()
    result = (
        score(
            args.score_queue,
            args.current_dir,
            args.reviews,
            args.adjudications,
            args.score_output,
        )
        if args.score_queue
        else build(
            args.current_dir,
            args.preceding_dir,
            args.output,
            args.rapid_size,
            expected_page_count=args.expected_pages,
            source_manifest_sha256=args.source_manifest_sha256,
            blind_salt=args.blind_salt,
            provenance_path=args.provenance,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
