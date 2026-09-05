"""Append-only, local correction sink for supervised retraining data."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class CorrectionExample:
    document_id: str
    field_name: str
    previous_value: str | None
    corrected_value: str
    crop_reference: str | None
    reviewer: str
    corrected_at: str
    tenant_id: str = "default"
    reason: str | None = None
    task_id: str | None = None
    claim_id: str | None = None
    patient_id: str | None = None
    provider_id: str | None = None
    batch_id: str | None = None
    source_group_id: str | None = None
    source_document_sha256: str | None = None
    crop_sha256: str | None = None
    page_number: int | None = None
    candidate_provenance: tuple[dict[str, Any], ...] = ()
    model_provenance: dict[str, str] | None = None
    route_id: str | None = None
    route_status: str | None = None
    review_reason_codes: tuple[str, ...] = ()
    usage_authority: str = "TRAINING_ONLY"
    runtime_acceptance_authority: bool = False


@dataclass(frozen=True)
class CorrectionPattern:
    field_name: str
    observed: str
    corrected: str
    occurrences: int
    distinct_documents: int
    distinct_reviewers: int
    agreement_ratio: float
    promotion_eligible: bool


class CorrectionSink(Protocol):
    def append(self, example: CorrectionExample) -> None: ...


class JsonlCorrectionSink:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(self, example: CorrectionExample) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def correction_example(
    document_id: str,
    field_name: str,
    previous_value: str | None,
    corrected_value: str,
    crop_reference: str | None,
    reviewer: str,
    tenant_id: str = "default",
    reason: str | None = None,
    **provenance: Any,
) -> CorrectionExample:
    return CorrectionExample(
        document_id=document_id,
        field_name=field_name,
        previous_value=previous_value,
        corrected_value=corrected_value,
        crop_reference=crop_reference,
        reviewer=reviewer,
        corrected_at=datetime.now(UTC).isoformat(),
        tenant_id=tenant_id,
        reason=reason,
        **provenance,
    )


@dataclass(frozen=True)
class CorrectionDatasetManifest:
    schema_version: str
    generated_at: str
    source_path: str
    source_sha256: str
    split_seed: str
    split_percentages: dict[str, int]
    record_counts: dict[str, int]
    source_group_counts: dict[str, int]
    output_sha256: dict[str, str]
    runtime_acceptance_authority: bool = False


def _canonical_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_group(row: dict[str, Any]) -> str:
    return str(row.get("source_group_id") or row.get("source_document_sha256") or row["document_id"])


_PARTITION_IDENTITY_FIELDS = (
    "source_group_id",
    "source_document_sha256",
    "document_id",
    "claim_id",
    "patient_id",
    "provider_id",
    "batch_id",
)


def _partition_identities(row: dict[str, Any]) -> set[str]:
    return {
        f"{field_name}:{row[field_name]}"
        for field_name in _PARTITION_IDENTITY_FIELDS
        if row.get(field_name) not in (None, "")
    }


def assert_partition_disjoint(partitions: dict[str, list[dict[str, Any]]]) -> None:
    """Reject direct or derived entity overlap between learning partitions."""
    identities = {
        split: set().union(*(_partition_identities(row) for row in rows))
        if rows else set()
        for split, rows in partitions.items()
    }
    names = sorted(identities)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = identities[left] & identities[right]
            if overlap:
                kinds = sorted({item.split(":", 1)[0] for item in overlap})
                raise ValueError(
                    f"partition identity leakage between {left} and {right}: "
                    + ",".join(kinds)
                )


def _split_for(source_group_id: str, seed: str, percentages: dict[str, int]) -> str:
    point = int(sha256(f"{seed}:{source_group_id}".encode()).hexdigest()[:8], 16) % 100
    boundary = 0
    for name in ("train", "calibration", "holdout"):
        boundary += percentages[name]
        if point < boundary:
            return name
    raise AssertionError("split percentages must cover the full range")


def export_correction_dataset(
    source_path: Path,
    output_directory: Path,
    *,
    seed: str = "correction-dataset-v1",
    percentages: dict[str, int] | None = None,
) -> CorrectionDatasetManifest:
    """Export raw corrections for offline learning without runtime authority.

    A source group is assigned as a unit, preventing document/source leakage
    across training, calibration, and locked holdout sets.
    """
    split_percentages = percentages or {"train": 70, "calibration": 15, "holdout": 15}
    if set(split_percentages) != {"train", "calibration", "holdout"}:
        raise ValueError("splits must be train, calibration, and holdout")
    if any(value < 0 for value in split_percentages.values()) or sum(split_percentages.values()) != 100:
        raise ValueError("split percentages must be non-negative and sum to 100")

    source_bytes = source_path.read_bytes() if source_path.is_file() else b""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid correction JSON on line {line_number}") from exc
        if row.get("runtime_acceptance_authority") is True:
            raise ValueError(f"correction line {line_number} improperly claims runtime authority")
        row["usage_authority"] = "TRAINING_ONLY"
        row["runtime_acceptance_authority"] = False
        rows.append(row)

    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in split_percentages}
    groups: dict[str, set[str]] = {name: set() for name in split_percentages}
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        for identity in _partition_identities(row):
            if identity in owner:
                union(index, owner[identity])
            else:
                owner[identity] = index

    components: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        components.setdefault(find(index), []).append(row)
    for component in components.values():
        identities = sorted(set().union(*(_partition_identities(row) for row in component)))
        locked = any(
            row.get("holdout_member") is True or row.get("dataset_split") == "holdout"
            for row in component
        )
        split = "holdout" if locked else _split_for("|".join(identities), seed, split_percentages)
        for row in component:
            group = _source_group(row)
            assigned[split].append({**row, "dataset_split": split, "source_group_id": group})
            groups[split].add(group)
    assert_partition_disjoint(assigned)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for split, split_rows in assigned.items():
        payload = "".join(_canonical_json(row) + "\n" for row in split_rows).encode("utf-8")
        target = output_directory / f"{split}.jsonl"
        target.write_bytes(payload)
        output_hashes[target.name] = sha256(payload).hexdigest()
    manifest = CorrectionDatasetManifest(
        schema_version="correction-dataset-v1",
        generated_at=datetime.now(UTC).isoformat(),
        source_path=str(source_path),
        source_sha256=sha256(source_bytes).hexdigest(),
        split_seed=seed,
        split_percentages=split_percentages,
        record_counts={name: len(items) for name, items in assigned.items()},
        source_group_counts={name: len(items) for name, items in groups.items()},
        output_sha256=output_hashes,
    )
    (output_directory / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


class CorrectionMemory:
    """Read bounded, field-scoped exemplars from append-only correction memory."""

    def __init__(self, path: Path, *, limit: int = 3) -> None:
        self._path = path
        self._limit = max(0, limit)

    def exemplars(self, field_name: str, tenant_id: str = "default") -> list[dict[str, str]]:
        if self._limit == 0 or not self._path.is_file():
            return []
        selected: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("field_name") != field_name or row.get("tenant_id", "default") != tenant_id:
                continue
            previous = str(row.get("previous_value") or "")
            corrected = str(row.get("corrected_value") or "")
            if not corrected or (previous, corrected) in seen:
                continue
            selected.append({"observed": previous, "corrected": corrected})
            seen.add((previous, corrected))
            if len(selected) >= self._limit:
                break
        return list(reversed(selected))

    def promotion_candidates(
        self,
        tenant_id: str = "default",
        *,
        minimum_documents: int = 5,
        minimum_reviewers: int = 2,
        minimum_agreement: float = 0.95,
    ) -> list[CorrectionPattern]:
        """Identify patterns for holdout testing; this never activates a route."""
        if not self._path.is_file():
            return []
        observations: dict[tuple[str, str], list[dict]] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("tenant_id", "default") != tenant_id:
                continue
            field = str(row.get("field_name") or "")
            observed = str(row.get("previous_value") or "")
            corrected = str(row.get("corrected_value") or "")
            if field and corrected:
                observations.setdefault((field, observed), []).append(row)

        candidates: list[CorrectionPattern] = []
        for (field, observed), rows in observations.items():
            corrected_counts: dict[str, int] = {}
            for row in rows:
                corrected = str(row.get("corrected_value") or "")
                corrected_counts[corrected] = corrected_counts.get(corrected, 0) + 1
            corrected, occurrences = max(corrected_counts.items(), key=lambda item: item[1])
            documents = {str(row["document_id"]) for row in rows if row.get("document_id")}
            reviewers = {str(row["reviewer"]) for row in rows if row.get("reviewer")}
            agreement = occurrences / len(rows)
            eligible = (
                len(documents) >= minimum_documents
                and len(reviewers) >= minimum_reviewers
                and agreement >= minimum_agreement
            )
            candidates.append(CorrectionPattern(
                field, observed, corrected, occurrences, len(documents), len(reviewers), agreement, eligible,
            ))
        return sorted(
            candidates,
            key=lambda item: (-int(item.promotion_eligible), -item.occurrences, item.field_name),
        )
