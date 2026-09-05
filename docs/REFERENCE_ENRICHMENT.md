# Governed reference enrichment

The enrichment pipeline closes review-routed fields only with independently
authorized evidence. OCR, Azure, evaluation truth and the current candidate are
never reference sources.

## Default operation

All providers in `config/reference_enrichment.yaml` are disabled. A default run
makes no external calls, preserves all pending decisions and emits
`AWAITING_AUTHORIZED_REFERENCE_SOURCE`.

```powershell
python -m evaluation.run_reference_enrichment `
  --workbook reference_decisions_governed_v3_corrected.xlsx `
  --output evaluation_results/reference_enrichment
```

## Provider promotion

An adapter may be enabled only after authorization, secrets, region, retention,
dataset version and lineage have been approved. Tier-A member matching requires
member ID plus DOB and a compatible name. The approved fallback requires DOB,
strong name and ZIP. Provider matching requires a checksum-valid exact NPI.
Name-only matching is prohibited.

Synthetic fixtures are always `TEST_ONLY`, never evaluation eligible. Downstream
lineage containing `extraction-v2`, `azure-fallback`, `unreviewed-ocr` or
`cdp-prediction` is rejected, including indirect lineage.

## Public reference snapshots

The production importer supports four public, non-PHI sources:

- CMS NPPES provider data: NPI, provider name, taxonomy, status and practice address.
  Tax ID/EIN is deliberately never copied into the snapshot.
- CDC ICD-10-CM diagnosis releases.
- CMS HCPCS Level II quarterly releases. Numeric CPT codes are not included because
  CPT is licensed by the AMA.
- CMS place-of-service codes.

Official release pages:

- `https://download.cms.gov/nppes/NPI_Files.html`
- `https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/`
- `https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/quarterly-update`
- `https://www.cms.gov/medicare/coding-billing/place-of-service-codes/code-sets`

Download a release, record its SHA-256 in the release ticket, and import it without
printing source records:

```bash
python -m scripts.import_public_reference \
  --dataset icd10cm \
  --source '/secure/releases/icd10cm-Code Descriptions-2026.zip' \
  --source-url 'https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/icd10cm-Code%20Descriptions-2026.zip' \
  --expected-sha256 <64-hex-release-sha256> \
  --destination config/reference_snapshots/icd10cm \
  --version FY2026 \
  --expires-at 2026-10-01T00:00:00+00:00 \
  --source-contract-id public-reference-policy-v1 \
  --approved-by data-governance
```

The command also accepts `--url` for a direct download, but only HTTPS URLs on CMS
or CDC hosts. It rejects unpinned downloads, unapproved redirect hosts, artifacts
over 4 GiB, duplicate keys and empty releases. Output is an indexed SQLite snapshot
plus a strict manifest containing the source SHA-256, normalized-store SHA-256,
release version, effective period, expiry and approval lineage.
The place-of-service importer also accepts the official CMS HTML code-set page,
which is the machine-readable table CMS publishes alongside its PDF download.

To activate a completed snapshot, set only its corresponding provider to
`authorized: true` and `enabled: true`, then run:

```bash
python scripts/check_reference_readiness.py config/reference_enrichment.yaml
```

Runtime lookups are field-scoped and read-only. A missing file, changed checksum,
not-yet-effective release or expired release becomes a provider error and abstains;
it cannot create machine acceptance. Member eligibility, provider Tax ID, CPT and
UB-04/NUBC revenue data remain private/licensed connectors and must not be inferred
or fabricated from public data.

## Historical backfill

`evaluation.run_reference_historical_backfill` seals the prediction hash and
inference timestamp before any finalized truth can be retrieved. The current
implementation intentionally stops at `SEALED_AWAITING_AUTHORIZED_HISTORICAL_SOURCE`.
