# Phase 9A runtime performance remediation

Status date: 2026-08-25. Branch: `phase9-external-1000-execution`.
Base Git SHA: `bffc616775f4df818d065fe4060ca901dd4f669b`.

The frozen 1,000-page benchmark was not executed. The development harness used
30 truth-blind pages selected with seed `20260824`, allocation Group A/B/C/D =
15/6/5/4, and selection SHA-256
`e29d5956c84fe51d9d50612b9b679a4906b984f94833c0f9d397b305d894b4f6`.

## Integration repairs

| Defect | Root cause | Exact repair | Regression test | GitHub at base SHA |
|---|---|---|---|---|
| Stale `_align_or_rescale` import | The runner retained a removed permissive geometry API after the worker moved to fail-closed `_resolve_geometry`. | Construct verified form identity, call `_resolve_geometry`, and run fixed ROI OCR only when registered geometry is authorized. | `test_production_holdout_runner_imports_with_current_geometry_api` | Absent; local branch patch only |
| Numeric TIFF suffix rejected | Intake trusted `.tif`-style suffixes although the frozen TIFF corpus uses `.001`, `.002`, etc. | Accept standard suffixes or little-/big-endian TIFF magic. | `test_manifest_accepts_tiff_with_numeric_page_suffix` | Absent; local branch patch only |
| `ZipInfo` sort crash | Python 3.14 does not order `ZipInfo` instances. | Sort archive members by `member.filename`. | `test_truth_blind_dataset_orders_zip_members_by_filename` | Absent; local branch patch only |

## Baseline

Reference host: 8 logical CPUs, CPU-only OCR. Complete OCR environment:
Python 3.11.15, PaddleOCR 2.7.3, PaddlePaddle 2.6.2, RapidOCR ONNX,
NumPy 1.26.4, and OpenCV 4.9. No internal thread cap was applied.

| Metric | Baseline |
|---|---:|
| Pages / failures | 30 / 0 |
| Elapsed | 3,608.41 s |
| Mean page wall | 119.89 s |
| P50 | 17.17 s |
| P95 | 138.81 s |
| P99 | 2,303.39 s |
| Mean CPU/page | 101.92 s |
| Throughput | 0.00831 pages/s |
| Mean worker RSS high-water | 729.53 MiB |
| P95 worker RSS high-water | 969.36 MiB |
| RapidOCR calls | 225 (7.50/page) |
| PaddleOCR calls | 19 (0.63/page) |
| Tesseract calls | 30 inferred from one routing call/page; counter missing in base runner |

Cold page 1 was 138.81 seconds. CMS pages averaged 93.91 seconds and their OCR
stage averaged 74.36 seconds. Non-CMS pages had a 7.04-second median. One Paddle
fallback stalled for 2,299.08 seconds of OCR wall time but used only 11.70 CPU
seconds for the entire page, consistent with a blocking model/cache/runtime event.

## Stage bottlenecks

Preparation sub-stages overlap the preparation total and are excluded from the
percentage denominator to avoid double counting.

| Stage | Total wall s | Mean s/page | P95 s | Share |
|---|---:|---:|---:|---:|
| OCR | 3,262.79 | 108.76 | 126.53 | 90.51% |
| Classification | 272.45 | 9.08 | 42.51 | 7.56% |
| Registration | 50.35 | 1.68 | 5.36 | 1.40% |
| Preparation | 8.88 | 0.30 | 0.47 | 0.25% |
| Layout | 1.42 | 0.05 | 0.13 | 0.04% |
| Evidence decision | 0.22 | 0.007 | 0.029 | <0.01% |
| Claim decision | 0.01 | <0.001 | 0.002 | <0.01% |

Measured diagnosis: CMS processing is CPU-bound in repeated regional RapidOCR;
the process performs up to 25 detector/recognizer calls after full-page Tesseract
routing OCR. The Paddle outlier is blocking/I/O-like. One uncapped worker saturated
the 8 CPUs. Two workers with cap 2 saturated CPU and reduced free host memory to
324 MiB after four pages. The current system is both CPU-bound and memory-constrained;
sequential orchestration is secondary to engine cost, and unrestricted nested
threading prevents safe page parallelism.

Template registry, reference images, OCR adapters, criticality policy, and claim
decision policy are now cached once per process. Page-specific data is not cached.
Preparation occurs once per page. Image decode, orientation, deskew, and denoise
together are below 0.3% and are not worthwhile optimization targets.

## Worker and thread experiments

| Configuration | Result |
|---|---|
| A: workers=1, defaults | Complete baseline; 0.00831 pages/s, P95 138.81 s |
| B: workers=2, cap=2 | Resource reject after 4 atomic checkpoints; 99.8% CPU, 324 MiB free |
| C: workers=4 | Preflight reject: measured P95 worker RSS implies about 3.8 GiB worker demand |
| D: workers=6 | Preflight reject: measured P95 worker RSS implies about 5.7 GiB worker demand |
| E: workers=8 | Preflight reject: measured P95 worker RSS implies about 7.6 GiB worker demand |

The higher configurations were deliberately not launched into known memory exhaustion.
This is a measured capacity rejection, not an untested performance claim.

## Sequential versus optimized

Thread caps are the only completed semantic-safe runtime optimization. On the same
frozen CMS page, cap 1 produced 140.77 s wall / 124.64 s CPU and cap 2 produced
109.16 s wall / 190.50 s CPU. Both were semantically identical to baseline.

| Metric | Before aggregate | After cap=2 microbenchmark |
|---|---:|---:|
| Mean latency | 119.89 s | 109.16 s (one CMS page) |
| P50 / P95 / P99 | 17.17 / 138.81 / 2303.39 s | Not claimed from one page |
| CPU sec/page | 101.92 s | 190.50 s |
| Pages/sec | 0.00831 | 0.00916 |
| Peak memory | 969.36 MiB P95 | Not independently sampled |
| RapidOCR calls | 7.50/page | 25 on CMS page, unchanged |
| PaddleOCR calls | 0.63/page | 0, route-specific |
| Tesseract calls | 1/page | 1/page, unchanged |

No staged structural-first router or shared full-page extraction observation was
promoted. Those changes can alter route or field values and require a complete
semantic-equivalence development run. The dominant next architectural task is to
replace per-field detector invocation with one canonical page observation plus
selective regional recognition, then prove output equivalence.

## Correctness and safety

Thread-cap comparisons were `IDENTICAL` for document ID, route, schema, field names,
normalized values, field dispositions, and claim disposition. No confidence or
decision threshold changed. The full runner now supports atomic page checkpoints,
deterministic ordering, duplicate prevention, and resume fingerprints binding corpus,
runtime manifest, code SHA, command, and expected page count. The final prediction
artifact and freeze are created only after complete coverage.

Unknown CPU, GPU, reviewer, cloud, and fixed-run prices remain `NOT_PROVIDED`.
The report API separately emits CPU seconds/page, wall seconds/page, peak memory,
and OCR calls/page without inventing monetary rates.

Tests: 54 passed, 1 existing dataset-dependent skip. Focused new/Phase 9 tests:
21 passed. No full 1,000-page run was started.

## Decision

`PERFORMANCE_GATE: REJECT`

`PHASE9_FULL_RUN: NOT_READY`

The P95 target is missed by 27.8x and throughput target by 120.3x. The frozen
benchmark remains untouched for promotion purposes.
