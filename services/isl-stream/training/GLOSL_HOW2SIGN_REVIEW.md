# GloSL How2Sign ASL candidate review

Status: **candidate only — not approved for training or redistribution**.

This document records the checks required before importing any GloSL–How2Sign row into SIGNBRIDGE's ASL training pipeline. A public dataset page, a license label, or a non-empty `gloss` cell is not sufficient evidence of consent, annotation validity, or permission for this project's intended use.

## Candidate facts to verify against the current source

The dataset page describes 34,305 ASL-to-English samples (61.15 hours), with 30,814 train, 1,527 dev, and 1,964 test rows. It lists `annotations.csv`, `card.json`, pose files, and video files; the pose representation is described as 543-point MediaPipe Holistic. The listed license is CC BY-NC 4.0 and the upstream source is How2Sign.

Treat these as source-reported facts, not an independent rights determination. Review the current `card.json`, source dataset terms, upstream How2Sign terms, and attribution requirements before any use. Do not assume the repackager can grant rights it did not receive. The official How2Sign site describes access as research-only and says the material may not be used commercially; its paper reports IRB-approved collection and participant consent for making recordings publicly available for research. That is useful provenance, but it is not blanket proof that every repackaged file, derived feature, trained checkpoint, or intended app deployment is authorized. A private family app is not automatically covered merely because it is non-commercial: confirm the scope of the original and repackaging terms for this exact use before training.

## Metadata and gloss-semantics review (23 Nov 2026)

The currently published GloSL README confirms that the CSV schema contains separate `text`, `gloss`, and `gloss_type` columns, and describes the corpus as ASL-to-English with 34,305 sentence samples. However, the README does **not** define the allowed `gloss_type` values, explain their meanings, or state row-level counts by type. The public dataset page also describes the split as imported How2Sign video-disjoint splits; that is not evidence of signer-disjoint splits.

The upstream How2Sign materials describe gloss as a linguistic annotation: ASL linguists annotated signs in ELAN, using capitalized English gloss tokens as approximate labels for ASL signs. A gloss is explicitly not an English translation. This makes the upstream corpus a plausible *candidate* for continuous ASL gloss research, but it does not prove that GloSL's `gloss` column is a faithful export of that tier or that every row has the same annotation type.

**Result of this review:** the schema is promising, but label suitability is **not yet established**. Do not train on the `gloss` column until the actual CSV's `gloss_type` distribution, sample labels, row quality, and `card.json` transformation/provenance details have been inspected. No annotation CSV or media corpus was downloaded as part of this review, so no row-level statistics are claimed.

## Local annotation inventory

Do not clone or download the full video/pose corpus just to inspect its metadata. Obtain only the annotation CSV and provenance files through an allowed route, then run:

```bash
python -m training.audit_glosl_annotations \
  --annotations /path/to/annotations.csv \
  --output /path/to/private-review/glosl-audit.json
```

Review all of the following in the report and source files:

- Exact `gloss_type` values and their documented meanings.
- Whether each gloss is a human-annotated ASL gloss, a generated label, or another representation. The How2Sign paper reports that ASL linguists produced gloss annotations using ELAN; verify that GloSL's `gloss` and `gloss_type` fields preserve that annotation source and tier semantics rather than assuming every non-empty value is an equivalent label.
- Empty/malformed glosses, IDs, signer IDs, and split labels.
- Duplicate IDs and video references crossing splits.
- Signer overlap across train/dev/test. The published split description is video-disjoint; do not describe it as signer-disjoint unless the audit proves that.
- Timestamp, frame-rate, frame-count, and video/pose alignment for a sampled set of clips.
- Full `card.json` provenance, source revision, transformation history, and upstream terms.

The audit script deliberately reports `training_approved: false`. It is an inventory tool, not an authorization mechanism.

## Required authorization ledger

Before converting any row to SIGNBRIDGE's training manifest, maintain a separate, private, reviewed ledger keyed by clip ID and signer ID. The ledger must contain real evidence and a reviewer/date for:

1. The applicable upstream and repackaging license/terms, with permitted use for this private non-commercial project confirmed.
2. Explicit consent/authorization covering the source recording and the intended processing/training use, or authoritative documentation that establishes the relevant permission.
3. Required attribution and citation text.
4. Permission to retain the derived pose/features and resulting model/checkpoints.
5. Any restrictions on redistribution, publication, or deletion.

Never synthesize `consent=yes`, `permission_status=approved`, or a license ID from the dataset's public visibility or README badge. Do not put private consent documents, family videos, or authorization records in this public repository.

## Import gates

A reviewed adapter may map GloSL `dev` to SIGNBRIDGE `validation` only after all checks pass. It must preserve original clip IDs, signer IDs, timestamps, source URLs, attribution, and the exact gloss label/type. It must not use the English `text` field as gloss.

Training remains blocked until:

- gloss semantics and provenance are verified;
- authorization is documented for every included clip/signer;
- all required train/validation/test splits are present and signer-disjoint for evaluation;
- media/pose alignment and feature format are validated;
- the private manifest passes the existing strict consent and path checks.

If signer overlap exists, do not silently accept it: create a genuinely signer-disjoint split only if the source permissions and data volume allow it, otherwise report the limitation and do not claim signer-independent evaluation.

## Current decision

**Do not download the 85 GB corpus or start training yet.** First inspect the small annotation/provenance files and obtain the required permissions. In particular, confirm whether the upstream research-only grant and the GloSL repackaging terms permit this specific private application use and retention of derived features/checkpoints. This review note does not certify legal rights, signer consent, gloss quality, or model readiness.


## Machine-readable GloSL card inspection (23 Nov 2026)

The public Hugging Face `card.json` was inspected directly (revision shown on the file page: `d714db7`). It confirms:

- `annotation_types` is an empty array, and `modalities` lists pose, text, and video — **not gloss**. This is a material warning: the README's generic CSV schema includes a `gloss`/`gloss_type` column, but the machine-readable card does not declare gloss as an annotation type. Do not assume the column contains human-annotated gloss labels.
- `signers` is null, so the card supplies no signer count.
- The split policy is explicitly imported How2Sign official train/validation/test and **video-disjoint**; signer-disjointness is not established.
- The provenance reports 34,305 clips and 133 clips missing video.
- The card says source RGB clips are not rehosted and should be obtained from How2Sign; its `video.hosted` field is true while the note says not rehosted. Treat this internal inconsistency as unresolved metadata, not as evidence that video is available or authorized.
- The card's access block asserts public visibility, CC BY-NC 4.0, redistributable=true, and reupload_ok=public-nc. These are repackager-provided assertions, not independent proof that upstream How2Sign grants every downstream use, nor proof of participant consent for this application's training and checkpoint retention.

The CSV's 6.8 MB file page confirms the file exists but its contents are too large for the page renderer. **No row-level CSV values or gloss_type counts were inspected.** The card's empty `annotation_types` makes the next step more cautious: retrieve and sample the actual CSV before deciding whether it is usable at all.

Updated decision: **not approved; gloss supervision is currently unsubstantiated by the machine-readable card.** Do not train or prepare a training manifest from this dataset on the strength of the README schema alone.


## Annotation CSV retrieval attempt (23 Nov 2026)

The Hugging Face Data Studio preview was opened. It reports a train preview of 114 rows, but the rendered table exposes only the `id` column and clip IDs; it does not expose `gloss`, `gloss_type`, `signer`, `text`, or timing columns. The page also reports that its DuckDB WASM SQL console failed to initialize. Direct browser retrieval of the CSV and the datasets-server rows endpoint was not available through the current inspection route.

Therefore:
- No CSV row contents were inspected.
- No `gloss_type` distribution or gloss examples can be reported.
- No signer overlap, empty gloss, timing, or video-reference counts can be independently recomputed.
- The previous claim that the card's empty `annotation_types` makes gloss supervision unsubstantiated remains a caution, not proof that every CSV gloss is invalid.

Next retrieval route: obtain only `annotations.csv` (6.8 MB) and `card.json` from the Hugging Face file download controls or HF CLI, then run the repository audit locally. Do not fetch `videos/` or `poses/` for this metadata-only step. Training remains blocked.
