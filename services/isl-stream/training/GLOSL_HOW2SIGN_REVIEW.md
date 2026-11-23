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
