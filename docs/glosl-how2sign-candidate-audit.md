# GloSL How2Sign ASL gloss candidate — preliminary audit

Checked on 2026-11-23. This is a candidate source for further verification, not an approved training dataset.

## Why it matters

The Hugging Face dataset card for `pedroodb/glosl-how2sign` describes:
- 34,305 ASL-to-English samples (61.15 hours), with train/dev/test counts.
- An `annotations.csv` schema containing `id, split, text, gloss, gloss_type, signer, start, end, fps, n_frames, video_ref, pose_file, video_file`.
- Video and MediaPipe Holistic pose files, with pose data re-extracted from How2Sign frontal sentence clips.
- A declared `CC BY-NC 4.0` license and attribution to How2Sign.

Dataset card: https://huggingface.co/datasets/pedroodb/glosl-how2sign
Source project: https://how2sign.github.io/
Original How2Sign paper: Duarte et al., CVPR 2021.

## Why this is not yet training-approved

The dataset card is a third-party repackaging, not the original How2Sign release. Before importing any rows or media, independently inspect and record:
1. The complete `card.json` provenance and diagnostics, including the exact source commit/release and any transformations.
2. The actual `gloss_type` values and annotation conventions; confirm the labels are ASL gloss, not English text or generated pseudo-gloss.
3. The source and rights for each video and pose file, including whether the repackager's redistribution is within the upstream license.
4. The original signer consent and whether it covers this repackaging and model training.
5. The exact signer-disjoint train/dev/test policy and any duplicate or near-duplicate clips.
6. Frame count, FPS, start/end semantics, video/pose alignment, and missing/corrupt file rates.
7. Whether the terms permit this private family app's intended use and any distribution of a trained checkpoint.

A dataset card's license field is a claim to verify, not legal advice or automatic permission. Do not download the 61-hour corpus until storage, rights, provenance and consent have been reviewed.

## SIGNBRIDGE import gate

- Do not use English `text` as gloss.
- Do not mark permission as approved or consent as yes merely to satisfy the manifest validator.
- Preserve original signer IDs and source splits; never randomly split clips from the same signer across partitions.
- Import only rows with verified gloss labels and approved source permission/consent.
- Keep media, pose files, derived features and checkpoints out of the public GitHub repository.
- The existing strict manifest converter remains the only path into training; this note does not alter its gates.

## Status

- Candidate continuous ASL gloss corpus identified: yes.
- Dataset card advertises gloss, split, signer and timestamps: yes.
- Full provenance and actual labels independently verified: no.
- Rights/consent cleared for SIGNBRIDGE training: no.
- Imported approved training manifest: no.
- Trained model or live translation: no.
