# How2Sign source audit for SIGNBRIDGE

Checked against the official How2Sign project materials on 2026-11-23.

## What is confirmed

- The project describes How2Sign as a continuous ASL dataset with more than 80 hours of multimodal/multiview video and sentence-level alignment for more than 35,000 sentences.
- The project poster lists gloss annotations among the dataset resources.
- The official downloader exposes RGB video, sentence-level RGB clips, 2D keypoints, English translations, and re-aligned English translations as selectable modalities.

Sources:
- Project: https://how2sign.github.io/
- Project poster: https://how2sign.github.io/images/poster/How2Sign_Poster.pdf
- Official downloader: https://github.com/how2sign/how2sign-data/blob/main/download_how2sign.sh

## Important blocker discovered

The official downloader currently labels gloss annotations as `TODO`; gloss is not one of the selectable download modalities in its final modality list. Therefore, the public download script alone does **not** provide a confirmed, ready-to-import gloss CSV for this pipeline.

The English translation CSVs are not ASL gloss. SIGNBRIDGE must not relabel English sentences as gloss or train the gloss CTC model against them.

## What this means for the pipeline

1. RGB clips and English translation files can be acquired separately, subject to reviewing the dataset's current terms and storage requirements.
2. The current `prepare_asl_training_manifest.py` intentionally requires expert-reviewed gloss labels, explicit signer consent, approved permission, signer-disjoint splits, and local media paths.
3. Before training, obtain the gloss annotations through an official approved source or create/review them with qualified ASL annotators and appropriate signer permissions.
4. Keep downloaded media and any family recordings out of the public GitHub repository. Do not start a large download until the required storage, rights, and annotation path are confirmed.

## Current status

- Continuous ASL corpus identified: yes.
- Official download route identified: yes.
- Public downloader's gloss export confirmed: no (marked TODO).
- Ready-to-train, permission-cleared gloss manifest: no.
- Trained model: no.
- Live sign translation: no.
