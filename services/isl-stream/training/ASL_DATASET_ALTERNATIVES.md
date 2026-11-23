# SIGNBRIDGE ASL dataset alternatives

Review date: 23 Nov 2026

## Decision

**Best new candidate for a small, permission-gated ASL prototype: CLERC Épée v0.4.** It is the first candidate found that publishes ASL gloss segments with timing, signer IDs, and ready-made keypoints in a small download. It is not approved for SIGNBRIDGE training yet: its public licence is CC BY-NC-SA 4.0, and the exact use of a trained checkpoint in a private family app plus share-alike obligations must be clarified with CLERC. The release is also a small pilot, not a broad continuous-translation corpus.

Keep production ASL inference disabled until permission, data audit, model adaptation, and held-out signer evaluation are complete.

The failed GloSL-How2Sign CSV audit is recorded in [GLOSL_HOW2SIGN_REVIEW.md](./GLOSL_HOW2SIGN_REVIEW.md): all 34,305 rows have empty gloss, gloss_type, signer, and usable timestamps. It is not a supervised gloss-training source as supplied.

**Important original-How2Sign finding:** the GloSL CSV failure does not prove that the original How2Sign project has no gloss annotations. However, the official `how2sign-data/download_how2sign.sh` script offers RGB videos/clips, frontal 2D keypoints, and English translation CSVs; its own final TODO section explicitly says `Gloss annotations`. Gloss is not an available modality in the published download script. Do not mistake English translation CSVs for ASL gloss labels.

## Candidate comparison

| Dataset | What it can support | Strengths | Blocking limits |
|---|---|---|---|
| **CLERC Épée v0.4** | Small continuous phrase/gloss prototype and temporal gloss decoding | 2,400 ASL clips; 8 Deaf signers; 179.9 minutes; 1,501 unique glosses; gloss segments have start/end times; signer IDs, FPS, frame counts and 128 keypoints/frame; no raw video download; written signer consent reported; includes a parallel 5-signer phrase grid and 3 separate signer registers | CC BY-NC-SA 4.0. Ask CLERC whether non-revenue private family-app use, model checkpoint retention, and share-alike obligations are allowed. Hugging Face viewer currently exposes one `train` split, not ready-made train/validation/test splits; we must construct signer-disjoint evaluation carefully. Pilot-scale and 179.9 minutes are not enough to claim broad continuous ASL translation. Keypoint schema (128×3) differs from our current 1,699-feature MediaPipe Holistic pipeline; it needs a dedicated adapter/model and cannot be fed into current training code unchanged. |
| ASLLRP SignStream 3 / NCSLGR | Linguistically annotated continuous ASL recognition research | Continuous signing, multiple synchronized camera views, XML annotations, manual English-based gloss labels and start/end frames; recently expanded corpus | Terms permit research/education but prohibit redistribution without permission and commercial use without explicit permission. Need permission for private family-app deployment and derived checkpoints. Need audit actual downloadable corpus metadata, signer counts, file formats and split options before selecting it. |
| ASL Citizen v1.0 | Isolated-sign dictionary lookup | 83,399 videos, 2,731 signs, 52 signers; consented collection and user-independent splits | Microsoft Research License limits use to non-commercial, non-revenue-generating research; prohibits distributing data or modifications and providing Materials as a standalone hosted solution. Not continuous sentence recognition. Obtain written clarification before use. |
| WLASL | Isolated word-level ASL recognition | Gloss labels, signer IDs, frame boundaries, FPS and splits in metadata | C-UDA agreement and source-video rights need review; academic/computational-use restrictions. Not continuous sentence recognition. |
| ChicagoFSWild+ | Continuous fingerspelling recognition | 55,232 naturally occurring fingerspelling sequences; labels, temporal boundaries and signer metadata | Fingerspelling only, not general ASL sentences; very large video archive. Rights must be checked before use. |
| Original How2Sign | Continuous ASL sentence/gloss research | >80 hours and multiple modalities; paper describes gloss annotation | Research-only / CC BY-NC 4.0. Official download script currently marks gloss annotations TODO and does not expose a gloss-download option. Ask maintainers whether gloss files are available separately and whether private family-app use and derived checkpoints are permitted. |
| BOBSL | Continuous BSL recognition | Human-annotated continuous sequences with sentence/gloss annotations | British Sign Language, not ASL; do not mix it into the ASL model. |

## Épée technical audit gates (before training)

1. Obtain written permission/clarification from CLERC for this exact non-revenue family accessibility app, model training, retaining an embedded checkpoint, and whether model weights are a share-alike adaptation.
2. Retrieve only the small metadata/keypoint release after confirming licence terms; no raw signer video is included in this release.
3. Audit all 2,400 rows: missing/duplicate clip IDs, signer counts, per-signer clip counts, segment count, valid start/end bounds against frame count/FPS, empty glosses, hand-slot verification, non-finite keypoints, and phrase overlap.
4. Build strict signer-disjoint folds. Do not randomly split clips: the same signer and same parallel phrase may otherwise leak across train and test. The public viewer lists one `train` split; the dataset's reported external held-out-signer benchmark is not a ready-made split in the downloadable table.
5. Create a dedicated 128-landmark feature adapter. The current MediaPipe Holistic extractor emits 1,699 features; those arrays are incompatible without a deliberate schema mapping and validation.
6. Train a small gloss-sequence baseline, report held-out signer WER/CER and per-gloss metrics, and compare with a simple baseline. Do not call it general translation or live recognition.
7. Keep English translation as a separate downstream task; gloss sequences are not direct English sentences.

## Other recommended permission requests

- Ask ASLLRP for written permission to use its continuous annotated corpus and derived checkpoint in this private family app, and for an inventory of current signer-disjoint partitions and XML annotation downloads.
- Ask How2Sign maintainers whether original gloss annotations are available separately, with signer IDs, clip IDs, timing/alignment and splits.

## Keep tasks separate

(a) isolated ASL sign dictionary lookup, (b) continuous ASL sentence recognition, and (c) fingerspelling are different tasks. A dictionary classifier must never be represented as sentence-level live translation. Do not train on YouTube tutorial footage or user-provided playlist videos without explicit rights and signer consent for ML training.

## Sources

- CLERC Épée v0.4 overview and dataset details: https://clerc.io/data
- CLERC Épée v0.4 dataset card and metadata preview: https://huggingface.co/datasets/CLERC-DATA/epee
- CLERC commercial licensing contact: florian@clerc.io
- ASLLRP corpus and terms: https://www.bu.edu/asllrp/
- ASLLRP continuous corpus information: https://www.bu.edu/asllrp/about-datasets.pdf
- Microsoft Research ASL Citizen overview: https://www.microsoft.com/en-us/research/project/asl-citizen/
- ASL Citizen Microsoft Research License: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
- WLASL repository and C-UDA terms reference: https://github.com/dxli94/WLASL
- ChicagoFSWild dataset page: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm
- Official How2Sign page and license: https://how2sign.github.io/
- Official How2Sign download script: https://github.com/how2sign/how2sign-data/blob/main/download_how2sign.sh
- How2Sign paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.pdf
- BOBSL: https://www.robots.ox.ac.uk/~vgg/data/bobsl/

This is a technical screening note, not legal advice or a rights determination.
