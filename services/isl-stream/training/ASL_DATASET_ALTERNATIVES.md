# SIGNBRIDGE ASL dataset alternatives

Review date: 23 Nov 2026

## Decision

No replacement dataset is approved for training or embedding in the SIGNBRIDGE family app yet. Keep ASL model inference disabled. Do not download large video archives until the intended-use terms are clarified.

The failed GloSL-How2Sign CSV audit is recorded in [GLOSL_HOW2SIGN_REVIEW.md](./GLOSL_HOW2SIGN_REVIEW.md): all 34,305 rows have empty gloss, gloss_type, signer, and usable timestamps. It is not a supervised gloss-training source as supplied.

**Important original-How2Sign finding:** the GloSL CSV failure does not prove that the original How2Sign project has no gloss annotations. However, I inspected the official `how2sign-data/download_how2sign.sh` script. It offers RGB videos/clips, frontal 2D keypoints, and English translation CSVs; its own final TODO section explicitly says `Gloss annotations`. Gloss is not an available modality in the published download script. Therefore, original How2Sign gloss annotations may exist elsewhere or be obtainable from the maintainers, but they are not currently exposed by the official download script. Do not mistake English translation CSVs for ASL gloss labels.

## Candidate comparison

| Dataset | What it can support | Strengths | Blocking limits |
|---|---|---|---|
| ASL Citizen v1.0 | Isolated-sign dictionary lookup | 83,399 videos, 2,731 signs, 52 signers; consented crowd-sourced collection; user-independent train/validation/test splits | Microsoft Research License limits use to non-commercial, non-revenue-generating research; prohibits distributing data or modifications and providing Materials as a standalone hosted solution. SIGNBRIDGE is a family-facing app, so do not assume an embedded model or hosted app is permitted. Obtain written clarification from Microsoft before use. Not continuous sentence recognition. |
| WLASL | Isolated word-level ASL recognition | Gloss labels, signer IDs, frame boundaries, FPS and splits are provided in metadata | C-UDA agreement must be read and accepted; source states academic/computational use only and no commercial usage. Public web videos may be unavailable or rights-restricted. Label quality concerns exist. Not continuous sentence recognition. Do not use in SIGNBRIDGE without confirming that this private app use and derived checkpoint are permitted. |
| ChicagoFSWild+ | Continuous fingerspelling recognition | 55,232 naturally occurring fingerspelling sequences; processed labels, start times, frame counts, partitions and signer IDs documented | This is fingerspelling, not general ASL sentences. Video archive is 82 GB. Obtain and review applicable terms and source-video rights before download or use. |
| Original How2Sign | Continuous ASL sentence/gloss research | Official project describes >80 hours and multiple modalities; paper describes gloss annotation | Research-only / CC BY-NC 4.0 terms; official download script currently marks gloss annotations TODO and does not expose a gloss-download option. English translation CSVs are not gloss. Ask maintainers whether gloss files are available separately and whether this private family app and derived checkpoints are permitted. |
| BOBSL | Continuous BSL recognition | Human-annotated continuous sequences with sentence/gloss annotations | British Sign Language, not ASL; do not mix it into the ASL model. |

## Recommended path

1. Ask Microsoft whether ASL Citizen's research-only license permits a non-revenue private family accessibility app to run an embedded model trained on the data, and whether retaining that checkpoint is permitted. No videos or checkpoint may be redistributed.
2. Ask How2Sign maintainers whether original gloss annotation files can be made available separately, with documentation for signer IDs, clip IDs, timing/alignment and splits; also clarify research-only terms for this exact non-revenue family app and retention of derived pose features/checkpoints.
3. If an annotation source is made available with permission, audit label coverage, signer-disjoint splits, clip availability, and temporal alignment before retrieving any video. Do not download the full archive first.
4. Keep separate model tasks: (a) isolated ASL sign dictionary lookup, (b) continuous ASL sentence recognition, and (c) fingerspelling. A dictionary classifier must never be represented as sentence-level live translation.
5. Do not train on YouTube tutorial footage or user-provided playlist videos without explicit rights and signer consent for ML training.

## Sources

- Microsoft Research ASL Citizen overview: https://www.microsoft.com/en-us/research/project/asl-citizen/
- ASL Citizen dataset description and split information: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/
- ASL Citizen Microsoft Research License: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
- WLASL repository and C-UDA terms reference: https://github.com/dxli94/WLASL
- ChicagoFSWild dataset page: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm
- Official How2Sign page and license: https://how2sign.github.io/
- Official How2Sign download script: https://github.com/how2sign/how2sign-data/blob/main/download_how2sign.sh
- How2Sign paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.pdf
- BOBSL: https://www.robots.ox.ac.uk/~vgg/data/bobsl/

This is a technical screening note, not legal advice or a rights determination.
