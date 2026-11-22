# ASL continuous-data preparation bridge

The ASL corpus metadata validator and the landmark CTC trainer intentionally use
different schemas. This bridge only converts a reviewed metadata CSV into the
input schema expected by `extract_landmarks.py`.

## Safety gates

The source CSV must include all fields in
`services/isl-stream/training/asl_corpus_manifest.py`, plus:

- `media_path`: a path relative to the private `--data-root`
- `consent`: explicit `yes`, `true`, or `1`

Only rows with `transcript_kind=gloss`, `permission_status=approved`, explicit
consent, and RGB video are accepted. An English translation is never converted
into a gloss. The bridge preserves source dataset, source clip ID, timestamps,
license, attribution and source URL in its intermediate CSV.

**Important clip boundary rule:** each `media_path` must already point to a
video file containing exactly the clip represented by that row. The bridge
retains `start_ms` and `end_ms` as provenance but does not crop a long source
video. Do not point it at a full recording and assume the timestamps will be
applied. Prepare the authorized clip files first, or add a separately reviewed
timestamp-aware extraction stage.

## Commands

From the repository root:

```bash
python services/isl-stream/training/asl_corpus_manifest.py /path/to/asl_metadata.csv

python services/isl-stream/training/prepare_asl_training_manifest.py \
  --manifest /path/to/asl_metadata_with_media.csv \
  --data-root /path/to/private/asl-data \
  --output /path/to/private/asl-extractor-input.csv

python services/isl-stream/training/extract_landmarks.py \
  --manifest /path/to/private/asl-extractor-input.csv \
  --data-root /path/to/private/asl-data \
  --output-manifest /path/to/private/asl-features.csv \
  --max-fps 30

python services/isl-stream/training/train_ctc.py \
  --manifest /path/to/private/asl-features.csv \
  --data-root /path/to/private/asl-data \
  --language ASL \
  --output /path/to/private/asl-baseline.pt \
  --causal
```

The extractor's output `.npy` files, manifests and checkpoints should remain
private and must not be committed to Git. The trainer requires genuine
expert-reviewed gloss labels and signer-disjoint train/validation/test splits.
It reports gloss recognition metrics; it does **not** produce sentence-level
English/Hindi translation or a production-ready live interpreter.

## Dataset note

How2Sign is a continuous ASL resource with more than 80 hours of multimodal
videos. Its official page states that it is for research purposes and uses
CC BY-NC 4.0; the page also warns that English translation alignment may not
perfectly match ASL clips. Confirm the current terms and the exact intended use
before downloading or processing any media. This project does not automatically
treat public availability as permission for every use.

- Official dataset and terms: https://how2sign.github.io/
- CVPR 2021 paper: https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html

No corpus has been downloaded, no signer consent has been inferred, and no ASL
model has been trained by adding this bridge.
