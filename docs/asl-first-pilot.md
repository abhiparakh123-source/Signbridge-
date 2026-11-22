# SIGNBRIDGE ASL-first research pilot

**Priority update:** ASL is the first model-development track while the ISL path is being investigated. ISL remains a separate supported-language target; this does not mean ASL signs can interpret ISL signing. A signer must actually use ASL for the ASL model to be useful.

## Dataset candidates and constraints

### Primary continuous-signing candidate: How2Sign

- Official project: https://how2sign.github.io/
- Describes more than 80 hours of continuous ASL video with English transcripts, depth and pose modalities.
- The project states the dataset is available for research purposes only and publishes its material under CC BY-NC 4.0. Preserve attribution, license link and modification notices.
- The project warns that sentence clips segmented from source video may not align perfectly with English translations; manually re-aligned timestamps are also provided.
- The official page lists very large RGB downloads (hundreds of GB per split/view). Start with the released pose/keypoint and clip metadata where permitted; do not download the full RGB corpus by default.
- This is the leading candidate for a continuous ASL prototype, but English transcript alignment is not a gloss annotation and does not by itself create reliable ASL gloss targets. Do not train a gloss CTC model by pretending English transcript words are ASL glosses.

### Complementary continuous linguistic corpus: ASLLRP / NCSLGR

- Project: https://www.bu.edu/asllrp/
- Continuous signing data include sentences, stories and dialogues, multiple camera views and linguistic annotations.
- Access terms state research/education use; redistribution and commercial use are restricted without permission. Review and accept the current terms before access.
- This is useful for understanding sign-level and non-manual annotations, subject to access and license constraints.

### Isolated vocabulary only: WLASL, MS-ASL, ASL Citizen

- WLASL: https://github.com/dxli94/WLASL
- MS-ASL: https://www.microsoft.com/en-us/research/project/ms-asl/
- ASL Citizen: https://www.microsoft.com/en-us/research/project/asl-citizen/
- These are isolated-sign recognition resources, not continuous sentence translation. Check each dataset's official terms and source-video rights before use. Use them only for a bounded vocabulary/encoder experiment, not as proof of fluent conversation translation.

## Pilot sequence

1. Confirm current official terms and access eligibility for How2Sign and ASLLRP before downloading or training.
2. Build a dataset-specific metadata importer that preserves source IDs, signer IDs, original splits, sentence IDs, timestamps, transcript provenance and license/attribution fields.
3. Use only released modalities and annotations that the dataset terms permit. Do not scrape source videos or bypass access controls.
4. Keep signer-independent splits and detect duplicate source clips across partitions.
5. Treat English transcript as a translation target, not ASL gloss. If gloss-level supervision is unavailable, choose a sequence-to-sequence translation baseline with documented alignment, or obtain expert gloss annotations; do not relabel transcript words as signs.
6. Evaluate sentence-level outputs and latency on held-out signers. Isolated-sign scores are reported separately.
7. Keep ASL and ISL vocabularies, checkpoints, metrics and model identifiers separate.

## Release rule

The ASL pilot is a research track until a real checkpoint is trained, independently evaluated with fluent Deaf ASL signers and integrated into the camera pipeline. The live service must continue to report recognition unavailable until those gates pass. ASL output must never be presented as ISL interpretation.


## ASL continuous-corpus metadata gate (implemented)

A metadata-only validator is now available at
`services/isl-stream/training/asl_corpus_manifest.py`. It accepts a CSV with
these columns:

`source_dataset, source_clip_id, signer_id, split, start_ms, end_ms, transcript, transcript_kind, modality, license_id, permission_status, attribution, source_url`.

Example command:

```bash
python services/isl-stream/training/asl_corpus_manifest.py path/to/asl_continuous_manifest.csv
```

- Allowed split names are train, validation and test; a signer may occur in only one split.
- Clip IDs must be unique within a source dataset and segment timestamps must be valid.
- Every row must record a license, attribution, source URL and explicit `permission_status=approved`.
- `transcript_kind` must explicitly be `translation` or `gloss`. How2Sign English transcripts are translation targets, not ASL gloss labels.
- This tool validates metadata only. It does not download media, verify legal rights, decode video, validate signing labels or train a model.
- It deliberately does not convert the English translation transcript into the existing gloss-CTC training format.
