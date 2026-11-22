# SIGNBRIDGE ISL streaming service

**Status: API scaffold only. There is no model and no sign translation.**

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

- `GET /healthz` reports whether the service process is alive.
- `GET /v1/recognition/status` returns HTTP 503 and `model_not_configured`.
- `WS /v1/recognition/session` explicitly reports that recognition is unavailable and closes with code 1013.

The service deliberately does not accept video frames or invent sample predictions until a validated model is integrated. This prevents the frontend from mistaking a demo response for a real translation.

## Model integration requirements

A future model adapter must return timestamped incremental hypotheses with:
- source sequence/frame timestamps and dropped-frame counters
- provisional versus stable segments
- calibrated confidence and an explicit abstain/uncertain state
- model version, supported vocabulary/domain and language variant
- processing and end-to-end latency metrics

A future streaming implementation must use authenticated, encrypted transport, explicit camera consent, bounded frame buffers, rate limits, disconnect cleanup and a documented retention policy. Do not log raw video or persist it by default.

## Release safety

Never speak an uncertain hypothesis as a confirmed sentence. Critical details (names, phone numbers, amounts, dates, addresses, medication and emergency instructions) require user confirmation. The service is not a substitute for a qualified interpreter or emergency service.


## Multiple sign languages

The family manifest requires `sign_language` with one of `ISL`, `ASL`, or `BSL`. Labels and model training must remain language-specific; never pool identical-looking gloss labels across languages without expert-reviewed mapping. The recorder downloads local clips tagged with the selected language. No sign recognition model is configured yet.

## Streaming protocol groundwork

`stream_protocol.py` defines strict validation for a future frame envelope and incremental model hypotheses. It checks language tags, monotonically increasing timestamps, image MIME types, base64 validity and a 2 MiB frame limit. It also prevents speech playback for provisional, uncertain or below-threshold hypotheses.

This is deliberately not wired to the WebSocket yet: without a real inference model, accepting camera frames would create the appearance of a working translator without producing reliable meaning. The 0.90 speech threshold is a conservative engineering default for gating, not a calibrated accuracy guarantee. It must be calibrated against held-out signer data before use.

## Experimental landmark CTC training baseline

The `training/` folder contains a small PyTorch Transformer + CTC baseline. It consumes pre-extracted, signer-reviewed `.npy` feature sequences, not raw video. The sample rows in `feature-manifest.template.csv` are placeholders, not real training examples. Each run selects exactly one sign language (ISL, ASL or BSL) from the combined feature manifest and ignores the other valid language rows. It requires explicit consent, checks signer-disjoint train/validation/test splits for the selected language, and saves the best validation-loss checkpoint plus test metrics.

Install optional training dependencies:

```bash
pip install -r requirements-training.txt
```

Example, after creating a private feature manifest and extracting features:

```bash
python training/train_ctc.py --manifest /private-data/manifest.csv --data-root /private-data --language ISL --output /private-models/isl-ctc.pt
```

**Important:** this is an experimental baseline, not a trained model, decoder, Hindi/English sentence translator or real-time service. The training command has not been run because no consented, labelled landmark dataset is present in this repository. The current service must continue reporting recognition unavailable until a model is trained, independently evaluated and integrated.

## Local landmark extraction (optional research step)

Install the separate extraction dependencies:

```bash
pip install -r requirements-extraction.txt
```

From a private family dataset folder containing `manifest.csv` and `clips/`, run:

```bash
python /path/to/Signbridge/services/isl-stream/training/extract_landmarks.py --manifest manifest.csv --data-root . --output-manifest features_manifest.csv --max-fps 60
```

This uses MediaPipe Holistic to extract a fixed 1,699-value vector per sampled frame: pose, face, left/right hand landmarks and presence flags. It uses a ceiling-based frame stride so the selected sample rate does not intentionally exceed the requested maximum, preserves temporal order, rejects duplicate IDs and sanitized filename collisions, and writes NumPy arrays under `features/`. It does not recognise signs. Review every source clip and gloss label; the source manifest's `label` field must contain expert-reviewed gloss token(s), not a translated sentence. Keep the generated feature arrays and manifest private.

Then train one language at a time:

```bash
python /path/to/Signbridge/services/isl-stream/training/train_ctc.py --manifest features_manifest.csv --data-root . --language ISL --output /private-models/isl-ctc.pt
```

The training report includes held-out gloss WER/CER and CTC loss. These are research metrics on the supplied split, not proof of sentence-level translation quality or generalisation. A model trained on a tiny family dataset must remain a private experiment and must not be used for safety-critical communication.

## Offline inference from a trained checkpoint

After a training run produces a checkpoint, test one extracted clip locally:

```bash
python training/inference.py --checkpoint /private-models/isl-ctc.pt --features /private-data/features/clip-001.npy --language ISL
```

The output is a research gloss sequence only. It is not translated into Hindi or English, does not carry calibrated confidence, and is not connected to the camera or WebSocket service. Do not use it to make decisions in real conversations until it has been evaluated with fluent signers and the end-to-end streaming system has passed safety testing.
