# SIGNBRIDGE low-latency continuous signing architecture

## Current implementation boundary

The `LandmarkCTCModel` is an experimental full-sequence Transformer CTC baseline. By default it uses bidirectional attention. An optional `--causal` training mode applies a causal attention mask so each frame output cannot attend to future frames. Both modes still recompute a complete supplied window; causal masking alone does not provide cached incremental inference, bounded-lookahead streaming, or a production real-time model. It consumes pre-extracted MediaPipe landmark arrays and emits experimental gloss tokens. It is not trained and does not translate ISL sentences into Hindi or English. The FastAPI WebSocket intentionally returns `recognition_unavailable` until a validated runtime model is installed.

Do not report camera preview FPS as recognition FPS. Do not infer accuracy or latency from an untrained model.

## Target live pipeline

1. **Capture:** Camera frames carry monotonically increasing sequence numbers and monotonic timestamps. Adapt capture resolution and sampling rate to device load; never queue an unbounded backlog.
2. **Preprocess:** Extract pose, both hands, face and non-manual cues. Preserve signer-relative geometry and temporal continuity. Face/hand loss is an explicit missing-data signal, not a guessed pose.
3. **Temporal recognition:** Use a causal or bounded-lookahead temporal encoder with a rolling context buffer. Process only new frames and keep state between windows. The full-window Transformer (including its causal-mask experiment) is a research comparator, not the production streaming model.
4. **Decode:** Use incremental CTC prefix decoding or another evaluated streaming decoder. Keep provisional hypotheses revisable; commit only when temporal stability and calibrated uncertainty criteria are met.
5. **Language:** Keep ISL, ASL and BSL as separate model/data pipelines. Never silently route an ASL or BSL stream into an ISL model.
6. **Translation:** Translate recognised sign-language meaning into Hindi or English in a separate layer. Gloss sequences are not grammatical sentences and must not be read aloud as if they were translations.
7. **Output:** Show provisional text immediately, stable text after evidence accumulates, and an uncertainty/correction affordance. Speak only after explicit user setting/consent and conservative confidence calibration; high-impact details require confirmation.

## Latency and quality measurement

Measure on real target phones and report each component separately:
- camera capture interval and dropped-frame percentage;
- landmark extraction p50/p95 time;
- model inference p50/p95 time per incremental chunk;
- time from the start of a sign/phrase to first useful provisional text;
- time to stable hypothesis;
- translation and speech synthesis time;
- end-to-end delay and backlog growth under sustained rapid signing;
- word/gloss error rate, character error rate, omissions, insertions and substitutions;
- signer-held-out performance, language/dialect coverage and uncertainty calibration.

A sensible engineering objective is to keep the live pipeline bounded and responsive, not to promise a universal "many words per second" number. Agree latency targets only after measuring real devices and Deaf signers. A fast but wrong transcript is not a successful accessibility feature.

## Backpressure and recovery

- Keep a bounded latest-frame queue; discard stale frames rather than building seconds of latency.
- Preserve sequence/timestamp telemetry without storing camera content by default.
- On overload, lower capture resolution or sampling rate and tell the user that recognition quality may be reduced.
- On disconnect, resume with a fresh session and explicit state reset; do not stitch unrelated streams.
- If hands/face are occluded or the signer leaves frame, surface uncertainty rather than inventing text.
- Provide a large pause/stop control and a manual text fallback.

## Evaluation gate before live activation

1. Obtain rights-cleared continuous signing examples and informed consent.
2. Have Deaf signers/qualified interpreters annotate phrase meaning, boundaries and relevant non-manual cues.
3. Keep signer-disjoint train/validation/test partitions.
4. Compare offline and causal/bounded-lookahead models using held-out signers and naturally fast conversational clips.
5. Calibrate confidence and test dangerous substitutions, negation, names, numbers, addresses and emergency phrases.
6. Run sustained end-to-end tests on low-, mid- and high-range Android phones and iPhones.
7. Keep live speech output disabled until error and uncertainty thresholds are agreed with the intended users.

## Current status

Architecture and test safeguards only. There is no trained continuous-sign model, no calibrated confidence, no production streaming inference, and no live sign-to-speech translation. The honest fallback remains text entry or a qualified human interpreter.


## Running the benchmark report

The dependency-free reporter accepts one JSON object per line, with monotonic
millisecond timestamps for each phrase. Example:

```json
{"captured_at_ms":1000,"first_provisional_at_ms":1080,"stable_at_ms":1250,"translated_at_ms":1320,"captured_frames":90,"dropped_frames":2}
{"captured_at_ms":2000,"first_provisional_at_ms":2070,"stable_at_ms":2210,"translated_at_ms":2290,"captured_frames":92,"dropped_frames":1}
```

Save records to a private local file (for example, `benchmark.jsonl`) and run:

```bash
cd services/isl-stream
python benchmark_latency.py /path/to/benchmark.jsonl
```

The report gives p50/p95 latency for first provisional text, stable text and
translation, plus dropped-frame percentage. Timestamps must come from the same
monotonic clock for each phrase. The reporter only summarises measurements;
it does not create model output or imply that live recognition is active.
Do not commit recordings, personal data or private benchmark files to GitHub.


## Prefix-emission experiment

`training/inference.py` includes `iter_causal_predictions`, which accepts successive landmark chunks and yields newly decoded gloss tokens. Because causal attention prevents future frames from changing earlier frame outputs, it can emit a prefix hypothesis as new chunks arrive. The implementation deliberately recomputes the entire accumulated prefix on each chunk and keeps that prefix in memory. Its compute cost grows with sequence length; it is a correctness-oriented experiment, not an efficient streaming runtime. CTC repeat state is carried across chunk boundaries so a token spanning two chunks is not duplicated. The live service remains disabled until a trained, evaluated model and efficient runtime exist.
