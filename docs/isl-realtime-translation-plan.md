# SIGNBRIDGE ISL real-time translation engineering plan

**Status: research and integration plan only. No recognition model is connected yet.**

This document defines the next engineering milestone. It deliberately separates camera capture from recognition, and recognition from sentence translation.

## Product target

A Deaf ISL signer should be able to sign naturally and continuously. SIGNBRIDGE should produce a live, revisable text interpretation and, after a phrase is stable enough, optional spoken output. It must not require the signer to pause after every sign.

The system must never silently turn an uncertain prediction into a confident sentence. This is an assistive communication tool, not a certified interpreter or emergency service.

## Research candidates reviewed

### 1. Continuous baseline: Sid-V5/isl-translator

- Public implementation: https://github.com/Sid-V5/isl-translator
- Described architecture: MediaPipe Holistic keypoints (hands, pose and face), ST-GCN spatial encoder, Transformer temporal encoder, CTC gloss decoder, optional IndicTrans2 gloss-to-text.
- Its README reports 30.77% word error rate on its continuous-recognition setup, trained from 4,287 INCLUDE word-level videos.
- The repository documents local inference requiring a GPU with at least 8 GB VRAM.
- The repository README says MIT, but the Git tree currently has no separate LICENSE file. Treat the code license as unverified until the owner supplies a complete license file; do not copy or redistribute code/weights on the README statement alone. Audit every dependency/model/data license too.
- Treat this as a research baseline to reproduce and evaluate, not a production-ready interpreter.

### 1b. Alternative architecture candidate: Danshivam/RTISLT

- Public implementation: https://github.com/Danshivam/RTISLT
- The repository includes an MIT LICENSE file for the repository code. This does **not** automatically license third-party datasets, pretrained checkpoints, or dependencies; audit those separately before use.
- Its README describes a landmark-based design with hands/body pose and a separate face/head non-manual stream, local/global temporal modeling and an English decoder.
- The README's own progress checklist marks the BiLSTM and iSign Transformer training/evaluation steps incomplete. Treat the architecture as a research design and inspect the implementation/checkpoints before relying on it; do not assume a trained continuous translator is ready to run.
- It is a candidate to compare against the Sid-V5 baseline, not a drop-in production model.

### 2. Continuous sentence data: ISL-CSLTR

- Dataset record: https://data.mendeley.com/datasets/kcmpdxky7p/1
- Describes 700 annotated videos, 100 spoken-language sentences and seven signers.
- The record lists CC BY 4.0; verify attribution requirements against the downloaded record before use.
- The record describes recordings from two native signers and four SASTRA students. This very small signer pool makes it useful for pipeline prototyping, not evidence of generalisation to the wider Deaf community.

### 3. Sentence-level video/gloss data: ISLVT

- Dataset record: https://data.mendeley.com/datasets/98mzk82wbb/1
- Describes 152 videos and 76 English-Marathi-ISL gloss sentence pairs, with front and side views.
- The record lists CC BY 4.0.
- Useful as a small supplementary evaluation/training resource; do not assume it represents everyday conversational ISL.

### 4. Larger isolated-sign resource: FDMSE-ISL

- Project page: https://cs.rkmvu.ac.in/~isl/dataset.html
- Describes 40,033 isolated videos, 2,002 signs and 20 Deaf native signers; frontal RGB recordings include 60 fps.
- The synchronized video data are available for research by request and subject to approval.
- This is valuable for vocabulary pretraining, but isolated words are not continuous sentence translation. Do not treat a 60 fps isolated-sign dataset as proof of fast conversational translation.

### 5. ISH-NEWS

- Paper: https://www.nature.com/articles/s41598-026-60893-0
- Describes 4,222 sentence videos and more than 6,500 words.
- The paper says raw videos cannot be redistributed because of copyright; annotations, splits, timestamps and preprocessing scripts are available.
- Use only in ways permitted by the paper and rights holders. Do not scrape, redistribute or train on source broadcast videos without the required rights. Annotation availability does not grant rights to the underlying broadcast video.

## Proposed model pipeline

1. **Capture:** continuous camera stream; record actual frame timestamps, frame drops, resolution and device capability. Capture FPS is a diagnostic, not recognition speed.
2. **Visual features:** extract both hands, upper-body pose and facial/non-manual cues. Preserve temporal order and confidence/visibility per landmark.
3. **Temporal recognition:** streaming spatial-temporal encoder with chunked/overlapping windows and a decoder that can revise partial output as context arrives. Avoid independent per-frame or isolated-gesture classification.
4. **Gloss and sentence layer:** recognise ISL gloss/structure, then translate into natural Hindi and English with an ISL-aware translation layer. Do not assume word-for-word English order.
5. **Output policy:** show provisional text distinctly; speak only stable segments; allow immediate pause/cancel/correction. Keep original gloss, translated text and confidence metadata separately.
6. **Safety layer:** low-confidence or out-of-distribution input triggers "please repeat" rather than a guessed sentence. Names, phone numbers, amounts, dates, addresses, medication and emergency instructions require explicit confirmation.
7. **Android runtime:** benchmark quantised/on-device inference versus secure streaming inference. Never upload camera video without clear opt-in, purpose limitation, retention controls and a privacy notice.

## Data and evaluation gates

- Obtain written permission and verify licenses, consent, attribution and commercial-use rights before using each dataset.
- Keep signer-independent train/validation/test partitions. No signer or near-duplicate clip may leak across splits.
- Include fluent Deaf ISL signers across regional variation, signing speeds, handedness, age, skin tones, clothing, camera quality, lighting, backgrounds and occlusion.
- Test natural conversational signing, rapid signing, co-articulation, short signs, pauses, repetitions, fingerspelling, facial grammar and non-manual markers.
- Report continuous sign/gloss error rate, sentence translation quality judged by Deaf reviewers, false insertions, missed signs, correction rate, dropped frames, end-to-end latency (capture to stable text), thermal behaviour and battery use.
- Publish results separately for seen and unseen signers and for standard/fast signing. Never claim "zero errors".

## Release gates

**Gate A — Reproducible research baseline:** reproduce the candidate's reported setup or document why it cannot be reproduced; establish a clean, signer-independent baseline.

**Gate B — Streaming prototype:** incremental outputs, partial revisions, timing and frame-drop telemetry; no fabricated predictions when the model is absent.

**Gate C — Deaf-led evaluation:** fluent signer review and measured error/latency targets agreed with community partners. Fix systematic errors before widening the test cohort.

**Gate D — Android pilot:** test on representative mid-range Android phones, not only a desktop GPU. Ensure camera, heat, network loss and battery do not cause silent failures.

**Gate E — Limited release:** clearly label supported vocabulary/domains, uncertainty, limitations and fallback options. Keep text entry and human interpreter alternatives available.

## Streaming service scaffold

A deliberately non-translating FastAPI scaffold now lives in `services/isl-stream/`. Its health endpoint reports process health; its recognition-status endpoint returns HTTP 503 with `model_not_configured`; its WebSocket session explicitly reports recognition unavailable and closes rather than returning guessed labels. This is an API boundary only, not an inference engine. It is not yet connected to the web or Android app and is not deployed.

## Immediate next actions

1. Contact the FDMSE-ISL team to request research access and clarify permitted use.
2. Download and inspect the CC BY 4.0 ISL-CSLTR and ISLVT records; verify attribution and dataset contents.
3. Audit the candidate model repository's complete license and dependencies before adapting code.
4. Create a separate Python model service and reproducible training/evaluation environment; do not add large model dependencies to the Vite frontend.
5. Build a streaming API contract and UI states only after the model can return timestamped, confidence-aware incremental predictions.
6. Keep the current camera page explicitly labelled as preview until a real model is integrated and validated.

## Current status

The existing SIGNBRIDGE web/PWA and Android shell do not contain a sign-language recognition model. The camera's 60 fps request only changes capture settings. This plan is not an implementation or a claim of translation capability.


## Family-first personal-use priorities

SIGNBRIDGE is being built for the user's parents and personal accessibility use, not as a commercial product. Prioritise their actual signing style, regional vocabulary, preferred output language (Hindi and/or English), and the devices they will use.

### Practical path

1. Start with a small, useful family vocabulary (greetings, needs, food, household objects, directions, appointments, family names and common questions), then expand based on real use.
2. With each parent's informed consent, record example clips in their normal signing environment and preferred camera position. Keep recordings private and local by default; allow deletion and avoid uploading raw video unless explicitly chosen.
3. Capture multiple examples per sign, including natural speed, transitions, both hands, facial expression and different lighting. A sign's meaning may depend on movement, context and non-manual features; do not reduce the task to hand-shape-only classification.
4. Build a personalised recogniser as an initial assistive layer, not a claim of full ISL translation. Clearly label supported vocabulary and abstain when uncertain.
5. Evaluate with the parents themselves: missed signs, incorrect substitutions, time-to-display, time-to-speech and whether the phrase was understood. Fix high-frequency failures first.
6. Add continuous sentence recognition only after the personalised vocabulary baseline works. A word-level model is not automatically a sentence translator.

### Personal dataset and consent

Family-recorded clips can be useful for adapting the app to the actual users, but they are not a substitute for broad signer coverage. Obtain each person's informed consent, explain what is recorded and where it is stored, and provide a simple delete/export path. Keep the data on-device or in a private user-controlled location unless the family explicitly chooses otherwise.

### Research baseline caveats

The Sid-V5 repository describes an ST-GCN + Transformer + CTC pipeline and reports 30.77% WER on its stated continuous-recognition setup. Its README says MIT, but the repository tree currently has no separate LICENSE file; verify the actual license with the author before copying code. Its README also says local inference requires a GPU with at least 8 GB VRAM, so it is not yet a proven Android-ready runtime. Treat it as an architecture reference, not a drop-in model or a performance guarantee.


## On-device perception candidate

Google MediaPipe Tasks provides Android live-stream hand landmark detection and a generic gesture recognizer. Its live-stream API processes asynchronously and can drop incoming frames when busy, which is preferable to building an ever-growing frame queue. However, the generic gesture model recognises a small predefined set of hand gestures; it is not an ISL translator. Hand landmarks alone also omit important facial, body and grammatical information. Use MediaPipe as a possible real-time feature-extraction component, not as the sign-language model itself. See the official [Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) and [Gesture Recognizer guide](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer).

## Family data preparation tools

The `services/isl-stream/family-dataset/` guide and `manifest.template.csv` define a private, local-first recording workflow. `dataset_tools.py` checks consent flags, local file paths, required labels and signer-disjoint splits. It never uploads or decodes videos. Automated tests cover consent, path traversal and signer leakage. These checks make a future family-specific dataset safer to organise; they do not create a trained model or validate the meaning of any sign.
