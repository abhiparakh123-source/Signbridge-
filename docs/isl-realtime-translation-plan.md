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
- The repository README says MIT; verify the repository's actual license file and every dependency/model/data license before copying code or redistributing weights.
- Treat this as a research baseline to reproduce and evaluate, not a production-ready interpreter.

### 2. Continuous sentence data: ISL-CSLTR

- Dataset record: https://data.mendeley.com/datasets/kcmpdxky7p/1
- Describes 700 annotated videos, 100 spoken-language sentences and seven signers.
- The record lists CC BY 4.0.
- Small signer count means it is useful for an initial pipeline and sanity checks, not sufficient evidence of generalisation.

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
- Use only in ways permitted by the paper and rights holders. Do not scrape, redistribute or train on source broadcast videos without the required rights.

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

## Immediate next actions

1. Contact the FDMSE-ISL team to request research access and clarify permitted use.
2. Download and inspect the CC BY 4.0 ISL-CSLTR and ISLVT records; verify attribution and dataset contents.
3. Audit the candidate model repository's complete license and dependencies before adapting code.
4. Create a separate Python model service and reproducible training/evaluation environment; do not add large model dependencies to the Vite frontend.
5. Build a streaming API contract and UI states only after the model can return timestamped, confidence-aware incremental predictions.
6. Keep the current camera page explicitly labelled as preview until a real model is integrated and validated.

## Current status

The existing SIGNBRIDGE web/PWA and Android shell do not contain a sign-language recognition model. The camera's 60 fps request only changes capture settings. This plan is not an implementation or a claim of translation capability.
