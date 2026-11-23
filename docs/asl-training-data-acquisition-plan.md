# SIGNBRIDGE ASL training-data acquisition plan

## Goal

Build a family-use ASL communication assistant in stages. Do not claim continuous
translation from isolated-word accuracy, subtitles, or a successful software test.
ASL is the current pilot; ISL and BSL require separate language-specific datasets,
labels, models and evaluations.

## 1. Candidate data sources

### How2Sign: primary continuous-signing research candidate

Project page: https://how2sign.github.io/

How2Sign reports more than 80 hours of multiview continuous ASL video, with
gloss annotations, English transcripts, depth and pose-related modalities. Its
published license is CC BY-NC 4.0 and it is described as research-only. The
project warns that English translations may not align perfectly with ASL video,
even where clips are manually realigned.

Actions:
- Review the full license and dataset terms before downloading or using it.
- Confirm whether a private family-use application is covered by the research
  purpose restriction; if unclear, request written permission from the authors.
- Obtain video, gloss annotation and metadata only through official project
  links. Do not use unrelated source-video content or scrape third-party sites.
- Preserve attribution, license, source identifiers and original signer splits.
- Audit gloss coverage, clip boundaries, frame rates, missing labels and signer
  overlap before converting to the project's training manifest.

### ASL Citizen: supplementary isolated-sign pretraining/evaluation only

Project page: https://www.microsoft.com/en-us/research/project/asl-citizen/
License: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
Download: https://www.microsoft.com/en-us/download/details.aspx?id=105253

ASL Citizen reports about 84,000 videos across 2,731 isolated signs from 52 Deaf
or hard-of-hearing signers. Microsoft explicitly recommends it for isolated
sign dictionary retrieval and cautions against treating it as continuous
sentence-signing data. Its terms restrict use to non-commercial, non-revenue
generating research, prohibit redistribution of the data/modified data, and
require respecting participant consent.

Actions:
- Use only if the intended activity is within the license; seek permission if
  family-use deployment is not clearly covered.
- Use it for a separate isolated-sign lookup/pretraining experiment, not as
  proof of sentence translation.
- Keep the raw dataset and derivatives private; do not put them in GitHub or a
  public app bundle.

## 2. Permissioned signer recordings

Once a baseline exists, recruit fluent Deaf ASL signers remotely through
community organisations, universities, interpreter/research networks or
community referrals. Pay fairly for their time and compensate annotation work.
Do not assume that an organisation or signer has agreed until explicit
permission is obtained.

Before recording, provide a plain-language consent form covering:
- purpose: private family accessibility research and model development;
- exact capture: video, audio if needed, face, hands and body;
- whether the recording can be used for training, evaluation, demonstration
  and/or publication (separate opt-ins);
- storage location, access controls, retention/deletion date and withdrawal
  process;
- whether derived landmarks/embeddings may be retained;
- compensation, contact route and no penalty for declining;
- no public upload, sale or unrelated reuse.

No family recording or participant video is to be committed to this repository.

## 3. What to record and label

Record natural continuous signing, not only alphabet or dictionary poses:
short conversations, requests, questions, corrections, names, numbers, dates,
places, negation, classifiers, spatial references, facial grammar and turn
boundaries. Include natural and fast signing, different camera distances,
lighting, backgrounds, dominant hands and signer styles.

Each clip must have:
- a stable pseudonymous signer ID;
- language and consent scope;
- source/session ID and clip timestamps;
- expert-reviewed ASL gloss sequence where appropriate;
- a separate English translation, never relabelled as gloss;
- labeler/reviewer IDs and disagreement/uncertainty flags;
- license/permission and provenance;
- split assignment by signer, never by randomly splitting clips from one signer.

A fluent Deaf ASL reviewer should validate glosses, non-manual markers and
translation meaning. Ambiguous clips should be excluded or explicitly marked,
not guessed.

## 4. Training sequence

1. Audit and validate source rights, labels and signer-disjoint splits.
2. Run the existing manifest preparation and landmark extraction pipeline.
3. Train the CTC gloss-recognition baseline on continuous, gloss-labelled clips.
4. Evaluate signer-disjoint gloss WER/CER and error categories; report results
   separately for normal and fast signing.
5. Add a separate context-aware ASL-to-English translation model only after
   recognition labels and alignment are adequate. English translation is not
   interchangeable with ASL gloss.
6. Integrate camera preprocessing and bounded streaming inference on Android.
7. Calibrate provisional/stable output, uncertainty and correction behaviour.
8. Test end-to-end with Deaf signers who were not in training, across phones,
   lighting and natural conversation speeds.

## 5. Release gates

- No checkpoint is deployed until dataset rights and consent are documented.
- Train/validation/test signer identities must be disjoint.
- No raw participant video is stored in GitHub, public artifacts or app assets.
- Report WER/CER, translation quality, latency p50/p95, dropped frames,
  false insertions, omissions and correction rate separately.
- Test names, numbers, dates, addresses and other high-impact details with
  explicit confirmation flows.
- A fast model with inaccurate output is not acceptable. Keep text fallback and
  qualified human interpreter options.
- ASL results must never be represented as ISL or BSL capability.

## Current project readiness

The repository already has consent/provenance manifest validation, signer-split
checks, landmark extraction, a CTC training baseline, checkpoint validation and
fail-closed API status. These are software foundations only. No licensed
dataset has yet been ingested, no real ASL checkpoint has been trained, and
camera-to-translation streaming remains disabled.

## Immediate practical next steps

1. Confirm license scope with How2Sign authors and Microsoft before using data
   for this specific private family app.
2. Obtain authorized downloads and inspect the actual gloss/translation labels.
3. Run a small, permission-compliant pilot subset through manifest validation
   and landmark extraction.
4. Arrange remote Deaf ASL review for a small labeled pilot and later
   signer-disjoint evaluation.
