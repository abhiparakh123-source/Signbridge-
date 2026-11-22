# SIGNBRIDGE implementation milestones

This checklist tracks what exists in the repository versus what still requires real training data, model weights, and device validation. It is a personal family accessibility project.

## Working foundation

- [x] Web/PWA interface and Android Capacitor shell.
- [x] ISL, ASL and BSL language selection.
- [x] Fast camera capture request (up to 60 fps where supported); this is capture only.
- [x] Local family clip recorder with explicit consent and language labels.
- [x] Dataset manifest validation, signer-disjoint split checks and path safety.
- [x] FastAPI recognition service scaffold that refuses to invent predictions.
- [x] Timestamped frame-envelope validation and hypothesis speech gating.
- [x] Automated Python and frontend build/smoke checks.

## Remaining implementation

### A. Data and model (critical path)

- [ ] Obtain and verify dataset access, attribution and allowed use for ISL research datasets.
- [ ] Confirm family members' signing languages and regional ISL variation; keep ISL, ASL and BSL datasets separate.
- [ ] Build reproducible video-to-landmark extraction for both hands, upper-body pose and face/non-manual cues.
- [ ] Build dataset quality review with fluent signer confirmation of every label.
- [ ] Reproduce a continuous ISL baseline and record signer-independent WER/CER and sentence-level human review.
- [ ] Train/fine-tune a streaming temporal model with incremental, revisable hypotheses.
- [ ] Calibrate confidence and abstention on unseen signers; do not equate model confidence with correctness.
- [ ] Add a Hindi/English sentence translation layer reviewed by fluent ISL users.
- [ ] Quantise and benchmark on the actual Android phones used by the family.

### B. Live experience

- [ ] Connect the camera to a real landmark pipeline (not merely the preview).
- [ ] Connect the inference model to the WebSocket protocol.
- [ ] Display provisional text separately from stable text; allow corrections and replay.
- [ ] Speak only stable, sufficiently reliable text; ask for repetition on uncertainty.
- [ ] Add frame-drop, latency, thermal and battery diagnostics.
- [ ] Add offline model operation where device memory and performance allow.

### C. Other languages and reverse communication

- [ ] Build and evaluate an ASL model and vocabulary separately.
- [ ] Build and evaluate a BSL model and vocabulary separately.
- [ ] Add spoken Hindi/English captions with clear indication when browser speech recognition is unavailable or uncertain.
- [ ] Implement true two-way live conversation mode with user-visible consent and transcript correction.
- [ ] Implement a phone-call audio bridge only after Android call/audio routing and consent/privacy constraints are verified; the current Call Assist is a preparation tool, not an automatic interpreter.

### D. Family validation

- [ ] Agree with parents on useful everyday scenarios and preferred Hindi/English output.
- [ ] Test natural and rapid signing, regional signs, fingerspelling, names, numbers and interruptions.
- [ ] Measure errors and end-to-end latency with each signer held out from training.
- [ ] Keep a manual text/typing fallback available at all times.
- [ ] Do not rely on SIGNBRIDGE alone for emergencies, medication, payments or other high-stakes decisions until independently validated for those scenarios.

## Streaming protocol currently implemented

`services/isl-stream/stream_protocol.py` validates timestamped frame envelopes, supported language tags, image MIME types, base64 payloads and a maximum decoded frame size. It validates incremental hypothesis states and gates speech so provisional/uncertain/low-confidence output cannot be spoken automatically.

The protocol is not yet connected to the WebSocket or the camera. No model is loaded, and no sign translation is available. The 0.90 speech gate is a conservative code default, not a measured accuracy guarantee.

## Engineering note: high frame rate is not enough

A camera configured for 60 fps does not guarantee 60 fps inference or accurate translation. Google MediaPipe live-stream tasks can drop incoming frames while processing; timestamps and dropped-frame telemetry therefore matter. Hand landmarks alone do not provide full sign-language meaning: the planned model must also account for body, face, temporal context and language grammar.

## Exit criteria for the first useful family pilot

The first pilot should focus on ISL and a clearly bounded set of everyday family conversations. It should produce incremental text at natural signing speed, visibly flag uncertainty, allow corrections, run reliably on the family's Android device, and be evaluated with signer-independent testing. ASL/BSL expansion follows as separate language work, not a relabeling of the ISL model.
