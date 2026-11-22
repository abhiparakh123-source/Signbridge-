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
