# SIGNBRIDGE streaming benchmark procedure

This procedure is for measuring a real device + model pipeline once one is
available. The current service does **not** perform live recognition, so no
SIGNBRIDGE camera-to-translation latency or recognition-speed result is claimed.

## Measure the stages separately

Record monotonic timestamps in milliseconds for each utterance:

- `captured_at_ms`: source frame/utterance capture time.
- `first_provisional_at_ms`: first provisional model hypothesis available.
- `stable_at_ms`: hypothesis becomes stable under the decoder's revision policy.
- `translated_at_ms`: sentence translation is ready for display or speech.
- `captured_frames` and `dropped_frames`: frame counters for the same measurement window.

Do not substitute camera preview FPS for landmark extraction throughput or model
inference throughput. Measure those independently, on the target phone, with the
same camera resolution, frame sampling policy, model checkpoint and thermal
conditions intended for family use.

## JSONL format

One JSON object per utterance, for example:

```json
{"captured_at_ms":1000,"first_provisional_at_ms":1040,"stable_at_ms":1180,"translated_at_ms":1260,"captured_frames":120,"dropped_frames":3}
```

All timestamps must use the same monotonic clock. Timestamps must be ordered.
Frame counters must be non-negative integers and dropped frames cannot exceed
captured frames. Do not include names, raw video, transcripts, or other family
identifiers in benchmark files unless separately consented and necessary.

## Generate the report

From this directory run:

```bash
python benchmark_latency.py /path/to/consented-benchmark.jsonl
```

The report includes p50 and p95 latency for first provisional output, stable
output and translation, plus aggregate dropped-frame percentage. Collect enough
utterances across multiple sessions to expose variability; report sample counts
alongside percentiles. Keep device model, OS version, model/checkpoint hash,
input resolution, sampling rate, app version, and test conditions in a separate
non-identifying run manifest.

## Release gates

Do not set a universal latency target until the target phone and representative
signing data are available. Establish thresholds with Deaf signers and
interpreters, then test:

1. landmark extraction throughput and queue growth;
2. model inference time and end-to-end provisional/stable/translation latency;
3. p50/p95 latency and dropped frames under sustained signing;
4. thermal throttling, memory use, background/foreground transitions and
   reconnect behaviour;
5. recognition quality (including signer-disjoint WER/CER or task-specific
   metrics) separately from speed.

A fast but incorrect hypothesis is not a successful translation. For names,
numbers, addresses, medication, emergencies and other consequential details,
the interface must preserve uncertainty and request confirmation rather than
silently guessing.
