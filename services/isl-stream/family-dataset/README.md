# Private family sign dataset preparation

This is an optional local-only preparation step for a future personalised recogniser. It does not upload video, run AI recognition, or translate signs.

## Folder layout

```text
family-sign-data/
  manifest.csv
  clips/
    clip-001.webm
    clip-002.webm
```

Copy `manifest.template.csv` to `manifest.csv`, replace example rows with real clip metadata, and place clips under `clips/`. The sample rows are placeholders; they do not define ISL signs.

## Consent and privacy

- Ask each parent for informed, voluntary consent before recording.
- Explain that video can show faces, homes and other identifying details.
- Keep this folder private on a trusted device; do not commit recordings to GitHub or upload them to a public service.
- Use pseudonymous signer IDs such as `parent-a`; do not put full names in the manifest.
- Delete clips whenever the signer asks. Do not record visitors or bystanders without their consent.

## Split rule

All clips from one signer must remain in a single split. For a family pilot, use separate signers for training/validation/test where possible. If there are only one or two signers, do not pretend the resulting evaluation measures generalisation; treat it as a personal prototype and test carefully in real conversations.

## Validate

From this directory:

```bash
python dataset_tools.py manifest.csv --root .
```

A passing check only confirms metadata, file paths, consent flags and signer split hygiene. It does not verify that a label is correct or that a model understands the sign.

## Recording guidance

Capture multiple natural examples per phrase, not just one frozen pose. Include the movement into and out of the sign, both hands when used, facial expression and upper body where relevant. Avoid forcing a signer's natural speed or style. Have the signer confirm the intended phrase label after each clip.
