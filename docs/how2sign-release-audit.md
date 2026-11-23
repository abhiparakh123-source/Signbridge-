# How2Sign local release audit

How2Sign's official page describes continuous ASL video and gloss annotation,
but the downloadable assets are presented separately and the page warns that
English sentence text may not align perfectly with signing. Some users have
also reported difficulty locating the gloss annotations. Therefore SIGNBRIDGE
must inspect the actual downloaded release rather than assume a gloss file is
present or that English translations are gloss labels.

Official source: https://how2sign.github.io/
Dataset download scripts: https://github.com/how2sign/how2sign-data

## Before downloading

- Confirm the CC BY-NC 4.0 research-only terms cover the intended private family
  research and development. Ask the dataset authors for written clarification
  if needed.
- Do not put raw videos, annotations, or derived participant data in GitHub.
- Download only from official dataset links or an author-authorized source.
- Keep the original license, attribution, source URL, version and checksums.

## Run a local inventory

After obtaining authorized files, place them in a private directory and run:

```bash
python services/isl-stream/training/audit_how2sign_release.py \
  --data-root /private/path/to/how2sign \
  --output /private/path/to/how2sign-audit.json
```

The report counts video files and inspects CSV/TSV headers. It flags tables with
an ID, split and nonempty gloss column as candidates for manual review. It does
not approve any table for training.

A candidate still needs:
- evidence that the gloss labels are genuine source annotations, not English
  translations or automatically generated pseudo-glosses;
- annotation convention and provenance review by a fluent Deaf ASL reviewer;
- signer identity mapping and signer-disjoint train/validation/test splits;
- media-to-annotation timestamp alignment checks;
- a documented license/permission decision for this specific use.

If the release contains only English translation tables, stop. Do not convert
those strings into glosses. The model may later use aligned translations as
targets in a separate sign-to-English translation experiment, but only after
the video alignment and legal scope are verified; that is not the current
gloss-CTC training input.

## Current boundary

The audit utility is an inventory helper, not a trained model or a data
download. No real dataset is currently committed to the repository, and no
real ASL checkpoint is claimed.
