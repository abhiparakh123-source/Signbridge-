# Run the SIGNBRIDGE ASL pilot on Kaggle (free tier)

This guide runs the existing ASL research notebook in your own Kaggle account. It does not create a paid cloud resource. Kaggle GPU availability and free-session quotas are controlled by Kaggle and are not guaranteed.

## What this pilot does

1. Checks the dataset metadata before downloading the larger files.
2. Verifies the audited 2,400-row inventory and signer-disjoint 1,200/600/600 split.
3. Downloads only metadata, annotation JSON and keypoint arrays (not RGB videos).
4. Converts keypoints into 384-dimensional landmark features.
5. Runs a short, three-epoch ASL gloss-recognition CTC baseline.
6. Saves a checkpoint, metrics and a pilot summary in the notebook output.

This is an offline research pilot. It is not sentence translation, live camera recognition, a production model, or phone-call interpretation.

## Steps

1. Sign in to Kaggle and open the [SIGNBRIDGE ASL training notebook](https://github.com/abhiparakh123-source/Signbridge-/blob/main/notebooks/signbridge_free_asl_training.ipynb).
2. Download the notebook file from GitHub (open the notebook, then use the Raw/download option) and create a new Kaggle Notebook by importing/uploading that `.ipynb` file.
3. In Kaggle Notebook settings, turn **Internet** on. Select **GPU** if Kaggle offers one to your account. If no GPU is available, the metadata audit and feature preparation may still run on CPU; model training will be slower and may exceed a free session.
4. Read the notebook's scope and dataset terms. The notebook records the approval confirmation already provided in this project; it does not invent a written permission document or override dataset license conditions.
5. Run cells from top to bottom. Do not skip the metadata preflight. If it stops because held-out glosses are missing from the training vocabulary or clips are too short for CTC alignment, stop there; do not download the large keypoint payload or move held-out signers into training to hide the issue.
6. The dataset step fetches only `metadata.csv`, `annotations/*.json` and `keypoints/*.npy`. It deliberately excludes RGB video.
7. Wait for the three-epoch pilot and artifact-summary cells to finish. If Kaggle disconnects, reopen the notebook and rerun as needed; downloaded data may need to be fetched again in a fresh session.
8. In Kaggle, use **Save Version** and ensure notebook outputs are saved. Download the `signbridge-artifacts` output folder if Kaggle presents it as a downloadable notebook output.

## Expected output

The output folder should contain:

- `signbridge-asl-pilot.pt` — research checkpoint
- `signbridge-asl-pilot.metrics.json` — training and held-out evaluation metrics
- `pilot-summary.json` — dataset count, checkpoint SHA-256, test metrics and limitations

Keep these artifacts private. Do not commit dataset files, private recordings, access tokens or checkpoints to the public GitHub repository.

## If something fails

- **Internet/download error:** confirm Kaggle Internet is enabled and retry the failed cell.
- **No GPU:** the notebook prints the available device. Training may be slow; do not pay for compute just to complete this pilot.
- **Metadata preflight fails:** preserve the error and stop before the keypoint download. It means the audited split/labels need review.
- **Session time limit:** save the notebook version and outputs if available; retry in a fresh free session. Kaggle's free quotas and session limits can change.
- **Missing output files:** verify the final artifact cell completed and use Kaggle's notebook output/version controls.

## What success means

A completed run demonstrates that this particular dataset preparation and baseline-training pipeline executed and produced measurable held-out metrics. It does not establish real-world ASL translation accuracy or readiness for your parents to rely on. Live recognition must remain disabled until a streaming inference path is integrated and evaluated with fluent signers.
