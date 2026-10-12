# Google overnight GPU run

Prepare with `.venv/bin/python explore/gpu_run/prepare.py` on the local machine.
The frozen bundle is `data/coruna/gpu_google_current/`. Images are hard links locally;
rsync transfers ordinary files. Later downloads do not change this run's inventory.
No credentials, `.env`, Apple data, or historical images are included.

Use one NVIDIA GPU with at least 24 GB VRAM, Python 3.12+, a CUDA-compatible driver,
at least 8 effective CPU cores, 32 GB RAM, and 100 GB container storage.
Check driver compatibility with the pinned Torch wheel during remote preflight.

Transfer the bundle with rsync over SSH. Then, in the remote project directory:

```bash
bash explore/gpu_run/bootstrap.sh
bash explore/gpu_run/launch.sh smoke
WORKERS=4 RUN_HOURS=8 bash explore/gpu_run/launch.sh run
```

The smoke run saves eight actual views into the same run. Inspect detections and
measure speed before continuing. Full inference skips completed views and projects
only the assigned unfinished Google views. It detects all 13 categories, pins the
Apple model revision and saves portable crop, mask and source paths. It does not
reuse legacy shop-sign results, so the remote run has no legacy asset dependency.

The time cap stops inference gracefully. Rental billing must be stopped separately
through Vast after results are copied and checked. Keep the API key on the local
controller. Do not upload it to the GPU machine.

Copy `data/coruna/gsv/multicategory/<run-id>/` back to the matching local directory,
including its config and every view file. Completed views are committed with atomic
`result.json` records; resume retries unfinished views. Copy again after shutdown
to capture the final completed records. Keep the frozen source manifest and run plan.
Verify crop/mask paths and counts before adding this run to the map refresh service
with `--extra-run`. New Google outputs remain separate until that integration.

## Cost selection

Compare projected total cost, including disk rental, model/image upload, result
download and installation time. Use a short actual SAM benchmark to compare views
per dollar. Vast's generic performance score is only a shortlist signal. Keep an
on-demand offer as the baseline; interruption risk and repeated setup can erase
the savings on a short job. Stop the instance promptly after copying outputs,
then destroy it after confirming the local copy to end retained-storage charges.
