# Hugging Face datasets and trained models

Exports and weights stay on the data disk. Uploads use the Hugging Face credential
store and default to private repositories. Never put a token in a command argument,
the repository, or an export directory.

The initial repositories under `mateo-19182` are:

- [coruna-typology-observations](https://huggingface.co/datasets/mateo-19182/coruna-typology-observations), the complete query-index metadata snapshot with 1,051,149 observations across 13 categories. It has an `all` configuration and one configuration per category. Images are linked through the existing crop deployment, not embedded.
- [coruna-graffiti-reviewed](https://huggingface.co/datasets/mateo-19182/coruna-graffiti-reviewed), the original SD 1.5 dataset. Its exact 225/29 split is pinned at commit `effe3744dbb412e4a214e432f622f8629f1d2621`.
- [coruna-graffiti-reviewed-v2](https://huggingface.co/datasets/mateo-19182/coruna-graffiti-reviewed-v2), the expanded FLUX dataset with 971 training images and 92 validation images, retaining the original 225/29 split inside those counts.
- [coruna-graffiti-sd15-lora](https://huggingface.co/mateo-19182/coruna-graffiti-sd15-lora), the completed adapter, training record, example outputs and a link to the dataset commit.
- [coruna-graffiti-flux2-klein-lora](https://huggingface.co/mateo-19182/coruna-graffiti-flux2-klein-lora), the completed FLUX.2 klein Base 4B adapter. It trained for 2,000 steps at rank 16 on v2 and includes three fixed-seed base/adapter sample pairs.
- [coruna-graffiti-captioned-v3](https://huggingface.co/datasets/mateo-19182/coruna-graffiti-captioned-v3), the same 971/92 image split with individual visual captions. Unverified lettering candidates are separate metadata and are excluded from training captions. Commit `d659780255f07e75490f4866658731117401f9da`.
- [coruna-graffiti-flux2-klein-captioned-lora](https://huggingface.co/mateo-19182/coruna-graffiti-flux2-klein-captioned-lora), a fresh 2,000-step adapter trained on v3, with a 128-token text cache, 10% caption dropout and fourteen fixed-seed comparison images. It uses ordinary prompts without a special city-name trigger. Commit `047f89ead7f10e8fd2a1e15f98e7a087b67c52be`.

These links require access to the private repositories. Apple/Google imagery and
graffiti artwork rights remain unresolved. The project's MIT license covers code.
It does not license these images. Dataset cards record that limitation.

## Export and upload

Install the optional packages in a separate environment:

```bash
export HUB_ENV=/mnt/data/projects/typology-machine/hub-venv
uv venv --python .venv/bin/python "$HUB_ENV"
uv pip install --python "$HUB_ENV/bin/python" -r explore/huggingface-requirements.txt
"$HUB_ENV/bin/hf" auth login
```

Create a new output directory for each export. The exporter refuses to overwrite
an existing package. It reads one frozen query snapshot, writes category Parquet
shards with a consistent schema, and checks the record count:

```bash
"$HUB_ENV/bin/python" explore/publish_huggingface.py archive \
  --output /mnt/data/projects/typology-machine/huggingface/new-export/observations
"$HUB_ENV/bin/python" explore/publish_huggingface.py upload \
  --package /mnt/data/projects/typology-machine/huggingface/new-export/observations \
  --repo-id mateo-19182/coruna-typology-observations
```

Package and upload a completed training run:

```bash
"$HUB_ENV/bin/python" explore/publish_huggingface.py run \
  --home /mnt/data/projects/typology-machine/graffiti-ai \
  --output /mnt/data/projects/typology-machine/huggingface/new-run \
  --namespace mateo-19182 --upload
```

Omit `--upload` to inspect the packages first. A prepared package can be uploaded
separately with the `upload` command, including after a failed network request.
Upload receipts live beside the package and record the repository URL and commit.
The model's `provenance.json` includes the reviewed dataset's exact Hub commit.

Uploads verify local checksums and inventory before contacting the Hub, then verify
the committed file inventory, sizes and LFS digests. Only files listed in the
package manifest can be uploaded. The Hub's streamed uploader splits large exports
into several commits and resumes completed files after a network failure. During
that upload the latest branch can be incomplete; use the final receipt's commit.
The uploader replaces files owned by the previous Typology package and accepts a
partial new upload only when its files match the prepared package. It refuses
repositories containing unrelated files or a different visibility. Git history
preserves earlier exports. It never uploads an entire model
home, cache, environment or generated-image directory.

Reviewed image exports require valid manifest hashes, image counts, readable files,
no extra training files, disjoint IDs/groups/panoramas, the geographic buffer and
cross-split perceptual-hash separation. They preserve JPEG bytes and split assignments.
The archive metadata inventory has no training split. Its `observations` split must
be reviewed and split before use for training.

New SD 1.5 runs can use `graffiti_model.py train --publish --namespace mateo-19182`
to upload the completed adapter and reviewed dataset. Publishing failure does not
change a successful training run to a failed training state.

## Loading private datasets

```python
from datasets import load_dataset

graffiti = load_dataset("mateo-19182/coruna-graffiti-reviewed-v2", token=True)
assert len(graffiti["train"]) == 971
assert len(graffiti["validation"]) == 92

# The exact original SD 1.5 training dataset:
original = load_dataset(
    "mateo-19182/coruna-graffiti-reviewed",
    revision="effe3744dbb412e4a214e432f622f8629f1d2621", token=True,
)

signs = load_dataset(
    "mateo-19182/coruna-typology-observations", "store_sign",
    split="observations", token=True,
)
```

Use a receipt's commit as `revision=` to reproduce an earlier export. Crop links
in the metadata archive depend on the deployment and can disappear. The reviewed
graffiti repository embeds its images and can be downloaded independently.

Use `run --dataset-repo-id NAMESPACE/DATASET` for a versioned dataset repository.
A run can also store `{"dataset_repo_id": "NAMESPACE/DATASET"}` in its local
`publishing.json`. The expanded FLUX run points to `coruna-graffiti-reviewed-v2`.
Its first attempt to update the original repository timed out while the Hub kept
that commit in progress, so the expanded dataset uses a separate repository.

## Checks

```bash
"$HUB_ENV/bin/python" -m unittest discover -s explore -p 'test_publish_huggingface.py'
```

These checks exercise image and split preservation, review gates, altered/extra
artifact rejection, Parquet loading and the FLUX training plan without using a GPU.
