# Hugging Face datasets and trained models

Exports and weights stay on the data disk. Uploads use the Hugging Face credential
store and default to private repositories. Never put a token in a command argument,
the repository, or an export directory.

The initial repositories under `mateo-19182` are:

- [coruna-typology-observations](https://huggingface.co/datasets/mateo-19182/coruna-typology-observations), the complete query-index metadata snapshot with 1,051,149 observations across 13 categories. It has an `all` configuration and one configuration per category. Images are linked through the existing crop deployment, not embedded.
- [coruna-graffiti-reviewed](https://huggingface.co/datasets/mateo-19182/coruna-graffiti-reviewed), the exact 225 training and 29 validation JPEGs used in the graffiti experiment, with original captions and provenance.
- [coruna-graffiti-sd15-lora](https://huggingface.co/mateo-19182/coruna-graffiti-sd15-lora), the completed adapter, training record, example outputs and a link to the dataset commit.

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
package manifest can be uploaded. The uploader replaces files owned by the previous
Typology package; it refuses repositories containing unrelated files or a different
visibility. Git history preserves earlier exports. It never uploads an entire model
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

graffiti = load_dataset("mateo-19182/coruna-graffiti-reviewed", token=True)
assert len(graffiti["train"]) == 225
assert len(graffiti["validation"]) == 29

signs = load_dataset(
    "mateo-19182/coruna-typology-observations", "store_sign",
    split="observations", token=True,
)
```

Use a receipt's commit as `revision=` to reproduce an earlier export. Crop links
in the metadata archive depend on the deployment and can disappear. The reviewed
graffiti repository embeds its images and can be downloaded independently.

## Checks

```bash
"$HUB_ENV/bin/python" -m unittest discover -s explore -p 'test_publish_huggingface.py'
```

These checks exercise image and split preservation, review gates, altered/extra
artifact rejection, Parquet loading and the FLUX training plan without using a GPU.
