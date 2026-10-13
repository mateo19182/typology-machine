# Coruña alphabet

Open `/alphabet.html` to compose words from photographs of local signs. The starter alphabet uses the installed RapidOCR Latin PP-OCRv5 recognition model. Its CTC alignment supplies individual character positions and confidence values. These are approximate character boundaries, not pixel masks; decorative lettering, clipped characters and recognition mistakes still need review. Each starter tile has a **Trim** control, and users can collect additional letters from the existing sign search.

The October 13 expansion processed 600 sign crops and produced 434 candidates. Visual review of the top 307 rejected 79 tiles and retained 228 variants across 43 characters, including all A–Z. The reviewed set is queued for publication after the storage migration finishes.

Build a larger starter candidate set with the query API running:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python explore/build_alphabet.py \
  --limit 600 --variants 12 --output data/coruna/alphabet-review-candidates.json
```

The default is 500 existing sign crops and eight variants per character. `--limit` is bounded at 1,500 and `--variants` at 16. Use `--output` to inspect candidates before publication; without it, the builder atomically replaces `data/coruna/site/alphabet-data.json`. The builder uses two CPU threads and caches line recognition under `data/coruna/alphabet-ocr-cache`. It does not rerun object detection or decode original panoramas. Generated crop data stays in the ignored data directory, not Git.

Selection requires line recognition confidence of at least 0.96 and character confidence of at least 0.98. Character boxes get at most four pixels of horizontal margin, bounded by the halfway point to neighboring boxes. Candidates must be at least 28 pixels tall and pass contrast and normalized sharpness checks. The model then reads each isolated tile again: it must return the same character at confidence 0.90 or higher. This rejects many clipped or mislabeled tiles. These thresholds are model scores and image statistics, not measured accuracy. Ranking favors readable resolution, contrast and sharpness; variants come from distinct source observations. Visual review still matters, especially for similar characters such as I, l and 1.

The page loads this saved alphabet; browsing does not run OCR. Custom letters live in browser storage. Returning visitors can use **Add new starter letters** to merge new variants without replacing their saved letters. **Download alphabet** saves a JSON backup containing letter images, crop rectangles and source links; **Import alphabet** restores it. The browser collection supports up to 500 variants and reports storage failures. **Print / save PDF** prints the current composition and its source list. Clicking a composed letter opens its source photograph. Outlined letters are placeholders for characters not yet collected.

Source photography and artwork retain their existing rights. Exporting a composition or alphabet does not grant an additional reuse licence.
