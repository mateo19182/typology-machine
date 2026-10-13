# Coruña alphabet

Open `/alphabet.html` to compose words from photographs of local signs. The starter alphabet uses the installed RapidOCR Latin PP-OCRv5 recognition model. Its CTC alignment supplies individual character positions and confidence values. These are approximate character boundaries, not pixel masks; decorative lettering, clipped characters and recognition mistakes still need review. Each starter tile has a **Trim** control, and users can collect additional letters from the existing sign search.

The current starter comes from 1,500 sign crops. The builder produced 1,445 candidates. `explore/refine_alphabet.py` then re-cut each one to the letter's ink, which left 382. Visual review rejected 30 of those. The published set has 289 variants, at most 12 per character, covering A–Z. W, Q and 2 include letters 40–55px tall. Ñ has only two small lowercase variants that are not re-cut; the source crops contain few usable ones.

Build a larger starter candidate set with the query API running:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python explore/build_alphabet.py \
  --limit 600 --variants 12 --output data/coruna/alphabet-review-candidates.json
```

Then tighten and filter the candidates before reviewing them:

```sh
OMP_NUM_THREADS=2 .venv/bin/python explore/refine_alphabet.py \
  data/coruna/alphabet-review-candidates.json data/coruna/alphabet-refined.json
```

Refinement thresholds each OCR box and takes the polarity from it, since ink is the minority there. It keeps the connected components over the middle of the box and re-cuts around them, padding by 12% of the letter height. Padding never extends past halfway to neighbouring ink. It rejects letters shorter than 56 source pixels (`--min-height`), letters whose ink reaches the edge of the search window (clipped), and letters closer to a neighbour than 4% of their height (touching). Each tight tile must still read as the same character. Lowercase forms are kept as variants of their uppercase letter.

The default is 500 existing sign crops and eight variants per character. `--limit` is bounded at 1,500 and `--variants` at 40. Use `--output` to inspect candidates before publication; without it, the builder atomically replaces `data/coruna/site/alphabet-data.json`. The builder uses two CPU threads and caches line recognition under `data/coruna/alphabet-ocr-cache`. It does not rerun object detection or decode original panoramas. Generated crop data stays in the ignored data directory, not Git.

Selection requires line recognition confidence of at least 0.96 and character confidence of at least 0.98. Character boxes get at most four pixels of horizontal margin, bounded by the halfway point to neighboring boxes. Candidates must be at least 28 pixels tall and pass contrast and normalized sharpness checks. The model then reads each isolated tile again: it must return the same character at confidence 0.90 or higher. This rejects many clipped or mislabeled tiles. These thresholds are model scores and image statistics, not measured accuracy. Ranking favors readable resolution, contrast and sharpness; variants come from distinct source observations. Visual review still matters, especially for similar characters such as I, l and 1.

The page loads this saved alphabet; browsing does not run OCR. Custom letters live in browser storage. When the published starter revision changes, returning visitors get the new starter letters automatically; letters they collected or trimmed themselves are kept. **Download alphabet** saves a JSON backup containing letter images, crop rectangles and source links; **Import alphabet** restores it. The browser collection supports up to 500 variants and reports storage failures. **Print / save PDF** prints the current composition and its source list. Hovering or focusing a composed letter reveals its street photograph behind the fixed letter tile. The photo keeps its aspect ratio; its corresponding letter is centred beneath the tile using the recorded crop geometry. On touch screens, tap once to preview and again to open the source. Escape, scrolling or resizing closes the preview. Clicking a composed letter opens its source photograph. Outlined letters are placeholders for characters not yet collected.

Source photography and artwork retain their existing rights. Exporting a composition or alphabet does not grant an additional reuse licence.
