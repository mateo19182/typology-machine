# Coruña alphabet

Open `/alphabet.html` to compose words from photographs of local signs. The starter alphabet uses the installed RapidOCR Latin PP-OCRv5 recognition model. Its CTC alignment supplies individual character positions and confidence values. These are approximate character boundaries, not pixel masks; decorative lettering, clipped characters and recognition mistakes still need review. Each starter tile has a **Trim** control, and users can collect additional letters from the existing sign search.

Rebuild the starter with the query API running:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python explore/build_alphabet.py --limit 100
```

The builder reads at most 100 existing sign crops by default, uses two CPU threads, and writes `data/coruna/site/alphabet-data.json` atomically. It does not rerun object detection or decode original panoramas. It keeps lines with recognition confidence of at least 0.96 and characters with confidence of at least 0.98, rejects mismatched alignment records and nearly blank crops, and selects up to three variants per character from distinct source observations. The confidence thresholds are model scores, not measured accuracy. Generated crop data belongs in the ignored data directory, not Git.

The page loads this saved alphabet; browsing does not run OCR. Custom letters live in browser storage. **Download alphabet** saves a JSON backup containing letter images, crop rectangles and source links; **Import alphabet** restores it. **Print / save PDF** prints the current composition and its source list. Clicking a composed letter opens its source photograph. Outlined letters are placeholders for characters not yet collected.

Source photography and artwork retain their existing rights. Exporting a composition or alphabet does not grant an additional reuse licence.
