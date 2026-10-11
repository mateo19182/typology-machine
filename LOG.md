# Log

Working journal for the eventual blog post. What we tried, what we found, what we decided and why.
Dead ends included on purpose. Newest entries at the bottom.

## 2026-10-11 · Day one

**Starting point.** Wanted a project in the spirit of angleref.com: gather a dataset, run or train
a model, wrap a creative concept around it. Brainstormed a dozen ideas (see IDEAS.md). Two stuck:
HorizonRef (a perspective reference tool for artists) and Typology Machine (Becher-style grids of
street objects, harvested from street-level imagery at city scale). Parked HorizonRef, went with the
typologies. First city: A Coruña, because it's home and because the galerías (the white glazed
balconies) seemed like the obvious first object.

**Mapillary, the first look.** Pulled the vector tiles for the Coruña bbox: 86k images, 7.5k of them
panoramas, 354 sequences, only 15 contributors. 73% of everything is a single 2019 dashcam drive along
the main roads. A contact sheet stratified by year showed what that means in practice: forward-facing,
low-res, lots of tunnel and night shots along the Mariña. The galerías are on upper floors and the
cameras point down the road. **Decision: drop the galerías.**

**Mapillary's own detections.** Mapillary runs segmentation on every image and triangulates "map
features", 3D-placed objects with an image list each. 21.6k of them in Coruña, about 100 classes:
street lights 3.8k, store signs 2.5k, ads 2.1k, poles 2k, trash cans 1k, manholes 943, benches 305.
Since this is free, we used it to get a first crop set without running any model ourselves: for each
feature, pick the image about 9 m away, fetch its detections, keep the polygon of the right class whose
horizontal position matches the bearing from camera to feature, crop with 15% padding. 24 threads,
cached JSON, about two hours. Result: 14.4k crops, 59 classes with a usable size, a local HTML viewer
with one grid per class.

What the crops taught us: duplicates everywhere (the same sign from five frames; one shop appeared 7
times in the top 40), quality all over the place, a few percent mislabeled (windows and laundry as
"store sign"), "trash can" is mostly wheelie bins, and the red Coruña lamp posts are gorgeous.

**Prior art.** Nobody has built the full pipeline (detect every instance, straighten, dedupe, cluster,
grid). Closest in method: StreetStyle (Matzen et al.), Doersch's "What makes Paris look like Paris".
Closest in look: Jenny Odell's satellite collections. Closest in spirit: Yufeng Zhao, especially
"there is only one way" (734,700 NYC one-way signs). Write-up in PRIOR_ART.md and a shared page.

**Decisions, afternoon.** Coruña only. Scraping is fine, paying for map APIs is not. Straightening
later. The time axis is interesting but secondary. Keep docs loose.

**Looking for better imagery.** Google Street View: the two unofficial Python libraries both fail
everywhere right now (Google changed an internal endpoint). KartaView and Panoramax: nothing in
Coruña. Then Apple Look Around via the `streetlevel` library, no key needed: **105k car panoramas in
the Coruña bbox, six faces each, the big ones at 5632×4864.** Night and day compared to the dashcam.
Capture dates are 2019 (14k), 2023 (5k), 2024 (14k), 2025 (11k), 2026 (61k), but Apple only serves
the latest pass per street, so it's a patchwork of dates, not a time machine. Google is the only real
time machine and it's down. **Decision: Apple is the main source from here.**

**SAM 3 vs Mapillary.** Meta gates the SAM 3 weights; an ungated mirror in Transformers format works.
Ran ten plain-English prompts on 24 Mapillary images plus one Apple face, side by side with
Mapillary's polygons. Not close: Mapillary's polygons are fragments (a lamp head, not the post), and
it misses most instances (1 traffic sign vs 51, 2 manholes vs 31). SAM 3 returns whole objects as
single instances and works on the Apple imagery unchanged. What we lose: Mapillary's 3D position per
object, which made cross-frame dedupe trivial. We'll have to do that ourselves.

**Decision, evening:** Apple imagery, SAM 3, start with store signs, run OCR on them. Figure out how
fast the whole thing can go before scaling up.

**Evening: making it fast.** Measured before scaling. Apple serves faces at about 25 per second with
16 threads (2 MB each at full zoom, HEIC). Decoding a face takes 0.3 s of CPU. SAM 3 takes 0.4 s per
face alone, 0.32 s in batches of four; batching barely helps, the model is compute-bound. First attempt
blew the GPU: Transformers upsamples every mask to the full 5632×4864 face. Post-processing at quarter
resolution fixed it (2 GB instead of OOM). Prompt wording matters more than expected: "store sign"
finds 5 signs on a face, "shop sign" and "storefront sign" find 0. Tiling the wide front face into
halves adds almost nothing.

Look Around faces: 0 front (124° wide), 1 right and 3 left (67°, tall), 2 back, 4 up, 5 down. For
shop signs the two side faces are the frontal view, so start there: two faces per pano, 0.5 s per face
all in. Panos are 2.5 m apart, far denser than needed; thinning to one per 6 m cell cuts the 2026 set
from 61k to about 20k panos. Faces aren't stored, only crops and quarter-res masks (re-downloading is
cheaper than 240 GB of HEIC).

First 15 crops from the port area: "store sign" is loose. It also grabs street banners, a street-name
plate and a port notice board, and the same banner shows up from three consecutive panos. So two
stages are coming: a "is this actually a shop's sign" filter, and dedupe across panos. OCR: RapidOCR's
default model is Chinese-trained and hallucinates CJK characters on Latin signs; trying EasyOCR
(Spanish) and Florence-2 next. Launched the old town and centre (3,450 panos, 6,876 faces) as the
first real slice.

**Night: first real slice.** Old town and centre, 3,450 panos, 6,876 side faces, 24.6 minutes,
17,767 "store sign" instances. SigLIP zero-shot sorting over the crops: shop 2.1k, "street plate"
5.5k (which on inspection is mostly small wall-mounted shop signs, so the label description was bad
but the split is still useful), notices 3.8k, traffic 3.4k, other 2.9k, tunnel exit signs 100.
Shop plus plate is the working set: 7,640 crops.

Dedupe by SigLIP similarity plus proximity (same sign within 40 m, cosine ≥ 0.88, different panos):
7,640 crops become 4,681 signs. Porcelanosa's fascia was captured 34 times, a crêperie 25 times.
So in the dense streets a sign is seen from about three to four thinned panos on average, and far
more where the car crawled.

Small lessons: a `pgrep -f` wait loop that matches its own command line never ends. Florence-2 with
three beams and batch 16 does not fit next to the 16 GB the other service holds. 

Started the frontend: a dark wall of tiles with three readings of the same signs, the photo, the
silhouette from the SAM mask, and the OCR'd words alone.

**Late: the GPU fell off the bus.** Florence-2 had read 1,698 of the 7,640 shop crops when CUDA
returned "unspecified launch failure" and nvidia-smi stopped seeing the card ("Unable to determine
the device handle for GPU0"). The card is still on the PCI bus and the other service that holds 16 GB
is still running, blind. Nothing in our code does anything exotic, so this looks like a driver or
hardware hiccup rather than our bug; needs a module reload or a reboot. The 240 signs on the website
got their words from Florence-2 on 32 CPU cores instead.
Follow-up: the kernel log says Xid 79 "GPU has fallen off the bus" then Xid 154 "Node Reboot
Required". First Xid in 60 days of uptime. The card is shared with `henad_inference` (a BiRefNet
background-removal service holding 16 GB) and Immich's CUDA machine learning. Plan: reboot, keep the
henad GPU container down for the night's runs, bring it back after.

## 2026-10-11 · later that night

Rebooted; the card came back clean, henad's inference container left down for the night. Launched the
whole 2026 set (33k thinned panos, 59.5k faces) and finished OCR on the old-town shop crops.

**Decisions:** OCR stays but isn't first class. Clusters get named by a vision-language model.

**Two ways to cluster.** K-means (24) on the 3.5k deduped old-town signs, once on DINOv2 embeddings
and once on SigLIP's. HDBSCAN first: it called 85% noise, so k-means. The two spaces sort the same
signs along different axes. DINOv2 sorts by form: hanging blade signs, long fascias, square plaques,
engraved stone. SigLIP sorts by meaning: street-name plates, "REBAJAS" sale posters, corporate
brands, lawyers' and doctors' nameplates, carved lettering. A typology wants the first; the second is
a catalogue of what the city's signs are *for*. Both worth keeping. Also visible in the SigLIP sheets:
street-name plates are still in the "shop" set, so the filter needs its calibration pass.
