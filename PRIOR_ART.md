# Prior art (searched 2026-10-11)

Related projects cover detection, crop collections, classification and mapping. The references
below are examples, not an exhaustive search or evidence that our combination is unique.

| Project | What it is | How it differs |
|---|---|---|
| [What Makes Paris Look Like Paris?](http://graphics.cs.cmu.edu/projects/whatMakesParis/) (Doersch et al., 2012) | Mines city-specific Street View patches (doors, balconies, railings) across 12 cities | Square patches rather than whole objects, no straightening, a paper figure rather than a typology |
| [Linking Past to Present](https://vision.luddy.iu.edu/projects/general-papers/linking-past-to-present.html) (2015) | Same method across two centuries of Paris architecture | Change over time, not across cities |
| [StreetStyle](http://streetstyle.cs.cornell.edu/) (Matzen et al., 2017) | Detect people, embed crops, 400 style clusters, rank by city | The closest method, but for clothing on Instagram |
| [Urban Mosaic](https://arxiv.org/abs/2008.13321) (NYU, 2020) | Similarity search over 7.7M NYC street images | Whole images; a planning tool |
| [Searchable.City](https://searchable.city/) (2025–26) | VLM tags every Street View image in NYC and LA | Whole images on a map |
| [all text in nyc](https://alltext.nyc/) (Yufeng Zhao, 2024) | OCR over every NYC panorama | Text only. In the [Show HN thread for a London clone](https://news.ycombinator.com/item?id=45995913), people asked for bins and street furniture |
| [Jenny Odell, Satellite Collections](https://www.jennyodell.com/satellite.html) (2009–11) | Hand-cut satellite objects in grids | The closest in look, but manual and top-down |
| [Terrapattern](https://studioforcreativeinquiry.org/project/terrapattern) (2016) | "Find similar" over satellite imagery | Satellite only, no grids |
| [Multiplicity](https://der-mo.net/projects/multiplicity) (Stefaner) | 25k Paris photos in a t-SNE grid | Whole photos |
| [STDL manholes](https://tech.stdl.ch/PROJ-STREETVIEW/), [Outerview hydrants](https://huggingface.co/datasets/Outerview/fire-hydrants-dataset) | Detection, triangulation, crop datasets | Infrastructure work, never presented as art |
| "Doors of Dublin" posters, Rafman's *Nine Eyes* | The folk and art versions | Hand-picked |

## Gaps to claim
- Object-level crops, perspective-corrected to a frontal view at a shared scale.
- Every instance, deduplicated, shown together (infrastructure projects deduplicate but never show).
- StreetStyle-style sub-type clustering for street objects, compared across cities.
- Open-vocabulary classes (SAM 3) instead of a fixed list.
- Becher presentation discipline: consistent scale, neutral ground, place and date captions, print-ready.
- Time: how a city's typology changes across capture years.

## Yufeng Zhao ([yufengzhao.com](https://www.yufengzhao.com/))
The artist behind all text in nyc. Relevant works:
- [there is only one way](https://thereisonlyoneway.com) (2024): 734,700 NYC "one way" signs from Street View, given out by scratch card. One class, every instance.
- [Aligned Waters](https://aligned-waters.yufeng.place) (2025): seascapes aligned by hand-labelled horizons. Alignment is the edit.
- [Desk Setup](https://desk-setup.yufeng.place) (2025): a 307-photo typology from Unsplash.
- [NYC Street Clock](https://clock.alltext.nyc) (2026): a clock built from 20k+ Street View captures.
- [Looking for Love (in NYC)](https://vimeo.com/1176785928) (2026): a performance of searching panoramas for "LOVE".
- [diffused·city](https://diffused.city) (2024): diffusion-model panoramas of real cities.

## Graffiti detection and Walzr projects, checked 2026-10-12

- [Walzr's SF Graffiti](https://walzr.com/sf-graffiti) collects city inspectors' violation photos scraped from the city website. This is a useful presentation and provenance reference, but it does not document a Street View graffiti detector. Its linked CSV includes request IDs, locations, dates and an AI beauty score; no detection boxes or masks.
- [Weather Watching](https://walzr.com/weather-watching) uses YOLO to detect and track people, selects a clear frame per person, and sends crops to Gemini for classification. Relevant to our detector, representative crop, classifier sequence. Video tracking does not solve our cross-panorama identity problem.
- [LA Damage](https://walzr.com/la-damage) shows color-coded damage classes on a map with before/after images from county, Street View and Compass sources. Its assessment data comes from the county assessor; the page does not establish an automated damage classifier.
- [17K-Graffiti](https://github.com/visual-ds/17K-Graffiti) provides a Street View graffiti detection precedent with Faster R-CNN. [Dataset and weights](https://zenodo.org/records/5899631) require approval and are restricted to academic use.
- [Graffitiview](https://github.com/vladysolonyi/graffitiview) collects panoramas, splits them and detects graffiti for an archive. Its README explicitly lists improving the model for panorama images as future work.
- [STORM](https://zenodo.org/records/3238357) supplies 1,022 smartphone images with graffiti/tagging boxes. [KGD](https://noonv.github.io/KGD/) has 1,927 images and a YOLOv5s study. These are possible training sources, not verified improvements over SAM3 on Coruña; check reuse terms and checkpoint availability first.

Proposed small detector comparison: separate SAM3 prompts with shared image features versus
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) on the same 30–50 full source
views, including graffiti-positive scenes and negatives. Review missed objects, false positives,
crop completeness, duplicate boxes and runtime. Existing crop-category reviews cannot measure
missed detections. Keep masks saved and classify rectangular crops. No detector replacement or
specialist training has been decided.

Local category-review snapshot: 40 human labels on sample `4a52d925395c30ed`; Intern-Decision-2B
matched 33/40 and Qwen3.5-2B matched 27/40. None of those human labels were graffiti or street-name
plates. These are preliminary results on a small reviewed subset, not a city-wide accuracy estimate.
