# Prior art (searched 2026-10-11)

Nobody seems to have done the full pipeline: automatic detection, perspective-corrected crops,
Becher-style grids per object type, sub-type clustering, and city comparison. The pieces exist separately.

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
