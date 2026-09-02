# Resolution-sensitive heuristic audit

Reference canvas: 768 × 768. Geometric lengths use
`scale = sqrt(width × height) / 768`; geometric areas use `scale²`.

| Module | Measurement | Classification | Resolution rule |
| --- | --- | --- | --- |
| `repair.reclaim_occluded` | morphology kernel, minimum component, feather | geometric | length / area normalized |
| `repair.fit_layer_tone` | minimum sample and cluster population | area | normalized by scale² |
| `repair.fit_edge_alpha` | inner/outside bands | geometric length | normalized by scale |
| `repair.fit_seam_residual` | seam-side band | geometric length | normalized by scale |
| `repair.clean_garment_orphans` | minimum area, main-mass distance | area / normalized diagonal | scale² / dimensionless |
| `repair.clean_garment_orphans` | two-pixel fringe | raster antialias footprint | intentionally constant; scaling to three pixels at 1024 overreaches valid cloth and fails the exact transfer gate |
| `semantic.semantic_warnings` | minimum iris/sclera evidence | area | normalized by scale² |
| `local_fidelity` | iris/sclera evidence, neckline contact band | area / geometric length | area normalized by scale²; eye padding derives from feature size and neckline band scales by length |
| `ownership` | contact/sample morphology, minimum recovered region | length / area | normalized by scale / scale² |
| `seams` | local band, gap closing, minimum run, validation slack | geometric length | normalized by scale |
| `matting` via WebUI | border ring and edge band | input-image length | caller supplies scale-normalized values |
| `layers.crop_head` | crop expansion | detected head-box ratio | already dimensionless (`w/5`, `h/5`) |
| `portrait_core` coverage/verdict | alpha, area and ratio thresholds | photometric / dimensionless | intentionally resolution-independent |
| fidelity / colour evidence | RGB error, chroma, contrast | photometric | intentionally resolution-independent |

The A002 regression is not caused by a postprocess size threshold alone. With
the same seed and steps, the v3 head diffusion emits `eyewhite` at 768 and does
not emit it at 1024. Postprocess normalization makes subsequent behaviour
predictable, while face-local validation exposes the upstream semantic loss.

## Update: there is no safe resolution

**Withdrawn**: this document previously recommended a fixed 768 head pass as
"the verified safe profile." A later portrait (768, untiled, same head/body
resolution as A-001) reproduced the identical failure -- `eyewhite` missing
from `raw_layers/`, not just trimmed by repair -- proving the loss is not
resolution-specific. A002's 768-vs-1024 comparison showed *a* difference, not
a *safe tier*: the v3 head diffusion's semantic prediction can fail on a given
character at any head resolution, 768 included.

The policy is now: **no head resolution is declared safe in advance.** Each
generation is judged after the fact by comparing the original against the
composite (`local_fidelity.py`'s `missing_visible_eyewhite`: a sclera visible
in the original and lost in the composite -- closed/stylised-shut eyes never
trigger it). Only when that specific loss is observed does
`generation._rescue_head_semantic` act, in two tiers:

1. **Deterministic derivation first** (`eyewhite_derivation.py`, no GPU call):
   copy the sclera straight from the original wherever the evidence around
   the existing `irides` layer is decisive (a real connected patch, mostly
   bright/low-chroma, sitting on head/face support). When this alone clears
   the warning, the resolution ladder below never runs at all.
2. Only if that is not decisive does it re-diffuse the head crop alone
   (never the body) at up to two higher profiles from
   `config/portrait_defaults.json`'s `head_rescue.ladder` (default
   `[896, 1024]`), keeping whichever attempt's eye local fidelity is best.

If both tiers are exhausted and the loss persists, it surfaces as an
explicit `missing_visible_eyewhite` warning and a `REWORK`-eligible bundle
rather than a silently accepted guess.
