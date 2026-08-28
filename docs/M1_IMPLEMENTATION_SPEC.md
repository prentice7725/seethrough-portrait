# M1 Implementation Specification

## Goal

Add a portrait-specific safety path to tk_seethrough that preserves the trusted
source silhouette even when semantic decomposition omits shoulders, sleeves, or
arms. M1 does not attempt to improve generation quality or implement a 2.5D rig.

## Runtime contract

### Inputs

| Input | Type | Required | Meaning |
| --- | --- | --- | --- |
| `original_rgba` | `uint8[H,W,4]` | yes | Source portrait and recovery pixels |
| `layers` | `dict[str, uint8[H,W,4]]` | yes | Generated semantic layers |
| `subject_mask` | mask | conditional | Independent foreground mask for opaque sources |
| `portrait_mode` | bool | yes | Enables portrait completeness rules |
| `silhouette_guard` | bool | yes | Enables clipping and remainder recovery |

An informative source alpha has priority over a supplied mask. If neither an
informative source alpha nor an independent mask exists, the generated union is
only LOW-confidence evidence and cannot produce a hard PASS.

### Outputs

| Output | Purpose |
| --- | --- |
| semantic layers | Existing tk_seethrough result, clipped to the trusted subject |
| `body_remainder` | Original pixels unexplained by semantic layers |
| `coverage_mask` | Pre-recovery union of semantic alpha |
| `missing_mask` | Positive difference between subject and semantic union |
| reconstruction preview | Semantic union plus `body_remainder` |
| JSON report | Coverage, recovery, semantic status, verdict, and reasons |

## Core functions

| Function | Responsibility |
| --- | --- |
| `resolve_subject_mask` | Select source alpha, supplied mask, segmentation, or guarded fallback |
| `compose_union_alpha` | Alpha-compose all valid semantic layers |
| `build_body_remainder` | Recover the exact residual alpha from original pixels |
| `apply_silhouette_guard` | Clip spill, calculate missing pixels, recover, and classify |
| `evaluate_portrait_layers` | Apply crop-aware FACE/HAIR/BODY completeness rules |
| `select_portrait_layers` | Rank auto-fill candidates using portrait coverage scoring |
| `build_portrait_report` | Keep recovery verdict separate from semantic verdict |

The remainder alpha uses the source-over inverse:

`A_remainder = (A_subject - A_generated) / (1 - A_generated)`

for pixels where the denominator is safe. This reconstructs partially
transparent source edges without double-counting alpha.

## Verdict rules

| Verdict | Meaning |
| --- | --- |
| `PASS` | Strong pre-recovery coverage with little or no remainder dependency |
| `SOFT_PASS` | Final silhouette is preserved but some pixels rely on the remainder |
| `REWORK` | Recovery succeeded, but semantic omission is too large for clean use |
| `FAIL` | Post-recovery silhouette still misses the required threshold |
| `SOFT_PASS_LOW_CONFIDENCE` | Coverage appears usable but no independent subject mask proves it |

Semantic completeness is reported independently. A 100% reconstructed
silhouette is not evidence that FACE/HAIR/BODY were separated correctly.

## M1 acceptance

1. Trusted-alpha A-001 resolves as HIGH confidence.
2. A simulated arm/sleeve omission is recovered pixel-for-pixel.
3. Post-recovery silhouette coverage is at least 0.999.
4. Large remainder dependency is classified as REWORK rather than PASS.
5. Existing node contracts remain compatible when portrait mode is disabled.
6. Core unit tests pass without loading diffusion models.

## Out of scope

- WebUI implementation (M2)
- folder batch processing (M3)
- rig implementation or automated rigging (M4 feasibility only)
- claims about real model quality before GPU A/B/C runs

