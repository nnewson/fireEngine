// Shared VDPM score ABI + apply policy — the ONE GLSL authority for the per-split ScoreOut struct
// layout, its four-channel decision reduction, and the mark / coarsen predicates, so vdpm_score.comp
// (which writes the record), vdpm_mark.comp + vdpm_coarsen.comp (the recorder), and the persistent
// apply kernel can't drift apart. Pure: NO buffer access, atomics, or state marking — a caller
// fetches the ScoreOut, calls these, then applies the result in its own execution model (mirrors how
// vdpm_repair_classify.glsl factors the repair policy). Included via -I shaders.
#ifndef VDPM_APPLY_CLASSIFY_GLSL
#define VDPM_APPLY_CLASSIFY_GLSL

// The per-split score record (std430, buffer_reference_align = 4, 24 B). MUST match
// render/ubo.hpp VdpmScoreOut and the CPU VdpmSplitScore layout.
struct ScoreOut
{
    float geometry;
    float uv;
    float normal;
    float tangent;
    float straddle;
    uint backface;
};

// Decision score = max of the four refine channels. `straddle` is excluded — it is the silhouette
// boost folded into the channels at score time, not a budget channel. Matches VdpmSplitScore::score.
float combinedScore(ScoreOut o)
{
    return max(max(o.geometry, o.uv), max(o.normal, o.tangent));
}

// Refine seed (mark): a FRONT-facing split whose screen error exceeds the refine budget. A
// back-facing split scores 0 on the CPU too, so the explicit backface gate and the raw max agree.
bool applyMarkRequired(ScoreOut o, float pixelBudget)
{
    return o.backface == 0u && combinedScore(o) > pixelBudget;
}

// Coarsen eligibility: a back-facing split, or one whose error dropped BELOW the coarsen budget (the
// hysteresis dead-band's lower edge). The exact complement of the recorder's "keep it" test
// (`backface == 0 && score >= coarsenBudget`). Leaf-gating + the collapse stay in the caller.
bool applyCoarsenEligible(ScoreOut o, float coarsenBudget)
{
    return o.backface != 0u || combinedScore(o) < coarsenBudget;
}

#endif // VDPM_APPLY_CLASSIFY_GLSL
