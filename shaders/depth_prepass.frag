#version 450

// Depth prepass: depth-only, no colour attachment. The fixed-function depth test writes
// gl_FragCoord.z; this stage produces no colour. It reuses shader.vert so the written depth matches
// the forward pass exactly.
//
// It is NOT empty any more, and the reason is the whole point of the pass: an `alphaMode: MASK`
// material's coverage is not its triangles. The forward shader discards fragments whose base-colour
// alpha falls below the material's cutoff, and a prepass that did not would write depth across the
// holes — so anything BEHIND a cutout failed the forward pass' LESS_OR_EQUAL test and never shaded
// (a leaf card's gaps read as background-coloured nothing), and SSAO, which reconstructs position
// and normal from this depth alone, occluded as if the cutout were a solid sheet.
//
// The test is the SHARED one (material.glsl), on the same UVs from the same vertex path and the same
// bindless material entry, because prepass and forward must discard the same fragments: a fragment
// the prepass keeps and the forward discards leaves a depth-only occluder, and the reverse leaves a
// shaded fragment whose depth nobody wrote.
#include "forward_push.glsl"
#include "material.glsl"

// Only what this stage reads. shader.vert writes more (normals, TBN, clip positions); a fragment
// stage need not declare outputs it ignores.
layout(location = 3) in vec2 fragTexCoord;
layout(location = 8) in vec2 fragTexCoord1;

void main()
{
    // GATED on the material's own cutoff, so an opaque draw pays one scalar SSBO read and NO texture
    // fetch — the prepass covers the whole screen, and this runs per fragment of it.
    //
    // The gate is exactly equivalent to running the test unconditionally, which is what makes it a
    // cost decision rather than a behavioural one — and the equivalence rests on an invariant that is
    // ENFORCED, not assumed. `toMaterialUBO` (graphics/material_binding.cpp) publishes a cutoff only
    // for MASK and 0 for every other mode, AND clamps the packed alpha into glTF's [0,1] and the
    // packed cutoff to >= 0. With alpha >= 0 the skipped test (`alpha < 0`) can never discard, here
    // or in the forward stage that shares this implementation. Without that clamp this gate would BE
    // a bug: a negative alpha discards in the forward pass while this stage keeps the fragment,
    // leaving a depth-only occluder. A MASK material authored with alphaCutoff 0 discards nothing in
    // EITHER pass, which is that value's meaning in the spec.
    if (material.materialParams.z > 0.0)
    {
        if (materialAlphaCutoutFails(materialAlpha(materialBaseColourTexel(fragTexCoord,
                                                                          fragTexCoord1))))
        {
            discard;
        }
    }
}
