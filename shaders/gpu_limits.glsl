// The values C++ and GLSL must agree on — ONE declaration, read by both languages. Data-layout
// limits, mostly, plus the encoding of any field whose meaning is split across the seam.
//
// This file is written in the subset that is simultaneously valid GLSL and valid C++, and it is
// included by both: shaders take it through the shaders/ include path, and
// `include/fire_engine/graphics/gpu_limits.hpp` includes it inside a namespace and re-exports every
// value under the engine's `k`-names. There is therefore no second copy to drift — the numbers, and
// the arithmetic deriving one from another, exist here and nowhere else.
//
// It exists because they DID drift-by-hand for a long time: `SHADOW_TOTAL_MATRIX_COUNT = 32` in
// shadow.vert, `SHADOW_POINT_MATRIX_BASE = 8` in shadow_depth.glsl and the caster counts in
// light_ubo.glsl were each a transcription of gpu_limits.hpp. A one-sided change to any of them
// compiles cleanly on both sides and then indexes the wrong region of a UBO: every index stays in
// range, so there is no validation error and no crash, just a shadow matrix read from a slot that
// belongs to another family.
//
// KEEP IT IN THE COMMON SUBSET. Only `const int NAME = <integer constant expression>;`, `//`
// comments, and this include guard are portable between GLSL 450 and C++. No `constexpr`, no
// `inline`, no namespaces, no unsigned suffixes, no `static_cast` — anything else compiles in one
// language and fails in the other, and the failure surfaces as a shader-compile error in a file
// that never mentions this one.

#ifndef FIRE_ENGINE_GPU_LIMITS_GLSL
#define FIRE_ENGINE_GPU_LIMITS_GLSL

// Cap on lights consumed by the forward shader's main lighting loop.
const int MAX_LIGHTS = 8;

// Skinning joint matrices per SkinUBO.
const int MAX_JOINTS = 64;

// Morph target weights per MorphUBO. The block stores them PACKED as vec4s, so the shader-side
// array length is the derived count below rather than the weight count itself — writing `weights[2]`
// by hand is how a ninth weight would land outside the block with the C++ side none the wiser. The
// division is exact by construction; `gpu_limits.hpp` asserts the divisibility, which GLSL cannot.
const int MAX_MORPH_TARGETS = 8;
const int MORPH_WEIGHT_VEC4_COUNT = MAX_MORPH_TARGETS / 4;

// GPU particle pool: each active emitter owns a contiguous slice of the pool.
const int MAX_PARTICLE_EMITTERS = 4;

// SSAO hemisphere kernel size — the UBO array length and the loop bound in ssao.frag.
const int SSAO_KERNEL_SIZE = 16;

// Directional cascade layers in the 2D-array shadow map.
const int SHADOW_CASCADE_COUNT = 4;

// Per-skinned-object self-shadow slots (LightUBO::selfShadowViewProj).
const int MAX_SKINNED_SELF_SHADOW_CASTERS = 4;

// Shadow casters for punctual lights. Independent of MAX_LIGHTS; excess punctual lights remain
// unshadowed.
const int MAX_SPOT_SHADOW_CASTERS = 4;
const int MAX_POINT_SHADOW_CASTERS = 4;

// Faces of a cube map.
const int CUBE_FACE_COUNT = 6;

// (The shadow matrix TABLE is gone. Every shadow path rasterises with `pc.lightViewProj`, the
// matrix of the view being recorded, so there is no per-object array of every shadow transform and
// no slot arithmetic — cascade/spot/point bases and a total count — to keep in step.)

// Which shadow-map families were RECORDED this frame, packed as a bitmask in
// LightUBO::shadowMapValidMask. Not a limit, but it lives here for the same reason the limits do:
// the renderer sets the bits and the receiver reads them, so a bit that means one thing on one side
// and another on the other is a silent misread. Both sides take these declarations from here.
//
// A family whose bit is CLEAR was not rendered this frame and its depth is stale. The receiver must
// answer "fully lit" for it rather than sample: the whole point of skipping a family is that its
// texture no longer describes the scene, and "the stale content happens to be harmless" is not a
// property anything checks.
const int SHADOW_MAP_VALID_CASCADES = 1;
const int SHADOW_MAP_VALID_WORLD_ONLY = 2;
const int SHADOW_MAP_VALID_SELF = 4;
const int SHADOW_MAP_VALID_SPOT = 8;
const int SHADOW_MAP_VALID_POINT = 16;

#endif // FIRE_ENGINE_GPU_LIMITS_GLSL
