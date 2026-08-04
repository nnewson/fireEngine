// The BINDLESS MATERIAL AUTHORITY, declared ONCE and shared by every stage that reads a material.
//
// Field offsets in a block depend on every field before them, so a second hand-written copy of
// `MaterialData` is a latent misread of everything past the first divergence — with no validation
// error and no crash (cmake/check_shader_blocks.cmake carries the story of the last one). SH-05 gave
// the shadow pass a fragment path that must apply the *visible* material's alpha cutout, so this
// block now has more than one reader and belongs here rather than inside shader.frag.
//
// CONTRACT for an including shader:
//   * declare its push-constant block, named `pc`, with a `uint materialIndex` member, BEFORE the
//     include (forward stages use ForwardPushConstants; shadow stages use shadow_push.glsl). This
//     file indexes materials[] through it, which is what makes "one material per draw, dynamically
//     uniform" true for every reader.
//   * bind the bindless set at SET 2. Every pipeline that opts into bindless declares set 2 for it,
//     including the shadow pipelines, whose otherwise-absent set 1 is declared empty purely so this
//     number never varies by pass (see Pipeline::createPipelineLayout).
//
// The C++ side is MaterialUBO in render/ubo.hpp; std430 here matches its std140 because every member
// is 16-byte aligned (vec4 / ivec4 / UvXform / ivec4[]).

// Runtime-sized bindless texture array below.
#extension GL_EXT_nonuniform_qualifier : require

// KHR_texture_transform packed per material texture slot. `offsetScale.xy` is the UV offset;
// `offsetScale.zw` is the UV scale (identity = 0,0,1,1). `rotation` is radians CCW. Matches the
// std140 stride of the matching C++ struct (16-byte vec4 + float, padded to 32 bytes).
struct UvXform {
    vec4 offsetScale;
    float rotation;
};

// Material texture slots, ordered to match MaterialTextureSlot in
// render/descriptor_bindings.hpp. uv[SLOT_*] indexes the UV-xform array.
const int SLOT_BASE_COLOUR = 0;
const int SLOT_EMISSIVE = 1;
const int SLOT_NORMAL = 2;
const int SLOT_METALLIC_ROUGHNESS = 3;
const int SLOT_OCCLUSION = 4;
const int SLOT_TRANSMISSION = 5;
const int SLOT_CLEARCOAT = 6;
const int SLOT_CLEARCOAT_ROUGHNESS = 7;
const int SLOT_CLEARCOAT_NORMAL = 8;
const int SLOT_THICKNESS = 9;

struct MaterialData {
    vec4 diffuseAlpha;
    vec4 emissiveRoughness;
    // .y = normal scale, .z = alphaCutoff (0 for every mode but MASK — see Object::toMaterialUBO,
    // which is why the cutout test is inert rather than wrong on an opaque material).
    vec4 materialParams;
    ivec4 textureFlags;
    // .x = occlusion-texture present, .y = occlusion's UV-set index.
    ivec4 extraFlags;
    // x=baseColor, y=emissive, z=normal, w=metallicRoughness UV-set index.
    ivec4 texCoordIndices;
    // KHR_materials_transmission + KHR_materials_ior. .x = transmissionFactor,
    // .y = texture-present flag, .z = transmission texCoord index, .w = ior.
    vec4 transmissionParams;
    // KHR_materials_clearcoat. .x = factor, .y = roughness, .z = normalScale.
    vec4 clearcoatParams;
    // .x = factor texture present, .y = roughness texture present,
    // .z = normal texture present (all 0 / 1 floats).
    vec4 clearcoatFlags;
    // .x = factor texCoord, .y = roughness texCoord, .z = normal texCoord
    // (encoded as floats).
    vec4 clearcoatTexCoords;
    // KHR_materials_volume.
    //   .x = thicknessFactor (world units, scaled by node max scale)
    //   .y = thickness texture present (0 / 1)
    //   .z = thickness texCoord index (0 / 1)
    //   .w = reserved (thickness rotation lives in uv[SLOT_THICKNESS].rotation).
    vec4 volumeParams;
    // .rgb = attenuationColor, .a = attenuationDistance (huge finite when
    // the spec says +infinity — see Object::toMaterialUBO).
    vec4 attenuation;
    UvXform uv[10];
    // Bindless index into the global `textures[]` array per material texture slot
    // (packed as 3 ivec4s; matTex(SLOT_*) unpacks). Valid only where the slot's
    // present-flag is set. Matches MaterialUBO::textureIndex in render/ubo.hpp.
    ivec4 textureIndex[3];
};

// Global materials SSBO (bindless set 2, binding 1), indexed per-draw by the push constant.
// `material` aliases this draw's entry so reads stay `material.*`.
layout(std430, set = 2, binding = 1) readonly buffer Materials {
    MaterialData materials[];
};
#define material materials[pc.materialIndex]

// Bindless material texture array (set 2, binding 0). Indexed by matTex(slot).
layout(set = 2, binding = 0) uniform sampler2D textures[];

// Unpack the per-slot bindless texture index from the packed ivec4[3].
int matTex(int slot) { return material.textureIndex[slot >> 2][slot & 3]; }

// KHR_texture_transform: scale → rotate → translate (CCW around origin).
// offsetScale.xy = offset, offsetScale.zw = scale.
vec2 applyUvTransform(vec2 uv, vec4 offsetScale, float rotation)
{
    vec2 scaled = uv * offsetScale.zw;
    float c = cos(rotation);
    float s = sin(rotation);
    vec2 rotated = vec2(c * scaled.x - s * scaled.y,
                        s * scaled.x + c * scaled.y);
    return rotated + offsetScale.xy;
}

// This material's UV for one texture slot: pick the authored TEXCOORD set, then transform it.
// glTF allows each slot to choose set 0 or 1, so both are passed in and the material chooses —
// keeping this file free of any particular stage's varying names.
vec2 materialSlotUv(int slot, int uvSetIndex, vec2 uv0, vec2 uv1)
{
    vec2 uv = uvSetIndex == 0 ? uv0 : uv1;
    return applyUvTransform(uv, material.uv[slot].offsetScale, material.uv[slot].rotation);
}

// The base-colour texel, or opaque white when the material carries no base-colour texture — so
// callers can multiply unconditionally.
vec4 materialBaseColourTexel(vec2 uv0, vec2 uv1)
{
    if (material.textureFlags.x != 1) {
        return vec4(1.0);
    }
    return texture(textures[matTex(SLOT_BASE_COLOUR)],
                   materialSlotUv(SLOT_BASE_COLOUR, material.texCoordIndices.x, uv0, uv1));
}

// Surface alpha: base-colour factor times the texel's alpha (1.0 when untextured).
float materialAlpha(vec4 baseColourTexel)
{
    return material.diffuseAlpha.a * baseColourTexel.a;
}

// THE alpha-cutout test — one implementation for every pass, which is the point of it living here.
// Shadow and forward must agree fragment-for-fragment: a cutout whose shadow is tested against a
// different cutoff (or a different UV transform) casts a silhouette its own surface does not have,
// and the disagreement looks like a shadow bias bug rather than a mask bug. `materialParams.z` is 0
// for every non-MASK material, so this is inert there rather than wrong.
bool materialAlphaCutoutFails(float alpha)
{
    return alpha < material.materialParams.z;
}
