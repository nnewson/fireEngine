#version 450
// textureSize() and similar queries on plain texture* uniforms (no sampler
// attached) need this — we use it for the shadow map array which is bound
// as a sampledImage so all shadow maps can share one comparison sampler.
#extension GL_EXT_samplerless_texture_functions : require

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    int hasSkin;
} ubo;

// KHR_texture_transform packed per material texture slot. `offsetScale.xy` is
// the UV offset; `offsetScale.zw` is the UV scale (identity = 0,0,1,1).
// `rotation` is radians CCW. Matches std140 stride of the matching C++ struct
// in render/ubo.hpp (16-byte vec4 + float, padded to 32 bytes).
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

layout(binding = 1) uniform MaterialUBO {
    vec4 diffuseAlpha;
    vec4 emissiveRoughness;
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
} material;

layout(binding = 2) uniform sampler2D texSampler;

layout(binding = 6) uniform sampler2D emissiveMap;
layout(binding = 7) uniform sampler2D normalMap;
layout(binding = 8) uniform sampler2D metallicRoughnessMap;
layout(binding = 9) uniform sampler2D occlusionMap;
layout(binding = 16) uniform sampler2D transmissionMap;
layout(binding = 17) uniform sampler2D clearcoatMap;
layout(binding = 18) uniform sampler2D clearcoatRoughnessMap;
layout(binding = 19) uniform sampler2D clearcoatNormalMap;
// KHR_materials_transmission F3 — captured post-opaque scene colour with mip
// chain. Transmissive draws sample this at a screen-space UV displaced by
// the refracted ray; roughness drives the mip level for frosted-glass blur.
layout(set = 1, binding = 12) uniform sampler2D sceneColorMap;
// KHR_materials_volume — thickness texture (G channel multiplies the volume
// thicknessFactor). Drives both the refracted exit point and the Beer-Lambert
// path length.
layout(binding = 21) uniform sampler2D thicknessMap;
// Shadow images bound as plain textures so one comparison sampler can be
// reused across CSM, spot, and point maps (Apple's per-stage sampler limit is
// 16). Combined samplers are constructed at use time via the GLSL sampler*()
// constructors.
layout(set = 1, binding = 1) uniform texture2DArray shadowMapTex;
layout(set = 1, binding = 4) uniform texture2DArray spotShadowMapTex;
layout(set = 1, binding = 5) uniform textureCubeArray pointShadowMapTex;
layout(set = 1, binding = 7) uniform sampler shadowCompareSampler;
layout(set = 1, binding = 8) uniform sampler shadowDebugSampler;
layout(set = 1, binding = 6) uniform texture2DArray shadowDebugImageTex;
layout(set = 1, binding = 2) uniform texture2DArray worldShadowMapTex;
layout(set = 1, binding = 3) uniform texture2DArray selfShadowMapTex;
layout(set = 1, binding = 9) uniform samplerCube irradianceMap;
layout(set = 1, binding = 10) uniform samplerCube prefilteredMap;
layout(set = 1, binding = 11) uniform sampler2D brdfLut;

struct LightData {
    // .xyz = world position (point/spot), .w = type (0=dir, 1=point, 2=spot)
    vec4 position;
    // .xyz = world forward, .w = range (point/spot; 0 = infinite). For point
    // shadow casters, .w is the effective range used by the shadow pass.
    vec4 direction;
    // .rgb = colour, .a = intensity
    vec4 colour;
    // .x = cos(innerCone), .y = cos(outerCone), .z = shadow index (-1 = none)
    vec4 cone;
};

const int MAX_LIGHTS = 8;
const int MAX_SPOT_SHADOW_CASTERS = 4;

// Forward globals — descriptor set 1 holds bindings shared by every draw
// (light UBO, shadow maps, IBL textures, sceneColor). Bound once per frame
// in Renderer; reused across all forward pipelines.
layout(set = 1, binding = 0) uniform LightUBO {
    mat4 cascadeViewProj[4];
    mat4 spotViewProj[MAX_SPOT_SHADOW_CASTERS];
    mat4 selfShadowViewProj[4];
    vec4 cascadeSplits;
    vec4 iblParams;
    vec4 shadowParams;
    vec4 pointSpotShadowParams;
    vec4 environmentParams;
    int  lightCount;
    int  _pad0;
    int  _pad1;
    int  _pad2;
    LightData lights[MAX_LIGHTS];
} light;

layout(push_constant) uniform ForwardPushConstants {
    int selfShadowSlot;
} pc;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in mat3 fragTBN;
layout(location = 7) in float fragViewDepth;
layout(location = 8) in vec2 fragTexCoord1;

layout(location = 0) out vec4 outColor;

vec2 pickUv(int index)
{
    return index == 0 ? fragTexCoord : fragTexCoord1;
}

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

const float PI = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution
float distributionGGX(float NdotH, float alpha)
{
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry term (single direction)
float geometrySchlickGGX(float cosTheta, float k)
{
    return cosTheta / (cosTheta * (1.0 - k) + k);
}

// Smith's method combining view and light geometry terms
float geometrySmith(float NdotV, float NdotL, float roughness)
{
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return geometrySchlickGGX(NdotV, k) * geometrySchlickGGX(NdotL, k);
}

// Schlick Fresnel approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// 16-tap Poisson disk for directional CSM PCF.
const vec2 poissonDisk[16] = vec2[16](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
);

// Per-pixel rotation hash so neighbouring fragments use different rotations
// of the same Poisson kernel. Stops the kernel pattern from showing as moiré.
mat2 poissonRotation(vec3 worldPos)
{
    float h = fract(sin(dot(worldPos.xy + worldPos.zx, vec2(12.9898, 78.233))) * 43758.5453);
    float c = cos(h * 6.283185);
    float s = sin(h * 6.283185);
    return mat2(c, -s, s, c);
}

float sampleDirectionalShadowFrom(texture2DArray shadowTex, vec3 worldPos, vec3 normal,
                                  vec3 lightDir, int cascade)
{
    vec3 sampleWorldPos = worldPos + normal * light.shadowParams.w * exp2(float(cascade));
    vec4 lightSpace = light.cascadeViewProj[cascade] * vec4(sampleWorldPos, 1.0);
    vec3 proj = lightSpace.xyz / lightSpace.w;
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.z < 0.0
        || any(lessThan(proj.xy, vec2(0.0)))
        || any(greaterThan(proj.xy, vec2(1.0)))) {
        return 1.0;
    }

    float minBias = light.shadowParams.x;
    float slopeBias = light.shadowParams.y;
    float baseBias = max(minBias, slopeBias * (1.0 - max(dot(normal, lightDir), 0.0)));
    // Far cascades cover proportionally more world per texel; bias must scale
    // with cascade size.
    float bias = baseBias * exp2(float(cascade));
    float receiverDepth = proj.z - bias;

    mat2 rot = poissonRotation(worldPos);
    float texelSize = 1.0 / float(textureSize(shadowTex, 0).x);
    float filterRadius = max(light.shadowParams.z, 0.0) * texelSize;

    float vis = texture(sampler2DArrayShadow(shadowTex, shadowCompareSampler),
                        vec4(proj.xy, float(cascade), receiverDepth));
    if (filterRadius <= 0.0) {
        return vis;
    }
    for (int i = 0; i < 16; ++i) {
        vec2 off = rot * poissonDisk[i] * filterRadius;
        vis += texture(sampler2DArrayShadow(shadowTex, shadowCompareSampler),
                       vec4(proj.xy + off, float(cascade), receiverDepth));
    }
    return vis / 17.0;
}

float sampleSelfShadow(vec3 worldPos, vec3 normal, vec3 lightDir, int slot)
{
    if (slot < 0 || slot >= 4) {
        return 1.0;
    }

    vec3 sampleWorldPos = worldPos + normal * light.shadowParams.w;
    vec4 lightSpace = light.selfShadowViewProj[slot] * vec4(sampleWorldPos, 1.0);
    vec3 proj = lightSpace.xyz / lightSpace.w;
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.z < 0.0
        || any(lessThan(proj.xy, vec2(0.0)))
        || any(greaterThan(proj.xy, vec2(1.0)))) {
        return 1.0;
    }

    float baseBias = max(light.shadowParams.x,
                         light.shadowParams.y * (1.0 - max(dot(normal, lightDir), 0.0)));
    float receiverDepth = proj.z - baseBias;
    return texture(sampler2DArrayShadow(selfShadowMapTex, shadowCompareSampler),
                   vec4(proj.xy, float(slot), receiverDepth));
}

int selectCascade(float viewDepth)
{
    int cascade = 3;
    for (int i = 0; i < 4; ++i)
    {
        if (viewDepth < light.cascadeSplits[i])
        {
            cascade = i;
            break;
        }
    }
    return cascade;
}

// Blend factor in the last 10% of a cascade's view-space range. 0.0 = pure
// current cascade; 1.0 = pure next cascade. Always 0.0 for the last cascade.
float cascadeBlendFactor(int cascade, float viewDepth)
{
    if (cascade >= 3)
        return 0.0;
    float cascadeStart = cascade == 0 ? 0.0 : light.cascadeSplits[cascade - 1];
    float cascadeEnd = light.cascadeSplits[cascade];
    float blendBand = (cascadeEnd - cascadeStart) * 0.1;
    float blendStart = cascadeEnd - blendBand;
    return clamp((viewDepth - blendStart) / blendBand, 0.0, 1.0);
}

float computeShadow(vec3 worldPos, vec3 normal, vec3 lightDir, int cascade, float viewDepth)
{
    float current = sampleDirectionalShadowFrom(shadowMapTex, worldPos, normal, lightDir, cascade);
    float t = cascadeBlendFactor(cascade, viewDepth);
    if (t <= 0.0)
        return current;
    float next = sampleDirectionalShadowFrom(shadowMapTex, worldPos, normal, lightDir, cascade + 1);
    return mix(current, next, t);
}

float computeWorldShadow(vec3 worldPos, vec3 normal, vec3 lightDir, int cascade, float viewDepth)
{
    float current =
        sampleDirectionalShadowFrom(worldShadowMapTex, worldPos, normal, lightDir, cascade);
    float t = cascadeBlendFactor(cascade, viewDepth);
    if (t <= 0.0)
        return current;
    float next =
        sampleDirectionalShadowFrom(worldShadowMapTex, worldPos, normal, lightDir, cascade + 1);
    return mix(current, next, t);
}

vec2 directionalShadowDepths(vec3 worldPos, vec3 normal, vec3 lightDir, int cascade)
{
    vec3 sampleWorldPos = worldPos + normal * light.shadowParams.w * exp2(float(cascade));
    vec4 lightSpace = light.cascadeViewProj[cascade] * vec4(sampleWorldPos, 1.0);
    vec3 proj = lightSpace.xyz / lightSpace.w;
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.z < 0.0
        || any(lessThan(proj.xy, vec2(0.0)))
        || any(greaterThan(proj.xy, vec2(1.0)))) {
        return vec2(0.0, 0.0);
    }

    float minBias = light.shadowParams.x;
    float slopeBias = light.shadowParams.y;
    float baseBias = max(minBias, slopeBias * (1.0 - max(dot(normal, lightDir), 0.0)));
    float bias = baseBias * exp2(float(cascade));
    float receiverDepth = proj.z - bias;
    float storedDepth = texture(sampler2DArray(shadowDebugImageTex, shadowDebugSampler),
                                vec3(proj.xy, float(cascade))).r;
    return vec2(receiverDepth, storedDepth);
}

void main() {
    vec3 N;
    if (material.textureFlags.z == 1) {
        vec2 uvNormal = applyUvTransform(pickUv(material.texCoordIndices.z),
                                         material.uv[SLOT_NORMAL].offsetScale,
                                         material.uv[SLOT_NORMAL].rotation);
        vec3 mapNormal = texture(normalMap, uvNormal).rgb * 2.0 - 1.0;
        mapNormal.xy *= material.materialParams.y;
        N = normalize(fragTBN * mapNormal);
    } else {
        N = normalize(fragNormal);
    }

    // Use the geometric mesh normal for shadow receiver bias; tangent-space
    // normal maps affect BRDF shading, not geometric visibility.
    vec3 shadowNormal = normalize(fragNormal);

    vec3 V = normalize(ubo.cameraPos.xyz - fragWorldPos);
    float NdotV = max(dot(N, V), 0.001);
    // KHR_materials_ior. Use the authored dielectric IOR to derive F0 instead
    // of the previous hard-coded 0.04 baseline. This keeps Air (IOR = 1.0)
    // close to fully transmissive while preserving the spec's default 1.5 →
    // ~0.04 reflectance when the extension is absent.
    float ior = max(material.transmissionParams.w, 1e-4);

    // Sample base colour texture once
    vec4 texColor = vec4(1.0);
    if (material.textureFlags.x == 1) {
        vec2 uvBase = applyUvTransform(pickUv(material.texCoordIndices.x),
                                       material.uv[SLOT_BASE_COLOUR].offsetScale,
                                       material.uv[SLOT_BASE_COLOUR].rotation);
        texColor = texture(texSampler, uvBase);
    }
    vec3 baseColor = material.diffuseAlpha.rgb * fragColor * texColor.rgb;

    // Alpha
    float alpha = material.diffuseAlpha.a;
    if (material.textureFlags.x == 1) {
        alpha *= texColor.a;
    }
    if (alpha < material.materialParams.z) discard;

    if (light.environmentParams.z > 0.5 && light.environmentParams.z < 1.5) {
        outColor = vec4(N * 0.5 + 0.5, alpha);
        return;
    }

    // KHR_materials_unlit. Skip BRDF/IBL/shadow entirely; output the textured
    // base colour directly. Post-process tonemap still runs on the HDR target.
    if (material.extraFlags.z == 1) {
        outColor = vec4(baseColor, alpha);
        return;
    }

    // Metallic/roughness — sample from texture if available
    float roughness = material.emissiveRoughness.a;
    float metallic = material.materialParams.x;
    if (material.textureFlags.w == 1) {
        vec2 uvMr = applyUvTransform(pickUv(material.texCoordIndices.w),
                                     material.uv[SLOT_METALLIC_ROUGHNESS].offsetScale,
                                     material.uv[SLOT_METALLIC_ROUGHNESS].rotation);
        vec4 mrSample = texture(metallicRoughnessMap, uvMr);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);

    float dielectricF0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    vec3 F0 = mix(vec3(dielectricF0), baseColor, metallic);
    float a = roughness * roughness;

    // Direct lighting loop — accumulate contributions from every light in
    // KHR_materials_clearcoat. Resolve the per-fragment clearcoat factor /
    // roughness / normal once, ahead of the per-light loop.
    float clearcoat = material.clearcoatParams.x;
    float ccRough = material.clearcoatParams.y;
    float ccNormalScale = material.clearcoatParams.z;
    if (material.clearcoatFlags.x > 0.5) {
        vec2 ccUv = applyUvTransform(pickUv(int(material.clearcoatTexCoords.x)),
                                     material.uv[SLOT_CLEARCOAT].offsetScale,
                                     material.uv[SLOT_CLEARCOAT].rotation);
        clearcoat *= texture(clearcoatMap, ccUv).r;
    }
    if (material.clearcoatFlags.y > 0.5) {
        vec2 ccRuv = applyUvTransform(pickUv(int(material.clearcoatTexCoords.y)),
                                      material.uv[SLOT_CLEARCOAT_ROUGHNESS].offsetScale,
                                      material.uv[SLOT_CLEARCOAT_ROUGHNESS].rotation);
        ccRough *= texture(clearcoatRoughnessMap, ccRuv).g;
    }
    ccRough = clamp(ccRough, 0.04, 1.0);
    float ccAlpha = ccRough * ccRough;

    vec3 N_cc = N;
    if (material.clearcoatFlags.z > 0.5) {
        vec2 ccNuv = applyUvTransform(pickUv(int(material.clearcoatTexCoords.z)),
                                      material.uv[SLOT_CLEARCOAT_NORMAL].offsetScale,
                                      material.uv[SLOT_CLEARCOAT_NORMAL].rotation);
        vec3 cnSamp = texture(clearcoatNormalMap, ccNuv).rgb * 2.0 - 1.0;
        cnSamp.xy *= ccNormalScale;
        N_cc = normalize(fragTBN * cnSamp);
    }

    // LightUBO::lights[]. Only the first directional (i==0, type==0) gets CSM
    // shadow; everything else is unshadowed.
    vec3 directDiffuse = vec3(0.0);
    vec3 directSpecular = vec3(0.0);
    float primaryDirectionalVisibility = 1.0;
    float primaryDirectionalNdotL = 0.0;
    for (int i = 0; i < light.lightCount && i < MAX_LIGHTS; ++i) {
        LightData L = light.lights[i];
        int type = int(L.position.w);

        // KHR_lights_punctual stores forward (light-to-target). Negate to get
        // the surface-to-light vector the BRDF wants.
        vec3 lightVec;
        float attenuation = 1.0;
        if (type == 0) {
            lightVec = normalize(-L.direction.xyz);
        } else {
            // Point/spot share the inverse-square + range-windowed falloff
            // from KHR_lights_punctual:
            //   windowing = clamp(1 - (d/range)^4, 0, 1)
            //   attenuation = windowing^2 / max(d^2, 0.01)
            // Range == 0 means "no range cutoff" — windowing collapses to 1.
            vec3 toLight = L.position.xyz - fragWorldPos;
            float dist = length(toLight);
            lightVec = toLight / max(dist, 1e-4);
            float range = L.direction.w;
            float windowing = (range > 0.0)
                ? clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0)
                : 1.0;
            attenuation = (windowing * windowing) / max(dist * dist, 0.01);

            if (type == 2) {
                // KHR_lights_punctual spot: cosTheta is the angle between
                // the spot's forward (light-to-target) and the light-to-frag
                // vector. lightVec points surface-to-light, so the
                // light-to-frag vector is -lightVec.
                float cosTheta = -dot(normalize(L.direction.xyz), lightVec);
                float spotFactor = clamp((cosTheta - L.cone.y)
                                         / max(L.cone.x - L.cone.y, 1e-4),
                                         0.0, 1.0);
                attenuation *= spotFactor * spotFactor;
            }

            int shIdx = int(L.cone.z + 0.5);
            if (shIdx >= 0 && attenuation > 0.0 && light.environmentParams.w <= 0.5) {
                float bias = light.pointSpotShadowParams.x;
                if (type == 2) {
                    vec4 sp = light.spotViewProj[shIdx] * vec4(fragWorldPos, 1.0);
                    vec3 proj = sp.xyz / max(sp.w, 1e-4);
                    proj.xy = proj.xy * 0.5 + 0.5;
                    if (proj.z >= 0.0 && proj.z <= 1.0
                        && all(greaterThanEqual(proj.xy, vec2(0.0)))
                        && all(lessThanEqual(proj.xy, vec2(1.0)))) {
                        float visibility = texture(
                            sampler2DArrayShadow(spotShadowMapTex, shadowCompareSampler),
                            vec4(proj.xy, float(shIdx), proj.z - bias));
                        attenuation *= visibility;
                    }
                } else if (type == 1) {
                    vec3 toFrag = fragWorldPos - L.position.xyz;
                    float dist = length(toFrag);
                    float range = max(L.direction.w, 1e-4);
                    float compareValue = clamp(dist / range - bias, 0.0, 1.0);
                    vec3 sampleDir = toFrag / max(dist, 1e-4);
                    float visibility = texture(
                        samplerCubeArrayShadow(pointShadowMapTex, shadowCompareSampler),
                        vec4(sampleDir, float(shIdx)), compareValue);
                    attenuation *= visibility;
                }
            }
        }

        vec3 lightColor = L.colour.rgb * L.colour.a * attenuation;
        vec3 H = normalize(lightVec + V);
        float NdotL = max(dot(N, lightVec), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);

        float D = distributionGGX(NdotH, a);
        float G = geometrySmith(NdotV, NdotL, roughness);
        vec3 F = fresnelSchlick(VdotH, F0);

        vec3 numerator = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001;
        vec3 specularContrib = (numerator / denominator) * lightColor * NdotL;
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        vec3 diffuseContrib = kD * baseColor * (1.0 / PI) * lightColor * NdotL;

        // KHR_materials_clearcoat — second specular lobe over the base BRDF.
        // Energy conservation: attenuate the underlying lobes by (1 − F_cc * clearcoat).
        vec3 cc_contrib = vec3(0.0);
        if (clearcoat > 0.0) {
            float NccDotL = max(dot(N_cc, lightVec), 0.0);
            float NccDotH = max(dot(N_cc, H), 0.0);
            float D_cc = distributionGGX(NccDotH, ccAlpha);
            // Kelemen visibility — separable, cheaper than Smith for clearcoat.
            float V_cc = 1.0 / (4.0 * VdotH * VdotH + 0.0001);
            float F_cc = 0.04 + (1.0 - 0.04) * pow(1.0 - VdotH, 5.0);
            float spec_cc = D_cc * V_cc * F_cc * NccDotL * clearcoat;
            cc_contrib = vec3(spec_cc) * lightColor;
            float ccAtt = 1.0 - F_cc * clearcoat;
            diffuseContrib *= ccAtt;
            specularContrib *= ccAtt;
        }

        if (i == 0 && type == 0) {
            primaryDirectionalNdotL = NdotL;
            int cascade = selectCascade(fragViewDepth);
            float shadow = 1.0;
            if (light.environmentParams.w <= 0.5) {
                if (ubo.hasSkin == 1) {
                    float worldShadow =
                        computeWorldShadow(fragWorldPos, shadowNormal, lightVec, cascade,
                                           fragViewDepth);
                    float selfShadow = sampleSelfShadow(fragWorldPos, shadowNormal, lightVec,
                                                        pc.selfShadowSlot);
                    shadow = min(worldShadow, selfShadow);
                } else {
                    shadow =
                        computeShadow(fragWorldPos, shadowNormal, lightVec, cascade,
                                      fragViewDepth);
                }
            }
            primaryDirectionalVisibility = shadow;
            diffuseContrib *= shadow;
            specularContrib *= shadow;
            cc_contrib *= shadow;
        }

        directDiffuse += diffuseContrib;
        directSpecular += specularContrib;
        directSpecular += cc_contrib;
    }

    if (light.environmentParams.z > 1.5 && light.environmentParams.z < 2.5) {
        outColor = vec4(vec3(primaryDirectionalNdotL), alpha);
        return;
    }

    if (light.environmentParams.z > 2.5 && light.environmentParams.z < 3.5) {
        outColor = vec4(vec3(primaryDirectionalVisibility), alpha);
        return;
    }

    if (light.environmentParams.z > 3.5 && light.environmentParams.z < 4.5) {
        int cascade = selectCascade(fragViewDepth);
        vec3 lightVec = normalize(-light.lights[0].direction.xyz);
        vec2 depths = directionalShadowDepths(fragWorldPos, shadowNormal, lightVec, cascade);
        float cascadeDebug = float(cascade) / 3.0;
        outColor = vec4(depths.x, depths.y, cascadeDebug, alpha);
        return;
    }

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 R = reflect(-V, N);
    float maxReflectionLod = light.iblParams.x;
    vec3 prefilteredColor = textureLod(prefilteredMap, R, roughness * maxReflectionLod).rgb;
    vec2 envBrdf = texture(brdfLut, vec2(NdotV, roughness)).rg;

    // Fdez-Aguera multi-scatter compensation. Recovers the energy the split-sum
    // single-scatter lobe loses on rough conductors.
    vec3 FssEss = F0 * envBrdf.x + envBrdf.y;
    float Ess = envBrdf.x + envBrdf.y;
    float Ems = 1.0 - Ess;
    vec3 Favg = F0 + (1.0 - F0) / 21.0;
    vec3 Fms = FssEss * Favg / (1.0 - Ems * Favg);
    vec3 multiScatter = Fms * Ems;

    vec3 iblKD = baseColor * (1.0 - FssEss - multiScatter) * (1.0 - metallic);
    vec3 diffuseIbl = irradiance * iblKD * light.iblParams.y;
    vec3 specularIbl = prefilteredColor * (FssEss + multiScatter) * light.iblParams.z;

    // Clearcoat IBL — sample the prefilter at the clearcoat normal/roughness
    // and attenuate the base IBL terms by the clearcoat Fresnel.
    vec3 clearcoatIbl = vec3(0.0);
    if (clearcoat > 0.0) {
        float NccDotV = max(dot(N_cc, V), 0.001);
        vec3 R_cc = reflect(-V, N_cc);
        vec3 prefilteredCc = textureLod(prefilteredMap, R_cc, ccRough * maxReflectionLod).rgb;
        vec2 envBrdfCc = texture(brdfLut, vec2(NccDotV, ccRough)).rg;
        float F_ccIbl = 0.04 + (1.0 - 0.04) * pow(1.0 - NccDotV, 5.0);
        clearcoatIbl = prefilteredCc * (0.04 * envBrdfCc.x + envBrdfCc.y) * clearcoat
                     * light.iblParams.z;
        float ccIblAtt = 1.0 - F_ccIbl * clearcoat;
        diffuseIbl *= ccIblAtt;
        specularIbl *= ccIblAtt;
    }

    float ao = 1.0;
    if (material.extraFlags.x == 1) {
        // glTF spec: occluded = lerp(colour, colour * sampled, strength).
        // Equivalent to ao = mix(1.0, sampled, strength) when applied as a
        // multiplier downstream.
        vec2 uvOcc = applyUvTransform(pickUv(material.extraFlags.y),
                                      material.uv[SLOT_OCCLUSION].offsetScale,
                                      material.uv[SLOT_OCCLUSION].rotation);
        float sampled = texture(occlusionMap, uvOcc).r;
        ao = mix(1.0, sampled, material.materialParams.w);
    }

    float environmentShadow = mix(1.0, primaryDirectionalVisibility, light.environmentParams.y);
    vec3 diffuseAmbientTerm = diffuseIbl * ao * environmentShadow;
    float specularAo = mix(1.0, ao, 0.25);
    float specularEnvironmentShadow = mix(1.0, environmentShadow, 0.5);
    vec3 specularAmbientTerm =
        specularIbl * specularAo * specularEnvironmentShadow
        + clearcoatIbl * specularAo * specularEnvironmentShadow;
    vec3 ambientTerm = diffuseAmbientTerm + specularAmbientTerm;

    // Emissive
    vec3 emissiveTerm = material.emissiveRoughness.rgb;
    if (material.textureFlags.y == 1) {
        vec2 uvEm = applyUvTransform(pickUv(material.texCoordIndices.y),
                                     material.uv[SLOT_EMISSIVE].offsetScale,
                                     material.uv[SLOT_EMISSIVE].rotation);
        emissiveTerm *= texture(emissiveMap, uvEm).rgb;
    }

    // KHR_materials_transmission (F2 — IBL-faked refraction). Per glTF spec,
    // the diffuse lobe is *attenuated* by (1 - transmission) and a separate
    // transmission lobe is added on top — specular is left intact. For glass
    // against the environment this is sufficient; proper scene-behind-glass
    // refraction (F3) would copy the HDR target into a sceneColor mip chain.
    float transmission = material.transmissionParams.x;
    if (material.transmissionParams.y > 0.5) {
        vec2 uvTrans = applyUvTransform(pickUv(int(material.transmissionParams.z)),
                                        material.uv[SLOT_TRANSMISSION].offsetScale,
                                        material.uv[SLOT_TRANSMISSION].rotation);
        transmission *= texture(transmissionMap, uvTrans).r;
    }

    vec3 transmittedLight = vec3(0.0);
    if (transmission > 0.0) {
        vec3 refractDir = refract(-V, N, 1.0 / ior);
        if (dot(refractDir, refractDir) < 1e-6) refractDir = R;

        // KHR_materials_volume — sample thickness, scale by node size. When
        // volume is absent thicknessFactor defaults to 0, worldThickness
        // becomes 0, the exit point matches the entry point, and Beer-Lambert
        // collapses to identity. F3 thin-surface fallback is preserved.
        float thickness = material.volumeParams.x;
        if (material.volumeParams.y > 0.5) {
            vec2 uvThick = applyUvTransform(pickUv(int(material.volumeParams.z)),
                                            material.uv[SLOT_THICKNESS].offsetScale,
                                            material.uv[SLOT_THICKNESS].rotation);
            thickness *= texture(thicknessMap, uvThick).g;
        }
        vec3 modelScale = vec3(length(ubo.model[0].xyz),
                               length(ubo.model[1].xyz),
                               length(ubo.model[2].xyz));
        float worldThickness = thickness * max(max(modelScale.x, modelScale.y), modelScale.z);

        // Exit point in world space, projected to screen UV.
        vec3 exitPos = fragWorldPos + refractDir * worldThickness;
        vec4 exitClip = ubo.proj * ubo.view * vec4(exitPos, 1.0);
        vec2 sampleUv = exitClip.xy / max(exitClip.w, 1e-4) * 0.5 + 0.5;
        // Vulkan screen UV has Y pointing down; clip-space y is inverted.
        sampleUv.y = 1.0 - sampleUv.y;
        sampleUv = clamp(sampleUv, vec2(0.0), vec2(1.0));

        float maxLod = float(textureQueryLevels(sceneColorMap) - 1);
        // Rough transmission blur should collapse as the interface disappears.
        // For IOR = 1.0 there is no refractive boundary, so even a high authored
        // roughness should not smear the background into a milky patch. Scale
        // the blur by interface strength and normalise against the glTF default
        // dielectric IOR of 1.5 (F0 ~= 0.04) so default materials preserve the
        // previous look while Air-like materials trend toward no blur.
        float defaultDielectricF0 = 0.04;
        float interfaceStrength = sqrt(clamp(dielectricF0 / defaultDielectricF0, 0.0, 1.0));
        float lod = roughness * interfaceStrength * maxLod;
        vec3 transmissionSample = textureLod(sceneColorMap, sampleUv, lod).rgb;

        // Beer-Lambert absorption over the path through the volume.
        // attenuationColor at attenuationDistance is the colour the light
        // takes after travelling that distance through the medium.
        vec3 attenColour = material.attenuation.rgb;
        float attenDist = material.attenuation.a;
        vec3 absorption = -log(max(attenColour, vec3(1e-5))) / max(attenDist, 1e-5);
        vec3 transmittance = exp(-absorption * worldThickness);

        transmittedLight = transmission * baseColor * transmissionSample * transmittance;
    }

    // Diffuse lobes are scaled — NOT replaced — by (1 - transmission). Specular
    // and emissive paths are unchanged.
    directDiffuse      *= (1.0 - transmission);
    diffuseAmbientTerm *= (1.0 - transmission);

    vec3 color = diffuseAmbientTerm + specularAmbientTerm
               + directDiffuse + directSpecular
               + transmittedLight + emissiveTerm;

    outColor = vec4(color, alpha);
}
