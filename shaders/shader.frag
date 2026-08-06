#version 450
// textureSize() and similar queries on plain texture* uniforms (no sampler
// attached) need this — we use it for the shadow map array which is bound
// as a sampledImage so all shadow maps can share one comparison sampler.
#extension GL_EXT_samplerless_texture_functions : require

// Per-object data (set 0, pushed per draw) — must match ObjectUBO in shader.vert / render/ubo.hpp.
layout(binding = 0) uniform ObjectUBO {
    mat4 model;
    int hasSkin;
    int _pad1;
    int _pad2;
    int _pad3;
    mat4 previousModel;
} ubo;

// Per-frame camera data (set 0, binding 29 — pushed per draw) — must match CameraUBO in shader.vert.
layout(binding = 29) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    mat4 currentViewProj;
    mat4 previousViewProj;
} camera;

// The material authority this draw reads lives in material.glsl (bindless set 2), which indexes
// materials[] through `pc.materialIndex` — so the push block is declared FIRST and the include
// follows it. Both are shared declarations; neither may be restated here.
#include "forward_push.glsl"
#include "material.glsl"
#include "shadow_bias.glsl"

// Shared palette for the two LOD debug views, so a level always means the same colour in both.
vec3 lodTint(uint level) {
    vec3 tints[4] = vec3[4](vec3(0.2, 0.9, 0.2),   // LOD0 green
                            vec3(0.95, 0.85, 0.1), // LOD1 yellow
                            vec3(0.9, 0.2, 0.2),   // LOD2 red
                            vec3(0.9, 0.2, 0.9));  // LOD3+ magenta
    return tints[min(level, 3u)];
}

// KHR_materials_transmission F3 — captured post-opaque scene colour with mip
// chain. Transmissive draws sample this at a screen-space UV displaced by
// the refracted ray; roughness drives the mip level for frosted-glass blur.
layout(set = 1, binding = 12) uniform sampler2D sceneColorMap;
// Screen-space AO + contact term (R = ambient occlusion, G = contact shadow),
// from the SSAO pass. Sampled at the fragment's screen UV to modulate ambient.
layout(set = 1, binding = 13) uniform sampler2D ssaoMap;
// Shadow images bound as plain textures so one hardware-PCF comparison sampler is
// reused across CSM, spot, and point maps. Combined samplers are constructed at
// use time via the GLSL sampler*() constructors.
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

// LightData, the limit constants and the LightUBO block itself all come from the shared include —
// the single declaration of a buffer this shader and skybox.frag both bind.
#define LIGHT_UBO_SET 1
#define LIGHT_UBO_BINDING 0
#include "light_ubo.glsl"

// Descriptor set 1 holds the bindings shared by every draw (the light UBO above, shadow maps, IBL
// textures, sceneColor). Bound once per frame in Renderer; reused across all forward pipelines.

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in mat3 fragTBN;
layout(location = 7) in float fragViewDepth;
layout(location = 8) in vec2 fragTexCoord1;
// Jitter-free clip positions from the vertex stage for TAA motion vectors.
layout(location = 9) in vec4 fragCurClip;
layout(location = 10) in vec4 fragPrevClip;

layout(location = 0) out vec4 outColor;
// TAA motion vector: per-pixel screen-space (UV) motion since the previous
// frame, used by the resolve to reproject history. Written before any early
// return so every shaded fragment produces a defined velocity.
layout(location = 1) out vec2 outVelocity;

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
// The PCF kernel, normalised to unit support (SH-07) — see shaders/poisson_taps.inl, which is the
// ONE place the numbers live. `tests/graphics/test_shadow_bias.cpp` includes that same file and
// checks the support, so the test pins the taps this shader actually samples rather than a copy of
// them.
#define POISSON_TAP(x, y) vec2(x, y)
const vec2 poissonDisk[16] = vec2[16](
#include "poisson_taps.inl"
);
#undef POISSON_TAP

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
    // SH-07: this cascade's OWN fitted metrics — footprint and depth conversion — instead of
    // exp2(cascade), which asserted that both doubled per cascade and was wrong about each
    // independently.
    vec4 metrics = light.cascadeBiasMetrics[cascade];
    ShadowBias bias = shadowBiasFor(metrics.x, metrics.y, dot(normal, lightDir),
                                    light.cascadeParams.y, light.shadowParams);

    // The normal offset is WORLD space and applied BEFORE projection, so each projection converts it
    // through its own mapping rather than through an assumed one.
    vec3 sampleWorldPos = worldPos + normal * bias.normalOffsetWorld;
    vec4 lightSpace = light.cascadeViewProj[cascade] * vec4(sampleWorldPos, 1.0);
    vec3 proj = lightSpace.xyz / lightSpace.w;
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.z < 0.0
        || any(lessThan(proj.xy, vec2(0.0)))
        || any(greaterThan(proj.xy, vec2(1.0)))) {
        return 1.0;
    }

    float receiverDepth = proj.z - bias.depth;

    mat2 rot = poissonRotation(worldPos);
    float texelSize = 1.0 / float(textureSize(shadowTex, 0).x);
    float filterRadius = max(light.cascadeParams.y, 0.0) * texelSize;

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

    // The self layer is its own tight ortho fit, so it has its own footprint and depth span — it is
    // NOT a cascade and never shared their scale.
    vec4 metrics = light.selfBiasMetrics[slot];
    ShadowBias bias = shadowBiasFor(metrics.x, metrics.y, dot(normal, lightDir), 0.0,
                                    light.shadowParams);
    vec3 sampleWorldPos = worldPos + normal * bias.normalOffsetWorld;
    vec4 lightSpace = light.selfShadowViewProj[slot] * vec4(sampleWorldPos, 1.0);
    vec3 proj = lightSpace.xyz / lightSpace.w;
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.z < 0.0
        || any(lessThan(proj.xy, vec2(0.0)))
        || any(greaterThan(proj.xy, vec2(1.0)))) {
        return 1.0;
    }

    float receiverDepth = proj.z - bias.depth;
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
    float blendBand = (cascadeEnd - cascadeStart) * light.cascadeParams.x;
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
    // The DEBUG readout must use the same law as the sampler it explains — a debug view that reports
    // a different bias than the one in effect is worse than no debug view.
    vec4 metrics = light.cascadeBiasMetrics[cascade];
    ShadowBias bias = shadowBiasFor(metrics.x, metrics.y, dot(normal, lightDir),
                                    light.cascadeParams.y, light.shadowParams);
    vec3 sampleWorldPos = worldPos + normal * bias.normalOffsetWorld;
    vec4 lightSpace = light.cascadeViewProj[cascade] * vec4(sampleWorldPos, 1.0);
    vec3 proj = lightSpace.xyz / lightSpace.w;
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.z < 0.0
        || any(lessThan(proj.xy, vec2(0.0)))
        || any(greaterThan(proj.xy, vec2(1.0)))) {
        return vec2(0.0, 0.0);
    }

    float receiverDepth = proj.z - bias.depth;
    float storedDepth = texture(sampler2DArray(shadowDebugImageTex, shadowDebugSampler),
                                vec3(proj.xy, float(cascade))).r;
    return vec2(receiverDepth, storedDepth);
}

void main() {
    // Motion vector first — several debug/unlit paths below return early, and
    // location 1 must be written on every path or the velocity is undefined.
    // NDC delta * 0.5 converts clip-space [-1,1] motion to UV-space [0,1] motion.
    vec2 curNdc = fragCurClip.xy / max(fragCurClip.w, 1e-6);
    vec2 prevNdc = fragPrevClip.xy / max(fragPrevClip.w, 1e-6);
    outVelocity = (curNdc - prevNdc) * 0.5;

    // DebugView::Velocity (5) — visualise the motion vector. Scaled so the
    // small per-frame UV motion is perceptible: R = |x|, G = |y|.
    if (light.environmentParams.z > 4.5 && light.environmentParams.z < 5.5) {
        outColor = vec4(abs(outVelocity) * 50.0, 0.0, 1.0);
        return;
    }

    vec3 N;
    if (material.textureFlags.z == 1) {
        vec2 uvNormal = materialSlotUv(SLOT_NORMAL, material.texCoordIndices.z,
                                       fragTexCoord, fragTexCoord1);
        vec3 mapNormal = texture(textures[matTex(SLOT_NORMAL)], uvNormal).rgb * 2.0 - 1.0;
        mapNormal.xy *= material.materialParams.y;
        N = normalize(fragTBN * mapNormal);
    } else {
        N = normalize(fragNormal);
    }

    // Double-sided geometry: when a back face is rasterised (cull-none blend /
    // double-sided pipelines) the interpolated normal still points along the
    // authored front, so flip it to face the viewer. Without this the view-
    // dependent terms (IBL reflection R = reflect(-V, N), transmission
    // refract(-V, N)) evaluate against a normal pointing away from the camera
    // and produce a bright patch that tracks the camera — most visible on thin
    // transmissive surfaces such as a lamp shade. Single-sided (back-face
    // culled) draws only rasterise front faces, so this is a no-op there.
    if (!gl_FrontFacing) {
        N = -N;
    }

    // Use the geometric mesh normal for shadow receiver bias; tangent-space
    // normal maps affect BRDF shading, not geometric visibility.
    vec3 shadowNormal = normalize(fragNormal);
    if (!gl_FrontFacing) {
        shadowNormal = -shadowNormal;
    }

    vec3 V = normalize(camera.cameraPos.xyz - fragWorldPos);
    float NdotV = max(dot(N, V), 0.001);
    // KHR_materials_ior. Use the authored dielectric IOR to derive F0 instead
    // of the previous hard-coded 0.04 baseline. This keeps Air (IOR = 1.0)
    // close to fully transmissive while preserving the spec's default 1.5 →
    // ~0.04 reflectance when the extension is absent.
    float ior = max(material.transmissionParams.w, 1e-4);

    // Sample base colour texture once (opaque white when the material carries none).
    vec4 texColor = materialBaseColourTexel(fragTexCoord, fragTexCoord1);
    vec3 baseColor = material.diffuseAlpha.rgb * fragColor * texColor.rgb;

    // Alpha, and the MASK cutout — both from material.glsl, the single implementation the shadow
    // pass' masked path uses too (a cutout must not cast a silhouette its own surface lacks).
    float alpha = materialAlpha(texColor);
    if (materialAlphaCutoutFails(alpha)) discard;

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
        vec2 uvMr = materialSlotUv(SLOT_METALLIC_ROUGHNESS, material.texCoordIndices.w,
                                   fragTexCoord, fragTexCoord1);
        vec4 mrSample = texture(textures[matTex(SLOT_METALLIC_ROUGHNESS)], uvMr);
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
        vec2 ccUv = materialSlotUv(SLOT_CLEARCOAT, int(material.clearcoatTexCoords.x),
                                   fragTexCoord, fragTexCoord1);
        clearcoat *= texture(textures[matTex(SLOT_CLEARCOAT)], ccUv).r;
    }
    if (material.clearcoatFlags.y > 0.5) {
        vec2 ccRuv = materialSlotUv(SLOT_CLEARCOAT_ROUGHNESS, int(material.clearcoatTexCoords.y),
                                    fragTexCoord, fragTexCoord1);
        ccRough *= texture(textures[matTex(SLOT_CLEARCOAT_ROUGHNESS)], ccRuv).g;
    }
    ccRough = clamp(ccRough, 0.04, 1.0);
    float ccAlpha = ccRough * ccRough;

    vec3 N_cc = N;
    if (material.clearcoatFlags.z > 0.5) {
        vec2 ccNuv = materialSlotUv(SLOT_CLEARCOAT_NORMAL, int(material.clearcoatTexCoords.z),
                                    fragTexCoord, fragTexCoord1);
        vec3 cnSamp = texture(textures[matTex(SLOT_CLEARCOAT_NORMAL)], ccNuv).rgb * 2.0 - 1.0;
        cnSamp.xy *= ccNormalScale;
        N_cc = normalize(fragTBN * cnSamp);
        // Match the base-normal back-face flip (N_cc inherits the already-
        // flipped N otherwise, but a clearcoat normal map rebuilds it here).
        if (!gl_FrontFacing) {
            N_cc = -N_cc;
        }
    }

    // LightUBO::lights[]. Only the first directional (i==0, type==0) gets CSM
    // shadow; everything else is unshadowed.
    vec3 directDiffuse = vec3(0.0);
    vec3 directSpecular = vec3(0.0);
    float primaryDirectionalVisibility = 1.0;
    float primaryDirectionalNdotL = 0.0;
    // Screen-space AO + contact term (R = ambient occlusion, G = contact shadow),
    // sampled once at the fragment's screen UV. .g attenuates the direct sun
    // inside the loop; .r modulates ambient below. Full-res target, so the pixel
    // coord maps straight to the texel.
    vec2 ssaoSample = textureLod(ssaoMap, gl_FragCoord.xy / vec2(textureSize(ssaoMap, 0)), 0.0).rg;
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
                // The GEOMETRIC normal, not the shaded one. `N` carries normal-map detail, and
                // biasing or displacing a receiver by it would let a texture physically move the
                // surface the shadow test is performed against — crawling and leaks that track the
                // map rather than the geometry. The directional and self paths use shadowNormal for
                // exactly this reason.
                float nDotLForBias = dot(shadowNormal, lightVec);
                if (type == 2) {
                    // SPOT. Both metrics are per fragment: the footprint grows with FORWARD depth,
                    // and the depth conversion falls as its square and is measured along the light
                    // RAY — hence the radial distance beside it.
                    vec4 metrics = light.spotBiasMetrics[shIdx];
                    vec3 toFrag = fragWorldPos - L.position.xyz;
                    float forwardDepth = dot(toFrag, normalize(L.direction.xyz));
                    float radialDepth = length(toFrag);
                    ShadowBias bias = shadowBiasFor(
                        metrics.x * max(forwardDepth, 0.0),
                        spotNormalizedDepthPerWorldUnit(metrics.y, metrics.z, forwardDepth,
                                                        radialDepth),
                        nDotLForBias, 0.0, light.shadowParams);
                    // Offset in WORLD space first, then project — the mapping converts it.
                    vec4 sp = light.spotViewProj[shIdx]
                              * vec4(fragWorldPos + shadowNormal * bias.normalOffsetWorld, 1.0);
                    vec3 proj = sp.xyz / max(sp.w, 1e-4);
                    proj.xy = proj.xy * 0.5 + 0.5;
                    if (proj.z >= 0.0 && proj.z <= 1.0
                        && all(greaterThanEqual(proj.xy, vec2(0.0)))
                        && all(lessThanEqual(proj.xy, vec2(1.0)))) {
                        float visibility = texture(
                            sampler2DArrayShadow(spotShadowMapTex, shadowCompareSampler),
                            vec4(proj.xy, float(shIdx), proj.z - bias.depth));
                        attenuation *= visibility;
                    }
                } else if (type == 1) {
                    // POINT. Footprint follows the MAJOR AXIS (what the face's projection divides
                    // by); the stored comparison stays RADIAL, and its conversion is the constant
                    // 1/range the metrics carry.
                    vec4 metrics = light.pointBiasMetrics[shIdx];
                    vec3 rawToFrag = fragWorldPos - L.position.xyz;
                    ShadowBias bias = shadowBiasFor(
                        metrics.x * pointMajorAxisDepth(rawToFrag), metrics.y, nDotLForBias, 0.0,
                        light.shadowParams);
                    // Offset BEFORE the distance is taken, so the radial depth compared against the
                    // map is the offset position's own.
                    vec3 toFrag =
                        (fragWorldPos + shadowNormal * bias.normalOffsetWorld) - L.position.xyz;
                    float dist = length(toFrag);
                    float range = max(L.direction.w, 1e-4);
                    float compareValue = clamp(dist / range - bias.depth, 0.0, 1.0);
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
            // Contact shadows (screen-space) further occlude the *direct* sun,
            // catching short-range contact the CSM misses. Ambient keeps the pure
            // CSM visibility above. ssaoSample.g is 1.0 when contact is disabled.
            float directShadow = shadow * ssaoSample.g;
            diffuseContrib *= directShadow;
            specularContrib *= directShadow;
            cc_contrib *= directShadow;
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
        vec2 uvOcc = materialSlotUv(SLOT_OCCLUSION, material.extraFlags.y,
                                    fragTexCoord, fragTexCoord1);
        float sampled = texture(textures[matTex(SLOT_OCCLUSION)], uvOcc).r;
        ao = mix(1.0, sampled, material.materialParams.w);
    }

    // Screen-space AO (R, sampled above) folds into the ambient occlusion term.
    if (light.environmentParams.z > 5.5 && light.environmentParams.z < 6.5) {
        outColor = vec4(vec3(ssaoSample.r), alpha);
        return;
    }
    // LOD debug tint: colour each mesh by its selected discrete LOD level.
    if (light.environmentParams.z > 6.5 && light.environmentParams.z < 7.5) {
        outColor = vec4(lodTint(pc.lodLevel), alpha);
        return;
    }
    // Shadow-LOD debug tint (SH-01): colour each mesh by the level its SHADOW draw selected — the
    // same palette, so this view and the LOD view can be read against each other. Shadow draws are
    // depth-only, so the forward draw carries the level here.
    if (light.environmentParams.z > 7.5 && light.environmentParams.z < 8.5) {
        // No shadow draw this frame: neutral grey rather than the LOD0 green, which would read as
        // "full detail chosen" in the very view built to find over-detailed shadow casters.
        // Mirrors kNoShadowLod in graphics/shadow_diagnostics.hpp.
        if (pc.shadowLodLevel == 0xFFFFFFFFu) {
            outColor = vec4(vec3(0.35), alpha);
            return;
        }
        outColor = vec4(lodTint(pc.shadowLodLevel), alpha);
        return;
    }
    ao *= ssaoSample.r;

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
        vec2 uvEm = materialSlotUv(SLOT_EMISSIVE, material.texCoordIndices.y,
                                   fragTexCoord, fragTexCoord1);
        emissiveTerm *= texture(textures[matTex(SLOT_EMISSIVE)], uvEm).rgb;
    }

    // KHR_materials_transmission (F2 — IBL-faked refraction). Per glTF spec,
    // the diffuse lobe is *attenuated* by (1 - transmission) and a separate
    // transmission lobe is added on top — specular is left intact. For glass
    // against the environment this is sufficient; proper scene-behind-glass
    // refraction (F3) would copy the HDR target into a sceneColor mip chain.
    float transmission = material.transmissionParams.x;
    if (material.transmissionParams.y > 0.5) {
        vec2 uvTrans = materialSlotUv(SLOT_TRANSMISSION, int(material.transmissionParams.z),
                                      fragTexCoord, fragTexCoord1);
        transmission *= texture(textures[matTex(SLOT_TRANSMISSION)], uvTrans).r;
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
            vec2 uvThick = materialSlotUv(SLOT_THICKNESS, int(material.volumeParams.z),
                                          fragTexCoord, fragTexCoord1);
            thickness *= texture(textures[matTex(SLOT_THICKNESS)], uvThick).g;
        }
        vec3 modelScale = vec3(length(ubo.model[0].xyz),
                               length(ubo.model[1].xyz),
                               length(ubo.model[2].xyz));
        float worldThickness = thickness * max(max(modelScale.x, modelScale.y), modelScale.z);

        // Exit point in world space, projected to screen UV (F3 refraction).
        vec3 exitPos = fragWorldPos + refractDir * worldThickness;
        vec4 exitClip = camera.proj * camera.view * vec4(exitPos, 1.0);
        vec2 sampleUv = exitClip.xy / max(exitClip.w, 1e-4) * 0.5 + 0.5;
        // Vulkan screen UV has Y pointing down; clip-space y is inverted.
        sampleUv.y = 1.0 - sampleUv.y;
        sampleUv = clamp(sampleUv, vec2(0.0), vec2(1.0));

        float maxLod = float(textureQueryLevels(sceneColorMap) - 1);
        // KHR_materials_ior: the apparent blur of TRANSMITTED light grows with
        // IOR (the interface distorts transmitted rays more) while the blur of
        // REFLECTED light stays roughness-only. Scale the transmission lod by an
        // interface strength derived from the IOR's dielectric F0, normalised so
        // the glTF default IOR 1.5 (F0 = 0.04) reproduces the plain roughness
        // blur. IOR 1.0 (F0 = 0) gives sharp transmission; higher IOR is
        // progressively blurrier.
        float interfaceStrength = sqrt(dielectricF0 / 0.04);
        float lod = roughness * interfaceStrength * maxLod;
        vec3 sceneSample = textureLod(sceneColorMap, sampleUv, lod).rgb;

        // Screen-space refraction only carries a coherent image for clear/frosted glass. A
        // thin-walled surface that is BOTH transmissive and emissive is a self-lit diffuser — a
        // paper lamp shade (LightsPunctualLamp / StainedGlassLamp), not glass. It transmits
        // diffusely (view-independent), so it must not sample the screen: otherwise the bright bulb
        // behind it is beamed onto the shade as a camera-tracking, aliased blob. Route those to a
        // view-independent irradiance tint instead. (Roughness can't be the discriminator — the
        // shade's roughnessFactor is textured and overlaps TransmissionTest's frosted 0.32–0.9
        // range; the emissive factor is a clean per-material constant: [1,1,1] shade vs 0 glass.)
        // Clear/frosted glass (not emissive) still refracts the roughness-blurred scene; a maximally
        // rough thin surface also goes diffuse; volumetric panels always refract.
        const float kEnvTint = 0.2;
        vec3 scatterTint =
            vec3(1.0) + kEnvTint * texture(irradianceMap, refractDir).rgb * light.iblParams.y;
        float volumetric = smoothstep(0.0, 0.001, thickness);
        float emissiveLevel = max(material.emissiveRoughness.r,
                                  max(material.emissiveRoughness.g, material.emissiveRoughness.b));
        float diffuseScatter =
            (1.0 - volumetric) * max(step(0.001, emissiveLevel), smoothstep(0.9, 1.0, roughness));
        vec3 surface = mix(sceneSample, scatterTint, diffuseScatter);

        // Beer-Lambert absorption over the path through the volume.
        // attenuationColor at attenuationDistance is the colour the light
        // takes after travelling that distance through the medium.
        vec3 attenColour = material.attenuation.rgb;
        float attenDist = material.attenuation.a;
        vec3 absorption = -log(max(attenColour, vec3(1e-5))) / max(attenDist, 1e-5);
        vec3 transmittance = exp(-absorption * worldThickness);

        transmittedLight = transmission * baseColor * surface * transmittance;
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
