#version 450

// Screen-space ambient occlusion + contact shadows, computed from the depth
// prepass. View-space position + normal are reconstructed from depth (no normal
// G-buffer); a hemisphere kernel estimates occlusion. R = AO, G = contact term.
// Mirrors SsaoUBO in include/fire_engine/render/ubo.hpp.

const int KERNEL_SIZE = 16; // == kSsaoKernelSize

layout(set = 0, binding = 0) uniform sampler2D depthTex;
layout(set = 0, binding = 1) uniform SsaoUBO
{
    mat4 proj;               // jittered projection the depth was rendered with
    vec4 kernel[KERNEL_SIZE]; // hemisphere samples (xyz), tangent space (+Z = normal)
    vec4 params;             // x=radius y=bias z=intensity (0=off) w=power
    vec4 contact;            // x=length y=steps z=sunEnabled(>0.5) w=edgeThreshold
    vec4 sunViewDir;         // xyz sun direction in view space
    vec4 screen;             // x=w y=h z=1/w w=1/h
} ssao;

layout(location = 0) in vec2 fragUv;
layout(location = 0) out vec2 outAo; // R = AO, G = contact

// Reconstruct view-space position from depth + UV using the projection's terms.
// clip = proj * view, so the perspective form inverts analytically — no matrix
// inverse needed. proj is column-major: proj[col][row].
vec3 viewPosFromDepth(vec2 uv, float depth)
{
    float a = ssao.proj[0][0];
    float b = ssao.proj[1][1];
    float c = ssao.proj[2][2];
    float d = ssao.proj[3][2];
    float jx = ssao.proj[2][0];
    float jy = ssao.proj[2][1];
    vec2 ndc = uv * 2.0 - 1.0;
    float vz = -d / (depth + c);
    float vx = -vz * (ndc.x + jx) / a;
    float vy = -vz * (ndc.y + jy) / b;
    return vec3(vx, vy, vz);
}

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Screen-space contact shadow: march from the surface toward the sun through the
// depth buffer; if scene geometry sits in front of the ray, the point is in a
// short-range contact shadow. Returns 1 = lit, 0 = fully occluded.
float contactShadow(vec3 viewPos, vec3 n)
{
    if (ssao.contact.z < 0.5)
    {
        return 1.0; // disabled
    }
    vec3 L = normalize(-ssao.sunViewDir.xyz); // toward the sun
    // Surfaces facing away from the sun are already dark from N·L / CSM — don't
    // march (a grazing ray skims the surface and self-shadows into a trail).
    float ndl = dot(n, L);
    if (ndl <= 0.05)
    {
        return 1.0;
    }

    float len = ssao.contact.x;
    int steps = max(int(ssao.contact.y), 1);
    float stepLen = len / float(steps);
    vec3 rayStep = L * stepLen;
    // bias: step off the surface so a near-coplanar sample (e.g. the floor under
    // the feet) doesn't self-occlude. thickness: the occluder-depth WINDOW. The
    // comparison below is only in view-space Z, so scale the window by the ray's
    // Z movement rather than the full 3D step length; otherwise grazing rays get
    // an over-wide depth window and leave thin contact streaks.
    float zStep = max(abs(rayStep.z), 1e-4);
    float bias = zStep;
    float thickness = zStep * 2.0;
    vec3 p = viewPos + L * bias;
    for (int i = 0; i < steps; ++i)
    {
        p += rayStep;
        vec4 clip = ssao.proj * vec4(p, 1.0);
        vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        {
            break;
        }
        float sd = texture(depthTex, uv).r;
        if (sd >= 1.0)
        {
            continue;
        }
        float sceneZ = viewPosFromDepth(uv, sd).z;
        // View forward is -Z: sceneZ > ray z means scene geometry is in front of
        // the ray. Only a thin window in front (a real contact occluder) counts;
        // anything closer than `bias` is the same surface, farther than
        // `thickness` is a separate object the ray actually passes behind.
        float diff = sceneZ - p.z;
        if (diff > bias && diff < thickness)
        {
            // Fade by march distance: near contact dark, far contact soft.
            return float(i) / float(steps);
        }
    }
    return 1.0;
}

void main()
{
    float depth = texture(depthTex, fragUv).r;
    // Background (cleared depth 1.0): fully lit, no occlusion, no contact shadow.
    if (depth >= 1.0)
    {
        outAo = vec2(1.0, 1.0);
        return;
    }

    vec3 viewPos = viewPosFromDepth(fragUv, depth);
    // Geometric normal from the depth derivatives (view space). Force it to face
    // the camera (+Z) — only camera-facing surfaces are visible here.
    vec3 n = normalize(cross(dFdx(viewPos), dFdy(viewPos)));
    if (n.z < 0.0)
    {
        n = -n;
    }

    // Per-pixel random rotation of the kernel; TAA + the blur pass denoise it.
    float rot = hash(fragUv * ssao.screen.xy) * 6.2831853;
    vec3 randomVec = vec3(cos(rot), sin(rot), 0.0);
    vec3 tangent = normalize(randomVec - n * dot(randomVec, n));
    vec3 bitangent = cross(n, tangent);
    mat3 TBN = mat3(tangent, bitangent, n);

    float radius = ssao.params.x;
    float bias = ssao.params.y;
    float occlusion = 0.0;
    for (int i = 0; i < KERNEL_SIZE; ++i)
    {
        vec3 samplePos = viewPos + (TBN * ssao.kernel[i].xyz) * radius;
        vec4 clip = ssao.proj * vec4(samplePos, 1.0);
        vec2 sampleUv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0)
        {
            continue;
        }
        float sampleDepth = texture(depthTex, sampleUv).r;
        if (sampleDepth >= 1.0)
        {
            continue;
        }
        float sampleViewZ = viewPosFromDepth(sampleUv, sampleDepth).z;
        // View forward is -Z, so a larger (less negative) z is closer to the
        // camera: the surface at sampleUv occludes when it sits in front of the
        // kernel sample point.
        float rangeCheck = smoothstep(0.0, 1.0, radius / max(abs(viewPos.z - sampleViewZ), 1e-4));
        occlusion += (sampleViewZ >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }
    float ao = 1.0 - (occlusion / float(KERNEL_SIZE)) * ssao.params.z;
    ao = pow(clamp(ao, 0.0, 1.0), ssao.params.w);

    // Contact shadows are unreliable at depth silhouettes — the ray skims the
    // foreground geometry and leaves thin "hair" streaks. Detect a depth step
    // against the 4-neighbours and fade contact to lit there, so the artifact
    // never reaches the foot/floor edge.
    vec2 ts = ssao.screen.zw;
    float zc = viewPos.z;
    float zr = viewPosFromDepth(fragUv + vec2(ts.x, 0.0), texture(depthTex, fragUv + vec2(ts.x, 0.0)).r).z;
    float zl = viewPosFromDepth(fragUv - vec2(ts.x, 0.0), texture(depthTex, fragUv - vec2(ts.x, 0.0)).r).z;
    float zu = viewPosFromDepth(fragUv + vec2(0.0, ts.y), texture(depthTex, fragUv + vec2(0.0, ts.y)).r).z;
    float zd = viewPosFromDepth(fragUv - vec2(0.0, ts.y), texture(depthTex, fragUv - vec2(0.0, ts.y)).r).z;
    float maxStep = max(max(abs(zr - zc), abs(zl - zc)), max(abs(zu - zc), abs(zd - zc)));
    float et = ssao.contact.w;
    float edge = smoothstep(et * 0.5, et * 1.5, maxStep); // 1 = strong silhouette

    float contact = mix(contactShadow(viewPos, n), 1.0, edge);

    outAo = vec2(ao, contact);
}
