#version 450

// Depth-aware (bilateral) blur of the R8G8 AO target. Smooths the SSAO / contact
// per-pixel sampling noise without bleeding across depth silhouettes (e.g. the
// foot/floor edge). 5x5: Gaussian spatial weight * a view-space-Z edge-stop.

layout(set = 0, binding = 0) uniform sampler2D aoTex;
layout(set = 0, binding = 1) uniform sampler2D depthTex;

layout(push_constant) uniform Push
{
    vec2 texelSize; // 1/width, 1/height
    float projC;    // proj[2][2]
    float projD;    // proj[3][2]
} pc;

layout(location = 0) in vec2 fragUv;
layout(location = 0) out vec2 outAo;

// View-space Z from depth (negative, camera looks -Z) — same analytic inverse as
// the SSAO pass, using only the two projection terms.
float linearZ(float depth)
{
    return -pc.projD / (depth + pc.projC);
}

void main()
{
    float centerDepth = texture(depthTex, fragUv).r;
    vec2 centerAo = texture(aoTex, fragUv).rg;
    if (centerDepth >= 1.0)
    {
        outAo = centerAo; // background — nothing to blur
        return;
    }

    float centerZ = linearZ(centerDepth);
    const int R = 2;            // 5x5 kernel
    const float sigmaZ = 0.1;   // view units; taps across a depth edge are rejected

    vec2 sum = vec2(0.0);
    float wsum = 0.0;
    for (int y = -R; y <= R; ++y)
    {
        for (int x = -R; x <= R; ++x)
        {
            vec2 uv = fragUv + vec2(float(x), float(y)) * pc.texelSize;
            float z = linearZ(texture(depthTex, uv).r);
            float wz = exp(-abs(z - centerZ) / sigmaZ);          // edge-stop
            float ws = exp(-float(x * x + y * y) / 8.0);          // spatial
            float w = wz * ws;
            sum += texture(aoTex, uv).rg * w;
            wsum += w;
        }
    }
    outAo = sum / max(wsum, 1e-4);
}
