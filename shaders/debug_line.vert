#version 450

// Physics debug wireframe lines. Reuses the standard Vertex input — only the
// position and colour attributes are consumed; the rest are ignored.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform Push
{
    mat4 viewProj; // jitter-free world -> clip
} pc;

layout(location = 0) out vec3 fragColor;

void main()
{
    gl_Position = pc.viewProj * vec4(inPos, 1.0);
    fragColor = inColor;
}
