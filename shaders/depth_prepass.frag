#version 450

// Depth prepass: depth-only, no colour attachment. The fixed-function depth test
// writes gl_FragCoord.z; the fragment shader itself produces nothing. Reuses
// shader.vert so the written depth matches the forward pass exactly.
void main()
{
}
