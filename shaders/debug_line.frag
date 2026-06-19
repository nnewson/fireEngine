#version 450

// Flat-coloured debug line. Writes into the HDR target before post-process, so
// scale a little above 1.0 to stay vivid through ACES tone-mapping.
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(fragColor * 1.5, 1.0);
}
