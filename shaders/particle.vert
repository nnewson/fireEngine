#version 450

// Instanced billboard particle vertex shader. One instance
// per pool slot; the quad is generated from gl_VertexIndex (6 verts = 2 tris) and
// faced toward the camera using the view matrix's world-space right/up axes.

struct Particle {
    vec4 posAge;     // xyz position, w age
    vec4 velLife;    // xyz velocity, w lifetime
    vec4 colourSize; // rgb colour * intensity, w half-size
};

layout(std430, binding = 0) readonly buffer Particles {
    Particle particles[];
};

struct EmitterGpu {
    vec4 posCone;
    vec4 velLifetime;
    vec4 colourSize;
    vec4 gravitySpawn;
};

layout(std140, binding = 1) uniform Frame {
    mat4 view;
    mat4 proj;
    float dt;
    uint frameCounter;
    uint emitterCount;
    uint particlesPerEmitter;
    EmitterGpu emitters[4];
} frame;

layout(location = 0) out vec2 fragUv;
layout(location = 1) out vec3 fragColour;
layout(location = 2) out float fragViewDepth; // eye-space distance, for soft particles

const vec2 kCorners[6] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
                                vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));

void main() {
    Particle p = particles[gl_InstanceIndex];
    float age = p.posAge.w;
    float lifetime = p.velLife.w;
    float halfSize = p.colourSize.w;
    bool dead = (lifetime <= 0.0) || (age >= lifetime) || (halfSize <= 0.0);

    vec2 corner = kCorners[gl_VertexIndex];
    fragUv = corner * 0.5 + 0.5;

    // World-space camera basis = rows of the view rotation.
    vec3 camRight = vec3(frame.view[0][0], frame.view[1][0], frame.view[2][0]);
    vec3 camUp = vec3(frame.view[0][1], frame.view[1][1], frame.view[2][1]);

    vec3 worldPos = p.posAge.xyz + (camRight * corner.x + camUp * corner.y) * halfSize;
    vec4 viewPos = frame.view * vec4(worldPos, 1.0);
    fragViewDepth = -viewPos.z;
    gl_Position = frame.proj * viewPos;

    // Fade out over lifetime; brighter when freshly spawned.
    float lifeFrac = (lifetime > 0.0) ? clamp(age / lifetime, 0.0, 1.0) : 1.0;
    fragColour = p.colourSize.rgb * (1.0 - lifeFrac);

    if (dead) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // clip degenerate dead quads
    }
}
