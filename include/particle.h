#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm/glm.hpp>

// --- Particle Structure ---
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float mass;
};

#endif // PARTICLE_H
