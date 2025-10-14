#ifndef SIMULATION_H
#define SIMULATION_H

#include "../include/particle.h"
#include <vector>

class Simulation {
public:
    Simulation(int numParticles);
    ~Simulation();

    void init();
    void update(float dt);
    const std::vector<Particle>& getHostParticles() const;

private:
    void recycle_particles();

    int numParticles;
    Particle blackHole;
    std::vector<Particle> particles_host;
    Particle* particles_device;

    const float G = 0.05f;
    const float instabilityFactor = 0.85f;
    const float softeningFactor = 0.1f;
};

#endif // SIMULATION_H