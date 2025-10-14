#include "../include/simulation.h"
#include "../include/utils.h"
#include <glm/gtc/random.hpp>
#include <cmath>

__global__ void updateParticles(Particle* particles, int numParticles, Particle blackHole, float G, float dt, float softeningFactor);

Simulation::Simulation(int numParticles) : numParticles(numParticles), particles_device(nullptr) {
    particles_host.resize(numParticles);
}

Simulation::~Simulation() {
    if (particles_device) {
        checkCudaErrors(cudaFree(particles_device));
    }
}

void Simulation::init() {
    blackHole.position = glm::vec3(0.0f);
    blackHole.mass = 80000.0f;

    for (int i = 0; i < numParticles; ++i) {
        float phi = glm::radians((float)(rand() % 36000) / 100.0f);
        float cos_theta = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float theta = acos(cos_theta);
        float radius = 40.0f + (rand() % 2000) / 100.0f;
        particles_host[i].position.x = radius * sin(theta) * cos(phi);
        particles_host[i].position.y = radius * cos(theta);
        particles_host[i].position.z = radius * sin(theta) * sin(phi);
        float perfectOrbitSpeed = std::sqrt((G * blackHole.mass) / radius);
        glm::vec3 tangent = glm::cross(particles_host[i].position, glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::length(tangent) < 0.01f) {
            tangent = glm::cross(particles_host[i].position, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        tangent = glm::normalize(tangent);
        particles_host[i].velocity = tangent * perfectOrbitSpeed * instabilityFactor;
        particles_host[i].mass = (float)(rand() % 10 + 1);
    }

    checkCudaErrors(cudaMalloc((void**)&particles_device, numParticles * sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(particles_device, particles_host.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice));
}

void Simulation::update(float dt) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(particles_device, numParticles, blackHole, G, dt, softeningFactor);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(particles_host.data(), particles_device, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost));
    recycle_particles();
}

void Simulation::recycle_particles() {
    for (int i = 0; i < numParticles; ++i) {
        if (glm::length(particles_host[i].position) < 1.0f) {
            float phi = glm::radians((float)(rand() % 36000) / 100.0f);
            float cos_theta = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            float theta = acos(cos_theta);
            float radius = 60.0f + (rand() % 2000) / 100.0f;
            particles_host[i].position.x = radius * sin(theta) * cos(phi);
            particles_host[i].position.y = radius * cos(theta);
            particles_host[i].position.z = radius * sin(theta) * sin(phi);
            float perfectOrbitSpeed = std::sqrt((G * blackHole.mass) / radius);
            glm::vec3 tangent = glm::cross(particles_host[i].position, glm::vec3(0.0f, 1.0f, 0.0f));
            if (glm::length(tangent) < 0.01f) { tangent = glm::cross(particles_host[i].position, glm::vec3(1.0f, 0.0f, 0.0f)); }
            tangent = glm::normalize(tangent);
            particles_host[i].velocity = tangent * perfectOrbitSpeed * instabilityFactor;
            checkCudaErrors(cudaMemcpy(&particles_device[i], &particles_host[i], sizeof(Particle), cudaMemcpyHostToDevice));
        }
    }
}

const std::vector<Particle>& Simulation::getHostParticles() const {
    return particles_host;
}

// The CUDA kernel remains the same as before
__global__ void updateParticles(Particle* particles, int numParticles, Particle blackHole, float G, float dt, float softeningFactor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    glm::vec3 pos = particles[i].position;
    glm::vec3 vel = particles[i].velocity;
    float mass = particles[i].mass;
    glm::vec3 r_bh = blackHole.position - pos;
    float dist_sq_bh = r_bh.x * r_bh.x + r_bh.y * r_bh.y + r_bh.z * r_bh.z;
    glm::vec3 force_bh = glm::vec3(0.0f);
    if (dist_sq_bh > 0.25f) {
        float dist_bh = sqrtf(dist_sq_bh);
        force_bh = (r_bh / dist_bh) * (G * blackHole.mass * mass) / dist_sq_bh;
    }
    glm::vec3 force_others = glm::vec3(0.0f);
    for (int j = 0; j < numParticles; j++) {
        if (i == j) continue;
        glm::vec3 r_ij = particles[j].position - pos;
        float dist_sq_ij = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        float inv_dist = rsqrtf(dist_sq_ij + softeningFactor * softeningFactor);
        float forceMag = (G * particles[j].mass * mass) * (inv_dist * inv_dist * inv_dist);
        force_others += r_ij * forceMag;
    }
    glm::vec3 total_force = force_bh + force_others;
    vel += (total_force / mass) * dt;
    pos += vel * dt;
    particles[i].position = pos;
    particles[i].velocity = vel;
}