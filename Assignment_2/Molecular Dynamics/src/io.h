#pragma once
#include <vector>
#include <string>

struct float3 {
    float x, y, z;
};

struct Particle {
    float3 pos;
    float3 vel;
    float mass;
};

bool read_particles(const std::string& filename, std::vector<Particle>& particles);
