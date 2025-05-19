#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

struct Particle {
    float3 pos;
    float3 vel;
    float3 acc;
    float mass;
};

void print_particles(const std::vector<Particle>& particles, int max_print = 5);
void check_cuda_error(const char* msg);