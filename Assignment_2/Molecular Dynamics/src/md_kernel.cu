#include "utils.h"
#include <iostream>

void print_particles(const std::vector<Particle>& particles, int max_print) {
    for (int i = 0; i < std::min((int)particles.size(), max_print); ++i) {
        const auto& p = particles[i];
        std::cout << "Particle " << i << ": pos=(" << p.pos.x << "," << p.pos.y << "," << p.pos.z
                  << ") vel=(" << p.vel.x << "," << p.vel.y << "," << p.vel.z
                  << ") mass=" << p.mass << std::endl;
    }
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}