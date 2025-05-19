#include "io.h"
#include "utils.h"
#include "md_kernel.cuh"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " input.txt dt nsteps sigma epsilon" << std::endl;
        return 1;
    }
    std::string input_file = argv[1];
    float dt = std::stof(argv[2]);
    int nsteps = std::stoi(argv[3]);
    float sigma = std::stof(argv[4]);
    float epsilon = std::stof(argv[5]);

    std::vector<Particle> particles;
    if (!read_particles(input_file, particles)) return 1;
    int N = particles.size();

    // Allocate device memory
    Particle* d_particles;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMemcpy(d_particles, particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice);

    for (int step = 0; step < nsteps; ++step) {
        launch_compute_forces(d_particles, N, sigma, epsilon);
        launch_integrate(d_particles, N, dt);
        // Optionally: copy back and write VTK every N steps
        // cudaMemcpy(particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
        // if (step % 100 == 0) write_vtk(...);
    }

    cudaMemcpy(particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    print_particles(particles, 5);
    return 0;
}