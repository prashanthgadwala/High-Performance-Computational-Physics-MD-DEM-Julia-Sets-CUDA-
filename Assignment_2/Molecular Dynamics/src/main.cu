#include "io.h"
#include "md_kernel.cuh"
#include "vtk_writer.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <string>

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

    const char* outdir_env = std::getenv("OUTPUT_DIR");
    std::string outdir = outdir_env ? outdir_env : "output";

    auto t_start = std::chrono::high_resolution_clock::now();

    launch_compute_forces(d_particles, N, sigma, epsilon);

    for (int step = 0; step < nsteps; ++step) {
        launch_integrate_first_half(d_particles, N, dt);
        launch_compute_forces(d_particles, N, sigma, epsilon);
        launch_integrate_second_half(d_particles, N, dt);

        // Output VTK every 100 steps
        if (step % 100 == 0) {
            cudaMemcpy(particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
            for (int i = 0; i < N; ++i) {
                std::cout << "Step " << step << " Particle " << i
                          << ": pos=(" << particles[i].pos.x << "," << particles[i].pos.y << "," << particles[i].pos.z << ")"
                          << " vel=(" << particles[i].vel.x << "," << particles[i].vel.y << "," << particles[i].vel.z << ")\n";
            }
            std::string vtkfile = outdir + "/step_" + std::to_string(step) + ".vtk";
            write_vtk(vtkfile, particles, step);
        }
        
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Total simulation time: " << elapsed << " s\n";
    std::cout << "Average time per step: " << (elapsed / nsteps) << " s\n";

    cudaMemcpy(particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    void print_particles(const std::vector<Particle>& particles, int max_print);
    print_particles(particles, 5);
    return 0;
}