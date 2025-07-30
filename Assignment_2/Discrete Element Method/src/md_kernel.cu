#include "io.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <iostream>
#include "cell_list.h"

void check_cuda_error(const char* msg);

// Lennard-Jones force calculation kernel
__global__ void compute_forces_kernel(Particle* particles, int N, float sigma, float epsilon,  float box_x, float box_y, float box_z, float rcut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 force = make_float3(0, 0, 0);
    float3 pi = particles[i].pos;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        float3 pj = particles[j].pos;
        float3 rij = {pj.x - pi.x, pj.y - pi.y, pj.z - pi.z}; 
        rij.x -= box_x * roundf(rij.x / box_x);
        rij.y -= box_y * roundf(rij.y / box_y);
        rij.z -= box_z * roundf(rij.z / box_z);
        float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
        float dist = sqrtf(r2);
        float overlap = (particles[i].radius + particles[j].radius) - dist;
        if (overlap > 0.0f) {
            float3 n = {rij.x / dist, rij.y / dist, rij.z / dist};
            // Relative velocity along normal
            float vrel = (particles[i].vel.x - particles[j].vel.x) * n.x +
                         (particles[i].vel.y - particles[j].vel.y) * n.y +
                         (particles[i].vel.z - particles[j].vel.z) * n.z;
            float k = 10000.0f;    // spring constant (tune as needed)
            float gamma = 1.0f;  // damping (tune as needed)
            float fn = - k * overlap + gamma * vrel;
            force.x += fn * n.x;
            force.y += fn * n.y;
            force.z += fn * n.z;
        }
    }
    particles[i].acc.x = force.x / particles[i].mass;
    particles[i].acc.y = force.y / particles[i].mass;
    particles[i].acc.z = force.z / particles[i].mass;

    const float g = 9.81f; 
    particles[i].acc.z += g; 
}

void launch_compute_forces(Particle* d_particles, int N, float sigma, float epsilon, float box_x, float box_y, float box_z, float rcut) {
    int block = 128;
    int grid = (N + block - 1) / block;
    compute_forces_kernel<<<grid, block>>>(d_particles, N, sigma, epsilon, box_x, box_y, box_z, rcut);
    check_cuda_error("compute_forces_kernel");
}

// Velocity Verlet integration kernel
__global__ void integrate_first_half_kernel(Particle* particles, int N, float dt, float box_x, float box_y, float box_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Velocity half-step update
    particles[i].vel.x += 0.5f * particles[i].acc.x * dt;
    particles[i].vel.y += 0.5f * particles[i].acc.y * dt;
    particles[i].vel.z += 0.5f * particles[i].acc.z * dt;

    // Position full-step update
    particles[i].pos.x += particles[i].vel.x * dt;
    particles[i].pos.y += particles[i].vel.y * dt;
    particles[i].pos.z += particles[i].vel.z * dt;
    
    // Periodic boundary conditions
    if (particles[i].pos.x < 0) particles[i].pos.x += box_x;
    if (particles[i].pos.x >= box_x) particles[i].pos.x -= box_x;
    if (particles[i].pos.y < 0) particles[i].pos.y += box_y;
    if (particles[i].pos.y >= box_y) particles[i].pos.y -= box_y;
    if (particles[i].pos.z < 0) particles[i].pos.z += box_z;
    if (particles[i].pos.z >= box_z) particles[i].pos.z -= box_z;
}

void launch_integrate_first_half(Particle* d_particles, int N, float dt, float box_x, float box_y, float box_z) {
    int block = 128;
    int grid = (N + block - 1) / block;
    integrate_first_half_kernel<<<grid, block>>>(d_particles, N, dt, box_x, box_y, box_z);
    check_cuda_error("integrate_first_half_kernel");
}

__global__ void integrate_second_half_kernel(Particle* particles, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Final velocity half-step update using new acceleration
    particles[i].vel.x += 0.5f * particles[i].acc.x * dt;
    particles[i].vel.y += 0.5f * particles[i].acc.y * dt;
    particles[i].vel.z += 0.5f * particles[i].acc.z * dt;
}

void launch_integrate_second_half(Particle* d_particles, int N, float dt) {
    int block = 128;
    int grid = (N + block - 1) / block;
    integrate_second_half_kernel<<<grid, block>>>(d_particles, N, dt);
    check_cuda_error("integrate_second_half_kernel");
}


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

void build_cell_list(const std::vector<Particle>& particles, CellList& clist, float box_x, float box_y, float box_z) {
    float cell_size = clist.cell_size;
    for (int i = 0; i < clist.cells.size(); ++i) clist.cells[i].clear();
    for (int i = 0; i < particles.size(); ++i) {
        int ix = int(particles[i].pos.x / cell_size);
        int iy = int(particles[i].pos.y / cell_size);
        int iz = int(particles[i].pos.z / cell_size);
        int idx = clist.cell_index(ix, iy, iz);
        clist.cells[idx].push_back(i);
    }
}

void compute_forces_cell_list(std::vector<Particle>& particles, const CellList& clist,
                            float sigma, float epsilon, float box_x, float box_y, float box_z, float rcut) {
    // Zero accelerations
    for (auto& p : particles) p.acc = {0,0,0};
    float rcut2 = rcut * rcut;
    for (int ix = 0; ix < clist.ncell_x; ++ix) {
        for (int iy = 0; iy < clist.ncell_y; ++iy) {
            for (int iz = 0; iz < clist.ncell_z; ++iz) {
                int idx = clist.cell_index(ix, iy, iz);
                for (int a : clist.cells[idx]) {
                    Particle& pa = particles[a];
                    float3 fa = {0,0,0};
                    // Loop over neighbor cells
                    for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        int jx = (ix + dx + clist.ncell_x) % clist.ncell_x;
                        int jy = (iy + dy + clist.ncell_y) % clist.ncell_y;
                        int jz = (iz + dz + clist.ncell_z) % clist.ncell_z;
                        int jidx = clist.cell_index(jx, jy, jz);
                        for (int b : clist.cells[jidx]) {
                            if (a == b) continue;
                            Particle& pb = particles[b];
                            float3 rij = {pb.pos.x - pa.pos.x, pb.pos.y - pa.pos.y, pb.pos.z - pa.pos.z};
                            // Minimal image
                            rij.x -= box_x * roundf(rij.x / box_x);
                            rij.y -= box_y * roundf(rij.y / box_y);
                            rij.z -= box_z * roundf(rij.z / box_z);
                            float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
                            float dist = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
                            float overlap = (pa.radius + pb.radius) - dist;
                            if (overlap > 0.0f) {
                                float3 n = {rij.x / dist, rij.y / dist, rij.z / dist};
                                float vrel = (pa.vel.x - pb.vel.x) * n.x +
                                                (pa.vel.y - pb.vel.y) * n.y +
                                                (pa.vel.z - pb.vel.z) * n.z;
                                float k = 10000.0f;
                                float gamma = 1.0f;
                                float fn = - k * overlap + gamma * vrel;
                                fa.x += fn * n.x;
                                fa.y += fn * n.y;
                                fa.z += fn * n.z;
                            }
                        }
                    }
                    pa.acc.x += fa.x / pa.mass;
                    pa.acc.y += fa.y / pa.mass;
                    pa.acc.z += fa.z / pa.mass;

                    const float g = 9.81f;
                    pa.acc.z += g; 
                }
            }
        }
    }
}
