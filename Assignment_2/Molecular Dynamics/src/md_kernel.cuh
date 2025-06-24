#pragma once
#include "io.h"
#include "cell_list.h" 

void launch_compute_forces(Particle* d_particles, int N, float sigma, float epsilon, 
                           float box_x, float box_y, float box_z, float rcut);
void launch_integrate_first_half(Particle* d_particles, int N, float dt, 
                                 float box_x, float box_y, float box_z);
void launch_integrate_second_half(Particle* d_particles, int N, float dt);

void build_cell_list(const std::vector<Particle>& particles, CellList& clist, float box_x, float box_y, float box_z);

void compute_forces_cell_list(std::vector<Particle>& particles, const CellList& clist,
                              float sigma, float epsilon, float box_x, float box_y, float box_z, float rcut);