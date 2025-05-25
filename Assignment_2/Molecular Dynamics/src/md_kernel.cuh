#pragma once
#include "io.h"

void launch_compute_forces(Particle* d_particles, int N, float sigma, float epsilon);
void launch_integrate_first_half(Particle* d_particles, int N, float dt);
void launch_integrate_second_half(Particle* d_particles, int N, float dt);
