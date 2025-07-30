#pragma once
#include "io.h"
#include <vector>

// Cell list structure for 3D binning
struct CellList {
    int ncell_x, ncell_y, ncell_z;
    float cell_size;
    std::vector<std::vector<int>> cells; // cell index -> list of particle indices

    CellList(int ncell_x, int ncell_y, int ncell_z, float cell_size)
        : ncell_x(ncell_x), ncell_y(ncell_y), ncell_z(ncell_z), cell_size(cell_size),
          cells(ncell_x * ncell_y * ncell_z) {}

    int cell_index(int ix, int iy, int iz) const {
        return ((ix + ncell_x) % ncell_x) +
               ((iy + ncell_y) % ncell_y) * ncell_x +
               ((iz + ncell_z) % ncell_z) * ncell_x * ncell_y;
    }
};

void build_cell_list(const std::vector<Particle>& particles, CellList& clist, float box_x, float box_y, float box_z);

void compute_forces_cell_list(std::vector<Particle>& particles, const CellList& clist,
                              float sigma, float epsilon, float box_x, float box_y, float box_z, float rcut);