#include "vtk_writer.h"
#include <fstream>
#include <iomanip>

constexpr double box_x = 10.0;
constexpr double box_y = 10.0;
constexpr double box_z = 10.0;

bool write_vtk(const std::string& filename, const std::vector<Particle>& particles, int step) {
    std::ofstream ofs(filename);
    if (!ofs) return false;

    ofs << "# vtk DataFile Version 4.0\n";
    ofs << "hesp visualization file\n";
    ofs << "ASCII\n";
    ofs << "DATASET UNSTRUCTURED_GRID\n";
    ofs << "POINTS " << particles.size() + 8 << " double\n";
    for (const auto& p : particles) {
        ofs << std::fixed << std::setprecision(6)
            << p.pos.x << " " << p.pos.y << " " << p.pos.z << "\n";
    }
    // Add 8 box corners after particle points
    ofs << 0 << " " << 0 << " " << 0 << "\n";
    ofs << box_x << " " << 0 << " " << 0 << "\n";
    ofs << box_x << " " << box_y << " " << 0 << "\n";
    ofs << 0 << " " << box_y << " " << 0 << "\n";
    ofs << 0 << " " << 0 << " " << box_z << "\n";
    ofs << box_x << " " << 0 << " " << box_z << "\n";
    ofs << box_x << " " << box_y << " " << box_z << "\n";
    ofs << 0 << " " << box_y << " " << box_z << "\n";

    // Write CELLS section: 12 lines for box edges
    int n_particles = particles.size();
    int n_box_pts = 8;
    int n_lines = 12;
    ofs << "CELLS " << n_lines << " " << n_lines * 3 << "\n";
    int base = n_particles;
    // Bottom face
    ofs << "2 " << base+0 << " " << base+1 << "\n";
    ofs << "2 " << base+1 << " " << base+2 << "\n";
    ofs << "2 " << base+2 << " " << base+3 << "\n";
    ofs << "2 " << base+3 << " " << base+0 << "\n";
    // Top face
    ofs << "2 " << base+4 << " " << base+5 << "\n";
    ofs << "2 " << base+5 << " " << base+6 << "\n";
    ofs << "2 " << base+6 << " " << base+7 << "\n";
    ofs << "2 " << base+7 << " " << base+4 << "\n";
    // Vertical edges
    ofs << "2 " << base+0 << " " << base+4 << "\n";
    ofs << "2 " << base+1 << " " << base+5 << "\n";
    ofs << "2 " << base+2 << " " << base+6 << "\n";
    ofs << "2 " << base+3 << " " << base+7 << "\n";

    // Write CELL_TYPES section: VTK_LINE = 3
    ofs << "CELL_TYPES " << n_lines << "\n";
    for (int i = 0; i < n_lines; ++i) ofs << "3\n";

    // Point data for particles only
    ofs << "POINT_DATA " << particles.size() << "\n";
    ofs << "SCALARS m double\n";
    ofs << "LOOKUP_TABLE default\n";
    for (const auto& p : particles) {
        ofs << p.mass << "\n";
    }
    ofs << "VECTORS v double\n";
    for (const auto& p : particles) {
        ofs << p.vel.x << " " << p.vel.y << " " << p.vel.z << "\n";
    }
    return true;
}