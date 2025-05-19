#include "vtk_writer.h"
#include <fstream>
#include <iomanip>

bool write_vtk(const std::string& filename, const std::vector<Particle>& particles, int step) {
    std::ofstream ofs(filename);
    if (!ofs) return false;

    ofs << "# vtk DataFile Version 3.0\n";
    ofs << "Molecular Dynamics step " << step << "\n";
    ofs << "ASCII\n";
    ofs << "DATASET POLYDATA\n";
    ofs << "POINTS " << particles.size() << " float\n";
    for (const auto& p : particles) {
        ofs << std::fixed << std::setprecision(6)
            << p.pos.x << " " << p.pos.y << " " << p.pos.z << "\n";
    }
    ofs << "\n";
    ofs << "VERTICES " << particles.size() << " " << 2 * particles.size() << "\n";
    for (size_t i = 0; i < particles.size(); ++i) {
        ofs << "1 " << i << "\n";
    }
    return true;
}