#include "io.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool read_particles(const std::string& filename, std::vector<Particle>& particles) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Cannot open input file " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Particle p;
        if (!(iss >> p.pos.x >> p.pos.y >> p.pos.z >> p.vel.x >> p.vel.y >> p.vel.z >> p.mass >> p.radius)) {
            std::cerr << "Error: Invalid line in input: " << line << std::endl;
            return false;
        }
        particles.push_back(p);
    }
    return true;
}