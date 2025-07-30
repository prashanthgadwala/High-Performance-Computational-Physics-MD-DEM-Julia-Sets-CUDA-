# High-Performance-Computational-Physics-MD-DEM-Julia-Sets-CUDA-
A collection of advanced computational physics projects for the HESP course featuring CUDA-accelerated Molecular Dynamics (MD) and Discrete Element Method (DEM) simulations, Julia set fractal visualization, and STREAM memory benchmarks. 


# HESP_SoSe_2025: Computational Physics Projects (MD, DEM, Julia Sets)

>This repository showcases advanced computational physics assignments for the HESP course (Summer Semester 2025), featuring CUDA-accelerated simulations, fractal visualization, and high-performance computing concepts. The codebase is designed for both learning and benchmarking, with a focus on clarity, modularity, and extensibility.

---

## 🚀 Project Highlights

- **Molecular Dynamics (MD) Simulation**  
  - CUDA-accelerated, supports Lennard-Jones potential, periodic boundaries, cell lists, and VTK output.
- **Discrete Element Method (DEM) Simulation**  
  - Particle-based contact dynamics, customizable input, VTK output for visualization.
- **Julia Set Visualization**  
  - CUDA fractal renderer, PNG output, customizable parameters.
- **STREAM Benchmark**  
  - Memory bandwidth benchmarking (CPU, OpenMP, CUDA).

---

## 🗂️ Project Structure

```text
HESP_SoSe_2025/
│
├── Assignment_1/
│   ├── Julia/                # Julia set CUDA code, PNG images, run script
│   └── stream benchmark/     # STREAM memory benchmark (CPU, OpenMP, CUDA)
│
├── Assignment_2/
│   ├── Molecular Dynamics/   # CUDA MD simulation, input/output, src
│   └── Discrete Element Method/ # DEM simulation, input/output, src
│
└── README.md
```

---

## 🧩 Conceptual Overview

### Molecular Dynamics (MD) & DEM

**Core Concepts:**
- Particle-based simulation
- Force calculation (Lennard-Jones, contact)
- Time integration (Velocity Verlet)
- Periodic boundary conditions
- Cell list optimization
- CUDA parallelization

**Simulation Flow:**

```text
┌──────────────┐
│ Read Input   │
└─────┬────────┘
      │
┌─────▼───────┐
│ Initialize  │
└─────┬───────┘
      │
┌─────▼─────────────┐
│ For each timestep │
│  ┌──────────────┐ │
│  │ Compute      │ │
│  │ Forces       │ │
│  └─────┬────────┘ │
│        │          │
│  ┌─────▼───────┐  │
│  │ Integrate   │  │
│  │ Positions   │  │
│  └─────┬───────┘  │
│        │          │
│  ┌─────▼───────┐  │
│  │ Write VTK   │  │
│  │ Output      │  │
│  └─────────────┘  │
└───────────────────┘
```

### Julia Set Visualization

**Concepts:**
- Complex number iteration
- Escape-time algorithm
- CUDA parallel pixel computation
- PNG image output

**Flowchart:**

```text
┌──────────────┐
│ Set Params   │
└─────┬────────┘
      │
┌─────▼─────────────┐
│ For each pixel    │
│  ┌──────────────┐ │
│  │ Iterate      │ │
│  │ Complex Eqn  │ │
│  └─────┬────────┘ │
│        │          │
│  ┌─────▼───────┐  │
│  │ Color pixel │  │
│  └─────────────┘  │
└───────────────────┘
```

---

## 🛠️ Build & Run Instructions

### Molecular Dynamics

```sh
cd Assignment_2/Molecular\ Dynamics
make
./build/md_sim input/particles_test1.txt 0.001 1000 1.0 1.0 10.0 10.0 10.0 2.5
# Args: input_file dt nsteps sigma epsilon box_x box_y box_z rcut
```

### Discrete Element Method (DEM)

```sh
cd Assignment_2/Discrete\ Element\ Method
make
./build/md_sim input/attact-particle.txt 0.001 1000 1.0 1.0 10.0 10.0 10.0 2.5
```

### Julia Set

```sh
cd Assignment_1/Julia
bash run.sh
# Output: PNG images in output/
```

---

## 📂 Input File Format

- **MD/DEM:** Plain text, each line = particle (x y z vx vy vz m r)
- **Julia:** Parameters set in `run.sh` or source

---

## 📊 Output & Visualization

- **MD/DEM:** `.vtk` files in `output/` (view in [ParaView](https://www.paraview.org/))
- **Julia:** PNG images in `output/`

---

## ⚡ Performance & Features

- CUDA acceleration for MD, DEM, Julia set
- Cell list for O(N) force computation
- Periodic boundaries, cut-off radius
- Modular code: easy to extend for new potentials or features
- STREAM benchmark for memory bandwidth (compare CPU, OpenMP, CUDA)

---

## 📖 Example Directory Tree

```text
Assignment_2/
├── Molecular Dynamics/
│   ├── build/
│   ├── output/
│   ├── src/
│   └── input/
└── Discrete Element Method/
    ├── build/
    ├── output/
    ├── src/
    └── input/
```

---

## 📝 Further Reading & References

- [LAMMPS Documentation](https://docs.lammps.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [ParaView User Guide](https://www.paraview.org/)
- [Wikipedia: Julia Set](https://en.wikipedia.org/wiki/Julia_set)

---

## 📜 License

This repository is for educational purposes as part of the HESP course.
