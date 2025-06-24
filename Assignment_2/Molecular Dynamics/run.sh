#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --job-name=md_sim
#SBATCH --output=md_sim.out

set -e

echo "========= Molecular Dynamics Simulation ========="

make

mkdir -p output

INPUT_FILES=(
    "src/input/stable-particle.txt"
    "src/input/attact-particle.txt"
    "src/input/repel-particle.txt"
    "src/input/block-particles.txt"
    "src/input/grid-particles.txt"
    "src/input/particles_test1.txt"
    "src/input/collision_cluster.txt"
    "src/input/repulsiveshell.txt"

)

DT=0.0001
NSTEPS=1000
SIGMA=1.0
EPSILON=1.0
BOX_X=10.0
BOX_Y=10.0
BOX_Z=10.0
RCUT=2.5

echo "Test Case | #Particles | Avg. Time/Step (s)"
echo "--------------------------------------------"

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    if [[ -f "$INPUT_FILE" ]]; then
        NUM_PARTICLES=$(grep -v '^#' "$INPUT_FILE" | wc -l)
        TEST_NAME=$(basename "$INPUT_FILE" .txt)
        OUTDIR="output/${TEST_NAME}"
        mkdir -p "$OUTDIR"
        AVG_TIME=$(OUTPUT_DIR="$OUTDIR" ./build/md_sim "$INPUT_FILE" $DT $NSTEPS $SIGMA $EPSILON $BOX_X $BOX_Y $BOX_Z $RCUT | grep "Average time per step" | awk '{print $5}')
        echo "$TEST_NAME | $NUM_PARTICLES | $AVG_TIME"
    else
        echo "$INPUT_FILE not found, skipping."
    fi
done

echo "Done! Check the output directory for results."