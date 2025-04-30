#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output

# Define an array of complex numbers for different Julia set configurations
configs=(
    "-0.8 0.2 classic"
    "0.285 0.01 spiral1"
    "-0.4 0.6 swirl"
    "-0.70176 -0.3842 dense"
    "0.355 0.355 eye"
    "-0.75 0.11 nebula"
)

# Compile the CUDA file
nvcc -O2 julia_iteration.cu -o julia

# Loop through the configurations and run the CUDA program for each
for config in "${configs[@]}"; do
    set -- $config
    c_real=$1
    c_imag=$2
    output_file="output/julia_${3}.png"

    echo "Running for c = ${c_real} + ${c_imag}i, saving to ${output_file}..."
    ./julia $c_real $c_imag $output_file
done

echo "All tests completed!"
