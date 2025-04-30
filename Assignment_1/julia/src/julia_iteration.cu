//Julia set polynomial
// polynomial p(z) = z^2 + c
// iteration rule z_{n+1} = z_n^2 +c

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Image dimensions and max iterations
const int WIDTH = 1024;
const int HEIGHT = 1024;
const int MAX_ITER = 128;
const float THRESHOLD = 10.0f;
const float X_MIN = -2.0f;
const float X_MAX = 2.0f;
const float Y_MIN = -2.0f;
const float Y_MAX = 2.0f;

__constant__ float2 c;  // Complex constant, stored in constant memory for GPU

// Julia iteration kernel
__device__ int juliaIterations(float2 z, int max_iter, float threshold) {
    int iter = 0;
    while (iter < max_iter && (z.x * z.x + z.y * z.y) < threshold * threshold) {
        float x = z.x * z.x - z.y * z.y + c.x;
        float y = 2.0f * z.x * z.y + c.y;
        z.x = x;
        z.y = y;
        ++iter;
    }
    return iter;
}

// Kernel to calculate the Julia set for each pixel
__global__ void juliaKernel(unsigned char* img, int width, int height, int max_iter, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float real = X_MIN + x * (X_MAX - X_MIN) / width;
    float imag = Y_MIN + y * (Y_MAX - Y_MIN) / height;
    float2 z = {real, imag};

    int iter = juliaIterations(z, max_iter, threshold);
    int idx = (y * width + x) * 3;
    unsigned char val = static_cast<unsigned char>(255.0f * iter / max_iter);
    img[idx + 0] = val;
    img[idx + 1] = val;
    img[idx + 2] = val;
}

// Save the image as PNG
void saveImage(const char* filename, unsigned char* data, int width, int height) {
    stbi_write_png(filename, width, height, 3, data, width * 3);
}

// Main function to initialize CUDA, launch the kernel, and save the image
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./julia <c_real> <c_imag> <output_filename>\n";
        return 1;
    }

    // Parse command line arguments
    float c_real = std::stof(argv[1]);
    float c_imag = std::stof(argv[2]);
    std::string outname = argv[3];

    // Copy complex constant to device constant memory
    float2 c_host = {c_real, c_imag};
    cudaMemcpyToSymbol(c, &c_host, sizeof(float2));

    // Allocate memory for the image on the host and device
    const int imgSize = WIDTH * HEIGHT * 3;
    unsigned char* h_img = new unsigned char[imgSize];
    unsigned char* d_img;
    cudaMalloc(&d_img, imgSize);

    // Launch kernel to compute Julia set
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    juliaKernel<<<gridSize, blockSize>>>(d_img, WIDTH, HEIGHT, MAX_ITER, THRESHOLD);
    cudaDeviceSynchronize();

    // Copy result back to host and save image
    cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost);
    saveImage(outname.c_str(), h_img, WIDTH, HEIGHT);

    // Clean up
    std::cout << "Saved: " << outname << std::endl;
    cudaFree(d_img);
    delete[] h_img;

    return 0;
}

