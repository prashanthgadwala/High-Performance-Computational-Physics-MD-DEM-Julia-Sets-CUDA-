#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include "lodepng.h"

// Image dimensions
const int WIDTH = 1024;
const int HEIGHT = 1024;
const int MAX_ITER = 300;

// Coordinate bounds for the Julia set
const float X_MIN = -2.0f;
const float X_MAX = 2.0f;
const float Y_MIN = -2.0f;
const float Y_MAX = 2.0f;

// CUDA kernel for Julia set generation
__global__ void juliaKernel(unsigned char* img, float cr, float ci) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= WIDTH || y >= HEIGHT)
        return;
    
    // Map pixel position to complex plane
    float zx = X_MIN + (X_MAX - X_MIN) * x / (WIDTH - 1);
    float zy = Y_MIN + (Y_MAX - Y_MIN) * y / (HEIGHT - 1);
    
    int iter = 0;
    float zx2 = zx * zx;
    float zy2 = zy * zy;
    
    // Standard Julia set iteration: z = z² + c
    while (iter < MAX_ITER && zx2 + zy2 < 4.0f) {
        float xtemp = zx2 - zy2 + cr;
        zy = 2 * zx * zy + ci;
        zx = xtemp;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
    }
    
    // Calculate the color based on iteration count
    int idx = 4 * (y * WIDTH + x);
    
    if (iter == MAX_ITER) {
        // Points in the set are black
        img[idx]     = 0;     // R
        img[idx + 1] = 0;     // G
        img[idx + 2] = 0;     // B
        img[idx + 3] = 255;   // Alpha
    } else {
        // Map iteration count to a smooth color
        float t = (float)iter / MAX_ITER;
        
        // RGB coloring (simple gradient)
        img[idx]     = (unsigned char)(255 * t);           // R
        img[idx + 1] = (unsigned char)(255 * (1.0f - t));  // G
        img[idx + 2] = (unsigned char)(128);               // B
        img[idx + 3] = 255;                                // Alpha
    }
}

// Helper function to save an image using LodePNG
void saveImage(const char* filename, const std::vector<unsigned char>& image) {
    unsigned error = lodepng::encode(filename, image, WIDTH, HEIGHT);
    if (error) {
        std::cerr << "PNG encoding error " << error << ": " << lodepng_error_text(error) << std::endl;
    } else {
        std::cout << "Image saved successfully to " << filename << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <c_real> <c_imag> <output_filename>\n";
        return 1;
    }
    
    float cr = std::stof(argv[1]);  // Real part of c
    float ci = std::stof(argv[2]);  // Imaginary part of c
    std::string outfile = argv[3];  // Output filename
    
    std::cout << "Generating Julia set for c = " << cr << " + " << ci << "i\n";
    
    // Allocate host memory for the image
    std::vector<unsigned char> h_img(WIDTH * HEIGHT * 4, 0);
    
    // Allocate device memory
    unsigned char* d_img = nullptr;
    cudaMalloc(&d_img, WIDTH * HEIGHT * 4);
    
    // Initialize device memory to zero
    cudaMemset(d_img, 0, WIDTH * HEIGHT * 4);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, 
                 (HEIGHT + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Launching kernel with grid: " << gridDim.x << "x" << gridDim.y 
              << ", block: " << blockDim.x << "x" << blockDim.y << std::endl;
    
    // Launch the kernel
    juliaKernel<<<gridDim, blockDim>>>(d_img, cr, ci);
    
    // Check for kernel launch errors
    //cudaGetLastError();

    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_img.data(), d_img, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_img);
    
    // Save the image
    saveImage(outfile.c_str(), h_img);
    
    return 0;
}