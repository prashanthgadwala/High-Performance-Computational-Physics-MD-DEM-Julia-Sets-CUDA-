
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <cmath>
#include "lodepng.h"

// --- Parameters ---
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;
constexpr int MAX_ITER = 300;
constexpr float X_MIN = -2.0f;
constexpr float X_MAX = 2.0f;
constexpr float Y_MIN = -2.0f;
constexpr float Y_MAX = 2.0f;

// --- Device coloring helper (HSV to RGB) ---
__device__ void hsv2rgb(float h, float s, float v, unsigned char& r, unsigned char& g, unsigned char& b) {
    float c = v * s;
    float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
    float m = v - c;
    float r1, g1, b1;
    if (h < 60)      { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120)  { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180)  { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240)  { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300)  { r1 = x; g1 = 0; b1 = c; }
    else               { r1 = c; g1 = 0; b1 = x; }
    r = (unsigned char)(255 * (r1 + m));
    g = (unsigned char)(255 * (g1 + m));
    b = (unsigned char)(255 * (b1 + m));
}

// --- CUDA kernel for Julia set ---
__global__ void juliaKernel(unsigned char* img, float cr, float ci) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float zx = X_MIN + (X_MAX - X_MIN) * x / (WIDTH - 1);
    float zy = Y_MIN + (Y_MAX - Y_MIN) * y / (HEIGHT - 1);
    int iter = 0;
    float zx2 = zx * zx;
    float zy2 = zy * zy;
    while (iter < MAX_ITER && zx2 + zy2 < 4.0f) {
        float xtemp = zx2 - zy2 + cr;
        zy = 2 * zx * zy + ci;
        zx = xtemp;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
    }
    int idx = 4 * (y * WIDTH + x);
    if (iter == MAX_ITER) {
        img[idx] = 0; img[idx+1] = 0; img[idx+2] = 0; img[idx+3] = 255;
    } else {
        float t = (float)iter / MAX_ITER;
        float h = 360.0f * t;
        unsigned char r, g, b;
        hsv2rgb(h, 1.0f, 1.0f, r, g, b);
        img[idx] = r; img[idx+1] = g; img[idx+2] = b; img[idx+3] = 255;
    }
}

// --- Host-side PNG save helper ---
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
    float cr = std::stof(argv[1]);
    float ci = std::stof(argv[2]);
    std::string outfile = argv[3];
    std::cout << "Generating Julia set for c = " << cr << " + " << ci << "i\n";

    std::vector<unsigned char> h_img(WIDTH * HEIGHT * 4, 0);
    unsigned char* d_img = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_img, WIDTH * HEIGHT * 4);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; return 1; }
    err = cudaMemset(d_img, 0, WIDTH * HEIGHT * 4);
    if (err != cudaSuccess) { std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_img); return 1; }

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);
    std::cout << "Launching kernel with grid: " << gridDim.x << "x" << gridDim.y << ", block: " << blockDim.x << "x" << blockDim.y << std::endl;
    juliaKernel<<<gridDim, blockDim>>>(d_img, cr, ci);
    err = cudaGetLastError();
    if (err != cudaSuccess) { std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_img); return 1; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_img); return 1; }
    err = cudaMemcpy(h_img.data(), d_img, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_img); return 1; }

    saveImage(outfile.c_str(), h_img);
    cudaFree(d_img);
    return 0;
}