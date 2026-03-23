#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_FILTER_SIZE 7

__constant__ float d_filter_lib[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

__global__ void convolve2DKernel(unsigned int *image, unsigned int *output, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int pad = N / 2;

    if (row < M && col < M) {
        float sum = 0.0f;
        for (int fi = 0; fi < N; fi++) {
            for (int fj = 0; fj < N; fj++) {
                int imgRow = row + fi - pad;
                int imgCol = col + fj - pad;
                if (imgRow >= 0 && imgRow < M && imgCol >= 0 && imgCol < M) {
                    sum += (float)image[imgRow * M + imgCol] * d_filter_lib[fi * N + fj];
                }
            }
        }
        int val = (int)(sum + 0.5f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        output[row * M + col] = (unsigned int)val;
    }
}

extern "C" void gpu_convolve2d(unsigned int *h_image, float *h_filter,
                               unsigned int *h_output, int M, int N) {
    size_t imgSize = M * M * sizeof(unsigned int);
    unsigned int *d_image, *d_output;

    cudaMalloc((void **)&d_image, imgSize);
    cudaMalloc((void **)&d_output, imgSize);

    cudaMemcpy(d_image, h_image, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter_lib, h_filter, N * N * sizeof(float));

    int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolve2DKernel<<<dimGrid, dimBlock>>>(d_image, d_output, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_output);
}
