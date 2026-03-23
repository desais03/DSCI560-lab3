#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_FILTER_SIZE 7

__constant__ float d_filter[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

__global__ void convolve2DGPU(unsigned int *image, unsigned int *output, int M, int N) {
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
                    sum += (float)image[imgRow * M + imgCol] * d_filter[fi * N + fj];
                }
            }
        }
        int val = (int)(sum + 0.5f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        output[row * M + col] = (unsigned int)val;
    }
}

void generateTestImage(unsigned int *image, int M) {
    srand(42);
    for (int i = 0; i < M * M; i++) {
        image[i] = rand() % 256;
    }
}

void saveImagePGM(const char *filename, unsigned int *image, int M) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%d %d\n255\n", M, M);
    for (int i = 0; i < M * M; i++) {
        unsigned char pixel = (unsigned char)image[i];
        fwrite(&pixel, 1, 1, fp);
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int filterChoice = (argc > 2) ? atoi(argv[2]) : 0;

    float edgeDetect3[] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    float sharpen3[] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
    float gaussianBlur5[] = {
        1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f,
        4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
        6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
        4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
        1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f
    };
    float sobelX3[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    float emboss3[] = {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    };
    float boxBlur7[] = {
        1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
        1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
        1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
        1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
        1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
        1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
        1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f
    };

    float *filter;
    int N;
    const char *filterName;

    switch (filterChoice) {
        case 0: filter = edgeDetect3; N = 3; filterName = "EdgeDetect3x3"; break;
        case 1: filter = sharpen3; N = 3; filterName = "Sharpen3x3"; break;
        case 2: filter = gaussianBlur5; N = 5; filterName = "GaussianBlur5x5"; break;
        case 3: filter = sobelX3; N = 3; filterName = "SobelX3x3"; break;
        case 4: filter = emboss3; N = 3; filterName = "Emboss3x3"; break;
        case 5: filter = boxBlur7; N = 7; filterName = "BoxBlur7x7"; break;
        default: filter = edgeDetect3; N = 3; filterName = "EdgeDetect3x3"; break;
    }

    size_t imgSize = M * M * sizeof(unsigned int);
    unsigned int *h_image = (unsigned int *)malloc(imgSize);
    unsigned int *h_output = (unsigned int *)malloc(imgSize);

    generateTestImage(h_image, M);

    unsigned int *d_image, *d_output;
    cudaMalloc((void **)&d_image, imgSize);
    cudaMalloc((void **)&d_output, imgSize);

    cudaMemcpy(d_image, h_image, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, filter, N * N * sizeof(float));

    int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    convolve2DGPU<<<dimGrid, dimBlock>>>(d_image, d_output, M, N);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    printf("CUDA Convolution (%s, M=%d, N=%d): %f ms\n", filterName, M, N, milliseconds);

    char outputFile[256];
    sprintf(outputFile, "output_gpu_%s_%d.pgm", filterName, M);
    saveImagePGM(outputFile, h_output, M);
    printf("Saved: %s\n", outputFile);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_image);
    cudaFree(d_output);
    free(h_image);
    free(h_output);
    return 0;
}
