#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convolve2D(unsigned int *image, int M,
                float *filter, int N,
                unsigned int *output) {
    int pad = N / 2;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            for (int fi = 0; fi < N; fi++) {
                for (int fj = 0; fj < N; fj++) {
                    int imgRow = i + fi - pad;
                    int imgCol = j + fj - pad;
                    if (imgRow >= 0 && imgRow < M && imgCol >= 0 && imgCol < M) {
                        sum += (float)image[imgRow * M + imgCol] * filter[fi * N + fj];
                    }
                }
            }
            int val = (int)(sum + 0.5f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            output[i * M + j] = (unsigned int)val;
        }
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
        case 0:
            filter = edgeDetect3; N = 3; filterName = "EdgeDetect3x3"; break;
        case 1:
            filter = sharpen3; N = 3; filterName = "Sharpen3x3"; break;
        case 2:
            filter = gaussianBlur5; N = 5; filterName = "GaussianBlur5x5"; break;
        case 3:
            filter = sobelX3; N = 3; filterName = "SobelX3x3"; break;
        case 4:
            filter = emboss3; N = 3; filterName = "Emboss3x3"; break;
        case 5:
            filter = boxBlur7; N = 7; filterName = "BoxBlur7x7"; break;
        default:
            filter = edgeDetect3; N = 3; filterName = "EdgeDetect3x3"; break;
    }

    unsigned int *image = (unsigned int *)malloc(M * M * sizeof(unsigned int));
    unsigned int *output = (unsigned int *)malloc(M * M * sizeof(unsigned int));

    generateTestImage(image, M);

    char inputFile[256];
    sprintf(inputFile, "input_%d.pgm", M);
    saveImagePGM(inputFile, image, M);

    clock_t start = clock();
    convolve2D(image, M, filter, N, output);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU Convolution (%s, M=%d, N=%d): %f seconds\n", filterName, M, N, elapsed);

    char outputFile[256];
    sprintf(outputFile, "output_cpu_%s_%d.pgm", filterName, M);
    saveImagePGM(outputFile, output, M);
    printf("Saved: %s\n", outputFile);

    free(image);
    free(output);
    return 0;
}
