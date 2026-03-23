import ctypes
import numpy as np
import time

lib = ctypes.cdll.LoadLibrary("./libconv.so")

lib.gpu_convolve2d.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

filters = {
    "EdgeDetect 3x3": (np.array([
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    ], dtype=np.float32), 3),
    "Sharpen 3x3": (np.array([
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    ], dtype=np.float32), 3),
    "Gaussian 5x5": (np.array([
        1/256.0,  4/256.0,  6/256.0,  4/256.0, 1/256.0,
        4/256.0, 16/256.0, 24/256.0, 16/256.0, 4/256.0,
        6/256.0, 24/256.0, 36/256.0, 24/256.0, 6/256.0,
        4/256.0, 16/256.0, 24/256.0, 16/256.0, 4/256.0,
        1/256.0,  4/256.0,  6/256.0,  4/256.0, 1/256.0
    ], dtype=np.float32), 5),
    "SobelX 3x3": (np.array([
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    ], dtype=np.float32), 3),
    "BoxBlur 7x7": (np.full(49, 1.0/49.0, dtype=np.float32), 7),
}

image_sizes = [512, 1024, 2048]

print("=" * 70)
print("CUDA Convolution via Python ctypes - Performance Benchmark")
print("=" * 70)

for M in image_sizes:
    np.random.seed(42)
    image = np.random.randint(0, 256, size=M * M, dtype=np.uint32)

    for fname, (fdata, N) in filters.items():
        output = np.zeros(M * M, dtype=np.uint32)

        start = time.time()
        lib.gpu_convolve2d(image, fdata, output, M, N)
        elapsed = time.time() - start

        print(f"Python->CUDA conv ({fname}, M={M}): {elapsed:.4f} sec")

    print()

print("=" * 70)
print("SciPy CPU Convolution Comparison")
print("=" * 70)

try:
    from scipy.ndimage import convolve as scipy_convolve

    for M in image_sizes:
        np.random.seed(42)
        image_2d = np.random.randint(0, 256, size=(M, M)).astype(np.float32)

        for fname, (fdata, N) in filters.items():
            kernel = fdata.reshape(N, N)
            start = time.time()
            _ = scipy_convolve(image_2d, kernel, mode='constant', cval=0.0)
            elapsed = time.time() - start
            print(f"SciPy CPU conv ({fname}, M={M}): {elapsed:.4f} sec")

        print()
except ImportError:
    print("SciPy not installed, skipping CPU comparison.")
