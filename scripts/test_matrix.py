import ctypes
import numpy as np
import time

lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

sizes = [256, 512, 1024, 2048]

for N in sizes:
    A = np.random.rand(N * N).astype(np.float32)
    B = np.random.rand(N * N).astype(np.float32)
    C = np.zeros(N * N, dtype=np.float32)

    start = time.time()
    lib.gpu_matrix_multiply(A, B, C, N)
    end = time.time()

    print(f"Python->CUDA library matmul (N={N}): {end - start:.4f} seconds")

    np_A = A.reshape(N, N)
    np_B = B.reshape(N, N)

    start_np = time.time()
    np_C = np.dot(np_A, np_B)
    end_np = time.time()

    print(f"NumPy matmul              (N={N}): {end_np - start_np:.4f} seconds")

    cuda_C = C.reshape(N, N)
    if np.allclose(cuda_C, np_C, atol=1e-1):
        print(f"  -> Verification PASSED (N={N})")
    else:
        max_diff = np.max(np.abs(cuda_C - np_C))
        print(f"  -> Verification WARNING: max diff = {max_diff:.6f} (N={N})")
    print()
