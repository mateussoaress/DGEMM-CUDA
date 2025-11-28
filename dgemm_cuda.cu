#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32      // tamanho do bloco para a versão CUDA otimizada
#define CPU_BLOCK_SIZE 64  // tamanho do bloco para a versão CPU otimizada (tiling)

// -----------------------------------------------------------------------------
// Macros / Funções auxiliares
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

void randomMatrix(double *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (double) rand() / RAND_MAX;
    }
}

// diferença relativa máxima entre CPU e GPU
double maxRelativeDiff(const double *seq, const double *gpu, int n) {
    double maxDiff = 0.0;
    const double eps = 1e-12;

    for (int i = 0; i < n; i++) {
        double denom = fabs(seq[i]) + eps;
        double rel = fabs(seq[i] - gpu[i]) / denom;
        if (rel > maxDiff) {
            maxDiff = rel;
        }
    }
    return maxDiff;
}

// medição simples de tempo em segundos (CPU)
double wall_time() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}


static void transpose_cpu(int n, const double *matrix, double *result) {
    int i, j, ii, jj;
    for (ii = 0; ii < n; ii += CPU_BLOCK_SIZE) {
        for (jj = 0; jj < n; jj += CPU_BLOCK_SIZE) {
            for (i = ii; i < ii + CPU_BLOCK_SIZE && i < n; i++) {
                for (j = jj; j < jj + CPU_BLOCK_SIZE && j < n; j++) {
                    result[j * n + i] = matrix[i * n + j];
                }
            }
        }
    }
}

void dgemm_seq(int n, double alpha,
               const double *A, const double *B,
               double beta, double *C)
{
    // Aplica beta em C, se necessário
    if (beta != 1.0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i * n + j] *= beta;
            }
        }
    }

    // Transpõe B para melhorar localidade de memória
    size_t bytes = (size_t)n * n * sizeof(double);
    double *BT = (double*) malloc(bytes);
    if (!BT) {
        fprintf(stderr, "Erro ao alocar memória para BT na dgemm_seq\n");
        exit(EXIT_FAILURE);
    }

    transpose_cpu(n, B, BT);

    // DGEMM com tiling + register blocking 2x2
    for (int ii = 0; ii < n; ii += CPU_BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += CPU_BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += CPU_BLOCK_SIZE) {

                int i_max = (ii + CPU_BLOCK_SIZE < n) ? (ii + CPU_BLOCK_SIZE) : n;
                int j_max = (jj + CPU_BLOCK_SIZE < n) ? (jj + CPU_BLOCK_SIZE) : n;
                int k_max = (kk + CPU_BLOCK_SIZE < n) ? (kk + CPU_BLOCK_SIZE) : n;

                for (int i = ii; i < i_max; i += 2) {
                    for (int j = jj; j < j_max; j += 2) {

                        // register blocking 2x2
                        double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;

                        for (int k = kk; k < k_max; k++) {
                            double a0k = A[i * n + k];
                            double a1k = (i + 1 < n) ? A[(i + 1) * n + k] : 0.0;

                            double b0k = BT[j * n + k];
                            double b1k = (j + 1 < n) ? BT[(j + 1) * n + k] : 0.0;

                            c00 += a0k * b0k;
                            c01 += a0k * b1k;
                            c10 += a1k * b0k;
                            c11 += a1k * b1k;
                        }

                        C[i * n + j] += alpha * c00;
                        if (j + 1 < n)           C[i * n + j + 1]       += alpha * c01;
                        if (i + 1 < n)           C[(i + 1) * n + j]     += alpha * c10;
                        if (i + 1 < n && j + 1 < n)
                                               C[(i + 1) * n + j + 1] += alpha * c11;
                    }
                }
            }
        }
    }

    free(BT);
}

// -----------------------------------------------------------------------------
// Kernel CUDA básico: 1 thread -> 1 elemento de C
// -----------------------------------------------------------------------------
__global__
void dgemm_basic_kernel(int n, double alpha,
                        const double *A, const double *B,
                        double beta, double *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row >= n || col >= n) return;

    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }

    double c_old = C[row * n + col];
    C[row * n + col] = alpha * sum + beta * c_old;
}

// -----------------------------------------------------------------------------
// Kernel CUDA otimizado com memória compartilhada (tiling)
// Cada bloco calcula um sub-bloco BLOCK_SIZE x BLOCK_SIZE de C
// -----------------------------------------------------------------------------
__global__
void dgemm_tiled_kernel(int n, double alpha,
                        const double *A, const double *B,
                        double beta, double *C) {
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    double sum = 0.0;

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < numTiles; ++m) {
        int a_col = m * BLOCK_SIZE + threadIdx.x;
        int b_row = m * BLOCK_SIZE + threadIdx.y;

        // Carrega tile de A em shared memory
        if (row < n && a_col < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Carrega tile de B em shared memory
        if (b_row < n && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Produto parcial do bloco
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        double c_old = C[row * n + col];
        C[row * n + col] = alpha * sum + beta * c_old;
    }
}

// -----------------------------------------------------------------------------
// Funções host que chamam os kernels
// -----------------------------------------------------------------------------
double dgemm_cuda_basic(int n, double alpha,
                        const double *A_h, const double *B_h,
                        double beta, double *C_h) {
    size_t bytes = (size_t) n * n * sizeof(double);

    double *A_d = NULL, *B_d = NULL, *C_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&A_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&B_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&C_d, bytes));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_h, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    dgemm_basic_kernel<<<grid, block>>>(n, alpha, A_d, B_d, beta, C_d);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / 1000.0; // retorna segundos
}

double dgemm_cuda_tiled(int n, double alpha,
                        const double *A_h, const double *B_h,
                        double beta, double *C_h) {
    size_t bytes = (size_t) n * n * sizeof(double);

    double *A_d = NULL, *B_d = NULL, *C_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&A_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&B_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&C_d, bytes));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_h, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    dgemm_tiled_kernel<<<grid, block>>>(n, alpha, A_d, B_d, beta, C_d);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / 1000.0; // segundos
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <n>\n", argv[0]);
        fprintf(stderr, "Ex.: %s 1024\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "n deve ser positivo\n");
        return EXIT_FAILURE;
    }

    printf("DGEMM CUDA - n = %d\n", n);

    size_t elements = (size_t) n * n;
    size_t bytes = elements * sizeof(double);

    // Aloca matrizes no host
    double *A           = (double*) malloc(bytes);
    double *B           = (double*) malloc(bytes);
    double *C_seq       = (double*) malloc(bytes);
    double *C_gpu_basic = (double*) malloc(bytes);
    double *C_gpu_tiled = (double*) malloc(bytes);

    if (!A || !B || !C_seq || !C_gpu_basic || !C_gpu_tiled) {
        fprintf(stderr, "Erro ao alocar memória no host\n");
        return EXIT_FAILURE;
    }

    srand(1234);
    randomMatrix(A, (int)elements);
    randomMatrix(B, (int)elements);

    // Inicializa C com zeros
    for (size_t i = 0; i < elements; ++i) {
        C_seq[i]       = 0.0;
        C_gpu_basic[i] = 0.0;
        C_gpu_tiled[i] = 0.0;
    }

    double alpha = 1.0;
    double beta  = 0.0;

    // -------------------------------------------------------------------------
    // 1) DGEMM sequencial (referência, versão OTIMIZADA)
    // -------------------------------------------------------------------------
    double t0 = wall_time();
    dgemm_seq(n, alpha, A, B, beta, C_seq);
    double t1 = wall_time();
    double time_seq = t1 - t0;

    printf("\n[CPU]  Tempo sequencial (otimizado): %.6f s\n", time_seq);

    // -------------------------------------------------------------------------
    // 2) DGEMM CUDA - versão básica (1 thread por elemento)
    // -------------------------------------------------------------------------
    double time_basic = dgemm_cuda_basic(n, alpha, A, B, beta, C_gpu_basic);
    printf("[GPU]  Tempo kernel básico (1 thread/elemento): %.6f s\n", time_basic);

    double diff_basic = maxRelativeDiff(C_seq, C_gpu_basic, (int)elements);
    printf("[VAL]  Diferença relativa máxima (básico) = %.3e\n", diff_basic);

    // -------------------------------------------------------------------------
    // 3) DGEMM CUDA - versão otimizada (memória compartilhada)
    // -------------------------------------------------------------------------
    double time_tiled = dgemm_cuda_tiled(n, alpha, A, B, beta, C_gpu_tiled);
    printf("[GPU]  Tempo kernel otimizado (shared)       : %.6f s\n", time_tiled);

    double diff_tiled = maxRelativeDiff(C_seq, C_gpu_tiled, (int)elements);
    printf("[VAL]  Diferença relativa máxima (shared) = %.3e\n", diff_tiled);

    // -------------------------------------------------------------------------
    // 4) Métricas de desempenho
    // -------------------------------------------------------------------------
    double speedup_basic = time_seq / time_basic;
    double speedup_tiled = time_seq / time_tiled;

    printf("\n[DESempenho] Speedup básico  (CPU->GPU) = %.3f\n", speedup_basic);
    printf("[DESempenho] Speedup shared  (CPU->GPU) = %.3f\n", speedup_tiled);

        // -------------------------------------------------------------------------
    // 5) Eficiência (didática: threads por bloco como "processadores virtuais")
    // -------------------------------------------------------------------------
    int P_basic = 16 * 16;       // 256 threads
    int P_tiled = BLOCK_SIZE * BLOCK_SIZE; // 32×32 = 1024 threads

    double efficiency_basic = speedup_basic / P_basic;
    double efficiency_tiled = speedup_tiled / P_tiled;

    printf("\n[EFICIENCIA] Eficiência básica  = %.6f\n", efficiency_basic);
    printf("[EFICIENCIA] Eficiência shared  = %.6f\n", efficiency_tiled);


    const double limit = 1e-8;
    if (diff_basic < limit && diff_tiled < limit) {
        printf("\n[OK] Implementações CUDA consideradas CORRETAS (Δ < %.1e)\n", limit);
    } else {
        printf("\n[ALERTA] Diferença numérica acima do limite (%.1e)\n", limit);
    }

    free(A);
    free(B);
    free(C_seq);
    free(C_gpu_basic);
    free(C_gpu_tiled);

    return 0;
}
