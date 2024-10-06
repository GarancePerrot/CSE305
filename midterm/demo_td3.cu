#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

void Add(double* x, double* y, double* res, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        res[i] = x[i] + y[i];
    }
}

//----------------------------------------------------

__global__
void AddGPUAux(double* x, double* y, double* res, size_t N, size_t chunk_size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t begin = chunk_size * index;
    size_t end = chunk_size * (index + 1);
    if (end > N) {
        end = N;
    }
    for (size_t i = begin; i < end; ++i){
        res[i] = x[i] + y[i];
    }
}

void AddGPU(double* x, double* y, double* res, size_t N) {
    const size_t BLOCKS_NUM = 48;
    const size_t THREADS_PER_BLOCK = 256;
    const size_t TOTAL_THREADS = BLOCKS_NUM  * THREADS_PER_BLOCK;

    // moving the data to device 
    double* xd;
    double* yd;
    double* resd;
    cudaMalloc(&xd, N * sizeof(double));
    cudaMalloc(&yd, N * sizeof(double));
    cudaMalloc(&resd, N * sizeof(double));
    cudaMemcpy(xd, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd, y, N * sizeof(double), cudaMemcpyHostToDevice);

    // computing on GPU
    size_t chunk_size = (N + TOTAL_THREADS + 1) / TOTAL_THREADS;
    AddGPUAux<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(xd, yd, resd, N, chunk_size);
    cudaDeviceSynchronize();

    // copying the result back
    cudaMemcpy(res, resd, N * sizeof(double), cudaMemcpyDeviceToHost);
  
    // Free memory
    cudaFree(xd);
    cudaFree(yd);
    cudaFree(resd);
}

//----------------------------------------------------

__global__
void AddGPUAux2(double* x, double* y, double* res) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    res[index] = x[index] + y[index];
}

void AddGPU2(double* x, double* y, double* res, size_t N) {
    const size_t THREADS_PER_BLOCK = 128;
    size_t BLOCKS_NUM = N / THREADS_PER_BLOCK;

    // moving the data to device 
    double* xd;
    double* yd;
    double* resd;
    cudaMalloc(&xd, N * sizeof(double));
    cudaMalloc(&yd, N * sizeof(double));
    cudaMalloc(&resd, N * sizeof(double));
    cudaMemcpy(xd, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd, y, N * sizeof(double), cudaMemcpyHostToDevice);

    // computing on GPU
    AddGPUAux2<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(xd, yd, resd);
    cudaDeviceSynchronize();

    // copying the result back
    cudaMemcpy(res, resd, N * sizeof(double), cudaMemcpyDeviceToHost);
  
    // Free memory
    cudaFree(xd);
    cudaFree(yd);
    cudaFree(resd);
}

//----------------------------------------------------

int main(int argc, char* argv[]) {
    // setting the random seed to get the same result each time
    srand(42);

    // taking as input, which algo to run
    int alg_ind = std::stoi(argv[1]);

    // Generating data
    size_t N = 1 << 27;
    double* x = (double*) malloc(N * sizeof(double));
    double* y = (double*) malloc(N * sizeof(double));
    for (size_t i = 0; i < N; ++i) {
          x[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
          y[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }
 
    // Allocating the result
    double* result = (double*) malloc(N * sizeof(double));
    auto start = std::chrono::steady_clock::now();
    switch (alg_ind) {
        case 0: 
            Add(x, y, result, N);
            break;
        case 1:
            AddGPU(x, y, result, N);
            break;
        case 2:
            AddGPU2(x, y, result, N);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time: " << elapsed << std::endl;
    
    delete[] x;
    delete[] y;
    delete[] result;
    return 0;
}
