#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>


void LocalMax(double* x, double* res, int N, int width) {
    for (int i = 0; i < N; ++i) {
        res[i] = x[i];
        int from = std::max(0, i - width);
        int to = std::min(N, i + width);
        for (int j = from; j < to; ++j) {
            res[i] = std::max(res[i], x[j]);
        }
    }
}

//----------------------------------------------------

//auxiliary for LocalMaxGPU
__global__
void LocalMaxGPUAux(double* x, double* res, size_t N, int width) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }
    double local_res = x[idx];
    int from = max(0, (int)(idx - width));
    int to = min((int)(N), (int)(idx + width));
    for (int j = from; j < to; ++j) {
        local_res = max(local_res, x[j]);
    }
    res[idx] = local_res;
}


//DOES NOT PASS THE TEST BUT I CANNOT FIND THE MISTAKE...
//----------------------------------------------------

void LocalMaxGPU(double* x, double* res, int N, int width) {
// Compute, for every i from 0 to N - 1, the largest value in the subarray of x with indices [i - width, i + width) 
// (capped at the boundaries of x) and write it to res[i]. 
// You are asked to use GPU and do the computation for res[i] in a separate thread.
    
    
    // const size_t TOTAL_THREADS = N;  there are N threads
    const size_t THREADS_PER_BLOCK = 256;
    const size_t BLOCKS_NUM = N / THREADS_PER_BLOCK;

    // moving the data to device 
    double* xd;
    double* resd;
    cudaMalloc(&xd, N * sizeof(double));
    cudaMalloc(&resd, N * sizeof(double));
    cudaMemcpy(xd, x, N * sizeof(double), cudaMemcpyHostToDevice);

    // computing on GPU
    LocalMaxGPUAux<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(xd, resd, N, width);
    cudaDeviceSynchronize();

    // copying the result back
    cudaMemcpy(res, resd, N * sizeof(double), cudaMemcpyDeviceToHost);
  
    // Free memory
    cudaFree(xd);
    cudaFree(resd);

}

//----------------------------------------------------

int main(int argc, char* argv[]) {
    // setting the random seed to get the same result each time
    srand(42);

    // Generating data
    int N = (1 << 22) + 1893;
    double* x = (double*) malloc(N * sizeof(double));
    for (int i = 0; i < N; ++i) {
          x[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }
    int width = 100;
 
    // Allocating the result
    double* result_seq = (double*) malloc(N * sizeof(double));
    double* result_cuda = (double*) malloc(N * sizeof(double));

    auto start = std::chrono::steady_clock::now();
    LocalMax(x, result_seq, N, width);
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time for sequential algorithm: " << elapsed << std::endl;

    // "warm-up" kernel
    // LocalMaxGPU(x, result_cuda, N, width);

    start = std::chrono::steady_clock::now();
    LocalMaxGPU(x, result_cuda, N, width);
    finish = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time for GPU algorithm: " << elapsed << std::endl;

    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (result_seq[i] != result_cuda[i]) {
            correct = false;
	    std::cout << i << " at " << result_seq[i] << " vs " << result_cuda[i] << std::endl;
        }
    }

    if (correct) {
        std::cout << "The result for sequential and GPU versions coincide" << std::endl;
    } else {
        std::cout << "There is a mismatch in the results" << std::endl;
    }
 
    free(x);
    free(result_seq);
    free(result_cuda);
    return 0;
}
