#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

//------------------------------------------------

double Sum(double* arr, size_t N) {
    double result = 0.;
    for (size_t i = 0; i < N; ++i) {
        result += arr[i];
    }
    return result;
}

//-------------------------------------------------

__global__
void SumGPUAux(double* d_data, double* d_partial_sum, size_t N, size_t chunk_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t begin = chunk_size * idx;
    size_t end = chunk_size * (idx + 1);
    if (end > N) {
        end = N;
    }
    double ps = 0.0;
    for (size_t i = begin; i < end; ++i) {
        ps += d_data[i];
    }
    d_partial_sum[idx] = ps;

    
}

/**
 * @brief Computes the sum of the array
 * @param arr - the pointer to the beginning of an array
 * @param N - the length of the array
//  */
double SumGPU(double* arr, size_t N) {

    const size_t BLOCKS_NUM = 48;
    const size_t THREADS_PER_BLOCK = 512;
    const size_t TOTAL_THREADS = BLOCKS_NUM  * THREADS_PER_BLOCK;
    
    // moving the data to device 
    double* d_data;
    double* d_partial_sum;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMalloc(&d_partial_sum, TOTAL_THREADS* sizeof(double));
    cudaMemcpy(d_data, arr, N * sizeof(double), cudaMemcpyHostToDevice);


    // computing on GPU
    size_t chunk_size = (N + TOTAL_THREADS + 1) / TOTAL_THREADS;
    SumGPUAux<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(d_data, d_partial_sum, N, chunk_size);
    cudaDeviceSynchronize();


    // this works but is more costly : 

    // // adding up the partial sums on CPU (retrieving the partial sums from memory)
    // double res = 0.0;
    // for (int i = 0; i < TOTAL_THREADS; i++) {
    //     double ps;
    //     cudaMemcpy(&ps, d_partial_sum + i, sizeof(double), cudaMemcpyDeviceToHost);
    //     res += ps;
    // }

    //retrieving the final result
    double* cpu_arr;
    cpu_arr = (double*)malloc(TOTAL_THREADS * sizeof(double));
    cudaMemcpy(cpu_arr, d_partial_sum, TOTAL_THREADS * sizeof(double), cudaMemcpyDeviceToHost);

    // computing the sum
    double res = 0.;
    for (size_t i = 0; i < TOTAL_THREADS; ++i){
        res += cpu_arr[i];
    }
    // free device memory
    cudaFree(d_data);
    cudaFree(d_partial_sum);
    free(cpu_arr);

    return res;
}

//---------------------------------------------------

int main(int argc, char* argv[]) {
    // setting the random seed to get the same result each time
    srand(42);

    // taking as input, which algo to run
    int alg_ind = std::stoi(argv[1]);

    // Generating data
    size_t N = 1 << 26;
    double* arr = (double*) malloc(N * sizeof(double));
    for (size_t i = 0; i < N; ++i) {
          arr[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }
 
    double result = 0.;
    auto start = std::chrono::steady_clock::now();
    switch (alg_ind) {
        case 0: 
            result = Sum(arr, N);
            break;
        case 1:
            result = SumGPU(arr, N);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time: " << elapsed << std::endl;
    std::cout << "Total result: " << result << std::endl;
    
    delete[] arr;
    return 0;
}


///usr/local/cuda-12.2.0/bin/nsys profile --stats=true -o profile ./sum 1


//(48) Multiprocessors, (128) CUDA Cores/MP:     6144 CUDA Cores
//Maximum number of threads per multiprocessor:  1536
//Maximum number of threads per block:           1024
//Max dimension size of a thread block (x,y,z): (1024, 1024, 64)


// N = 1 << 26, GPU actual computation time : 
// cuda_gpu_kern_sum : around 2.4%  of the execution time  

//     sum 0  (in  μs)  |  sum 1 (in  μs) |   computation time from CPU to GPU 
//     103 101          |    90 154       |   - 13% 
