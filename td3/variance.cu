#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

//------------------------------------------------

double Variance(double* arr, size_t N) {
    double sum = 0.;
    double sum_squares = 0.;
    for (size_t i = 0; i < N; ++i) {
        sum += arr[i];
        sum_squares += arr[i] * arr[i];
    }
    return sum * sum / (1. * N * N)  - sum_squares / (1. * N);
}

//-------------------------------------------------

__global__
void VarianceGPUAux(double* d_data, double* d_simple_sum, double* d_sqr_sum, size_t N, size_t chunk_size){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t begin = chunk_size * idx;
    size_t end = chunk_size * (idx + 1);
    if (end > N) {
        end = N;
    }
    double simple_sum = 0.;
    double sqr_sum = 0.;
    for (size_t i = begin; i < end; ++i){
        simple_sum += d_data[i];
        sqr_sum += d_data[i] * d_data[i];
    }
    d_simple_sum[idx] = simple_sum;
    d_sqr_sum[idx] = sqr_sum;

}

/**
 * @brief Computes the variance of the array
 * @param arr - the pointer to the beginning of an array
 * @param N - the length of the array
 */
double VarianceGPU(double* arr, size_t N) {
    
    const size_t BLOCKS_NUM = 48;
    const size_t THREADS_PER_BLOCK = 512;
    const size_t TOTAL_THREADS = BLOCKS_NUM  * THREADS_PER_BLOCK;
    
    // moving the data to device
    double* d_data;
    double* d_simple_sum;
    double* d_sqr_sum; 
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMalloc(&d_simple_sum, TOTAL_THREADS* sizeof(double));
    cudaMalloc(&d_sqr_sum, TOTAL_THREADS* sizeof(double));
    cudaMemcpy(d_data, arr, N * sizeof(double), cudaMemcpyHostToDevice);


    // computing on GPU
    size_t chunk_size = (N + TOTAL_THREADS + 1) / TOTAL_THREADS;
    VarianceGPUAux<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(d_data, d_simple_sum, d_sqr_sum, N, chunk_size);
    cudaDeviceSynchronize();


    //retrieving the final result
    double* cpu_arr;
    double* cpu_arr_sqr;
    cpu_arr = (double*)malloc(TOTAL_THREADS * sizeof(double));
    cpu_arr_sqr = (double*)malloc(TOTAL_THREADS * sizeof(double));
    cudaMemcpy(cpu_arr, d_simple_sum, TOTAL_THREADS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_arr_sqr, d_sqr_sum, TOTAL_THREADS * sizeof(double), cudaMemcpyDeviceToHost);

    // computing the sum
    double simple_sum = 0.0;
    double sqr_sum = 0.0;
    for (size_t i = 0; i < TOTAL_THREADS; ++i){
        simple_sum += cpu_arr[i];
        sqr_sum += cpu_arr_sqr[i];
    }
    double new_N = std::stod(std::to_string(N)); // to convert size_t into double (I looked on the internet)
    double var = ( simple_sum * simple_sum) / (new_N * new_N) - sqr_sum / new_N ; 


    // free device memory
    cudaFree(d_data);
    cudaFree(d_simple_sum);
    cudaFree(d_sqr_sum);
    free(cpu_arr);
    free(cpu_arr_sqr);

    return var;
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
            result = Variance(arr, N);
            break;
        case 1:
            result = VarianceGPU(arr, N);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time: " << elapsed << std::endl;
    std::cout << "Total result: " << result << std::endl;
 
    delete[] arr;   
    return 0;
}



// N = 1 << 26, GPU actual computation time : 
// cuda_gpu_kern_sum : around 2.5%  of the execution time  

//     variance 0  (in  μs)  |  variance 1 (in  μs) |   computation time from CPU to GPU 
//       103 444             |     94 511           |   - 8%

