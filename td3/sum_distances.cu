#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

__device__
double DistKer(double* p, double* q, size_t dim) {
    double result = 0;
    for (size_t i = 0; i < dim; ++i) {
        result += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return std::sqrt(result);
}

double Dist(double* p, double* q, size_t dim) {
    double result = 0;
    for (size_t i = 0; i < dim; ++i) {
        result += (p[i] - q[i]) * (p[i] - q[i]);
        //std::cout << "p : " << p[i] << ", q : " << q[i] << std::endl;
        //std::cout << "ps : "<< (p[i] - q[i]) * (p[i] - q[i]) << std::endl;
    }
    //std::cout << "res : "<< std::sqrt(result) << std::endl;
    return std::sqrt(result);
}

//------------------------------------------------

double SumDistances(double* arr, size_t dim, size_t N) {
    double result = 0.;
    for (size_t i = 0; i < N; ++i) {
        double* p = arr + i * dim;
        //std::cout << "  " << p[0] ;
        for (size_t j = i + 1; j < N; ++j) {
            result += Dist(p, arr + j * dim, dim); 
        }
    }
    return result;
}

//-------------------------------------------------



double Sum(double* arr, size_t N) {
    double result = 0.;
    for (size_t i = 0; i < N; ++i) {
        result += arr[i];
    }
    return result;
}

__global__
void SumDistancesAux(double* d_data, double* d_partial_sum, size_t dim, size_t TOTAL_THREADS) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("   %lu", idx);  // outputs 0 ... 31 
    if (idx >= TOTAL_THREADS) return;
    double res = 0.;
    double* p = d_data + idx * dim;
    //printf( "%d  ", (int)(p[0]));
    for (size_t j = idx + 1; j < TOTAL_THREADS; ++j) {
        double* q = d_data + j * dim;
        res += DistKer(p, q, dim);
        //printf("   %f", res);
    }
    d_partial_sum[idx] = res;
}



/**
 * @brief Computes the sum of pairwise distances between the points
 * @param arr - the pointer to the beginning of an array of length N * dim representing N points
 *        of dimension dim each (each point is represented by dim consecutive elements)
 * @param dim - dimension of the ambient space
 * @param N - the number of points
 */

double SumDistancesGPU(double* arr, size_t dim, size_t N) {
    const size_t TOTAL_THREADS = N*dim ; 
    const size_t THREADS_PER_BLOCK = 256;
    size_t BLOCKS_NUM = TOTAL_THREADS / THREADS_PER_BLOCK ; 

    // moving the data to device 
    double* d_data;
    double* d_partial_sum;
    cudaMalloc(&d_data, TOTAL_THREADS * sizeof(double));
    cudaMalloc(&d_partial_sum, N * sizeof(double));
    cudaMemcpy(d_data, arr, TOTAL_THREADS * sizeof(double), cudaMemcpyHostToDevice);

    // computing on GPU
    SumDistancesAux<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(d_data, d_partial_sum, dim, N);
    cudaDeviceSynchronize();

    // copying the result back
    double* cpu_arr;
    cpu_arr = (double*)malloc(N * sizeof(double));
    cudaMemcpy(cpu_arr, d_partial_sum, N * sizeof(double), cudaMemcpyDeviceToHost);
    double res = Sum(cpu_arr, N);  

    // Free memory
    cudaFree(d_data);
    cudaFree(d_partial_sum);

    return res;
}

//---------------------------------------------------

int main(int argc, char* argv[]) {
    // setting the random seed to get the same result each time
    srand(42);

    // taking as input, which algo to run
    int alg_ind = std::stoi(argv[1]);

    // Generating data
    size_t N = 3900;
    size_t dim = 5;

    double* arr = (double*) malloc(N * dim * sizeof(double));
    for (size_t i = 0; i < dim * N; ++i) {
          //arr[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
          arr[i] = i;
          //std::cout << arr[i] << "  " << std::endl;
    }

    double result = 0.;
    auto start = std::chrono::steady_clock::now();
    switch (alg_ind) {
        case 0: 
            result = SumDistances(arr, dim, N);
            break;
        case 1:
            result = SumDistancesGPU(arr, dim, N);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time: " << elapsed << std::endl;
    std::cout << "Total result: " << result << std::endl;
    
    delete[] arr;
    return 0;
}


// N = 5000, dim 5, GPU actual computation time : 
// cuda_gpu_kern_sum : around 6.5%  of the execution time 


// Varying N : 

//      N      |     sum_distances 0  (in  μs)  |  sum_distances 1  (in  μs) |   computation time from CPU to GPU 

//   2 500     |               29 661           |           51 449           |             + 73%
//   3 000     |               42 793           |           52 194           |             + 21%
//   3 500     |               57 383           |           51 885           |             + 9% 
//   3 800     |               72 966           |           76 812           |             + 5% 
//   4 000     |               75 109           |           59 177           |             - 21%
//  5 000      |              139 618           |           49 396           |             - 65%
//  10 000     |              463 397           |           81 705           |             - 82%
//  20 000     |            1 885 402           |          107 297           |             - 94%

// We see that computing on GPU shows a significant performance improvement for large datasets (N = 4 000 and above) as N increases. 
// For instance at N = 20 000, the GPU algorithm is almost 18 times faster than the CPU algorithm. 
// However for smaller datasets (tipping point at N < 4000) the CPU algorithm performs better.
