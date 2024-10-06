#include <chrono>
#include <iostream>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>

typedef long double Num;
typedef std::vector<long double>::const_iterator NumIter;

//-----------------------------------------------------------------------------

void SumMapThread(NumIter begin, NumIter end, long double& result) {
    Num local_result = 0;
    while (begin != end) {
        local_result += *begin;
        ++begin;
    }
    result = local_result;
}

/**
 * @brief the values in in [begin, end)
 * @param begin Start iterator
 * @param end End iterator
 * @param result Reference to the result variable
 */
Num SumParallel(NumIter begin, NumIter end, size_t num_threads) {
    size_t length = end - begin;
    if (length == 0) {
        return 0.;
    }
    size_t block_size = length / num_threads;
    std::vector<Num> results(num_threads, 0.);
    std::vector<std::thread> workers(num_threads - 1);
    NumIter start_block = begin;
    for (size_t i = 0; i < num_threads - 1; ++i) {
        NumIter end_block = start_block + block_size;
        workers[i] = std::thread(&SumMapThread, start_block, end_block, std::ref(results[i]));
        start_block = end_block;
    }
    SumMapThread(start_block, end, results[num_threads - 1]);

    for (size_t i = 0; i < num_threads - 1; ++i) {
        workers[i].join();
    }

    Num total_result = 0.;
    for (size_t i = 0; i < results.size(); ++i) {
        total_result += results[i];
    }
    return total_result;
}

//-----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    int N = std::stoi(argv[1]);
    int num_threads_begin = std::stoi(argv[2]);
    int num_threads_end = num_threads_begin + 1;
    if (argc > 3) {
        num_threads_end = std::stoi(argv[3]);
    }

    // Generating large random vector
    std::vector<Num> test(N, 0.);
    for (size_t i = 0; i < N; ++i) {
        test[i] = ((double) rand()) / RAND_MAX;
    }

    for (size_t num_threads = num_threads_begin; num_threads < num_threads_end; ++num_threads) {
        auto start = std::chrono::steady_clock::now();
        SumParallel(test.begin(), test.end(), num_threads);
        auto finish = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
        std::cout << "# threads " << num_threads << ", running time " << elapsed << " microseconds" << std::endl;
    }
}
