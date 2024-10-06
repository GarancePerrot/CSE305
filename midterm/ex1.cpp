#include <chrono>
#include <iostream>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>
#include <cmath>


typedef long double Num;
typedef std::vector<long double>::const_iterator NumIter;

//-----------------------------------------------------------------------------

Num VectorProductSeq(NumIter begin_x, NumIter begin_y, NumIter end_x) {
    Num result = 0.;
    while (begin_x != end_x) {
        result += (*begin_x) * (*begin_y);
        ++begin_x;
        ++begin_y;
    }
    return result;
}

//-----------------------------------------------------------------------------
//auxiliary for VectorProductParallel

void VectorProductThread(NumIter begin_x, NumIter begin_y, NumIter end_x, long double& result) {
    Num local_result = 0.;
    while (begin_x != end_x) {
        local_result += (*begin_x) * (*begin_y);
        ++begin_x;
        ++begin_y;
    }
    result = local_result;
}

//-----------------------------------------------------------------------------


Num VectorProductParallel(NumIter begin_x, NumIter begin_y, NumIter end_x, size_t num_threads) {
    size_t length = end_x - begin_x;
    if (length == 0) {
        return 0.;
    }

    size_t block_size = length / num_threads;
    std::vector<Num> results(num_threads, 0.);
    std::vector<std::thread> workers(num_threads - 1);
    NumIter start_block_x = begin_x;
    NumIter start_block_y = begin_y;
    for (size_t i = 0; i < num_threads - 1; ++i) {
        NumIter end_block_x = start_block_x + block_size;
        NumIter end_block_y = start_block_y + block_size;
        workers[i] = std::thread(&VectorProductThread, start_block_x,start_block_y, end_block_x, std::ref(results[i]));
        start_block_x = end_block_x;
        start_block_y = end_block_y;
        
    }
    VectorProductThread(start_block_x, start_block_y,end_x, results[num_threads - 1]);

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

int main(int argc, char** argv) {
    std::cout << "Testing correctness" << std::endl;
    const size_t NUM_TESTS = 1000;
    size_t tests_passed = 0;
    for (size_t i = 0; i < NUM_TESTS; ++i) {
        size_t len = (rand() % 100) + 10;
        std::vector<Num> test_x;
        std::vector<Num> test_y;
        for (size_t j = 0; j < len; ++j) {
            test_x.push_back(rand() % 100);
            test_y.push_back(rand() % 100);
        }
        Num correct = VectorProductSeq(test_x.begin(), test_y.begin(), test_x.end());

        size_t num_threads = 1 + (rand() % 5);
        Num result = VectorProductParallel(test_x.begin(), test_y.begin(), test_x.end(), num_threads);

        if (fabs(correct - result) < 0.01) {
            ++tests_passed;
        }
    }
    std::cout << "Passed " << tests_passed << " correctness tests out of " << NUM_TESTS << std::endl;
    std::cout << "NB: Passing all the tests does not imply the correctness of the code and does not guarantee the full grade" << std::endl;  


    std::cout << std::endl;
    std::cout << "Checking the speedup" << std::endl;

    // Generating large random vector
    const size_t N = 1 << 25;
    std::vector<Num> test_x(N, 0.);
    std::vector<Num> test_y(N, 0.);
    for (size_t i = 0; i < N; ++i) {
        test_x[i] = ((double) rand()) / RAND_MAX;
        test_y[i] = ((double) rand()) / RAND_MAX;
    }
    VectorProductSeq(test_x.begin(), test_y.begin(), test_x.end());
    auto start = std::chrono::steady_clock::now();
    Num res = VectorProductSeq(test_x.begin(), test_y.begin(), test_x.end());
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    std::cout << "Timing for the sequential version in " << elapsed << " microseconds" << std::endl;


    for (size_t num_threads = 1; num_threads < 8; ++num_threads) {
        start = std::chrono::steady_clock::now();
        Num res_par = VectorProductParallel(test_x.begin(), test_y.begin(), test_x.end(), num_threads);
        if (fabs(res - res_par) > 0.01) {
            std::cout << "Result is incorrect" << std::endl;
        }
        finish = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
        std::cout << "# threads " << num_threads << ", running time " << elapsed << " microseconds" << std::endl;
    }
}
