#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>

const size_t N = 1000000;

void IncrementerNaive(size_t& c) {
    for (size_t i = 0; i < N; ++i) {
        ++c;
    }
}

void IncrementNaively() {
    size_t c = 0;
    std::thread A(&IncrementerNaive, std::ref(c));
    std::thread B(&IncrementerNaive, std::ref(c));
    A.join();
    B.join();
    std::cout << "Total value " << c << std::endl;
}

//------------------------------------------------

void IncrementerAtomic(std::atomic<size_t>& c) {
    for (size_t i = 0; i < N; ++i) {
        c.fetch_add(1);
    }
}

void IncrementAtomically() {
    std::atomic<size_t> c(0);
    std::thread A(&IncrementerAtomic, std::ref(c));
    std::thread B(&IncrementerAtomic, std::ref(c));
    A.join();
    B.join();
    std::cout << "Total value " << c << std::endl;
}

//------------------------------------------------

void IncrementerMutex(size_t& c, std::mutex& m) {
    for (size_t i = 0; i < N; ++i) {
        m.lock();
        ++c;
        m.unlock();
    }
}

void IncrementWithMutex() {
    size_t c = 0;
    std::mutex m;
    std::thread A(&IncrementerMutex, std::ref(c), std::ref(m));
    std::thread B(&IncrementerMutex, std::ref(c), std::ref(m));
    A.join();
    B.join();
    std::cout << "Total value " << c << std::endl;
}

//------------------------------------------------

int main(int argc, char* argv[]) {
    auto start = std::chrono::steady_clock::now();
    IncrementNaively();
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    std::cout << "Elapsed time " << elapsed << " microseconds" << std::endl;
}
