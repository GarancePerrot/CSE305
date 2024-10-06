#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

void wait_and_set(bool& flag) {
    std::this_thread::sleep_for(std::chrono::seconds(15));
    flag = true;
}

void wait_and_set2(bool& flag, std::condition_variable& cv) {
    std::this_thread::sleep_for(std::chrono::seconds(15));
    flag = true;
    cv.notify_all();
}

int main() {
    // Busy waiting
    bool flag = false;
    std::thread a(wait_and_set, std::ref(flag));
    while (!flag) {}
    std::cout << "Yes!" << std::endl;
    a.join();

    // Condition variable
    flag = false;
    std::condition_variable cv;
    std::mutex m;
    std::unique_lock<std::mutex> lk(m);
    std::thread b(wait_and_set2, std::ref(flag), std::ref(cv));
    while (!flag) {
        cv.wait(lk);
    }
    std::cout << "Yes!" << std::endl;
    b.join();
}