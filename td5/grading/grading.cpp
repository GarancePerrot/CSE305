#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>

#include "../gradinglib/gradinglib.hpp"
#include "td5.cpp"
#include <limits>

namespace tdgrading {

using namespace testlib;

//----------------------------------------------------------------------------

int test_safe_unbounded_queue_lock(std::ostream &out, const std::string test_name) {
    std::string fun_name = "SafeUnboundedQueueLock";

    start_test_suite(out, test_name);

    std::vector<int> res;

    // one-thread test
    SafeUnboundedQueueLock<int> Q;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 100000; ++i) {
        Q.push(i);
    }
    bool preserved = true;
    for (size_t i = 0; i < 100000; ++i) {
        int res = Q.pop_nonblocking(); 
        if (res != i) {
            print(out, "Order is not preserved with single-thread pushing");
            preserved = false;
        }
    }
    auto end = std::chrono::steady_clock::now();
    if (preserved) {
        res.push_back(1);
    } else {
        res.push_back(0);
    }

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (duration > 2000) {
        print(out, "Your queue is pretty slow. Are you using std::queue?");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    // multiple threads pushing
    SafeUnboundedQueueLock<int> push_parallel;
    std::thread even([](SafeUnboundedQueueLock<int>& q) -> void {
            for (size_t i = 0; i < 100000; ++i) {
                q.push(2 * i);
            }
        }, std::ref(push_parallel)        
    );
    std::thread odd([](SafeUnboundedQueueLock<int>& q) -> void {
            for (size_t i = 0; i < 100000; ++i) {
                q.push(2 * i + 1);
            }
        }, std::ref(push_parallel)
    );
    even.join();
    odd.join();
    int max_odd = -1;
    int max_even = -2;
    int total = 0;
    bool reversed = false;
    while (!push_parallel.is_empty()) {
        int next = push_parallel.pop_nonblocking();
        if (next % 2 == 0) {
            if (next <= max_even) {
                reversed = true;
                print(out, "Order is reversed while pushing");
            }
            max_even = std::max(max_even, next);
        } else {
            if (next <= max_odd) {
                reversed = true;
                print(out, "Order is reversed while pushing");
            }
            max_odd = std::max(max_odd, next);
        }
        ++total;
    }
    if (reversed) {
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    if (total < 200000) {
        print(out, "Elements lost while pushing");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    // multiple threads reading
    SafeUnboundedQueueLock<int> pop_parallel;
    for (size_t i = 0; i < 1000000; ++i) {
        pop_parallel.push(i);
    }

    auto popper = [](SafeUnboundedQueueLock<int>& q, std::vector<int>& dest) -> void {
        while (!q.is_empty()) {
            int a = q.pop_nonblocking();
            dest.push_back(a);
            if ((a == 999999) || (a == 999998)) {
                break;
            }
        }
    };
    std::vector<int> A;
    std::vector<int> B;
    std::thread ta(popper, std::ref(pop_parallel), std::ref(A));
    std::thread tb(popper, std::ref(pop_parallel), std::ref(B));
    ta.join();
    tb.join();
    std::vector<int> merged(A.size() + B.size());
    std::merge(A.begin(), A.end(), B.begin(), B.end(), merged.begin());
    if (merged.size() != 1000000) {
        print(out, "Wrong number of popped elements ", merged.size(), " instead of 1000000");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    bool not_sequence = false;
    for (size_t i = 0; i < merged.size(); ++i) {
        if (merged[i] != i) {
            print(out, "Popped numbers are not the same as pushed");
            not_sequence = true;
        }
    }
    if (not_sequence) {
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    
    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//---------------------------------------------------------------------

int test_safe_unbounded_queue_lock_block(std::ostream &out, const std::string test_name) {
    std::string fun_name = "SafeUnboundedQueueLock::pop_blocking";

    start_test_suite(out, test_name);

    std::vector<int> res;

    // reader-writer
    SafeUnboundedQueueLock<int> read_write;
    std::thread writer([](SafeUnboundedQueueLock<int>& q) -> void {
            for (size_t i = 0; i < 1000; ++i) {
                q.push(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }, std::ref(read_write)
    );
    std::thread reader([](SafeUnboundedQueueLock<int>& q) -> void {
            for (size_t i = 0; i < 1000; ++i) {
                q.pop_blocking();
            }
        }, std::ref(read_write)
    );
    writer.join();
    reader.join();

    if (!read_write.is_empty()) {
        print(out, "Queue not empty after 1000 reads and 1000 writes");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    
    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}


//------------------------------------------------------------------------

int test_divide_once_even(std::ostream& out, const std::string test_name) {
    std::string fun_name = "DivideOnceEven";
    
    start_test_suite(out, test_name);    
    std::vector<int> res;

    std::mutex m;
    std::condition_variable iseven;
    int n = 5;
    std::thread t(&DivideOnceEven, std::ref(iseven), std::ref(m), std::ref(n));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    iseven.notify_all();
    if (n == 5) {
        res.push_back(1);
    } else {
        res.push_back(0);
    }

    n = 6;
    iseven.notify_all();
    t.join();
    if (n == 3) {
        res.push_back(1);
    } else {
        res.push_back(0);
    }

    std::thread t1(&DivideOnceEven, std::ref(iseven), std::ref(m), std::ref(n));
    std::thread t2(&DivideOnceEven, std::ref(iseven), std::ref(m), std::ref(n));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    n = 20;
    iseven.notify_all();
    t1.join();
    t2.join();
    if (n == 5) {
        res.push_back(1);
    } else {
        res.push_back(0);
    }

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//----------------------------------------------------------------------------

int test_safe_unbounded_queue_cv(std::ostream &out, const std::string test_name) {
    std::string fun_name = "SafeUnboundedQueueCV";

    start_test_suite(out, test_name);

    std::vector<int> res;

    // one-thread test
    SafeUnboundedQueueCV<int> Q;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 100000; ++i) {
        Q.push(i);
    }
    bool preserved = true;
    for (size_t i = 0; i < 100000; ++i) {
        int res = Q.pop();    
        if (res != i) {
            print(out, "Order is not preserved with single-thread pushing");
            preserved = false;
        }
    }
    auto end = std::chrono::steady_clock::now();
    if (preserved) {
        res.push_back(1);
    } else {
        res.push_back(0);
    }

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (duration > 2000) {
        print(out, "Your queue is pretty slow. Are you using std::queue?");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    // multiple threads pushing
    SafeUnboundedQueueCV<int> push_parallel;
    std::thread even([](SafeUnboundedQueueCV<int>& q) -> void {
            for (size_t i = 0; i < 100000; ++i) {
                q.push(2 * i);
            }
        }, std::ref(push_parallel)        
    );
    std::thread odd([](SafeUnboundedQueueCV<int>& q) -> void {
            for (size_t i = 0; i < 100000; ++i) {
                q.push(2 * i + 1);
            }
        }, std::ref(push_parallel)
    );
    even.join();
    odd.join();
    int max_odd = -1;
    int max_even = -2;
    int total = 0;
    bool reversed = false;
    while (!push_parallel.is_empty()) {
        int next = push_parallel.pop();
        if (next % 2 == 0) {
            if (next <= max_even) {
                reversed = true;
                print(out, "Order is reversed while pushing");
            }
            max_even = std::max(max_even, next);
        } else {
            if (next <= max_odd) {
                reversed = true;
                print(out, "Order is reversed while pushing");
            }
            max_odd = std::max(max_odd, next);
        }
        ++total;
    }
    if (reversed) {
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    if (total < 200000) {
        print(out, "Elements lost while pushing");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    // multiple threads reading
    SafeUnboundedQueueCV<int> pop_parallel;
    for (size_t i = 0; i < 1000000; ++i) {
        pop_parallel.push(i);
    }

    auto popper = [](SafeUnboundedQueueCV<int>& q, std::vector<int>& dest) -> void {
        while (!q.is_empty()) {
            int a = q.pop();
            dest.push_back(a);
            if ((a == 999999) || (a == 999998)) {
                break;
            }
        }
    };
    std::vector<int> A;
    std::vector<int> B;
    std::thread ta(popper, std::ref(pop_parallel), std::ref(A));
    std::thread tb(popper, std::ref(pop_parallel), std::ref(B));
    ta.join();
    tb.join();
    std::vector<int> merged(A.size() + B.size());
    std::merge(A.begin(), A.end(), B.begin(), B.end(), merged.begin());
    if (merged.size() != 1000000) {
        print(out, "Wrong number of popped elements ", merged.size(), " instead of 1000000");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    bool not_sequence = false;
    for (size_t i = 0; i < merged.size(); ++i) {
        if (merged[i] != i) {
            print(out, "Popped numbers are not the same as pushed");
            not_sequence = true;
        }
    }
    if (not_sequence) {
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    // reader-writer
    SafeUnboundedQueueCV<int> read_write;
    std::thread writer([](SafeUnboundedQueueCV<int>& q) -> void {
            for (size_t i = 0; i < 1000; ++i) {
                q.push(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }, std::ref(read_write)
    );
    std::thread reader([](SafeUnboundedQueueCV<int>& q) -> void {
            for (size_t i = 0; i < 1000; ++i) {
                q.pop();
            }
        }, std::ref(read_write)
    );
    writer.join();
    reader.join();

    if (!read_write.is_empty()) {
        print(out, "Queue not empty after 1000 reads and 1000 writes");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 4,
  "names" : ["td5.cpp::SafeUnboundedQueueLock", "td5.cpp::SafeUnboundedQueueLock::pop_blocking", "td5.cpp::DivideOnceEven", "td5.cpp::SafeUnboundedQueueCV"],
  "points" : [5, 5, 5, 5]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 4;
    std::string const test_names[total_test_cases] = {"SafeUnboundedQueueLock", "SafeUnboundedQueueLock::pop_blocking", "DivideOnceEven", "SafeUnboundedQueueCV"};
    int const points[total_test_cases] = {5, 5, 5, 5};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
        test_safe_unbounded_queue_lock, test_safe_unbounded_queue_lock_block, test_divide_once_even, test_safe_unbounded_queue_cv
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
