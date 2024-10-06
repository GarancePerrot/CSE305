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
#include "td8.cpp"
#include <limits>

namespace tdgrading {

using namespace testlib;
using namespace std;

//-----------------------------------------------------------------------------

const unsigned int NUM_ELEMS_MULTISET = 1001;

void adder(MultiSetList& S) {
    std::string a = "a";
    for (size_t i = 0; i < NUM_ELEMS_MULTISET; ++i) {
        S.add(a);
        a = a + "a";
    }
}

void remover(MultiSetList& S) {
    std::string a = "a";
    for (size_t i = 0; i < NUM_ELEMS_MULTISET; ++i) {
        S.remove(a);
        a = a + "a";
    }
}

int test_multiset(std::ostream &out, const std::string test_name) {
    std::string fun_name = "MultiSetList";

    start_test_suite(out, test_name);

    std::vector<int> res;

    MultiSetList S;
    std::thread a1(&adder, std::ref(S));
    std::thread a2(&adder, std::ref(S));
    std::thread a3(&adder, std::ref(S));
    std::thread a4(&adder, std::ref(S));
    a1.join();
    a2.join();
    a3.join();
    a4.join();
    std::thread r1(&remover, std::ref(S));
    std::thread r2(&remover, std::ref(S));
    r1.join();
    r2.join();
    std::string a = "a";
    for (size_t i = 0; i < NUM_ELEMS_MULTISET; ++i) {
        if (S.contains(a) != 2) {
            print(out, "Wrong multiplicity with add/remove: expected 2, got ", S.contains(a));
            res.push_back(0);
        } else {
            res.push_back(1);
        }
        a = a + "a";
    }

    std::thread r3(&remover, std::ref(S));
    std::thread r4(&remover, std::ref(S));
    std::thread r5(&remover, std::ref(S));
    r3.join();
    r4.join();
    r5.join();
    a = "a";
    for (size_t i = 0; i < NUM_ELEMS_MULTISET; ++i) {
        if (S.contains(a) != 0) {
            print(out, "Wrong multiplicity with add/remove: expected 0, got ", S.contains(a));
            res.push_back(0);
        } else {
            res.push_back(1);
        }
        a = a + "a";
    }

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

const size_t NUM_ELEMS_MONOTONIC = 1000;

void inserter(MonotonicLockFreeSet& S, std::string starter) {
    for (size_t i = 0; i < NUM_ELEMS_MONOTONIC; ++i) {
        S.add(starter);
        starter = starter + "aaa";
    }
}

int test_monotonic(std::ostream &out, const std::string test_name) {
    std::string fun_name = "MonotonicLockFreeSet";

    start_test_suite(out, test_name);

    std::vector<int> res;

    MonotonicLockFreeSet S;
    std::thread t1(&inserter, std::ref(S), "aaa");
    std::thread t2(&inserter, std::ref(S), "a");
    std::thread t3(&inserter, std::ref(S), "aa");
    std::thread t4(&inserter, std::ref(S), "aaa");
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    std::string target = "a";
    for (size_t i = 0; i < 3 * NUM_ELEMS_MONOTONIC; ++i) {
        if (!S.contains(target)) {
            print(out, "Elements are lost");
            res.push_back(0);
        } else {
            res.push_back(1);
        }
        if (S.add(target)) {
            print(out, "Adding existing element does not always return false");
            res.push_back(0);
        } else {
            res.push_back(1);
        }
        target = target + "a";
    }

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

void setter(AtomicMarkable<int>& at, int* ref, bool mark, int& success) {
    std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 5));
    int* exp_ref = ref;
    bool exp_mark = mark;
    while ((exp_ref == ref) and (exp_mark == mark) and !at.compare_and_set(exp_ref, exp_mark, NULL, true)) {}
    if ((exp_ref == ref) and (exp_mark == mark)) {
        success = 1;
    }
}

int test_atomicmarkable(std::ostream &out, const std::string test_name) {
    std::string fun_name = "AtomicMarkable";

    start_test_suite(out, test_name);

    std::vector<int> res;
 

    int a = 5;
    int b = 6;
    AtomicMarkable<int> at(nullptr, false);
    int* n = nullptr;
    bool exp = false;
    while (!at.compare_and_set(n, exp, &a, true)) {
        n = NULL;
        exp = false;
    }
    if ((*at.get(exp) != 5) or !exp) {
        print(out, "Wrong assignment");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    if (at.attempt_mark(&b, false)) {
        print(out, "Marking is successful when it should not be");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    if (!at.attempt_mark(&a, false)) {
        print(out, "Marking is not successful when it should be");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    AtomicMarkable<int> am(&a, false);
    size_t num_threads = 10;
    std::vector<int> success(num_threads, 0);
    std::vector<std::thread > workers(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers[i] = std::thread(&setter, std::ref(am), &a, false, std::ref(success[i]));
    }
    std::for_each(workers.begin(), workers.end(), [](std::thread& t) {t.join();});
    int sum = std::accumulate(success.begin(), success.end(), 0);
    if (sum == 0) {
        print(out, "Nobody could update the object via busy waiting");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    if (sum > 1) {
        print(out, "It is possible that two threads will update a single object");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}


int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 3,
  "names" : ["td8.cpp::MultiSetList", "td8.cpp::MonotonicLockFreeSet", "td8.cpp::AtomicMarkable"],
  "points" : [10, 10, 0]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 3;
    std::string const test_names[total_test_cases] = {"MultiSetList", "MonotonicLockFreeSet", "AtomicMarkable"};
    int const points[total_test_cases] = {10, 10, 0};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      test_multiset, test_monotonic, test_atomicmarkable
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
