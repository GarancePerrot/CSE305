#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <chrono>
#include <future>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>

#include "../gradinglib/gradinglib.hpp"
#include "td6.cpp"
#include <limits>

namespace tdgrading {

using namespace testlib;
using namespace std;

//-----------------------------------------------------------------------------

// Auxiliary functions for testing the Account class
void withdraw(Account& a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a.withdraw(1);
    }
}

void add(Account& a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a.add(1);
    }
}

int test_account(std::ostream &out, const std::string test_name) {
    std::string fun_name = "Account";

    start_test_suite(out, test_name);
    std::vector<int> res;
  
    Account A(2000000);
    std::thread t1(&withdraw, std::ref(A), 1000000);
    std::thread t2(&withdraw, std::ref(A), 1000000);
    t1.join();
    t2.join();
    if (A.get_amount() != 0 ) {
        print(out, "Parallel withdrawals from an account interleave, consider using lock\n");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    A.add(100);
    std::thread t7(&withdraw, std::ref(A), 101);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (A.get_amount() != 0) {
        print(out, "Withdrawal not blocked or blocked too early");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    A.add(42);
    t7.join();
    if (A.get_amount() != 41) {
        print(out, "Withdrawal not blocked or blocked too early");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    Account B(0);
    std::thread t3(&add, std::ref(B), 1000000);
    std::thread t4(&add, std::ref(B), 1000000);
    t3.join();
    t4.join();
    if (B.get_amount() != 2000000 ) {
        print(out, "Parallel additions from an account interleave, consider using lock\n");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    
    B.add(2999999);
    std::thread t9(&add, std::ref(B), 3);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (B.get_amount() != Account::DEFAULT_MAX_AMOUNT) {
        print(out, "Additions are not blocked properly");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    B.withdraw(5000000);
    t9.join();
    if (B.get_amount() != 2) {
        print(out, "Additions are not blocked properly");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
 
    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

int test_change_max(std::ostream &out, const std::string test_name) {
    std::string fun_name = "Account::change_max";

    start_test_suite(out, test_name);
    std::vector<int> res;
  
    Account A(0);
    bool a = A.change_max(100);
    if (a) {
        res.push_back(1);
    } else {
        print(out, "Wrong return value");
        res.push_back(0);
    }

    std::thread t(&add, std::ref(A), 200);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (A.get_amount() != 100) {
        print(out, "Something wrong with blocking / changing the max");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    bool b = A.change_max(200);
    if (b) {
        res.push_back(1);
    } else {
        print(out, "Wrong return value");
        res.push_back(0);
    }
    t.join();
    if (A.get_amount() != 200) {
         print(out, "Something wrong with blocking / changing the max");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    bool c = A.change_max(100);
    if (!c) {
        res.push_back(1);
    } else {
        print(out, "Wrong return value");
        res.push_back(0);
    }
 
    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}


//-----------------------------------------------------------------------------

void inserted(CoarseBST& T, int& success) {
    for (size_t i = 0; i < 1000; ++i) {
        if (T.add(i)) {
            ++success;
        }
    }
}

int test_bst(std::ostream &out, const std::string test_name) {
    std::string fun_name = "CoarseBST";

    start_test_suite(out, test_name);
    std::vector<int> res;
  
    CoarseBST T;
    int success_a = 0;
    int success_b = 0;
    std::thread t1(&inserted, std::ref(T), std::ref(success_a));
    std::thread t2(&inserted, std::ref(T), std::ref(success_b));
    t1.join();
    t2.join();
    if (success_a + success_b != 1000) {
        print(out, "Lost or duplicated insertions");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    for (size_t i = 0; i < 1000; ++i) {
        if (!T.contains(i)) {
            print(out, "Lost element");
            res.push_back(0);
        } else {
            res.push_back(1);
        }
    }
    for (size_t i = 1000; i < 1100; ++i) {
        if (T.contains(i)) {
            print(out, "Extra element");
            res.push_back(0);
        } else {
            res.push_back(1);
        }
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
  "total" : 3,
  "names" : [
      "td6.cpp::Account_test",
      "td6.cpp::Account_change_max",
      "td6.cpp::CoarseBST"
  ],
  "points" : [7, 7, 6]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 3;
    std::string const test_names[total_test_cases] = {
        "Account_test", "Account_change_max", "CoarseBST_test"
    };
    int const points[total_test_cases] = {7, 7, 6};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
        test_account, test_change_max, test_bst
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
