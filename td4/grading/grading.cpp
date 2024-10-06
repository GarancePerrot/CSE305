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
#include "td4.cpp"
#include <limits>

namespace tdgrading {

using namespace testlib;
using namespace std;

//-----------------------------------------------------------------------------

int test_find_parallel(std::ostream &out, const std::string test_name) {
    std::string fun_name = "FindParallel";

    start_test_suite(out, test_name);

    std::vector<int> res;
   
    for (size_t i = 0; i < 2000; ++i) {
        size_t len = (rand() % 30000) + 10;
        if (i < 2) {
            len = i + 1;
        }
        int* test = new int[len];
        for (size_t j = 0; j < len; ++j) {
            test[j] = rand() % (len / (i % 2 == 0 ? 9 : 200) + 1);
        }
        int count = std::count(test, test + len, test[0]);
        bool correct = (count >= 10);
        bool student_result = FindParallel<int>(test, len, test[0], 10, (rand() % 5) + 1);
        delete[] test;
        res.push_back(test_eq(
           out, fun_name, student_result, correct
        ));
    }

    size_t N = 10000000;
    int* long_vec = new int[N];
    for (size_t i = 0; i < N; ++i) {
        long_vec[i] = rand() % 100;
    }
    for (size_t i = 0; i < 5; ++i) {
        long_vec[i] = 101;
        long_vec[5000000 + i] = 101;
    }
    auto start = std::chrono::steady_clock::now();
    FindParallel<int>(long_vec, N, 101, 10, 2);
    auto end = std::chrono::steady_clock::now();
    auto rt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (rt > 10) {
        print(out, "It seems that you do not terminate after finding the necessary number of occurences, too slow");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    for (size_t i = 0; i < N; ++i) {
        long_vec[i] = rand() % 100;
    }
    for (size_t i = 0; i < 5; ++i) {
        long_vec[i] = 101;
        long_vec[3333333 + i] = 101;
        long_vec[6666666 + i] = 101;
    }
    start = std::chrono::steady_clock::now();
    FindParallel<int>(long_vec, N, 101, 10, 3);
    end = std::chrono::steady_clock::now();
    rt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (rt > 10) {
        print(out, "It seems that you do not terminate after finding the necessary number of occurences, too slow");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    delete[] long_vec;

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

// Auxiliary functions for testing the Account class
void withdraw(Account& a) {
    for (size_t i = 0; i < 1000000; ++i) {
        a.withdraw(1);
    }
}

void add(Account& a) {
    for (size_t i = 0; i < 1000000; ++i) {
        a.add(1);
    }
}

void transfer(Account& a, Account& b) {
    for (size_t i = 0; i < 1000; ++i) {
        Account::transfer(1, a, b);
    }
}

void generate_accounts1(std::vector<unsigned int>& result) {
    for (size_t i = 0; i < 100000; ++i) {
        Account a;
        result.push_back(a.get_id());
    }
}

void generate_accounts2(std::vector<unsigned int>& result) {
    for (size_t i = 0; i < 100000; ++i) {
        Account a(0);
        result.push_back(a.get_id());
    }
}

int check_deadlock() {
    Account B(1000);
    Account C(1000);
    std::thread t1(&transfer, std::ref(B), std::ref(C));
    std::thread t2(&transfer, std::ref(C), std::ref(B));
    t1.join();
    t2.join();
    return 1;
}

int test_account(std::ostream &out, const std::string test_name) {
    std::string fun_name = "Account";

    start_test_suite(out, test_name);
    std::vector<int> res;
  
    Account A(2000000);
    std::thread t1(&withdraw, std::ref(A));
    std::thread t2(&withdraw, std::ref(A));
    t1.join();
    t2.join();
    if (A.get_amount() != 0 ) {
        print(out, "Parallel withdrawals from an account interleave, consider using lock\n");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    A.add(1000000);
    std::thread t7(&withdraw, std::ref(A));
    std::thread t8(&withdraw, std::ref(A));
    t7.join();
    t8.join();
    if (A.get_amount() != 0) {
        print(out, "Parallel withdrawals may lead to negative funds\n");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    Account B(0);
    std::thread t3(&add, std::ref(B));
    std::thread t4(&add, std::ref(B));
    t3.join();
    t4.join();
    if (B.get_amount() != 2000000 ) {
        print(out, "Parallel additions from an account interleave, consider using lock\n");
        res.push_back(0);
    } else {
        res.push_back(1);
    }
    
    std::future<int> f = std::async(std::launch::async, &check_deadlock);
    std::future_status status = f.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        print(out, "Deadlocks in parallel transfer\n");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    std::vector<unsigned int> ids1;
    std::vector<unsigned int> ids2;
    std::thread t5(&generate_accounts1, std::ref(ids1));
    std::thread t6(&generate_accounts2, std::ref(ids2));
    t5.join();
    t6.join();
    ids1.insert(ids1.end(), ids2.begin(), ids2.end());
    std::sort(ids1.begin(), ids1.end());
    for (auto it = ids1.begin(); it != ids1.end() - 1; ++it) {
        if (*it == *(it + 1)) {
            print(out, "Parallel creation of accounts yields equal ids");
            res.push_back(0);
        } else {
            res.push_back(1);
        }
    }

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

void aux_massiv_withdraw(std::vector<Account*> accounts, int count) {
    for (size_t i = 0; i < count; ++i) {
        Account::massiv_withdraw(accounts, 1);
    }
}

int test_massiv_withdraw(std::ostream &out, const std::string test_name) {
    std::string fun_name = "massiv_withdraw";

    start_test_suite(out, test_name);
    std::vector<int> res;
  
    Account A(1000);
    Account B(800);
    Account C(799);
    std::vector<Account*> abc = {&A, &B, &C};
    std::vector<Account*> acb = {&A, &C, &B};
    std::vector<Account*> bac = {&B, &A, &C};
    std::vector<Account*> bca = {&B, &C, &A};
    std::vector<Account*> cab = {&C, &A, &B};
    std::vector<Account*> cba = {&C, &B, &A};

    std::thread t1(&aux_massiv_withdraw, abc, 100);
    std::thread t2(&aux_massiv_withdraw, acb, 100);
    std::thread t3(&aux_massiv_withdraw, bac, 100);
    std::thread t4(&aux_massiv_withdraw, bca, 100);
    std::thread t5(&aux_massiv_withdraw, cab, 100);
    std::thread t6(&aux_massiv_withdraw, cba, 100);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();

    res.push_back(test_eq(out, "account A", A.get_amount(), 400));
    res.push_back(test_eq(out, "account B", B.get_amount(), 200));
    res.push_back(test_eq(out, "account C", C.get_amount(), 199));

    Account::massiv_withdraw(abc, 199);
    bool overdraft = Account::massiv_withdraw(cab, 1);
    res.push_back(test_eq(out, "No removing", overdraft, false));
    res.push_back(test_eq(out, "account A", A.get_amount(), 201));
    res.push_back(test_eq(out, "account B", B.get_amount(), 1));
    res.push_back(test_eq(out, "account C", C.get_amount(), 0));

    return end_test_suite(out, test_name,
                          accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

template <typename T>
void SlowInc(int& target, T& l) {
    l.lock();
    int tmp = target + 1;
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    target = tmp;
    l.unlock();
}

void TryUnlock(MyLock& l, bool& result) {
    l.unlock();
    result = l.try_lock();
}

void TryLock(MyLock& l, bool& result) {
    for (size_t i = 0; i < 1000; ++i) {
        if (l.try_lock()) {
            result = true;
            return;
        }
    }
}

int test_my_lock(std::ostream &out, const std::string test_name) {
    std::string fun_name = "MyLock";

    start_test_suite(out, test_name);

    std::vector<int> res;
   
    int x = 1;
    MyLock l;
    std::thread C(&SlowInc<MyLock>, std::ref(x), std::ref(l));
    std::thread D(&SlowInc<MyLock>, std::ref(x), std::ref(l));
    C.join();
    D.join();
    if (x != 3) {
        print(out, "Looks like the lock does not block");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    MyLock looo;
    looo.lock();
    bool unlocked = false;
    std::thread A(&TryUnlock, std::ref(looo), std::ref(unlocked));
    A.join();
    if (unlocked) {
        print(out, "A non-owner thread can release a lock, this should not be so");
        res.push_back(0);
    } else {
        res.push_back(1);
    }

    MyLock a;
    a.lock();
    bool got_lock = false;
    std::thread B(&TryLock, std::ref(a), std::ref(got_lock));
    B.join();
    if (got_lock) {
        print(out, "Lock can be got by two distinct threads");
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
  "names" : [
      "td4.cpp::FindParallel_test",
      "td4.cpp::Account_test",
      "td4.cpp::MassivWithdrawal_test",
      "td4.cpp::MyLock"
  ],
  "points" : [5, 5, 5, 5]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 4;
    std::string const test_names[total_test_cases] = {
        "MaxParallel_test",
        "Account_test",
        "MassivWithdrawal_test",
        "MyLock_test"
    };
    int const points[total_test_cases] = {5, 5, 5, 5};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
        test_find_parallel, test_account, test_massiv_withdraw, test_my_lock
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
