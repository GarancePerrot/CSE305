#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <chrono>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>

#include "../gradinglib/gradinglib.hpp"
#include "td2.cpp"
#include <limits>

namespace tdgrading {

using namespace testlib;
using namespace std;

//-----------------------------------------------------------------------------

int test_altsumparallel(std::ostream &out, const std::string test_name) {
    std::string fun_name = "AltSumParallel";

    start_test_suite(out, test_name);

    std::vector<int> res;

    for (size_t i = 0; i < 100; ++i) {
        size_t len = (rand() % 10000) + 10;
        if (i < 2) {
            len = i;
        }
        std::vector<int> test;
        for (size_t j = 0; j < len; ++j) {
            test.push_back(rand() % 100);
        }
	int correct = 0;
	int curr_sign = 1;
	for (auto it = test.begin(); it != test.end(); ++it) {
	    correct += *it * curr_sign;
	    curr_sign *= -1;
	}

        size_t num_threads = 1 + (rand() % 5);
        int student_result = AltSumParallel(test, num_threads);
        res.push_back(test_eq(
            out, fun_name, student_result, correct
        ));
    }

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//---------------------------------------------------------------------------

int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 1,
  "names" : [
      "td2.cpp::AltSum_test"
  ],
  "points" : [10]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 1;
    std::string const test_names[total_test_cases] = {
        "AltSum_test",
    };
    int const points[total_test_cases] = {10};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
        test_altsumparallel
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
