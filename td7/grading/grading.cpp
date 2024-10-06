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
#include "td7.cpp"
#include <limits>

namespace tdgrading {

using namespace testlib;
using namespace std;

//-----------------------------------------------------------------------------

void simple_inserter(BoundedSetList& s, std::string& str) {
    s.add(str);
}

int test_bounded(std::ostream &out, const std::string test_name) {
    std::string fun_name = "BoundedSetList";

    start_test_suite(out, test_name);

    std::vector<int> res;

    BoundedSetList s(4);
    std::vector<std::string> items{"a", "b", "c", "d", "e", "f", "g"};
    std::vector<std::thread> inserters;
    for (auto it = items.begin(); it != items.end(); ++it) {
        inserters.emplace_back(std::thread(&simple_inserter, std::ref(s), std::ref(*it)));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::vector<size_t> sizes;
    for (size_t i = 0; i < items.size(); ++i) {
        sizes.push_back(s.get_count());
        if (s.contains(items[i])) {
            res.push_back(1);
            s.remove(items[i]);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            print(out, "An element got lost after insetion");
            res.push_back(0);
        }
    }
    
    std::for_each(inserters.begin(), inserters.end(), [](std::thread& t) {t.join();});
    if (sizes != std::vector<size_t>{4, 4, 4, 4, 3, 2, 1}) {
        print(out, "The sizes do not behave as expected, we are expecting size to behave like 4. 4, 4, 4, 3, 2, 1");
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
  "total" : 1,
  "names" : ["td7.cpp::BoundedSetList"],
  "points" : [10]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 1;
    std::string const test_names[total_test_cases] = {"BoundedSetList"};
    int const points[total_test_cases] = {10,};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
        test_bounded
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
