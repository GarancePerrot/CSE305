#include <chrono>
#include <climits>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

#include "CoarseSetList.cpp"
#include "SetList.cpp"

template <typename T>
int benchmark_single_thread(T& set, int count) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        set.add(std::to_string(rand()));
    }    
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    return elapsed;
}

template <typename T>
void worker_thread(int id, T& set, int num_insertions) {
    srand(id + time(0));
    for (int i = 0; i < num_insertions; ++i) {
        set.add(std::to_string(rand()));
    }
    return;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./set_benchmarker num_thread num_insertions" << std::endl;
        return 0;
    }

    int num_threads = std::stoi(argv[1]);
    int num_insertions = std::stoi(argv[2]);

    if (num_threads == 1) {
        // Timing for coarse-grained
        CoarseSetList CL;
        int elapsed = benchmark_single_thread(CL, num_insertions);
        std::cout << "Time for coarse-grained version is " << elapsed << " microseconds" << std::endl;
    
        // Timing for fine-grained
        SetList L;
        elapsed = benchmark_single_thread(L, num_insertions);        
        std::cout << "Time for fine-grained version is " << elapsed << " microseconds" << std::endl;
    } else {

        // int n = num_insertions;
        // for(int i = 0; i<6 ; i++){

        //     n *= 2;

            // Timing for coarse-grained
            std::vector<std::thread> threads(num_threads);
            CoarseSetList CL; 
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < num_threads; ++i) {
                threads[i] = std::thread(worker_thread<CoarseSetList>, i + 1, std::ref(CL), num_insertions);
            }
            for (int i = 0; i< num_threads; i++) {
                threads[i].join();
            }
            auto finish = std::chrono::steady_clock::now();
            auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
            std::cout << "Time for coarse-grained version is " << elapsed_1 << " microseconds" << std::endl;

            // Timing for fine-grained
            SetList L;
            start = std::chrono::steady_clock::now();
            for (int i = 0; i < num_threads; ++i) {
                threads[i] = std::thread(worker_thread<SetList>, i + 1, std::ref(L), num_insertions);
            }
            for (int i = 0; i< num_threads; i++) {
                threads[i].join();
            }
            finish = std::chrono::steady_clock::now();
            auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
            std::cout << "Time for fine-grained version is " << elapsed_2 << " microseconds" << std::endl;

            // std::cout << "          " << num_threads << "         |        " << n << "        |      "<< elapsed_1 <<"       |      "<< elapsed_2 <<"      "<< std::endl;

        // }
    }
    return 0;
}


/*

SPACE TO REPORT AND ANALYZE THE RUNTIMES


The command 'nproc' displays 24, the number of logical processor on my machine. 
Below are the results of tests for a different number of threads and a different number of insertions :


    NB OF THREADS    |   NB OF INSERTIONS  |  COARSE-GRAINED (μs) |   FINE-GRAINED (μs)  
                     |                     |                      |               
          25         |        1 000        |      1 269 049       |      436 601      
          25         |        4 000        |      1 116 311       |      365 917      
          25         |        8 000        |      1 231 340       |      378 131      
          25         |       16 000        |      1 225 718       |      380 588      
          25         |       32 000        |      1 302 669       |      364 337      
          25         |       64 000        |      1 358 850       |      365 217 

          30         |        1 000        |      1 718 183       |      623 928
          30         |        2 000        |      1 802 093       |      570 779      
          30         |        4 000        |      1 826 850       |      532 219      
          30         |        8 000        |      1 909 595       |      533 240      
          30         |       16 000        |      1 940 098       |      556 390      
          30         |       32 000        |      1 892 828       |      522 518      
          30         |       64 000        |      1 803 909       |      528 915 

          35         |        1 000        |      2 585 253       |      755 450
          35         |        2 000        |      2 393 287       |      770 822      
          35         |        4 000        |      2 508 484       |      704 180      
          35         |        8 000        |      2 453 637       |      685 241      
          35         |       16 000        |      2 533 593       |      685 629      
          35         |       32 000        |      2 482 405       |      694 893      
          35         |       64 000        |      2 482 373       |      719 827

          40         |        1 000        |      3 475 904       |      968 701
          40         |        2 000        |      3 462 310       |      974 956      
          40         |        4 000        |      3 372 491       |      895 806      
          40         |        8 000        |      3 219 450       |      929 359      
          40         |       16 000        |      3 406 329       |      904 657      
          40         |       32 000        |      3 480 343       |      875 845      
          40         |       64 000        |      3 621 538       |      875 741   
 

Comparison of coarse-grained VS fine-grained : Coarse-grained ListSet performs better with a low nb of threads (25)
for all insertion counts as it needs less synchronization than fine-grained.
Fine-grained ListSet hows significant improvement over coarse-grained as the number of threads 
increases, suggesting that it minimizes contention on smaller data chunks with multiple threads.

Impact of thread count : For fine-grained, the larger the number of threads, the more improvement we obtain 
as the number of insertions increases, indicating an effective parallel processing.
However for coarse-grained, the optimal nb of threads is 35, otherwize their is no improvement related to thread count
as the number of insertions increases. 

To sum up : for a low nb of threads, coarse-grained ListSet might be preferable due to lower synchronization overhead,
in contrast for more threads, fine-grained ListSet is a better choice due to its ability to manage concurrent access more efficiently.


 */
