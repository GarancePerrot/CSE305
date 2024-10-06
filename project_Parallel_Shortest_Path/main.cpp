#include "dijkstra.cpp"
#include "delta_stepping.cpp"




// Compare distances from the 2 algorithms to check is they match
template <typename T>
void compare_distances(const std::vector<T>& dist1, const std::vector<T>& dist2) {
    if (dist1.size() != dist2.size()) {
        std::cerr << "Error: Distance vectors are of different sizes." << std::endl;
        return;
    }

    //double mse = 0;
    int diff_count = 0;
    for (size_t i = 0; i < dist1.size(); i++) {
        if (dist1[i] != dist2[i]) {
            diff_count++;
            // if (dist1[i] != INT_MAX && dist2[i] != INT_MAX) { 
            //     mse += pow(dist1[i] - dist2[i], 2);
            // }
        }
    }

    // if (diff_count > 0) {
    //     mse /= diff_count; 
    // }

    // Print MSE and nb of different values
    // std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Number of different values: " << diff_count << std::endl;
}




template <typename T>
int continue_main(Graph<T> g, int run_algo, int delta, int print_graph, int print_dist, int num_threads){

    if (print_graph){
        g.print_graph(); 
    }

    double t1 = 0;
    double t2 = 0;
    double t3 = 0;
    double best_time = INF;
    std::vector<T> dist_dijkstra;
    std::vector<T> dist_delta_stepping;
    std::vector<T> dist_delta_stepping_threads;

    int new_delta = delta;

    // running and chronometring the algorithms: 

    if (run_algo == 1 || run_algo == 4 || run_algo == 5 || run_algo == 0){// Run dijkstra algo
        std::cout << "\nResults with Dijkstra algo\n";
        std::chrono::steady_clock::time_point begin_dijkstra = std::chrono::steady_clock::now();
        dist_dijkstra = dijkstra(0, g, print_dist);
        std::chrono::steady_clock::time_point end_dijkstra = std::chrono::steady_clock::now();
        t1 = std::chrono::duration_cast<std::chrono::microseconds>(end_dijkstra - begin_dijkstra).count() ;
        std::cout << "Total time with Dijkstra: " << t1 << " micro seconds" << std::endl;

    }

    if (run_algo == 2 || run_algo == 4 || run_algo == 6 || run_algo == 0){ // Run delta-stapping
        if (delta==1000){
            new_delta = 1;
            int best_num_delta = 100;
            while (new_delta<50){
                std::cout << "\nResults with delta stepping algo, delta = " << new_delta << std::endl;
                std::chrono::steady_clock::time_point begin_delta_stepping = std::chrono::steady_clock::now();
                // dist_delta_stepping_threads = delta_stepping_parallel(0, g, delta, num_threads, print_dist);
                dist_delta_stepping = delta_stepping(0, g, new_delta, print_dist);
                std::chrono::steady_clock::time_point end_delta_stepping = std::chrono::steady_clock::now();
                t2 = std::chrono::duration_cast<std::chrono::microseconds>(end_delta_stepping - begin_delta_stepping).count() ;
                if (t2 < best_time){
                    best_time = t2;
                    best_num_delta = new_delta;
                }
                std::cout << "Total time with Delta stepping: " << t2 << "  micro seconds" << std::endl;
                new_delta += 2;
            }
            new_delta = best_num_delta;
            std::cout << "\nBest results with delta stepping algo, best delta = " << best_num_delta << ", best time = " << best_time << std::endl;
        }
        else{
            std::cout << "\nResults with delta stepping algo, delta = " << delta << std::endl;
            std::chrono::steady_clock::time_point begin_delta_stepping = std::chrono::steady_clock::now();
            dist_delta_stepping = delta_stepping(0, g, delta, print_dist);
            std::chrono::steady_clock::time_point end_delta_stepping = std::chrono::steady_clock::now();
            t2 = std::chrono::duration_cast<std::chrono::microseconds>(end_delta_stepping - begin_delta_stepping).count();
            std::cout << "Total time with Delta stepping: " << t2 << "  micro seconds" << std::endl;
        }
    }

    if (run_algo == 3 || run_algo == 5 || run_algo == 6 || run_algo == 0){ // Run delta-stepping threads
        if (num_threads == 1000){
            int new_num_threads = 1;
            int best_num_threads = 0;
            while (new_num_threads<25){
                std::cout << "\nResults with delta stepping threads algo, delta = " << new_delta << " , nb of threads = " << new_num_threads << std::endl;
                std::chrono::steady_clock::time_point begin_delta_stepping_threads = std::chrono::steady_clock::now();
                // dist_delta_stepping_threads = delta_stepping_parallel(0, g, delta, num_threads, print_dist);
                dist_delta_stepping_threads = delta_stepping_Par(0, g, new_delta, new_num_threads, print_dist);
                std::chrono::steady_clock::time_point end_delta_stepping_threads = std::chrono::steady_clock::now();
                t3 = std::chrono::duration_cast<std::chrono::microseconds>(end_delta_stepping_threads - begin_delta_stepping_threads).count() ;
                if (t3 < best_time){
                    best_time = t3;
                    best_num_threads = new_num_threads;
                }
                std::cout << "Total time with Delta stepping threads: " << t3 << "  micro seconds" << std::endl;
                new_num_threads += 1;
            }
            std::cout << "\nBest results with delta stepping threads algo, delta = " << new_delta << " , best nb of threads = " << best_num_threads << " , best time = " << best_time << std::endl;
        }
        
        else {        
            std::cout << "\nResults with delta stepping threads algo, delta = " << new_delta << " , nb of threads = " << num_threads << std::endl;
            std::chrono::steady_clock::time_point begin_delta_stepping_threads = std::chrono::steady_clock::now();
            // dist_delta_stepping_threads = delta_stepping_parallel(0, g, delta, num_threads, print_dist);
            dist_delta_stepping_threads = delta_stepping_Par(0, g, new_delta, num_threads, print_dist);
            std::chrono::steady_clock::time_point end_delta_stepping_threads = std::chrono::steady_clock::now();
            t3 = std::chrono::duration_cast<std::chrono::microseconds>(end_delta_stepping_threads - begin_delta_stepping_threads).count() ;
            std::cout << "Total time with Delta stepping threads: " << t3 << "  micro seconds" << std::endl;
        }
        

    }

    //comparison of algorithms (results & runtimes): 

    if (run_algo == 4 || run_algo == 0){// compare dijkstra & delta-stepping: 
        std::cout << "\nComparing Dijkstra / delta-stepping "<< std::endl;
        double speed_up = t1/t2; 
        std::cout << "Speed up: " << speed_up << std::endl;
        compare_distances(dist_dijkstra, dist_delta_stepping);
    }

    if (delta == 1000){
        if (run_algo == 5 || run_algo == 0){// compare dijkstra & delta-stepping threads:
            std::cout << "\nComparing Dijkstra / delta-stepping threads "<< std::endl; 
            double speed_up = t1/t3; 
            std::cout << "Speed up: " << speed_up << std::endl;
            compare_distances(dist_dijkstra, dist_delta_stepping_threads);
        }

        if (run_algo == 6 || run_algo == 0){// compare delta-stepping & delta-stepping threads: 
            std::cout << "\nComparing delta-stepping / delta-stepping threads "<< std::endl;
            double speed_up = best_time/t3; 
            std::cout << "Speed up: " << speed_up << std::endl;
            compare_distances(dist_delta_stepping, dist_delta_stepping_threads);
        }
        }

    else if (num_threads == 1000){
        if (run_algo == 5 || run_algo == 0){// compare dijkstra & delta-stepping threads:
            std::cout << "\nComparing Dijkstra / delta-stepping best number of threads "<< std::endl; 
            double speed_up = t1/best_time; 
            std::cout << "Speed up: " << speed_up << std::endl;
            compare_distances(dist_dijkstra, dist_delta_stepping_threads);
        }

        if (run_algo == 6 || run_algo == 0){// compare delta-stepping & delta-stepping threads: 
            std::cout << "\nComparing delta-stepping / delta-stepping best number of threads "<< std::endl;
            double speed_up = t2/best_time; 
            std::cout << "Speed up: " << speed_up << std::endl;
            compare_distances(dist_delta_stepping, dist_delta_stepping_threads);
        }
    }
    else{
        if (run_algo == 5 || run_algo == 0){// compare dijkstra & delta-stepping threads:
            std::cout << "\nComparing Dijkstra / delta-stepping threads "<< std::endl; 
            double speed_up = t1/t3; 
            std::cout << "Speed up: " << speed_up << std::endl;
            compare_distances(dist_dijkstra, dist_delta_stepping_threads);
        }

        if (run_algo == 6 || run_algo == 0){// compare delta-stepping & delta-stepping threads: 
            std::cout << "\nComparing delta-stepping / delta-stepping threads "<< std::endl;
            double speed_up = t2/t3; 
            std::cout << "Speed up: " << speed_up << std::endl;
            compare_distances(dist_delta_stepping, dist_delta_stepping_threads);
        }
    }
    

    return 0;
}





// For all type of graphs
// int type_graph = 0 for small graph, 1 for txt graph, 2 for random graph 
// int run_algo = 1 dijkstra ; 2 delta-stepping ; 3 DS-threads ; 4 compare dijkstra & DS ; 5 compare dijkstra & DS threads ; 6 compare DS & DS threads ; 0 compare all
// int type_weight = 0 int ; 1 double (positive edge weights) ; 2 not uniform distribution
// int delta = 0 if want to use computed value, or = value if want a specific value ; or = 1000 if wants multiple value of delta testing
// int num_threads = 0 if want to use computed value (g.suggestOptimalNumberOfThreads()), or = value if want a specific value, or = 1000 if wants multiple value of threads testing
// bool print_dist = if want to print the resulting distances or not, it affects the running time so put 0 preferably
// bool print_graph = whether or not want to print the graph


// For txt graph
// std::string name_of_txt = "txt_graph_1000.txt" for example
//    -> for type_weight = 0 (integer weights), choose between txt_graph_1000.txt, txt_graph_10000.txt, txt_graph_100000.txt
//    -> for type_weight = 1 (positive real weights), choose between txt_graph_1000_d.txt, txt_graph_10000_d.txt, txt_graph_100000_d.txt

// int num_vertices = nb vertices

// For random graph
// int num_vertices = nb vertices
// int num_edges =  nb edges
// int min_weight = min weight (>0 positive weights)
// int max_weight = max weight




// Examples to run:

// First run: g++ main.cpp -o test

// Then:
// For testing an existing small graph
// ./test 0 [run_algo] [type_weight] [delta] [num_threads] [print_dist] [print_graph] 
// ./test 0 0 0 32 6 1 1

// For testing a graph from a txt file
// ./test 1 name_of_txt_file [num_vertices] [run_algo] [type_weight] [delta] [num_threads] [print_dist] [print_graph]
// ./test 1 ./txt_graphs/txt_graph_1000.txt 1000 0 0 0 1000 0 0

// For testing a random graph
// ./test 2 [num_vertices] [num_edges] [min_weight] [max_weight] [run_algo] [type_weight] [delta] [num_threads] [print_dist] [print_graph]
// ./test 2 5000 500000 1 5 0 0 0 0 0 0



int main(int argc, char* argv[]) {

    int type_graph = std::atoi(argv[1]);

    std::string name_of_txt;
    int type_weight, num_vertices, num_edges, min_weight, max_weight;
    int run_algo, delta, num_threads;
    bool print_dist, print_graph;

    switch (type_graph) {
    case 0: // Small graph
        break;

    case 1: // txt graph
        name_of_txt = argv[2];
        num_vertices = std::atoi(argv[3]);
        break;

    case 2: // Random graph
        num_vertices = std::atoi(argv[2]);
        num_edges = std::atoi(argv[3]);
        min_weight = std::atoi(argv[4]);
        max_weight = std::atoi(argv[5]);
        break;

    default:
        std::cerr << "Invalid type_graph, valid options are 0, 1, or 2." << std::endl;
        return 1;
    }

    run_algo = std::atoi(argv[argc-6]);
    type_weight = std::atoi(argv[argc-5]);
    delta = std::atoi(argv[argc-4]);
    num_threads = std::atoi(argv[argc-3]);
    print_dist = std::atoi(argv[argc-2]);
    print_graph = std::atoi(argv[argc-1]);


    if (type_graph == 0){
        std::cout << "\nGenerating a small graph\n";
        if (type_weight == 1){ // creating a graph with real-valued positive edge weights
            Graph<double> g(6); //change with the nb of vertices in the fixed graph
            g.gen_small_graph_real();
            if (delta==0){
                delta = g.findDelta();
            }
            if (num_threads==0){
                num_threads = g.suggestOptimalNumberOfThreads();
            }
            return continue_main(g, run_algo, delta, print_graph, print_dist, num_threads);
        } 
        else {
            Graph<int> g(6);
            g.gen_small_graph_int();
            if (delta==0){
                delta = g.findDelta();
            }
            if (num_threads==0){
                num_threads = g.suggestOptimalNumberOfThreads();
            }
            return continue_main(g, run_algo, delta, print_graph, print_dist, num_threads); 
        }
    }

    else if (type_graph == 1){
        std::cout << "\nGenerating a graph via text file\n";
        if (type_weight == 1){ // creating a graph with real-valued positive edge weights
            if (name_of_txt == "txt_graph_1000.txt" || name_of_txt == "txt_graph_10000.txt" || name_of_txt == "txt_graph_100000.txt"){
                std::cout << "File Error : Please use a txt file with double eidge weights : txt_graph_1000_d.txt, txt_graph_10000_d.txt, txt_graph_100000_d.txt";
                return 1;
            }
            Graph<double> g(num_vertices);
            g.gen_graph_from_txt(name_of_txt);
            if (delta==0){
                delta = g.findDelta();
            }
            if (num_threads==0){
                num_threads = g.suggestOptimalNumberOfThreads();
            }
            return continue_main(g, run_algo, delta, print_graph, print_dist, num_threads);
        } 
        else {
            if (name_of_txt == "txt_graph_1000_d.txt" || name_of_txt == "txt_graph_10000_d.txt" || name_of_txt == "txt_graph_100000_d.txt"){
                std::cout << "File Error : Please use a txt file with integer edge weights : txt_graph_1000.txt, txt_graph_10000.txt, txt_graph_100000.txt";
                return 1;
            }
            Graph<int> g(num_vertices); //change with the nb of vertices in the txt graph
            g.gen_graph_from_txt(name_of_txt);
            if (delta==0){
                delta = g.findDelta();
            }
            if (num_threads==0){
                num_threads = g.suggestOptimalNumberOfThreads();
            }
            return continue_main(g, run_algo, delta, print_graph, print_dist, num_threads);
        }
    }

    else if (type_graph == 2){
        std::cout << "\nGenerating a random graph\n";
        if (type_weight == 1){ // creating a graph with real-valued positive edge weights (run_algo does not call dijkstra)
            Graph<double> g(num_vertices);
            g.gen_random_graph(type_weight, num_vertices, num_edges, min_weight, max_weight);
            if (delta==0){
                delta = g.findDelta();
            }
            if (num_threads==0){
                num_threads = g.suggestOptimalNumberOfThreads();
            }
            return continue_main(g, run_algo, delta, print_graph, print_dist, num_threads);
        } 
        else {
            Graph<int> g(num_vertices);
            g.gen_random_graph(type_weight, num_vertices, num_edges, min_weight, max_weight);
            if (delta==0){
                delta = g.findDelta();
            }
            if (num_threads==0){
                num_threads = g.suggestOptimalNumberOfThreads();
            }
            return continue_main(g, run_algo, delta, print_graph, print_dist, num_threads);
        }
    }

}


