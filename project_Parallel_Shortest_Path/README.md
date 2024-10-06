# CSE305-Project-Parallel-shortest-paths

### Authors  
- Clémence Mottez
- Garance Perrot

### Introduction  
Project as part of our Concurrent and Distributed Computing class at Ecole Polytechnique.  
Finding shortest distances in a graph is one the fundamental problems in computer science with
numerous applications (route planning by CityMapper and Google/Yandex-maps, for example).  
We have already seen classical algorithms (BFS, Dijkstra, etc) for this task in our algorithms course.
However, these algorithms are inherently sequential and hard to parallelize (although parallele versions exist).

### Aim  
In this project, we implement and benchmark one of the most standard shortest path algorithms, ∆-stepping algorithm.
We implement the algorithm, run it on a set of benchmarks, and compare it with non-parallel algorithms including single-thread ∆-stepping and Dijkstra.

### How to run the project
First run: ```make```  

Then run either:   
- For testing an existing small graph  
```./test 0 [run_algo] [type_weight] [delta] [num_threads] [print_dist] [print_graph]```  
For example you can run: ```./test 0 0 0 32 6 1 1```  
  
- For testing a graph from a txt file  
```./test 1 name_of_txt_file [num_vertices] [run_algo] [type_weight] [delta] [num_threads] [print_dist] [print_graph]```  
For example you can run: ```./test 1 ./txt_graphs/txt_graph_1000.txt 1000 0 0 0 1000 0 0```  

- For testing a random graph  
```./test 2 [num_vertices] [num_edges] [min_weight] [max_weight] [run_algo]  [type_weight] [delta] [num_threads] [print_dist] [print_graph]```  
For example you can run: ```./test 2 5000 500000 1 1 0 0 0 1000 0 0```  
  
The different arguments are explained bellow:   
int type_graph = 0 for small graph, 1 for txt graph, 2 for random graph     
int run_algo = 1 dijkstra ; 2 delta-stepping ; 3 DS-threads ; 4 compare dijkstra & DS ; 5 compare dijkstra & DS threads ; 6 compare DS & DS threads ; 0 compare all    
int type_weight = 0 int ; 1 double (positive edge weights) ; 2 not uniform distribution     
int delta = 0 if want to use computed value, or = value if want a specific value ; or = 1000 if wants multiple value of delta testing   
int num_threads = 0 if want to use computed value (g.suggestOptimalNumberOfThreads()), or = value if want a specific value, or = 1000 if wants multiple value of threads testing   
bool print_dist = if want to print the resulting distances or not, it affects the running time so put 0 preferably    
bool print_graph = whether or not want to print the graph     
int num_threads = 0 if want to use computed value, or = value if want a specific value       
  
For txt graph  
std::string name_of_txt = txt_graph_1000.txt for example      
    -> for type_weight = 0 (integer weights), choose between txt_graph_1000.txt, txt_graph_10000.txt, txt_graph_100000.txt       
    -> for type_weight = 1 (positive real weights), choose between txt_graph_1000_d.txt, txt_graph_10000_d.txt, txt_graph_100000_d.txt    
int num_vertices = nb vertices    

For random graph  
int num_vertices = nb vertices  
int num_edges =  nb edges  
int min_weight = min weight (>0 positive weights)  
int max_weight = max weight  

### What is should give you?  
For example when running ```./test 0 0 0 32 6 1 1``` (running on a small existing graph, all the programs, integer weights, with delta = 32, nb of threads = 6, printing the resulting distances and the graph):  
```
Generating a small graph  
Vertex 0 has edges to:  
 -> 1 with weight 10  
 -> 3 with weight 3  
 -> 4 with weight 10  
Vertex 1 has edges to:  
 -> 2 with weight 5  
Vertex 2 has edges to:  
 -> 4 with weight 1  
Vertex 3 has edges to:  
 -> 2 with weight 2  
 -> 1 with weight 6  
Vertex 4 has edges to:  
 -> 2 with weight 3  
 -> 3 with weight 3  
 -> 1 with weight 1  
Vertex 5 has edges to:  
 -> 4 with weight 10  
 -> 2 with weight 2  
 -> 0 with weight 6  
  
Results with Dijkstra algo  
Distance from 0 to 0 is 0  
Distance from 0 to 1 is 7  
Distance from 0 to 2 is 5  
Distance from 0 to 3 is 3  
Distance from 0 to 4 is 6  
Distance from 0 to 5 is infinity  
Total time with Dijkstra: 13 ms  
   
Results with delta stepping algo, delta = 32  
Distance from 0 to 0 is 0  
Distance from 0 to 1 is 7  
Distance from 0 to 2 is 5  
Distance from 0 to 3 is 3  
Distance from 0 to 4 is 6  
Distance from 0 to 5 is infinity  
Total time with Delta stepping: 17 ms  
  
Results with delta stepping threads algo, delta = 32 , nb of threads = 6  
Distance from 0 to 0 is 0  
Distance from 0 to 1 is 7  
Distance from 0 to 2 is 5  
Distance from 0 to 3 is 3  
Distance from 0 to 4 is 6  
Distance from 0 to 5 is infinity  
Total time with Delta stepping threads: 14 ms  
  
Comparing Dijkstra / delta-stepping   
Speed up: 0.764706  
Number of different values: 0  

Comparing Dijkstra / delta-stepping threads   
Speed up: 0.928571  
Number of different values: 0  

Comparing delta-stepping / delta-stepping threads  
Speed up: 1.21429  
Number of different values: 0  
```


