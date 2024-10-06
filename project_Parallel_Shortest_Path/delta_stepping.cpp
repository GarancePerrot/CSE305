
#include "graph.h"





// relax node u with new dist x
template <typename T>
void relax(int w, T x, std::vector<T>& dist, std::vector<std::deque<int>>& buckets, int delta) {
    if (x >= dist[w]) { // early exit if no update needed
        return;
    }
    // if x < tent(w) then
    if (dist[w] != INT_MAX) { 
        int oldBucketIdx = static_cast<int>(dist[w] / delta); // index of old bucket
        auto& oldBucket = buckets[oldBucketIdx];
        oldBucket.erase(std::remove(oldBucket.begin(), oldBucket.end(), w), oldBucket.end()); // remove w from its current bucket  
    }
    int newBucketIdx = static_cast<int>(x / delta);  // index of new bucket
    
    if (newBucketIdx >= buckets.size()) {
        buckets.resize(newBucketIdx + 1);
    }

    buckets[newBucketIdx].push_back(w);                                   // insert w into new bucket

    dist[w] = x;                                                          // update new distance
    
}


// creates requests for the edges of type isLight and of the nodes in R
// returns a set of edges
template <typename T>
std::vector<Edge<T>> findRequests(const std::deque<int>& Vprime, const Graph<T>& graph, int delta, const std::vector<T>& dist, bool isLight) {
    std::vector<Edge<T>> requests; 
    for (int v : Vprime) {                                   // v in V'
        for (const auto& e : graph.get_adjacent(v)) {        // get (v, w)
            int w = e.dest; 
            T weight = e.weight;                             // c(v, w) which is different from c(w,v), that is what caused problems earlier !!
            if ((isLight && weight <= delta) || (!isLight && weight > delta)) { // (v,w) in E_kind (edge of the good type, light or heavy)
                Edge<T> e(w, dist[v] + weight);              // tent(v) + c(v,w)
                requests.push_back(e);                       // put (w, newDist) in set of requests
            }
        }
    }
    return requests;                                        // return {(w, tent(v)+c(v,w)) | v in V' ∧ (v,w) in E_kind}
}

// do relaxations, may move nodes between buckets
template <typename T>
void relaxRequests(const std::vector<Edge<T>>& requests, const Graph<T>& graph, std::vector<T>& dist, std::vector<std::deque<int>>& buckets, int delta) {
    for (auto req : requests) {            // for each (w ,x) in Req 
        int w = req.dest;
        T x = req.weight;
        relax(w, x, dist, buckets, delta);  // do relax(w, x)
    }
}

// helper function to find the smallest non-empty bucket for delta-stepping
int find_smallest_non_empty_bucket(const std::vector<std::deque<int>>& buckets) {
  int min_size = INT_MAX; 
  int min_bucket = -1;

  for (size_t i = 0; i < buckets.size(); ++i) {
    if (!buckets[i].empty()) {
      int size = buckets[i].size();
      if (size < min_size) {
        min_size = size;
        min_bucket = i;
      }
    }
  }
  return min_bucket;
}


template <typename T>                                                        //  for reference : fig 1 p. 123 of paper 
std::vector<T> delta_stepping(int source, const Graph<T>& graph, int delta, bool print_dist) {
    int n = graph.size();
    std::vector<T> dist(n, INT_MAX);                                        // line 1: for each v in V do tent(v) = inf 
    int b = graph.nb_buckets(delta);
    std::vector<std::deque<int>> buckets(b); 

    dist[source] = 0;                                                       // line 2
    buckets[0].push_back(source); // Insert source node with distance 0     // line 2 : relax(s,0)

    while (true){                                                           // line 3 : while !isEmpty(B) do
        int i = find_smallest_non_empty_bucket(buckets);                    // line 4 : i := min {j>= 0 | B[j] != ∅}
        if (i == -1) break;                                                 // line 3 
        std::deque<int> R;                                                  // line 5 : R := ∅
        while (!buckets[i].empty()) {                                       // line 6 : while B[i] != ∅ do
            std::vector<Edge<T>> lightRequests = findRequests(buckets[i], graph, delta, dist, true); // line 7 : Req := findRequests(B[i], light)
            // move elements of buckets[i] to the end of R, empties buckets[i] : 
            R.insert(R.end(), std::make_move_iterator(buckets[i].begin()), std::make_move_iterator(buckets[i].end()));
            buckets[i].clear();                             // line 8 & 9 : R := R ∪ B[i], B[i] := ∅
            relaxRequests(lightRequests, graph, dist, buckets, delta);      // line 10 : relaxRequests(Req)
            
        }        
        // Relax heavy edges
        std::vector<Edge<T>> heavyRequests = findRequests(R, graph, delta, dist, false);  // line 11 : Req := findRequests(R, heavy)
        relaxRequests(heavyRequests, graph, dist, buckets, delta);          // line 12 : relaxRequests(Req)
        
    }
  

    // Print the distances
    if (print_dist){
        for (int i = 0; i < n; ++i) {
            std::cout << "Distance from " << source << " to " << i << " is ";
            if (dist[i] == INT_MAX){
                std::cout << "infinity" << std::endl;
            }
            else{
                std::cout << dist[i] << std::endl;
            }
        }
    }

    return dist;
}


// we use a mutex to ensure that updates to the distance vector and bucket lists are atomic
template <typename T>
void relaxPar(int u, T x, std::vector<T>& dist, std::vector<std::deque<int>>& buckets, int delta, std::mutex& m) {
    std::lock_guard<std::mutex> lock(m);
    relax(u, x, dist, buckets, delta);
}


// edge relaxation for an entire bucket can be done in parallel so we use threads
// we use multiple threads to process the requests and a mutex to manage shared resources safely
template <typename T>
void relaxRequestsPar(const std::vector<Edge<T>>& requests, const Graph<T>& graph, std::vector<T>& dist, std::vector<std::deque<int>>& buckets, int delta, int nb_threads) {
    // mutex to protect concurrent access to buckets
    std::mutex mutex;

    // divide requests into chunks for each thread
    int chunk_size = (requests.size() + nb_threads - 1) / nb_threads;
    std::vector<std::thread> threads(nb_threads);

    // Thread function to process a chunk of requests
    auto relax_chunk = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            auto req = requests[i];
            int w = req.dest;
            T x = req.weight;
            relaxPar(w, x, dist, buckets, delta, mutex); 
        }
    };

    for (int i = 0; i < nb_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, (int)requests.size());
        threads[i] = std::thread(relax_chunk, start, end);
    }


    for (auto& thread : threads) {
        thread.join();
    }
}


template <typename T>
std::vector<Edge<T>> findRequestsPar(const std::deque<int>& Vprime, const Graph<T>& graph, int delta, const std::vector<T>& dist, bool isLight, int nb_threads) {
    
    std::vector<Edge<T>> requests;
    // Use a mutex to protect the shared requests vector
    std::mutex mutex;
    size_t chunk_size = (Vprime.size() + nb_threads - 1) / nb_threads;
    std::vector<std::thread> threads(nb_threads);

    auto process_chunk = [&](int start, int end) {
        // Create a local vector to store requests for each thread
        std::vector<Edge<T>> local_requests;
        for (int i = start; i < end; ++i) {
            
            auto it = std::next(Vprime.begin(), i);
            int v = *it;
            for (const auto& e : graph.get_adjacent(v)) {        
                int w = e.dest; 
                T weight = e.weight;                             
                if ((isLight && weight <= delta) || (!isLight && weight > delta)) { 
                    Edge<T> e(w, dist[v] + weight); 
                    local_requests.push_back(e);                     
                }
            }
        }
        // Acquire the mutex and add the local requests to the shared vector
        std::lock_guard<std::mutex> lock(mutex);
        requests.insert(requests.end(), local_requests.begin(), local_requests.end());
    };

    for (int i = 0; i < nb_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min((int)(start + chunk_size), (int)Vprime.size());
        threads[i] = std::thread(process_chunk, start, end);
    } 

    for (auto& thread : threads) {
        thread.join();
    }

    return requests;
}



template <typename T>
std::vector<T> delta_stepping_Par(int source, const Graph<T>& graph, int delta, int nb_threads, bool print_dist) {
    int n = graph.size();
    std::vector<T> dist(n, INT_MAX);
    int b = graph.nb_buckets(delta);
    std::vector<std::deque<int>> buckets(b);

    dist[source] = 0;
    buckets[0].push_back(source);

    std::vector<Edge<T>> lightRequests;
    std::vector<Edge<T>> heavyRequests;
    int new_nb_thread = 0;

    while (true) {
        int i = find_smallest_non_empty_bucket(buckets);
        if (i == -1) break;

        std::deque<int> R;
        while (!buckets[i].empty()) {
            if ((int)buckets[i].size() > 2 * nb_threads){ // dynamic thread count adjustment based on the size of the bucket
                new_nb_thread = std::min((int)buckets[i].size(), nb_threads);

                lightRequests = findRequestsPar(buckets[i], graph, delta, dist, true, new_nb_thread);
                R.insert(R.end(), std::make_move_iterator(buckets[i].begin()), std::make_move_iterator(buckets[i].end()));
                buckets[i].clear();
            
                if ((int)buckets.size() >= nb_threads){ // dynamic thread count adjustment based on the size of the bucket
                    // new_nb_thread = (int)buckets.size();
                    relaxRequestsPar(lightRequests, graph, dist, buckets, delta, nb_threads);
                }
                else if ((int)buckets.size() <= 2){ 
                    relaxRequests(lightRequests, graph, dist, buckets, delta);
                }
                else {
                    new_nb_thread = (int)buckets.size();
                    relaxRequestsPar(lightRequests, graph, dist, buckets, delta,  new_nb_thread);      // line 10 : relaxRequests(Req)
                }
            }
            else {
                //std::cout << "here" <<std::endl;
                lightRequests = findRequests(buckets[i], graph, delta, dist, true); // line 7 : Req := findRequests(B[i], light)
                // move elements of buckets[i] to the end of R, empties buckets[i] : 
                R.insert(R.end(), std::make_move_iterator(buckets[i].begin()), std::make_move_iterator(buckets[i].end()));
                buckets[i].clear();                             // line 8 & 9 : R := R ∪ B[i], B[i] := ∅
                
                if ((int)buckets.size() >= nb_threads){ // dynamic thread count adjustment based on the size of the bucket
                    // new_nb_thread = (int)buckets.size();
                    relaxRequestsPar(lightRequests, graph, dist, buckets, delta, nb_threads);
                }
                else if ((int)buckets.size() <= 2){ 
                    relaxRequests(lightRequests, graph, dist, buckets, delta);
                }
                else {
                    new_nb_thread = (int)buckets.size();
                    relaxRequestsPar(lightRequests, graph, dist, buckets, delta,  new_nb_thread);      // line 10 : relaxRequests(Req)
                }
            }
            
        }

        if ((int)R.size() > 2 * nb_threads){ // dynamic thread count adjustment based on the size of the bucket
            new_nb_thread = std::min((int)R.size(), nb_threads);
            heavyRequests = findRequestsPar(R, graph, delta, dist, false, nb_threads);

            if ((int)buckets.size() >= nb_threads){ // dynamic thread count adjustment based on the size of the bucket
                relaxRequestsPar(heavyRequests, graph, dist, buckets, delta, nb_threads);
            }
            else if ((int)buckets.size() <= 2){ 
                    relaxRequests(heavyRequests, graph, dist, buckets, delta);
                }
            else {
                new_nb_thread = (int)buckets.size();
                relaxRequestsPar(heavyRequests, graph, dist, buckets, delta, new_nb_thread);
            }
        }
        else {
            heavyRequests = findRequests(R, graph, delta, dist, false);

            if ((int)buckets.size() >= nb_threads){ // dynamic thread count adjustment based on the size of the bucket
                // new_nb_thread = (int)buckets.size();
                relaxRequestsPar(heavyRequests, graph, dist, buckets, delta, nb_threads);
            }
            else if ((int)buckets.size() <= 2){ 
                    relaxRequests(heavyRequests, graph, dist, buckets, delta);
                }
            else {
                new_nb_thread = (int)buckets.size();
                relaxRequestsPar(heavyRequests, graph, dist, buckets, delta, new_nb_thread);
            }
        }       
    }

    if (print_dist) {
        for (int i = 0; i < n; ++i) {
            std::cout << "Distance from " << source << " to " << i << " is ";
            if (dist[i] == INT_MAX) {
                std::cout << "infinity" << std::endl;
            } else {
                std::cout << dist[i] << std::endl;
            }
        }
    }

    return dist;
}



// _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

// Other attempt to parallelize that failed



/*
// Better version of relaxRequestsPar but crashes for some sizes or some delta
// The program crashes because of the size of the node_mutexes
// The problem is that if we allocate too much mutexes to it, it slows down a lot, and if we don't allocate enough it crashes
template <typename T>
void relaxPar(int u, T x, std::vector<T>& dist, std::vector<std::deque<int>>& buckets, int delta, std::vector<std::mutex>& node_mutexes) {
    int bucketIdx = static_cast<int>(x / delta); // Calculate which bucket/mutex to use

    if (bucketIdx < 0 || bucketIdx >= node_mutexes.size()) {
        std::cerr << "Error: bucketIdx out of range: " << bucketIdx << std::endl;
        return; 
    }

    std::lock_guard<std::mutex> lock(node_mutexes[bucketIdx]); // Lock the specific mutex
    relax(u, x, dist, buckets, delta); // Protected operation
}
template <typename T>
void relaxRequestsPar(const std::vector<Edge<T>>& requests, const Graph<T>& graph, std::vector<T>& dist, std::vector<std::deque<int>>& buckets, int delta, int nb_threads) {
    // The program crashes because of the size of the node_mutexes
    // The problem is that if we allocate too much mutexes to it, it slows down a lot, and if we don't allocate enough it crashes
    std::vector<std::mutex> node_mutexes(graph.size()*buckets.size());

    // Divide requests into chunks for each thread
    int chunk_size = (requests.size() + nb_threads - 1) / nb_threads;
    std::vector<std::thread> threads(nb_threads);

    // Thread function to process a chunk of requests
    auto relax_chunk = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            auto req = requests[i];
            int w = req.dest;
            T x = req.weight;
            relaxPar(w, x, dist, buckets, delta, node_mutexes);
        }
    };

    for (int i = 0; i < nb_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, (int)requests.size());
        threads[i] = std::thread(relax_chunk, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

*/

/*
// Failed attempt to implement a static workload assignment where the nodes
//are assigned to threads at the beginning of the algorithm and
//only the assigned thread can relax edges leading to a node
// for reference, see pseudocode of Algorithm 1 of paper " Engineering a Parallel Δ-stepping Algorithm"
// loop 1 works, crashes during loop 2 


int find_sneb(const std::vector<std::list<int>>& buckets) {
  int min_size = INT_MAX; 
  int min_bucket = -1;

  for (size_t i = 0; i < buckets.size(); ++i) {
    if (!buckets[i].empty()) {
      int size = buckets[i].size();
      if (size < min_size) {
        min_size = size;
        min_bucket = i;
      }
    }
  }
  return min_bucket;
}


template <typename T>
void relaxReq(const std::vector<Edge2<T>>& requests, const Graph<T>& graph, const std::vector<T>& dist, const std::vector<std::list<int>>& buckets, int delta) {
    //std::cout<<"relaxreq"<<std::endl;
    for (auto req : requests) {           
        int w = req.dest;
        T x = req.weight;
        relax(w, x, dist, buckets, delta); 
    }
}

template <typename T>
std::vector<Edge2<T>> GenReq(int v, std::vector<Edge2<T>>& R, const Graph<T>& graph, int delta, const std::vector<T>& dist, bool isLight) {
    
    for (const Edge<T> e : graph.get_adjacent(v)) {        
        int u = e.dest; 
        T w = e.weight;                             
        if ((isLight && w <= delta) || (!isLight && w > delta)) { 
            Edge2<T> e(v, u, w);         
            R.push_back(e);                     
        }
    }
    return R;                                       
}

template <typename T>  
void loop1(int v, std::list<int>& bucket, std::vector<Edge2<T>>& lightRequests, std::vector<Edge2<T>>& heavyRequests, const Graph<T>& graph, int delta, const std::vector<T>& dist){

    bucket.remove(v);  // remove v from bucket
    lightRequests = GenReq(v, lightRequests, graph, delta, dist, true);
    heavyRequests = GenReq(v, heavyRequests, graph, delta, dist, false);

}


template <typename T>
void relAX(const Edge2<T>& req, std::vector<T>& dist, std::vector<std::list<int>>& buckets, int delta){
    int v = req.from;
    int u = req.dest;
    T w = req.weight;
    if (dist[v] + w < dist[u]){
        int i = std::floor(static_cast<int>(dist[u])/delta);
        int j = std::floor(static_cast<int>(dist[v]+ w)/delta);
        buckets[i].remove(u);
        buckets[j].push_back(u);
        dist[u] = dist[v] + w;
    }

}


template <typename T>
void loop2_3(const Edge2<T>& req, std::vector<std::list<int>>& buckets, std::vector<Edge2<T>>& Requests, const Graph<T>& graph, int delta, std::vector<T>& dist){
    Requests.erase(std::remove(Requests.begin(), Requests.end(), req), Requests.end());
    relAX(req, dist, buckets, delta);

}



template <typename T>                                                        
std::vector<T> DS_par(int source, const Graph<T>& graph, int delta, int nb_threads, bool print_dist) {
    int n = graph.size();
    std::vector<T> dist(n, INT_MAX);                                        
    std::vector<std::list<int>> buckets(n+1);  

    //initialize buckets
    for (int i = 1; i < n+1 ; i++){
        buckets[n].push_back(i);            // corresponds to B_inf
    }
    buckets[0].push_back(source);
    dist[source] = 0;   

    //(randomly: not for now) partition nodes among threads
    std::vector<int> thread_idx(n);

    std::vector<std::thread> threads(nb_threads);
    std::mutex mutex;
    int chunk_size = (n + nb_threads - 1) / nb_threads;
    for (int i=0 ; i< nb_threads ; i++){
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, n);
        for (int j= start; j < end ; j++){
            thread_idx[j] = i;
        }
    }

    while (true){                                                          

        int k = find_sneb(buckets);  
        if (k == -1) break;                                                 
        std::vector<Edge2<T>> lightRequests;
        std::vector<Edge2<T>> heavyRequests; 

        while (!buckets[k].empty()) { 

            for (int v : buckets[k]){    // loop 1
                int j = thread_idx[v];  // find thread associated to node
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    threads[j] = std::thread(loop1<T>, v, std::ref(buckets[k]), std::ref(lightRequests), std::ref(heavyRequests), std::ref(graph), delta, std::ref(dist));
                }
                
            }

            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }

            std::cout << "loop 1 works" << std::endl;

            for (Edge2<T> req : lightRequests){ // loop 2
                int v = req.dest;
                int j = thread_idx[v]; 
                std::cout << " node " << v << " ,thread " << j << std::endl;
                
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    threads[j] = std::thread(loop2_3<T>, std::ref(req), std::ref(buckets), std::ref(lightRequests), std::ref(graph), delta, std::ref(dist));
                    std::cout << " out of thread " << std::endl;
                }
                
            }

            
            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }

            std::cout << "loop 2 works" << std::endl;
 
        }    
        for ( Edge2<T> req : heavyRequests){ // loop 3
            int v = req.dest;
            int j = thread_idx[v];
            {
                std::lock_guard<std::mutex> lock(mutex);
                threads[j] = std::thread(loop2_3<T>, std::ref(req), std::ref(buckets), std::ref(heavyRequests), std::ref(graph), delta, std::ref(dist));
            }
            
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        std::cout << "loop 3 works" << std::endl;

    }

    return dist;
}

*/