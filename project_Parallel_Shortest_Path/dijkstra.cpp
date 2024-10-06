
#include "graph.h"


template <typename T>
std::vector<T> dijkstra(int source, Graph<T>& graph, bool print_dist) {
    int n = graph.size();
    std::vector<T> dist(n, INT_MAX); 
    std::vector<bool> visited(n, false); 
    std::priority_queue< Edge<T>, std::vector<Edge<T>>, CompareEdge<T>> pq;

    // Initialize priority queue with the source node
    Edge<T> e1(0, source);
    pq.push(e1); // (distance, vertex)
    dist[source] = 0;

    while (!pq.empty()) {
        T u = pq.top().weight;
        pq.pop();

        // If vertex already been visited, continue to the next
        if (visited[u]) continue;
        visited[u] = true;

        // Check each adjacent vertex of u
        for (const auto& edge : graph.get_adjacent(u)) {
            int v = edge.dest;
            T weight = edge.weight;

            // Only consider this vertex if it has not been visited
            if (!visited[v] && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                Edge<T> e2(dist[v],v);
                pq.push(e2); // Push updated distance and vertex
            }
        }
    }


    if (print_dist){
        for (int i = 0; i < n; ++i) {
            std::cout << "Distance from " << source << " to " << i << " is ";
            if (dist[i] == INF)
                std::cout << "infinity" << std::endl;
            else
                std::cout << dist[i] << std::endl;
        }
    }

    return dist;
}