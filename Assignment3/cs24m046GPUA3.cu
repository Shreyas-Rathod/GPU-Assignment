// %%writefile cs24m046.cu

#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

#define MOD 1000000007
#define BLOCK_SIZE 1024

using namespace std;

struct Edge {
    int src, dest, weight;
};

__global__ void reset_min_edges(int* min_edge, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) min_edge[idx] = -1;
}

__device__ int find_root(int u, int* parent) {
    while (parent[u] != u) {
        parent[u] = parent[parent[u]];  // Path compression
        u = parent[u];
    }
    return u;
}

__global__ void find_min_edges(Edge* edges, int E, int* parent, int* min_edge) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    Edge e = edges[idx];
    int u = e.src, v = e.dest, w = e.weight;

    int rootU = find_root(u, parent);
    int rootV = find_root(v, parent);
    if (rootU == rootV) return;

    // Atomic update for rootU's min edge
    int* addr = &min_edge[rootU];
    int old = *addr;
    while (old == -1 || (old != -1 && edges[old].weight > w)) {
        int prev = atomicCAS(addr, old, idx);
        if (prev == old) break;
        old = prev;
    }

    // Atomic update for rootV's min edge
    addr = &min_edge[rootV];
    old = *addr;
    while (old == -1 || (old != -1 && edges[old].weight > w)) {
        int prev = atomicCAS(addr, old, idx);
        if (prev == old) break;
        old = prev;
    }
}

__global__ void merge_components(Edge* edges, int* parent, int* min_edge, 
                               unsigned long long* mst_weight, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V) return;

    int current = idx;
    if (parent[current] != current) return;

    int edge_idx = min_edge[current];
    if (edge_idx == -1) return;

    Edge e = edges[edge_idx];
    int rootU = find_root(e.src, parent);
    int rootV = find_root(e.dest, parent);
    if (rootU == rootV) return;

    // Ensure deterministic merging
    if (rootU > rootV) {
        int temp = rootU;
        rootU = rootV;
        rootV = temp;
    }

    if (atomicCAS(&parent[rootV], rootV, rootU) == rootV) {
        atomicAdd(mst_weight, e.weight);
    }
}

int main() {
    int V, E;
    cin >> V >> E;

    vector<Edge> edges_host(E);
    vector<string> types(E);

    // Read input data
    for (int i = 0; i < E; i++) {
        cin >> edges_host[i].src >> edges_host[i].dest >> edges_host[i].weight >> types[i];
    }

    // Device memory pointers
    Edge* edges_dev;
    int *parent_dev, *min_edge_dev;
    unsigned long long *mst_weight_dev;

    // Allocate device memory
    cudaMalloc(&edges_dev, E * sizeof(Edge));
    cudaMalloc(&parent_dev, V * sizeof(int));
    cudaMalloc(&min_edge_dev, V * sizeof(int));
    cudaMalloc(&mst_weight_dev, sizeof(unsigned long long));

    // Initialize parent array
    vector<int> parent_host(V);
    for (int i = 0; i < V; i++) parent_host[i] = i;
    cudaMemcpy(parent_dev, parent_host.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();

    // Adjust weights and copy to device
    for (int i = 0; i < E; i++) {
        if (types[i] == "green") edges_host[i].weight *= 2;
        else if (types[i] == "traffic") edges_host[i].weight *= 5;
        else if (types[i] == "dept") edges_host[i].weight *= 3;
    }
    cudaMemcpy(edges_dev, edges_host.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);

    // Initialize MST weight and min_edge array
    cudaMemset(mst_weight_dev, 0, sizeof(unsigned long long));
    cudaMemset(min_edge_dev, -1, V * sizeof(int));

    int max_iterations = log2(V) + 1;

    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset minimum edges
        int blocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reset_min_edges<<<blocks, BLOCK_SIZE>>>(min_edge_dev, V);
        
        // Find minimum edges
        blocks = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;
        find_min_edges<<<blocks, BLOCK_SIZE>>>(edges_dev, E, parent_dev, min_edge_dev);
        
        // Merge components
        blocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
        merge_components<<<blocks, BLOCK_SIZE>>>(edges_dev, parent_dev, min_edge_dev, mst_weight_dev, V);
        
        cudaDeviceSynchronize();
    }

    // Get final MST weight
    unsigned long long mst_weight = 0;
    cudaMemcpy(&mst_weight, mst_weight_dev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // Output results
    cout << mst_weight % MOD << endl;
    // cout << elapsed.count() << " s\n";

    // Cleanup
    cudaFree(edges_dev);
    cudaFree(parent_dev);
    cudaFree(min_edge_dev);
    cudaFree(mst_weight_dev);

    return 0;
}
