%%writefile main.cu

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>
using namespace std;

#define INF 1000000000LL
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CUDA kernels
__global__ void initializeDistanceMatrix(long long *dist, long long *nxt, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * n;
    if (idx < total) {
        long long i = idx / n, j = idx % n;
        if (i == j) {
            dist[idx] = 0;
            nxt[idx]  = i;
        } else {
            dist[idx] = INF;
            nxt[idx]  = -1;
        }
    }
}

__global__ void updateDistanceWithRoads(int *roads, long long m, long long *dist, long long *nxt, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        long long u = roads[4*i + 0];
        long long v = roads[4*i + 1];
        long long w = roads[4*i + 2];
        long long idx_uv = u * n + v;
        long long idx_vu = v * n + u;
        
        // Use atomicMin to ensure race condition safety
        unsigned long long old_uv = atomicMin((unsigned long long*)&dist[idx_uv], (unsigned long long)w);
        if ((unsigned long long)w < old_uv) nxt[idx_uv] = v;
        
        unsigned long long old_vu = atomicMin((unsigned long long*)&dist[idx_vu], (unsigned long long)w);
        if ((unsigned long long)w < old_vu) nxt[idx_vu] = u;
    }
}

__global__ void floydWarshallStep(long long *dist, long long *nxt, long long n, long long k) {
    long long i = blockIdx.y * blockDim.y + threadIdx.y;
    long long j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        long long ij = i*n + j;
        long long ik = i*n + k;
        long long kj = k*n + j;
        
        long long d = dist[ik] + dist[kj];
        if (d < dist[ij]) {
            dist[ij] = d;
            nxt[ij]  = nxt[ik];
        }
    }
}

// Kernel for parallel path finding
__global__ void findNearestShelterKernel(
    long long *dist, long long N, 
    long long *cities, int num_cities,
    long long *shelter_cities, long long *shelter_capacities, int num_shelters,
    long long *best_shelters, long long *best_distances, long long *best_capacities
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cities) {
        long long city_id = cities[idx];
        long long best = -1, bd = INF;
        long long cap = 0;
        
        for (int s = 0; s < num_shelters; ++s) {
            long long sc = shelter_cities[s];
            long long d = dist[city_id * N + sc];
            if (d < bd) {
                bd = d;
                best = sc;
                cap = shelter_capacities[s];
            }
        }
        
        best_shelters[idx] = best;
        best_distances[idx] = bd;
        best_capacities[idx] = cap;
    }
}

// Kernel to check if city is a shelter
__global__ void checkIfShelterKernel(
    long long *cities, int num_cities,
    long long *shelter_cities, long long *shelter_capacities, int num_shelters,
    bool *is_shelter, long long *shelter_caps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cities) {
        long long city_id = cities[idx];
        bool found = false;
        long long cap = 0;
        
        for (int s = 0; s < num_shelters; ++s) {
            if (shelter_cities[s] == city_id) {
                found = true;
                cap = shelter_capacities[s];
                break;
            }
        }
        
        is_shelter[idx] = found;
        shelter_caps[idx] = cap;
    }
}

// Host helpers
vector<long long> reconstructPath(long long *nxt, long long s, long long t, long long n) {
    vector<long long> p;
    if (nxt[s*n + t] < 0) return p;
    p.push_back(s);
    while (s != t) {
        s = nxt[s*n + t];
        p.push_back(s);
    }
    return p;
}

bool canElderlyReach(const vector<long long>& path, int *roads, long long m, long long maxd) {
    long long tot = 0;
    for (size_t i = 0; i+1 < path.size(); ++i) {
        long long u = path[i], v = path[i+1];
        // find that segment
        for (long long r = 0; r < m; ++r) {
            long long ru = roads[4*r+0], rv = roads[4*r+1];
            if ((ru==u && rv==v) || (ru==v && rv==u)) {
                tot += roads[4*r+2];
                break;
            }
        }
        if (tot > maxd) return false;
    }
    return true;
}

struct Drop { long long city, prime_age, elderly; };
struct Path { vector<long long> cities; vector<Drop> drops; };

// Parallel implementation of evacuation simulation
vector<Path> simulateEvacuation(
    int *roads, long long m,
    long long *shelter_city, long long *shelter_capacity, int S,
    long long *city, long long *pop, int P,
    long long max_distance_elderly,
    long long *dist, long long *nxt, long long N
) {
    vector<Path> out(P);
    
    // Arrays for parallel kernel results
    bool *is_shelter_host = new bool[P];
    long long *shelter_caps_host = new long long[P];
    long long *best_shelters_host = new long long[P];
    long long *best_distances_host = new long long[P];
    long long *best_capacities_host = new long long[P];
    
    // GPU memory allocation
    bool *d_is_shelter;
    long long *d_shelter_caps;
    long long *d_city, *d_shelter_city, *d_shelter_capacity;
    long long *d_best_shelters, *d_best_distances, *d_best_capacities;
    
    cudaMalloc(&d_is_shelter, P * sizeof(bool));
    cudaMalloc(&d_shelter_caps, P * sizeof(long long));
    cudaMalloc(&d_city, P * sizeof(long long));
    cudaMalloc(&d_shelter_city, S * sizeof(long long));
    cudaMalloc(&d_shelter_capacity, S * sizeof(long long));
    cudaMalloc(&d_best_shelters, P * sizeof(long long));
    cudaMalloc(&d_best_distances, P * sizeof(long long));
    cudaMalloc(&d_best_capacities, P * sizeof(long long));
    
    // Copy data to GPU
    cudaMemcpy(d_city, city, P * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelter_city, shelter_city, S * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelter_capacity, shelter_capacity, S * sizeof(long long), cudaMemcpyHostToDevice);
    
    // Launch parallel kernels
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((P + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    checkIfShelterKernel<<<gridSize, blockSize>>>(
        d_city, P, 
        d_shelter_city, d_shelter_capacity, S,
        d_is_shelter, d_shelter_caps
    );
    
    findNearestShelterKernel<<<gridSize, blockSize>>>(
        dist, N,
        d_city, P,
        d_shelter_city, d_shelter_capacity, S,
        d_best_shelters, d_best_distances, d_best_capacities
    );
    
    // Copy results back to host
    cudaMemcpy(is_shelter_host, d_is_shelter, P * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(shelter_caps_host, d_shelter_caps, P * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_shelters_host, d_best_shelters, P * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_distances_host, d_best_distances, P * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_capacities_host, d_best_capacities, P * sizeof(long long), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_is_shelter);
    cudaFree(d_shelter_caps);
    cudaFree(d_city);
    cudaFree(d_shelter_city);
    cudaFree(d_shelter_capacity);
    cudaFree(d_best_shelters);
    cudaFree(d_best_distances);
    cudaFree(d_best_capacities);
    
    // Process results
    for (int i = 0; i < P; ++i) {
        long long ci = city[i];
        long long pa = pop[2*i+0], el = pop[2*i+1];
        Path path;
        
        if (is_shelter_host[i]) {
            // Current city is a shelter
            path.cities.push_back(ci);
            long long cap = shelter_caps_host[i];
            long long dropE = min(el, cap);
            long long dropP = min(pa, cap - dropE);
            path.drops.push_back({ci, dropP, dropE});
        } else {
            // Need to evacuate to nearest shelter
            long long best = best_shelters_host[i];
            long long cap = best_capacities_host[i];
            
            if (best < 0) {
                path.cities.push_back(ci);
                path.drops.push_back({ci, pa, el});
            } else {
                auto sp = reconstructPath(nxt, ci, best, N);
                path.cities = sp;
                
                // Determine furthest point elderly can reach
                long long edrop = ci;
                for (size_t j = 0; j < sp.size(); ++j) {
                    vector<long long> sub(sp.begin(), sp.begin()+j+1);
                    if (canElderlyReach(sub, roads, m, max_distance_elderly)) {
                        edrop = sp[j];
                    } else {
                        break;
                    }
                }
                
                // elderly drop
                if (el > 0) {
                  path.drops.push_back({ edrop, 0, el });
                }
                
                // prime-age drop
                if (pa > 0) {
                  long long dropE = (edrop == best ? el : 0);
                  long long avail = cap - dropE;
                  long long dropP = std::min(pa, avail);
                  path.drops.push_back({ best, dropP, dropE });
                }
            }
        }
        out[i] = path;
    }
    
    delete[] is_shelter_host;
    delete[] shelter_caps_host;
    delete[] best_shelters_host;
    delete[] best_distances_host;
    delete[] best_capacities_host;
    
    return out;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]); // Read input file from command-line argument
    if (!infile) {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }
    
    long long num_cities;
    infile >> num_cities;

    long long num_roads;
    infile >> num_roads;

    int *roads = new int[num_roads * 4]; 
    for (int i = 0; i < num_roads; i++) {
        infile >> roads[4*i] >> roads[4*i+1] >> roads[4*i+2] >> roads[4*i+3];
    }

    int num_shelters;
    infile >> num_shelters;

    long long *shelter_city     = new long long[num_shelters];
    long long *shelter_capacity = new long long[num_shelters];
    for (int i = 0; i < num_shelters; i++) {
        infile >> shelter_city[i] >> shelter_capacity[i];
    }

    int num_populated_cities;
    infile >> num_populated_cities;

    long long *city = new long long[num_populated_cities];
    long long *pop  = new long long[num_populated_cities * 2];
    for (long long i = 0; i < num_populated_cities; i++) {
        infile >> city[i] >> pop[2*i] >> pop[2*i+1];
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;
    infile.close();

    // set your answer to these variables
    long long *path_size    = new long long[num_populated_cities];
    long long **paths       = new long long*[num_populated_cities];
    long long *num_drops    = new long long[num_populated_cities];
    long long ***drops      = new long long**[num_populated_cities];

    auto timeStart = chrono::high_resolution_clock::now(); // Start timing

    // ----- CUDA-accelerated all-pairs shortest paths -----

    long long N = num_cities, M = num_roads;
    long long *d_dist, *d_nxt;
    int     *d_roads;
    
    // Use cudaMallocHost for pinned memory to optimize data transfer
    long long *h_dist, *h_nxt;
    cudaMallocHost(&h_dist, N*N*sizeof(long long));
    cudaMallocHost(&h_nxt, N*N*sizeof(long long));
    
    cudaMalloc(&d_dist, N*N*sizeof(long long));
    cudaMalloc(&d_nxt,  N*N*sizeof(long long));
    cudaMalloc(&d_roads, M*4*sizeof(int));
    
    // Use asynchronous copy for better performance
    cudaMemcpyAsync(d_roads, roads, M*4*sizeof(int), cudaMemcpyHostToDevice);

    // Phase 1: Initialize distance matrix
    dim3 b1(BLOCK_SIZE), g1((N*N + BLOCK_SIZE-1)/BLOCK_SIZE);
    initializeDistanceMatrix<<<g1,b1>>>(d_dist, d_nxt, N);
    
    // Phase 2: Update with road information
    dim3 g2((M + BLOCK_SIZE-1)/BLOCK_SIZE);
    updateDistanceWithRoads<<<g2,b1>>>(d_roads, M, d_dist, d_nxt, N);

    // Phase 3: Floyd-Warshall algorithm - optimize block size for 2D grid
    dim3 b2(16,16), g3((N+15)/16,(N+15)/16);
    for (long long k = 0; k < N; ++k) {
        floydWarshallStep<<<g3,b2>>>(d_dist, d_nxt, N, k);
    }

    // Asynchronously copy results back to host
    cudaMemcpyAsync(h_dist, d_dist, N*N*sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(h_nxt,  d_nxt,  N*N*sizeof(long long), cudaMemcpyDeviceToHost);
    
    // Synchronize to ensure copy is complete
    cudaDeviceSynchronize();
    
    // Free device memory early to make room for evacuation simulation
    cudaFree(d_dist);
    cudaFree(d_nxt);
    cudaFree(d_roads);

    // simulate evacuation on host with parallel components
    vector<Path> evac = simulateEvacuation(
        roads, M,
        shelter_city, shelter_capacity, num_shelters,
        city, pop, num_populated_cities,
        max_distance_elderly,
        h_dist, h_nxt, N
    );

    // Build output arrays in parallel where possible
    for (int i = 0; i < num_populated_cities; ++i) {
        auto &P = evac[i];
        path_size[i] = P.cities.size();
        paths[i] = new long long[P.cities.size()];
        for (size_t j = 0; j < P.cities.size(); ++j)
            paths[i][j] = P.cities[j];

        num_drops[i] = P.drops.size();
        drops[i] = new long long*[P.drops.size()];
        for (size_t j = 0; j < P.drops.size(); ++j) {
            drops[i][j] = new long long[3];
            drops[i][j][0] = P.drops[j].city;
            drops[i][j][1] = P.drops[j].prime_age;
            drops[i][j][2] = P.drops[j].elderly;
        }
    }
    
    // Free pinned memory
    cudaFreeHost(h_dist);
    cudaFreeHost(h_nxt);
    // ------------------------------------------------------

    auto timeEnd = chrono::high_resolution_clock::now(); 
    chrono::duration<double> timeElapsed = timeEnd - timeStart;

    // cout << timeElapsed.count() << " s\n";

    ofstream outfile(argv[2]);
    if (!outfile) {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }
    
    for(long long i = 0; i < num_populated_cities; i++){
        for(long long j = 0; j < path_size[i]; j++){
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    for(long long i = 0; i < num_populated_cities; i++){
        for(long long j = 0; j < num_drops[i]; j++){
            for(int k = 0; k < 3; k++){
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }
    
    // Clean up
    delete[] roads;
    delete[] shelter_city;
    delete[] shelter_capacity;
    delete[] city;
    delete[] pop;
    
    for(long long i = 0; i < num_populated_cities; i++) {
        delete[] paths[i];
        for(long long j = 0; j < num_drops[i]; j++) {
            delete[] drops[i][j];
        }
        delete[] drops[i];
    }
    
    delete[] path_size;
    delete[] paths;
    delete[] num_drops;
    delete[] drops;

    return 0;
}
