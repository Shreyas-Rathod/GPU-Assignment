#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    int H = blockIdx.y * blockDim.y + threadIdx.y;
    int W = blockIdx.x * blockDim.x + threadIdx.x;
    int K = blockIdx.z;

    if (H < h && W < w && K < k)
    {
        long int sum = 0;
        __shared__ long int sharedFilter[4096];
        int rsc = r*s*c;

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int i = 0; i < rsc; i++) sharedFilter[i] = filter[(K * rsc) + i];
        }
        __syncthreads();

        for (int C = 0; C < c; C++)
        {
            for (int R = 0; R < r; R++)
            {
                for (int S = 0; S < s; S++)
                {
                    int matrix_row = H + R - r / 2;
                    int matrix_col = W + S - s / 2;

                    if (matrix_row >= 0 && matrix_row < h && matrix_col >= 0 && matrix_col < w) 
                      sum += matrix[(C * h + matrix_row) * w + matrix_col] * sharedFilter[(C * r * s) + (R * s) + S];
                }
            }
        }
        result[(K * h + H) * w + W] = sum;
    }
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * cf * k];
    for (long int i = 0; i < r * s * cf * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    auto start = std::chrono::high_resolution_clock::now();

    long int *d_matrix, *d_filter, *d_result;
    cudaMalloc(&d_matrix, h * w * c * sizeof(long int));
    cudaMalloc(&d_filter, r * s * cf * k * sizeof(long int));
    cudaMalloc(&d_result, h * w * k * sizeof(long int));

    cudaMemcpy(d_matrix, h_mat, h * w * c * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, r * s * cf * k * sizeof(long int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y, k);

    dkernel<<<gridDim, blockDim>>>(d_matrix, d_filter, d_result, h, w, c, r, s, k);

    cudaMemcpy(h_ans, d_result, h * w * k * sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_filter);
    cudaFree(d_result);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;

    cudaDeviceSynchronize();

    std::ofstream file("cuda.out");
    if (file.is_open()){
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else{
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open()){
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else{
        std::cout << "Unable to open file";
    }
// to print output
/* 
    for (long int i = 0; i < h * k; i++){
        for (long int j = 0; j < w; j++){
            cout << h_ans[i * w + j] << " ";
        }
        cout << "\n";
    }
*/
    return 0;
}
