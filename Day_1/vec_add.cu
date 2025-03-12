#include <iostream>
#include <cuda_runtime.h>
using namespace std;


__global__ 
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float A[3] = {1.0, 2.0, 3.0};
    float B[3] = {1.0, 2.0, 3.0};
    float C[3] = {};

    float* Ap = A;
    float* Bp = B;
    float* Cp = C;

    int n = 3;
    int size = n * sizeof(float);

    float *A_d, *B_d, *C_d;
    
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, Ap, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, Bp, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/32.0), 32>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cout << "[" ;
    for (int i=0; i < n; i++) {
        cout << C[i];
        if (i < n - 1){
            cout << ", ";
        }
    }
    cout << "]" << "\n";

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
