// Noah Backus and Cameron Brown
// CS222 Program 4
// 5/6/21

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 16

// matrix multiplication kernel, uses GPU to optimize multiplication
__global__ void mxmKernel( float *A, float *B, float *C, size_t pitch, int n) {
        float sum = 0.0;
        int col = threadIdx.x + blockDim.x * blockIdx.x;
        int row = threadIdx.y + blockDim.y * blockIdx.y;
        int numEltsPerRow = pitch / sizeof(float); // #elements in each padded row
        int k;
        if (col < N && row < N) {
                sum = 0.0;
                for (k = 0; k < N; k++)
                        sum += A[row * numEltsPerRow + k] * B[k * numEltsPerRow + col];
                C[row * numEltsPerRow + col] = sum;
        }
}

// function multiplies two given matrices on the CPU
void matrixMult(float matA[N][N], float matB[N][N], float matProduct[N][N]) {
    int i, j, k;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            matProduct[i][j] = 0;
            for (k = 0; k < N; ++k)
                matProduct[i][j] += matA[i][k] * matB[k][j];
        }
    }
}


int main() {

        float *d_A, *d_B, *d_C;
        float h_A[N][N], h_B[N][N], h_C[N][N];
        size_t pitch;

        cudaMallocPitch((void**)&d_A, &pitch, N*sizeof(float), N);
        cudaMallocPitch((void**)&d_B, &pitch, N*sizeof(float), N);
        cudaMallocPitch((void**)&d_C, &pitch, N*sizeof(float), N);

        // populate matrices with random values from 0 to 1
        float newAVal, newBVal;
        float expectedMatrix[N][N];
        for (int i = 0; i<N; ++i) {
                for (int j = 0; j < N; ++j) {
                        newAVal = drand48();
                        newBVal = drand48();
                        h_A[i][j] = newAVal;
                        h_B[i][j] = newBVal;
                }
        }

        // perform matrix multiplication on cpu side
        matrixMult(h_A, h_B, expectedMatrix);

        // copy host matrices from cpu to gpu
        cudaMemcpy2D(d_A, pitch, h_A, N*sizeof(float), N*sizeof(float), N, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_B, pitch, h_B, N*sizeof(float), N*sizeof(float), N, cudaMemcpyHostToDevice);

        dim3 dimGrid(1, 1);
        dim3 dimBlock(N, N);
        mxmKernel<<<dimGrid, dimBlock>>>( d_A, d_B, d_C, pitch, N);

        cudaMemcpy2D(h_C, N*sizeof(float), d_C, pitch, N*sizeof(float), N, cudaMemcpyDeviceToHost);

        float total = 0;

        for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                        float error = fabs(expectedMatrix[i][j] - h_C[i][j]) / fabs(expectedMatrix[i][j]);

                        total += error;

                }
        }

        float totalError = total / (N * N);

        // print product matrix
        for (int i = 0; i < N; ++i) {
                for(int j = 0; j < N; ++j) {
                        printf("  %-5f", h_C[i][j]);
                }
                printf("\n");
        }

        printf("\n");
        // print expected product matrix
        for (int i = 0; i < N; ++i) {
                for(int j = 0; j < N; ++j) {
                        printf("  %-5f", expectedMatrix[i][j]);
                }
                printf("\n");
        }

        printf("\nAverage Error: %-15f \n", totalError);
        return 0;
}
