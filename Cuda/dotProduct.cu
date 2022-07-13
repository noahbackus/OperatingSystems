// Noah Backus and Cameron Brown
// CS222 Program 4
// 5/6/21

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 65536
#define BLOCK_SIZE 256

__global__ void dotp( float *U, float *V, float *partialSum, int n ) {

        __shared__ float localCache[BLOCK_SIZE];
        int tidx = threadIdx.x;
        localCache[tidx] = U[tidx] * V[tidx];

        int cacheIndex = threadIdx.x;
        int i = blockDim.x / 2;
        while (i > 0) {
                if (cacheIndex < i)
                        localCache[cacheIndex] = localCache[cacheIndex] + localCache[cacheIndex + i];
                __syncthreads();
                i = i / 2;
        }

        if (cacheIndex == 0)
                partialSum[blockIdx.x] = localCache[cacheIndex];

}

int main() {

        struct timeval t1, t2, t3, t4;
        float elaspedTimeMem, elaspedTimeNoMem, elaspedTimeCpu;

        float U[N], V[N];
        float *dev_U, *dev_V, *dev_partialSum;

        float error = 0.0;

        int numBlocks = 256;
        int threadsPerBlock = 256;

        float partialSum[numBlocks];

        cudaMalloc( (void **) &dev_U, N*sizeof(float) );
        cudaMalloc( (void **) &dev_V, N*sizeof(float) );
        cudaMalloc( (void **) &dev_partialSum, numBlocks*sizeof(float) );

        //U = (float *) malloc(N * sizeof(float));
        //V = (float *) malloc(N * sizeof(float));
        //partialSum = (float *) malloc(numBlocks * sizeof(float));
        for (int i=0; i<N; ++i) {
                U[i] = drand48();
                V[i] = drand48();
                error += U[i] * V[i];
        }

        gettimeofday(&t1, NULL);

        cudaMemcpy( dev_U, U, N*sizeof(float), cudaMemcpyHostToDevice );
        cudaMemcpy( dev_V, V, N*sizeof(float), cudaMemcpyHostToDevice );

        gettimeofday(&t2, NULL);

        dotp<<<numBlocks, threadsPerBlock>>>( dev_U, dev_V, dev_partialSum, N );

        cudaDeviceSynchronize(); // wait for GPU threads to complete; again, not necessary but good pratice
        cudaMemcpy( partialSum, dev_partialSum, numBlocks*sizeof(float), cudaMemcpyDeviceToHost );

        gettimeofday(&t3, NULL);

        // finish up on the CPU side
        float gpuResult = 0.0;
        for (int i=0; i<numBlocks; ++i)
                gpuResult = gpuResult + partialSum[i];

        gettimeofday(&t4, NULL);

        float finalError = 0.0;

        if (error > 0){
                finalError = fabs(error - gpuResult) / fabs(error);
                printf("Error: %.10f\n", finalError);
        }

        elaspedTimeMem = (t3.tv_sec - t1.tv_sec) * 1000.0;
        elaspedTimeMem += (t3.tv_usec - t1.tv_usec) / 1000.0;

        elaspedTimeNoMem = (t3.tv_sec - t2.tv_sec) * 1000.0;
        elaspedTimeNoMem += (t3.tv_usec - t2.tv_usec) / 1000.0;

        elaspedTimeCpu = (t4.tv_sec - t3.tv_sec) * 1000.0;
        elaspedTimeCpu += (t4.tv_usec - t3.tv_usec) / 1000.0;

        printf("GPU Result: %.10f\n", gpuResult);
        printf("Error Value: %.10f\n", error);

        printf("GPU with Memcpy: %f ms\n", elaspedTimeMem);
        printf("GPU no Memcpy: %f ms\n", elaspedTimeNoMem);
        printf("CPU: %f ms\n", elaspedTimeCpu);

        return 0;

}
