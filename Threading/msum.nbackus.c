#include "msum.nbackus.h"
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>

#define NUMTHREADS 4
#define NUMELEMENTS 64000

double A[NUMELEMENTS];

void *doSum(void *data);

int main() {
    int i;
    double sum = 0, tsum = 0;
    // Init thread stuff
    pthread_t tid[NUMTHREADS];
    pthread_attr_t attr;

    // Init array
    for(i = 0; i < NUMELEMENTS; ++i) {
        A[i] = (double) drand48();
        sum += A[i];
    }

    // Create structs
    SumInfo sums[NUMTHREADS];
    sums[0].firstIndex = 0;
    sums[0].lastIndex = 16000;

    sums[1].firstIndex = 16001;
    sums[1].lastIndex = 32000;

    sums[2].firstIndex = 32001;
    sums[2].lastIndex = 48000;

    sums[3].firstIndex = 48001;
    sums[3].lastIndex = 64000;

    // Start threads
    for(i = 0; i < NUMTHREADS; ++i) {
        pthread_create(&tid[i], NULL, doSum, &sums[i]);
    }

    // Wait for finish
    for(i = 0; i < NUMTHREADS; ++i) {
        pthread_join(tid[i], NULL);
    }

    for(i = 0; i < NUMTHREADS; ++i) {
        tsum += sums[i].theSum;
        printf("\nThread %i Sum: %f\n", i, sums[i].theSum);
    }
    printf("\nThe Sum: %f\n", sum);
    printf("Thread Sum: %f\n", tsum);

    return 0;
}

void *doSum(void *data) {
    // Init suminfo
    SumInfo *info;

    info = (SumInfo *) data;
    info->theSum = 0;

    // Loop through indexes
    for(int i = info->firstIndex; i < info->lastIndex; ++i) {
        info->theSum += A[i];
    }

    pthread_exit(0);
}