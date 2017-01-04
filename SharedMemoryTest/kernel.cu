#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <algorithm>

//#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <sys/time.h>
#endif

__global__ void staticReverse(unsigned int *d, int n)
{
	__shared__ unsigned int s[64];
	int t = threadIdx.x;
	int tr = n - t - 1;
	s[t] = d[t];
	__syncthreads();
	d[t] = s[tr];
}

int main()
{
    const int arraySize = 5;
    unsigned int a[arraySize] = { 1, 2, 3, 4, 5 };

	unsigned int *d_vals;

	cudaMalloc(&d_vals, sizeof(unsigned int) * arraySize);
	cudaMemcpy(d_vals, a, sizeof(unsigned int) * arraySize, cudaMemcpyHostToDevice);

    printf("{1,2,3,4,5}  = {%d,%d,%d,%d,%d}\n",
        a[0], a[1], a[2], a[3], a[4]);

	staticReverse << <1, 5>> >(d_vals, arraySize);
	
	cudaMemcpy(a, d_vals, sizeof(unsigned int) * arraySize, cudaMemcpyDeviceToHost);
	printf("after {1,2,3,4,5}  = {%d,%d,%d,%d,%d}\n",
		a[0], a[1], a[2], a[3], a[4]);

    return 0;
}
