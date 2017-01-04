#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string>
#include "gputimer.h"

int compare(float* h_in, float* h_out, float* h_out_shared, float* h_cmp, int ARRAY_SIZE){
	int failure = 0;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (h_out[i] != h_cmp[i]) {
			fprintf(stderr, "ERROR: h_in[%d] is %f, h_out[%d] is %f, h_cmp[%d] is %f\n",
				i, h_in[i], i, h_out[i], i, h_cmp[i]);
			failure = 1;
		}
		if (h_out_shared[i] != h_cmp[i]) {
			fprintf(stderr, "ERROR: h_in[%d] is %f, h_out_shared[%d] is %f, h_cmp[%d] is %f\n",
				i, h_in[i], i, h_out_shared[i], i, h_cmp[i]);
			failure = 1;
		}
	}

	if (failure == 0)
	{
		printf("Success! Your smooth code worked!\n");
	}

	return failure;
}

// Reference
// Your code executed in 0.010048 ms
__global__ void smooth(float * v_new, const float * v) {
	int myIdx = threadIdx.x * gridDim.x + blockIdx.x;
	int numThreads = blockDim.x * gridDim.x;
	int myLeftIdx = (myIdx == 0) ? 0 : myIdx - 1;
	int myRightIdx = (myIdx == (numThreads - 1)) ? numThreads - 1 : myIdx + 1;
	float myElt = v[myIdx];
	float myLeftElt = v[myLeftIdx];
	float myRightElt = v[myRightIdx];
	v_new[myIdx] = 0.25f * myLeftElt + 0.5f * myElt + 0.25f * myRightElt;
}

//
// Your code
// Your code executed in 0.012512 ms
// udacity server
__global__ void smooth_shared(float * v_new, const float * v) {
	extern __shared__ float s[];
	// TODO: Complete the rest of this function
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	s[threadIdx.x + 1] = v[myId];

	if (threadIdx.x == 0)	{
		if (myId == 0)	s[0] = v[0];
		else
		{
			s[0] = v[myId - 1];
		}
	}
	else if (threadIdx.x == blockDim.x - 1)
	{
		if (myId == 4095)	s[threadIdx.x + 2] = v[4095];
		else
		{
			s[threadIdx.x + 2] = v[myId + 1];
		}
	}

	__syncthreads();

	v_new[myId] = s[threadIdx.x] * 0.25f + s[threadIdx.x + 1] * 0.5f + s[threadIdx.x + 2] * 0.25f;


} 
 
int main(int argc, char **argv)
{

	const int ARRAY_SIZE = 4096;
	const int BLOCK_SIZE = 256;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	float h_cmp[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];
	float h_out_shared[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		// generate random float in [0, 1]
		h_in[i] = (float)rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_cmp[i] = (0.25f * h_in[(i == 0) ? 0 : i - 1] +
			0.50f * h_in[i] +
			0.25f * h_in[(i == (ARRAY_SIZE - 1)) ? ARRAY_SIZE - 1 : i + 1]);
	}

	// declare GPU memory pointers
	float * d_in, *d_out, *d_out_shared;

	// allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);
	cudaMalloc((void **)&d_out_shared, ARRAY_BYTES);

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// launch the kernel 
	smooth << <ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE >> >(d_out, d_in);
	GpuTimer timer;
	timer.Start();
	smooth_shared << <ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE, (BLOCK_SIZE + 2) * sizeof(float) >> >(d_out_shared, d_in);
	//smooth << <ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE, (BLOCK_SIZE + 2) * sizeof(float) >> >(d_out_shared, d_in);
	timer.Stop();

	printf("Your code executed in %g ms\n", timer.Elapsed());
	// cudaEventSynchronize(stop);
	// float elapsedTime;
	// cudaEventElapsedTime(&elapsedTime, start, stop);    

	// copy back the result from GPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out_shared, d_out_shared, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// testing for correctness
	compare(h_in, h_out, h_out_shared, h_cmp, ARRAY_SIZE);

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_out_shared);
}
