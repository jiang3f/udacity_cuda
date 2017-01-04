
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 1

__global__ void YourKernel(int d_A[BLOCK_SIZE][BLOCK_SIZE], int d_B[BLOCK_SIZE][BLOCK_SIZE])
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//if (row >= h || col >= w)	return;
	/* whatever you wanna do with d_A[][] and d_B[][] */
}
int main()
{
	int d_A[BLOCK_SIZE][BLOCK_SIZE];
	int d_B[BLOCK_SIZE][BLOCK_SIZE];

	/* d_A initialization */

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	dim3 dimGrid(GRID_SIZE, GRID_SIZE); // 1*1 blocks in a grid

	YourKernel <<<dimGrid, dimBlock >>>(d_A, d_B); //Kernel invocation
}