/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

#define ANDY_SOLUTION	88

#ifdef ANDY_SOLUTION
__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals, int qnum)
{
	// this kernel will be running in 64 x 256 threads.
	// Each thread block has 256 threads running parallelly. It will in turn to generate 16 histograms value to the bin(one coarse bin).
	// Therefore, to avoid the race condition, we will allocate 256 x 16 x sizeof(unsigned int) buffer in shared memory to store the histogram value.

	__shared__ unsigned int s[4096];

	for (int i = threadIdx.x; i < 4096; i += blockDim.x)
	{
		s[i] = 0;
	}

	__syncthreads();


	unsigned int nBin;
	unsigned int nCoarseBin;		// first 4 of 10 bit is coarse bin id.
	int a = blockIdx.x / qnum;
	int b = blockDim.x*qnum;
	
	for (int i = (threadIdx.x + blockDim.x*((blockIdx.x) % qnum)); i < numVals; i += b)
	{
		nBin = vals[i];
		nCoarseBin = (nBin >> 4) & 0x3f;		// first 4 of 10 bit is coarse bin id.
		if (nCoarseBin == a)
		{
			s[threadIdx.x + (nBin & 0xf) * 256] ++;
		}
	}
	
	__syncthreads();
	
	if (threadIdx.x < 16)
	{
		unsigned int histoVal = 0;

		for (int i = 0; i < 256; i++)
		{
			histoVal += s[i + threadIdx.x * 256];
		}
		histo[(blockIdx.x/qnum) * 16 * qnum + threadIdx.x * qnum + (blockIdx.x%qnum)] = histoVal;
	}
	
}

/*
CUDA Device Query...
There are 1 CUDA devices.

CUDA Device #0
Major revision number:         5
Minor revision number:         0
Name:                          Quadro M2000
Total global memory:           0
Total shared memory per block: 49152
Total registers per block:     65536
Warp size:                     32
Maximum memory pitch:          2147483647
Maximum threads per block:     1024
Maximum dimension 0 of block:  1024
Maximum dimension 1 of block:  1024
Maximum dimension 2 of block:  64
Maximum dimension 0 of grid:   2147483647
Maximum dimension 1 of grid:   65535
Maximum dimension 2 of grid:   65535
Clock rate:                    1137000
Total constant memory:         65536
Texture alignment:             512
Concurrent copy and execution: Yes
Number of multiprocessors:     5
Kernel execution timeout:      No

Press any key to exit...*/

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
	yourHisto << <640, 256>> >(d_vals, d_histo, numElems, 10);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

#else

#define STRIDE 888
//#define OFFSET 888

__global__
void yourHisto(const unsigned int* const vals, //INPUT
unsigned int* const histo,      //OUPUT
int numVals)
{
	//TODO fill in this kernel to calculate the histogram
	//as quickly as possible

	//Although we provide only one kernel skeleton,
	//feel free to use more if it will help you
	//write faster code

	__shared__ unsigned int s[4096];

	for (int i = threadIdx.x; i < 4096; i += blockDim.x)
	{
		s[i] = 0;
	}

	__syncthreads();

	// all elements will be divided into 256 consecutive buffers. So the number of elements is scanned in a thread is numVals/256

	unsigned int nBin;
	unsigned int nCoarseBin;		// first 4 of 10 bit is coarse bin id.

#ifdef OFFSET
	unsigned int nElements = numVals / blockDim.x;

	for (int i = threadIdx.x*nElements; i < (threadIdx.x + 1)*nElements; i++)
#endif

#ifdef STRIDE
		for (int i = threadIdx.x; i < numVals; i += blockDim.x)
#endif
		{
		nBin = vals[i];
		nCoarseBin = (nBin >> 4) & 0x3f;		// first 4 of 10 bit is coarse bin id.
		if (nCoarseBin == blockIdx.x)
		{
			s[threadIdx.x + (nBin & 0xf) * 256] ++;
		}
		}

	__syncthreads();

	if (threadIdx.x < 16)
	{
		unsigned int histoVal = 0;

		for (int i = 0; i < 256; i++)
		{
			histoVal += s[i + threadIdx.x * 256];
		}
		histo[blockIdx.x * 16 + threadIdx.x] = histoVal;
	}

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
	unsigned int* const d_histo,      //OUTPUT
	const unsigned int numBins,
	const unsigned int numElems)
{
	//TODO Launch the yourHisto kernel
	yourHisto << <64, 256 >> >(d_vals, d_histo, numElems);

	//if you want to use/launch more than one kernel,
	//feel free
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

#endif