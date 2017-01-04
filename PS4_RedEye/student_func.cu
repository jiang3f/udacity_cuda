//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

//#define PS4_DEBUG

__global__ void CountElementsInBin( unsigned int* inputVal, 
									unsigned int* radix, 
									int shiftOff, 
									const size_t elements, 
									unsigned int* bin)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (myId >= elements)	return;
#ifdef PS4_DEBUG	
	unsigned int binId = (inputVal[myId] >> shiftOff) & 0x3;		// 4 bits radix
#else
	unsigned int binId = (inputVal[myId] >> shiftOff) & 0xff;		// 8 bits radix
#endif
	radix[myId] = binId;
	atomicAdd(&bin[binId], 1);
}

__global__ void WriteOutput(unsigned int* inputVal, 
							unsigned int* inputPos, 
							unsigned int* outputVal, 
							unsigned int* outputPos, 
							unsigned int* bin,
							unsigned int* radix,
							const size_t elements,
							const size_t numBin)
{
	const int myId = blockIdx.x * blockDim.x + threadIdx.x;

	if (myId >= numBin)	return;

	size_t count = 0;

	for (size_t i = 0; i < elements; i++)
	{
		if (radix[i] == myId)
		{
			outputVal[count + bin[myId]] = inputVal[i];
			outputPos[count + bin[myId]] = inputPos[i];
			count++;
		}
	}

}

__global__ void Blelloch_scan_FistHalf(unsigned int  *d_cdf, const size_t numBin, int step)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (myId >= numBin)	return;

	if ((myId % (2 << step)) == ((2 << step) - 1))
	{
		atomicAdd(&d_cdf[myId], d_cdf[myId - (1 << step)]);
	}
}

__global__ void Blelloch_scan_SecondHalf(unsigned int *d_cdf, const size_t numBin, int step)
{
	const int myId = blockIdx.x * blockDim.x + threadIdx.x;

	if (myId >= numBin)	return;

	if ((myId % (2 << step)) == ((2 << step) - 1))
	{
		unsigned int temp = d_cdf[myId];
		atomicAdd(&d_cdf[myId], d_cdf[myId - (1 << step)]);
		d_cdf[myId - (1 << step)] = temp;
	}
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
	//TODO
	//PUT YOUR SORT HERE
	const int numThreads = 192;

#ifdef PS4_DEBUG
	// Allocate memory for bins. Radix is 4 (2 bits).
	int radix_bits = 2;
#else
	// 8 bits
	int radix_bits = 8;
#endif
	int numBins = 1 << radix_bits;

	// Allocate memory for bins. Radix is 4 (2 bits).
	unsigned int  *d_bin;
	checkCudaErrors(cudaMalloc(&d_bin, sizeof(unsigned int) * numBins));
	checkCudaErrors(cudaMemset(d_bin, 0, sizeof(unsigned int) * numBins));
	
	unsigned int *d_radix;
	checkCudaErrors(cudaMalloc(&d_radix, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMemset(d_radix, 0, sizeof(unsigned int) * numElems));

	size_t elements = numElems;

#ifdef PS4_DEBUG
	unsigned int input[] = { 2, 8, 7, 4, 9, 1, 7, 10 };
	elements = 8;
	checkCudaErrors(cudaMemcpy(d_inputVals, input, elements * sizeof(unsigned int), cudaMemcpyHostToDevice));

	unsigned int *outputVal = new unsigned int[elements];
	unsigned int *outputPos = new unsigned int[elements];
	unsigned int *radix = new unsigned int[elements];
	unsigned int* bin = new unsigned int[numBins];

#endif

	int shiftoff = 0;
	int sortsteps = (sizeof(unsigned int) * 8) / radix_bits;

	for (int n = 0; n < sortsteps; n++)
	{
		checkCudaErrors(cudaMemset(d_bin, 0, sizeof(unsigned int) * numBins));

		if (n % 2 == 0)
		{
			CountElementsInBin << <(numElems + numThreads - 1) / numThreads, numThreads >> > (d_inputVals, d_radix, shiftoff, elements, d_bin);
		}
		else
		{
			CountElementsInBin << <(numElems + numThreads - 1) / numThreads, numThreads >> > (d_outputVals, d_radix, shiftoff, elements, d_bin);
		}
		//		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#ifdef PS4_DEBUG
		cudaMemset(&d_bin, 2, 4 * sizeof(unsigned int));
		checkCudaErrors(cudaMemcpy(radix, d_radix, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(bin, d_bin, numBins* sizeof(unsigned int), cudaMemcpyDeviceToHost));

#endif
		int step = 0;
		int bins = numBins;

		while (bins > 1)
		{
			bins = bins >> 1;
			step++;
		}

		for (int i = 0; i < step; i++)
		{
			Blelloch_scan_FistHalf << <1, numBins >> >(d_bin, numBins, i);
		}

		//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
#ifdef PS4_DEBUG
		checkCudaErrors(cudaMemcpy(bin, d_bin, numBins* sizeof(unsigned int), cudaMemcpyDeviceToHost));
#endif

		checkCudaErrors(cudaMemset(&d_bin[numBins - 1], 0, sizeof(unsigned int)));

		for (int i = (step - 1); i >= 0; i--)

		{
			Blelloch_scan_SecondHalf << <1, numBins >> >(d_bin, numBins, i);
		}

#ifdef PS4_DEBUG
		//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpy(bin, d_bin, numBins* sizeof(unsigned int), cudaMemcpyDeviceToHost));
#endif
		if (n % 2 == 0)
		{
			WriteOutput << <1, numBins >> > (d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_bin, d_radix, elements, numBins);
		}
		else
		{
			WriteOutput << <1, numBins >> > (d_outputVals, d_outputPos, d_inputVals, d_inputPos, d_bin, d_radix, elements, numBins);
		}
		//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#ifdef PS4_DEBUG
		if (n % 2 == 0)
		{
			checkCudaErrors(cudaMemcpy(outputVal, d_outputVals, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(outputPos, d_outputPos, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		}
		else
		{
			checkCudaErrors(cudaMemcpy(outputVal, d_inputVals, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(outputPos, d_inputPos, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		}
#endif
		shiftoff += radix_bits;
	}

	if (sortsteps % 2 == 0)
	{
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, elements * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, elements * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	}
	checkCudaErrors(cudaFree(d_bin));
	checkCudaErrors(cudaFree(d_radix));

#ifdef PS4_DEBUG
	delete outputVal;
	delete outputPos;
	delete radix;
	delete bin;
#endif

}
