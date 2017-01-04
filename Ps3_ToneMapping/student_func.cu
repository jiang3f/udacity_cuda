/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/
/*
result:
	Your code compiled! 
	Your code printed the following output: 
	Your code executed in 0.568768 ms 
	
	Good job!. 
	Your image matched perfectly to the reference image 
*/
#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void get_maxmin_1d(const float * d_in, float * d_out, int len, const int step, bool ismax)
{
	int pos;
	pos = threadIdx.x;

	if ((pos % (2 << step)) == 0)
	{
		d_out[pos] = d_in[pos];

		if ((pos + (2 << step) - 1) < len)
		{
			if ((ismax && (d_in[pos] < d_in[pos + (2 << step) - 1])) ||
				(!ismax && (d_in[pos] > d_in[pos + (2 << step) - 1])))
			{
				d_out[pos] = d_in[pos + (2 << step) - 1];
			}
		}
	}


}


__global__ void get_maxmin_2d(const float * d_in, float * d_out, const size_t numRows, const size_t numCols, const int step, bool ismax)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	if ((thread_1D_pos % (2 << step)) == 0)
	{
		d_out[thread_1D_pos] = d_in[thread_1D_pos];

		if ((thread_1D_pos + (2 << step) - 1) < numCols * numRows)
		{
			if ((ismax && (d_in[thread_1D_pos] < d_in[thread_1D_pos + (2 << step) - 1])) ||
				(!ismax && (d_in[thread_1D_pos] > d_in[thread_1D_pos + (2 << step) - 1])))
			{
				d_out[thread_1D_pos] = d_in[thread_1D_pos + (2 << step) - 1];
			}
		}
	}

	
}

__global__ void get_max(float * d_in, int len, const int step)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	//if (myId >= len)	return;
	if ((myId + (1 << step)) >= len)	return;

	if ((myId % (2 << step)) == 0)
	{
		float temp = d_in[myId + (1 << step)];
		if (temp > d_in[myId])
		{
			d_in[myId] = temp;
		}
	}
}

__global__ void get_min(float * d_in, int len, const int step)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	//if (myId >= len)	return;
	if ((myId + (1 << step)) >= len)	return;

	if ((myId % (2 << step)) == 0)
	{
		float temp = d_in[myId + (1 << step)];
		if (temp < d_in[myId])
		{
			d_in[myId] = temp;
		}
	}
}

__global__ void histogram(const float * d_in, unsigned int* const d_cdf, float lumMin, float lumRang, const size_t len, const size_t numBins)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (myId >= len)	return;

	int bin = (d_in[myId] - lumMin) / lumRang * numBins;

	atomicAdd(&d_cdf[bin], 1);

}

__global__ void Blelloch_scan_FistHalf(unsigned int *d_cdf, const size_t numBin, int step)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (myId >= numBin)	return;

	if ((myId % (2<<step)) == ((2<<step) - 1))
	{
		atomicAdd(&d_cdf[myId], d_cdf[myId - (1<<step)]);
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

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
#if 0
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
  */
	size_t numPixels = numCols * numRows;

	size_t channelSize = sizeof(float) * numPixels;
	
	int step = 0;
	while (numPixels > 1)
	{
		numPixels = numPixels >> 1;
		step++;
	}

	for (int i = 0; i < step; i++)
	{
		get_maxmin << <gridSize, blockSize >> >(d_logLuminance, d_maxmin, numRows, numCols, i, true);
	}

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&max_logLum, d_maxmin, sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < step; i++)
	{
		get_maxmin << <gridSize, blockSize >> >(d_logLuminance, d_maxmin, numRows, numCols, i, false);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&min_logLum, d_maxmin, sizeof(float), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	/*
		2) subtract them to find the range
	*/
	float lumRange = max_logLum - min_logLum;

	/*
	3) generate a histogram of all the values in the logLuminance channel using
		the formula : bin = (lum[i] - lumMin) / lumRange * numBins
	*/
	histogram << <gridSize, blockSize >> >(d_logLuminance, d_cdf, min_logLum, lumRange, numRows, numCols, numBins);

	/*
	4) Perform an exclusive scan(prefix sum) on the histogram to get
		the cumulative distribution of luminance values(this should go in the
		incoming d_cdf pointer which already has been allocated for you)       * /
	*/

	step = 0;
	int bins = numBins;

	while (bins > 1)
	{
		bins = bins >> 1;
		step ++;
	}

	for (int i = 0; i < step; i++)
	{
		Blelloch_scan_FistHalf << <1, numBins >> >(d_cdf, numBins, i);
	}

	checkCudaErrors(cudaMemset(&d_cdf[numBins - 1], 0, sizeof(unsigned int)));

	for (int i = step; i > 0; i--)
	{
		Blelloch_scan_SecondHalf << <1, numBins>> >(d_cdf, numBins, i);
	}

	checkCudaErrors(cudaFree(d_maxmin));
#else
	const int numThreads = 192;

	float * d_input;
	size_t input_size = numCols * numRows;
	checkCudaErrors(cudaMalloc(&d_input, sizeof(float) * input_size));
	checkCudaErrors(cudaMemcpy(d_input, d_logLuminance, sizeof(float) * input_size, cudaMemcpyDeviceToDevice));

	int step = 0;
	size_t temp = input_size;
	do
	{
		temp = temp >> 1;
		step++;
	} while (temp> 1);

	if ((2 << (step - 1)) < (input_size))	step++;

	// get maximum 
	for (int i = 0; i < step; i++)
	{
		get_max << < (input_size + numThreads - 1) / numThreads, numThreads >> >(d_input, input_size, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&max_logLum, d_input, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(d_input, d_logLuminance, sizeof(float) * input_size, cudaMemcpyDeviceToDevice));

	// get minimum
	for (int i = 0; i < step; i++)
	{
		get_min << < (input_size + numThreads - 1) / numThreads, numThreads >> >(d_input, input_size, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&min_logLum, d_input, sizeof(float), cudaMemcpyDeviceToHost));

	/*
	2) subtract them to find the range
	*/
	float lumRange = max_logLum - min_logLum;

	/*
	3) generate a histogram of all the values in the logLuminance channel using
	the formula : bin = (lum[i] - lumMin) / lumRange * numBins
	*/
	histogram << <(input_size + numThreads - 1) / numThreads, numThreads >> >(d_logLuminance, d_cdf, min_logLum, lumRange, input_size, numBins);

	/*
	4) Perform an exclusive scan(prefix sum) on the histogram to get
	the cumulative distribution of luminance values(this should go in the
	incoming d_cdf pointer which already has been allocated for you)       * /
	*/

	step = 0;
	int bins = numBins;

	while (bins > 1)
	{
		bins = bins >> 1;
		step++;
	}

	for (int i = 0; i < step; i++)
	{
		Blelloch_scan_FistHalf << < (numBins + numThreads - 1) / numThreads, numThreads >> >(d_cdf, numBins, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemset(&d_cdf[numBins-1], 0, sizeof(unsigned int)));

	for (int i = (step-1); i >= 0; i--)
	{
		Blelloch_scan_SecondHalf << <  (numBins + numThreads - 1) / numThreads, numThreads >> >(d_cdf, numBins, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// free cuda memory
	checkCudaErrors(cudaFree(d_input));

#endif
}


void your_histogram_and_prefixsum_test()
{
	/*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	store in min_logLum and max_logLum
	*/
#if 0
	float h_input[] = { 2, 4, 3, 3, 1, 7, 4, 5, 7, 0, 9, 4, 3, 2 };
	size_t input_size = 14;

	// alloc memory in CUDA
	//allocate memory for the cdf of the histogram
	float * d_input;
	checkCudaErrors(cudaMalloc(&d_input, sizeof(float) * input_size));
	checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice));

	float * d_maxmin;
	checkCudaErrors(cudaMalloc(&d_maxmin, sizeof(float) * input_size));

	int step = 0;
	size_t temp = input_size;
	do
	{
		temp = temp >> 1;
		step++;
	} while (temp> 1);

	if ((2 << (step - 1)) < (input_size))	step++;
	
	// Get max and min
	float min_logLum;
	float max_logLum;
	checkCudaErrors(cudaMemcpy(d_maxmin, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice));
	for (int i = 0; i < step; i++)
	{
		get_maxmin << <1, input_size>> >(d_input, d_maxmin, input_size, i, true);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&max_logLum, d_maxmin, sizeof(float), cudaMemcpyDeviceToHost));

	// Get min
	checkCudaErrors(cudaMemcpy(d_maxmin, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice));
	for (int i = 0; i < step; i++)
	{
		get_maxmin << <1, input_size >> >(d_input, d_maxmin, input_size, i, false);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&min_logLum, d_maxmin, sizeof(float), cudaMemcpyDeviceToHost));


	/*
	2) subtract them to find the range
	*/
	float lumRange = max_logLum - min_logLum;

	/*
	3) generate a histogram of all the values in the logLuminance channel using
	the formula : bin = (lum[i] - lumMin) / lumRange * numBins
	*/
	unsigned int* d_cdf;
	int numBins = 0;

	checkCudaErrors(cudaMalloc(&d_cdf, sizeof(unsigned int) * numBins));
	checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int) * numBins));


	histogram << <1, input_size>> >(d_input, d_cdf, min_logLum, lumRange, numBins);

	unsigned int *h_cdf = new unsigned int[numBins * sizeof(unsigned int)];
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));


	/*
	4) Perform an exclusive scan(prefix sum) on the histogram to get
	the cumulative distribution of luminance values(this should go in the
	incoming d_cdf pointer which already has been allocated for you)       * /
	*/

	step = 0;
	int bins = numBins;

	while (bins > 1)
	{
		bins = bins >> 1;
		step++;
	}

	for (int i = 0; i < step; i++)
	{
		Blelloch_scan_FistHalf << <1, numBins >> >(d_cdf, numBins, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemset(&d_cdf[numBins - 1], 0, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));


	for (int i = step; i > 0; i--)
	{
		Blelloch_scan_SecondHalf << <1, numBins >> >(d_cdf, numBins, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// free cuda memory
#else
	float h_input[] = { 2.0f, 4.0f, 3.0f, 3.0f, 1.0f, 7.0f, 4.0f, 5.0f, 7.0f, 0.0f, 9.0f, 4.0f, 3.0f, 2.0f };
	float * h_output = new float[14];
	size_t input_size = 14;

	float *d_input;
	checkCudaErrors(cudaMalloc(&d_input, sizeof(float) * 14));
	checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(float) * 14, cudaMemcpyHostToDevice));

	float min_logLum;
	float max_logLum;
	int step = 0;
	size_t temp = input_size;
	do
	{
		temp = temp >> 1;
		step++;
	} while (temp> 1);

	if ((2 << (step - 1)) < (input_size))	step++;

	// get maximum 
	for (int i = 0; i < step; i++)
	{
		get_max << < 1, 14 >> >(d_input, 14, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&max_logLum, d_input, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(float) * 14, cudaMemcpyHostToDevice));

	// get minimum
	for (int i = 0; i < step; i++)
	{
		get_min << < 1, 14 >> >(d_input, 14, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&min_logLum, d_input, sizeof(float), cudaMemcpyDeviceToHost));


	unsigned int h_cdf[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	
	unsigned int * d_cdf;
	checkCudaErrors(cudaMalloc(&d_cdf, sizeof(unsigned int) * 8));
	checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice));

	for (int i = 0; i < 3;i ++)
	{
		Blelloch_scan_FistHalf << < 1, 8 >> >(d_cdf, 8, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemset(&d_cdf[7], 0, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	for (int i = 2; i >= 0; i--)
	{
		Blelloch_scan_SecondHalf << < 1, 8 >> >(d_cdf, 8, i);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost));


#endif
}