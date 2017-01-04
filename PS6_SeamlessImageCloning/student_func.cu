//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <algorithm>
#include <string>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>


//#define SOLUTION1	1
#define SOLUTION2	1

#if defined(SOLUTION1)

#define VERIFY_BORDER 999
//#define USE_DEST_IMG 999

__global__ void isMask(const uchar4 * d_sourceImg, int* d_mask, int numPixels)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (myId > numPixels)	return;

	if (d_sourceImg[myId].x != 255 ||
		d_sourceImg[myId].y != 255 ||
		d_sourceImg[myId].z != 255 )
	{
		d_mask[myId] = 1;
	}
}

__global__ void isBorder(const uchar4* d_sourceImg, int* d_mask, int* d_interior, int* d_border, int width, int height)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (myId > (width*height))	return;

	if (d_mask[myId] == 0)	return;

	int neighbors[4];
	neighbors[0] = myId - width;
	neighbors[1] = myId - 1;
	neighbors[2] = myId + 1;
	neighbors[3] = myId + width;

	if (d_mask[neighbors[0]] == 1 &&
		d_mask[neighbors[1]] == 1 &&
		d_mask[neighbors[2]] == 1 &&
		d_mask[neighbors[3]] == 1 )
	{
		d_interior[myId] = 1;
	}
	else
	{
		d_border[myId] = 1;
	}

}

__global__ void JacobiProcess(float* d_imageGuessPrev, float* d_imageGuessNext, float *d_sourceImage, float *d_destinationImage, int width, int height, const int * interior, const int* border)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (myId > (width * height))	return;

	if (interior[myId] == 0 && border[myId] == 0)	return;

	int neighbors[4];
	neighbors[0] = myId - width;
	neighbors[1] = myId - 1;
	neighbors[2] = myId + 1;
	neighbors[3] = myId + width;

	float sum1 = 0, sum2 = 0;
	int nNeighbors = 4;
	for (int i = 0; i < 4; i++)
	{
		if (interior[neighbors[i]])
		{
			sum1 += d_imageGuessPrev[neighbors[i]];
		}
		else if (border[neighbors[i]])
		{
			sum1 += d_destinationImage[neighbors[i]];
		}
		else
			nNeighbors--;

		sum2 += d_sourceImage[myId] - d_sourceImage[neighbors[i]];
	}

	float newVal = (sum1 + sum2) / nNeighbors;

	d_imageGuessNext[myId] = min(255.0, max(0.0, newVal)); //clamp to [0, 255]

}

__global__ void ReplaceInterior(uchar4* d_blendImage, float* imageGuess, int* interior, int ch)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (interior[myId] == 1)
	{
		float val = imageGuess[myId];
		if (ch == 0)		d_blendImage[myId].x = (unsigned char)val;
		else if (ch == 1)	d_blendImage[myId].y = (unsigned char)val;
		else if (ch == 2)	d_blendImage[myId].z = (unsigned char)val;
	}
}

void your_blend(const uchar4* const h_sourceImg,  //IN
	const size_t numRowsSource, const size_t numColsSource,
	const uchar4* const h_destImg, //IN
	uchar4* const h_blendedImg) //OUT
{

	/* To Recap here are the steps you need to implement */
	size_t numPixels = numRowsSource * numColsSource;

	int numThreads = 1024;
	int grid = (numPixels + numThreads - 1) / numThreads;
	/*
	 1) Compute a mask of the pixels from the source image to be copied
	 The pixels that shouldn't be copied are completely white, they
	 have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
	 */
	uchar4* d_sourceImg;
	checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	int *d_mask;
	checkCudaErrors(cudaMalloc(&d_mask, sizeof(int) * numPixels));
	checkCudaErrors(cudaMemset(d_mask, 0, sizeof(int) * numPixels));
	isMask << <grid, numThreads >> >(d_sourceImg, d_mask, numPixels);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	/*
	2) Compute the interior and border regions of the mask.  An interior
	pixel has all 4 neighbors also inside the mask.  A border pixel is
	in the mask itself, but has at least one neighbor that isn't.
	*/
	int *d_interior, *d_border;
	checkCudaErrors(cudaMalloc(&d_interior, sizeof(int) * numPixels));
	checkCudaErrors(cudaMemset(d_interior, 0, sizeof(int) * numPixels));
	checkCudaErrors(cudaMalloc(&d_border, sizeof(int) * numPixels));
	checkCudaErrors(cudaMemset(d_border, 0, sizeof(int) * numPixels));
	isBorder << <grid, numThreads >> >(d_sourceImg, d_mask, d_interior, d_border, numColsSource, numRowsSource);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#ifdef VERIFY_BORDER
	int *h_mask, *h_interior, *h_border;
	h_mask = new int[numPixels];
	h_interior = new int[numPixels];
	h_border = new int[numPixels];

	for (int i = 0; i < numPixels; i++)
	{
		if (h_sourceImg[i].x != 255 ||
			h_sourceImg[i].y != 255 ||
			h_sourceImg[i].z != 255)
			h_mask[i] = 1;
		else
			h_mask[i] = 0;
	}

	for (int i = 0; i < numPixels; i++)
	{
		if (h_mask[i] == 1)
		{
			if (h_mask[i - 1] == 1 &&
				h_mask[i + 1] == 1 &&
				h_mask[i - numColsSource] == 1 &&
				h_mask[i + numColsSource] == 1)
			{
				h_interior[i] = 1;
				h_border[i] = 0;
			}
			else
			{
				h_interior[i] = 0;
				h_border[i] = 1;
			}
		}
		else
		{
			h_interior[i] = 0;
			h_border[i] = 0;
		}
	}

	int *h_mask_ref, *h_interior_ref, *h_border_ref;
	h_mask_ref = new int[numPixels];
	h_interior_ref = new int[numPixels];
	h_border_ref = new int[numPixels];
	checkCudaErrors(cudaMemcpy(h_mask_ref, d_mask, sizeof(int) * numPixels, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_interior_ref, d_interior, sizeof(int) * numPixels, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_border_ref, d_border, sizeof(int) * numPixels, cudaMemcpyDeviceToHost));

	int ret = 0;
	int n = 0;

	for (n = 0; n < numPixels; n++)
	{
		if (h_mask[n] != h_mask_ref[n])
		{
			ret = 1;
			break;
		}
	}

	printf("Compare h_mask to h_mask_ref, result = %d, n = %d\n", ret, n);

	for (n = 0; n < numPixels; n++)
	{
		if (h_interior[n] != h_interior_ref[n])
		{
			ret = 1;
			break;
		}
	}
	
	printf("Compare h_interior to h_interior_ref, result = %d, n = %d\n", ret, n);
	ret = 0;
	for (n = 0; n < numPixels; n++)
	{
		if (h_border[n] != h_border_ref[n])
		{
			ret = 1;
			break;
		}
	}
	printf("Compare h_border to h_border_ref, result = %d, n = %d\n", ret, n);
	
	delete[]h_mask;
//	delete[]h_interior;
	delete[]h_border;
	delete[]h_mask_ref;
	delete[]h_interior_ref;
	delete[]h_border_ref;

#endif
	/*
	3) Separate out the incoming image into three separate channels

 	4) Create two float(!) buffers for each color channel that will
	act as our guesses.  Initialize them to the respective color
	channel of the source image since that will act as our intial guess.
	*/

	float * h_image = new float[numPixels];

	float * d_sourceImage, * d_destinationImage;
	checkCudaErrors(cudaMalloc(&d_sourceImage, sizeof(float) * numPixels));
	checkCudaErrors(cudaMalloc(&d_destinationImage, sizeof(float) * numPixels));

	float * d_imageGuessPrev, *d_imageGuessNext;
	checkCudaErrors(cudaMalloc(&d_imageGuessPrev, sizeof(float) * numPixels));
	checkCudaErrors(cudaMalloc(&d_imageGuessNext, sizeof(float) * numPixels));

#ifdef USE_DEST_IMG
#ifdef HOST2HOST
	memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * numPixels);
#else
	//checkCudaErrors(cudaMemcpy(d_sourceImg, h_destImg, 4 * numPixels, cudaMemcpyHostToDevice));
	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_sourceImage, 4 * numPixels, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
#endif
#else
	checkCudaErrors(cudaMemcpy(d_sourceImg, h_destImg, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	for (int ch = 0; ch < 3; ch++)
	{
		for (int i = 0; i < numPixels; i++)
		{
			if (ch == 0)	h_image[i] = (float)h_sourceImg[i].x;
			else if (ch == 1)	h_image[i] = (float)h_sourceImg[i].y;
			else if (ch == 2)	h_image[i] = (float)h_sourceImg[i].z;
		}

		checkCudaErrors(cudaMemcpy(d_sourceImage, h_image, sizeof(float) * numPixels, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_imageGuessPrev, h_image, sizeof(float) * numPixels, cudaMemcpyHostToDevice));

		for (int i = 0; i < numPixels; i++)
		{
			if (ch == 0)	h_image[i] = (float)h_destImg[i].x;
			else if (ch == 1)	h_image[i] = (float)h_destImg[i].y;
			else if (ch == 2)	h_image[i] = (float)h_destImg[i].z;
		}
		checkCudaErrors(cudaMemcpy(d_destinationImage, h_image, sizeof(float) * numPixels, cudaMemcpyHostToDevice));

		/*
		5) For each color channel perform the Jacobi iteration described
		above 800 times.
		*/
		for (int step = 0; step < 800; step++)
		{
			if((step%2) == 0)
				JacobiProcess << <grid, numThreads >> >(d_imageGuessPrev, d_imageGuessNext, d_sourceImage, d_destinationImage, numColsSource, numRowsSource, d_interior, d_mask);
			else
				JacobiProcess << <grid, numThreads >> >(d_imageGuessNext, d_imageGuessPrev, d_sourceImage, d_destinationImage, numColsSource, numRowsSource, d_interior, d_mask);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}

		/*
		6) Create the output image by replacing all the interior pixels
			in the destination image with the result of the Jacobi iterations.
			Just cast the floating point values to unsigned chars since we have
			already made sure to clamp them to the correct range.
		*/

		/*
		Since this is final assignment we provide little boilerplate code to
			help you.Notice that all the input / output pointers are HOST pointers.

			You will have to allocate all of your own GPU memory and perform your own
			memcopies to get data in and out of the GPU memory.

			Remember to wrap all of your calls with checkCudaErrors() to catch any
			thing that might go wrong.After each kernel call do:

			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			to catch any errors that happened while executing the kernel.
		*/
		//ReplaceInterior << <grid, numThreads >> >(d_sourceImg, d_imageGuessPrev, d_interior, ch);
		checkCudaErrors(cudaMemcpy(h_image, d_imageGuessPrev, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		for (int i = 0; i < numPixels; i++)
		{
			if (h_interior[i] == 1)
			{
				if (ch == 0)	h_blendedImg[i].x = (float)h_image[i];
				else if (ch == 1)	h_blendedImg[i].y = (float)h_image[i];
				else if (ch == 2)	h_blendedImg[i].z = (float)h_image[i];
			}
			else
			{
				if (ch == 0)	h_blendedImg[i].x = h_destImg[i].x;
				else if (ch == 1)	h_blendedImg[i].y = h_destImg[i].y;
				else if (ch == 2)	h_blendedImg[i].z = h_destImg[i].z;
			}
		}
	}
	//checkCudaErrors(cudaMemcpy(h_blendedImg, d_sourceImage, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
#endif


	delete []h_image;

	checkCudaErrors(cudaFree(d_sourceImg));
	checkCudaErrors(cudaFree(d_sourceImage));
	checkCudaErrors(cudaFree(d_destinationImage));
	checkCudaErrors(cudaFree(d_imageGuessPrev));
	checkCudaErrors(cudaFree(d_imageGuessNext));
}
#endif // defined(SOLUTION1)

#if defined(SOLUTION2)

/*
	Your code compiled! 
	Your code printed the following output: Your code executed in 176.446304 ms Good job!. 
	Your image matched perfectly to the reference image 
*/
__global__ void isMask(const uchar4 * d_sourceImg, int* d_mask, int numPixels)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (myId > numPixels)	return;

	if (d_sourceImg[myId].x != 255 ||
		d_sourceImg[myId].y != 255 ||
		d_sourceImg[myId].z != 255)
	{
		d_mask[myId] = 1;
	}
}

__global__ void isBorder(const uchar4* d_sourceImg, int* d_mask, int* d_interior, int* d_border, int width, int height)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (myId > (width*height))	return;

	if (d_mask[myId] == 0)	return;

	int neighbors[4];
	neighbors[0] = myId - width;
	neighbors[1] = myId - 1;
	neighbors[2] = myId + 1;
	neighbors[3] = myId + width;

	if (d_mask[neighbors[0]] == 1 &&
		d_mask[neighbors[1]] == 1 &&
		d_mask[neighbors[2]] == 1 &&
		d_mask[neighbors[3]] == 1)
	{
		d_interior[myId] = 1;
	}
	else
	{
		d_border[myId] = 1;
	}

}

__global__ void separateChannel(uchar4* d_sourceImg, uchar4* d_destImg, float* d_sourceImage, float* d_destinationImage, float* d_imageGuessPrev, int numPixels, int channel)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (myId > numPixels)	return;

	if (channel == 0)
	{
		d_sourceImage[myId] = d_sourceImg[myId].x;
		d_destinationImage[myId] = d_destImg[myId].x;
		d_imageGuessPrev[myId] = d_sourceImg[myId].x;
	}
	else if (channel == 1)
	{
		d_sourceImage[myId] = d_sourceImg[myId].y;
		d_destinationImage[myId] = d_destImg[myId].y;
		d_imageGuessPrev[myId] = d_sourceImg[myId].y;
	}
	else if (channel == 2)
	{
		d_sourceImage[myId] = d_sourceImg[myId].z;
		d_destinationImage[myId] = d_destImg[myId].z;
		d_imageGuessPrev[myId] = d_sourceImg[myId].z;
	}
}

__global__ void ReplaceInterior(uchar4* d_blendImage, float* imageGuess, int* interior, int ch)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (interior[myId] == 1)
	{
		float val = imageGuess[myId];
		if (ch == 0)		d_blendImage[myId].x = (unsigned char)val;
		else if (ch == 1)	d_blendImage[myId].y = (unsigned char)val;
		else if (ch == 2)	d_blendImage[myId].z = (unsigned char)val;
	}
}

__global__ void JacobiProcess(float* d_imageGuessPrev, float* d_imageGuessNext, float *d_sourceImage, float *d_destinationImage, int width, int height, const int * interior, const int* border)
{
	int myId = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (myId > (width * height))	return;

	if (interior[myId] == 0 && border[myId] == 0)	return;

	int neighbors[4];
	neighbors[0] = myId - width;
	neighbors[1] = myId - 1;
	neighbors[2] = myId + 1;
	neighbors[3] = myId + width;

	float sum1 = 0, sum2 = 0;
	int nNeighbors = 4;
	for (int i = 0; i < 4; i++)
	{
		if (interior[neighbors[i]])
		{
			sum1 += d_imageGuessPrev[neighbors[i]];
		}
		else if (border[neighbors[i]])
		{
			sum1 += d_destinationImage[neighbors[i]];
		}
		else
			nNeighbors--;

		sum2 += d_sourceImage[myId] - d_sourceImage[neighbors[i]];
	}

	float newVal = (sum1 + sum2) / nNeighbors;

	d_imageGuessNext[myId] = min(255.0, max(0.0, newVal)); //clamp to [0, 255]

}

void your_blend(const uchar4* const h_sourceImg,  //IN
	const size_t numRowsSource, const size_t numColsSource,
	const uchar4* const h_destImg, //IN
	uchar4* const h_blendedImg) //OUT
{

	/* To Recap here are the steps you need to implement */
	size_t numPixels = numRowsSource * numColsSource;

	int numThreads = 1024;
	int grid = (numPixels + numThreads - 1) / numThreads;

	/*
	1) Compute a mask of the pixels from the source image to be copied
	The pixels that shouldn't be copied are completely white, they
	have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
	*/
	uchar4* d_sourceImg;
	checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	int *d_mask;
	checkCudaErrors(cudaMalloc(&d_mask, sizeof(int) * numPixels));
	checkCudaErrors(cudaMemset(d_mask, 0, sizeof(int) * numPixels));
	isMask << <grid, numThreads >> >(d_sourceImg, d_mask, numPixels);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	/*
	2) Compute the interior and border regions of the mask.  An interior
	pixel has all 4 neighbors also inside the mask.  A border pixel is
	in the mask itself, but has at least one neighbor that isn't.
	*/
	int *d_interior, *d_border;
	checkCudaErrors(cudaMalloc(&d_interior, sizeof(int) * numPixels));
	checkCudaErrors(cudaMemset(d_interior, 0, sizeof(int) * numPixels));
	checkCudaErrors(cudaMalloc(&d_border, sizeof(int) * numPixels));
	checkCudaErrors(cudaMemset(d_border, 0, sizeof(int) * numPixels));
	isBorder << <grid, numThreads >> >(d_sourceImg, d_mask, d_interior, d_border, numColsSource, numRowsSource);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	/*
	3) Separate out the incoming image into three separate channels

	4) Create two float(!) buffers for each color channel that will
	act as our guesses.  Initialize them to the respective color
	channel of the source image since that will act as our intial guess.
	*/

	float * d_sourceImage, *d_destinationImage;
	checkCudaErrors(cudaMalloc(&d_sourceImage, sizeof(float) * numPixels));
	checkCudaErrors(cudaMalloc(&d_destinationImage, sizeof(float) * numPixels));

	float * d_imageGuessPrev, *d_imageGuessNext;
	checkCudaErrors(cudaMalloc(&d_imageGuessPrev, sizeof(float) * numPixels));
	checkCudaErrors(cudaMalloc(&d_imageGuessNext, sizeof(float) * numPixels));

	uchar4* d_destImg;
	checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	for (int ch = 0; ch < 3; ch++)
	{
		separateChannel << <grid, numThreads >> >(d_sourceImg, d_destImg, d_sourceImage, d_destinationImage, d_imageGuessPrev, numPixels, ch);

		/*
		5) For each color channel perform the Jacobi iteration described
		above 800 times.
		*/
		for (int step = 0; step < 800; step++)
		{
			if ((step % 2) == 0)
				JacobiProcess << <grid, numThreads >> >(d_imageGuessPrev, d_imageGuessNext, d_sourceImage, d_destinationImage, numColsSource, numRowsSource, d_interior, d_mask);
			else
				JacobiProcess << <grid, numThreads >> >(d_imageGuessNext, d_imageGuessPrev, d_sourceImage, d_destinationImage, numColsSource, numRowsSource, d_interior, d_mask);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}

		/*
		6) Create the output image by replacing all the interior pixels
		in the destination image with the result of the Jacobi iterations.
		Just cast the floating point values to unsigned chars since we have
		already made sure to clamp them to the correct range.
		*/

		/*
		Since this is final assignment we provide little boilerplate code to
		help you.Notice that all the input / output pointers are HOST pointers.

		You will have to allocate all of your own GPU memory and perform your own
		memcopies to get data in and out of the GPU memory.

		Remember to wrap all of your calls with checkCudaErrors() to catch any
		thing that might go wrong.After each kernel call do:

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		to catch any errors that happened while executing the kernel.
		*/
		ReplaceInterior << <grid, numThreads >> >(d_destImg, d_imageGuessPrev, d_interior, ch);
	}

	checkCudaErrors(cudaMemcpy(h_blendedImg, d_destImg, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));


	checkCudaErrors(cudaFree(d_sourceImg));
	checkCudaErrors(cudaFree(d_destImg));
	checkCudaErrors(cudaFree(d_sourceImage));
	checkCudaErrors(cudaFree(d_destinationImage));
	checkCudaErrors(cudaFree(d_imageGuessPrev));
	checkCudaErrors(cudaFree(d_imageGuessNext));
}
#endif