#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <fstream>
#include "utils.h"
#include "timer.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <algorithm>

//#define SMALLER	88

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

#include "reference_calc.h"

#define ANDY_SOLUTION	88


void computeHistogram(const unsigned int *const d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems);

int main(void)
{
#ifdef SMALLER
	const unsigned int numBins = 512;
	const unsigned int numElems = 100 * numBins;
#else
  const unsigned int numBins = 1024;
  const unsigned int numElems = 100000 * numBins;
#endif
  const float stddev = 100.f;

  unsigned int *vals = new unsigned int[numElems];
  unsigned int *h_vals = new unsigned int[numElems];
  unsigned int *h_studentHisto = new unsigned int[numBins];
  unsigned int *h_refHisto = new unsigned int[numBins];

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
  srand(GetTickCount());
#else
  timeval tv;
  gettimeofday(&tv, NULL);

  srand(tv.tv_usec);
#endif

  //make the mean unpredictable, but close enough to the middle
  //so that timings are unaffected
#ifdef SMALLER
  unsigned int mean = rand() % 100 + 46;
#else
  unsigned int mean = rand() % 100 + 462;
#endif
  //Output mean so that grading can happen with the same inputs
  std::cout << mean << std::endl;

  thrust::minstd_rand rng;

  thrust::random::normal_distribution<float> normalDist((float)mean, stddev);

  // Generate the random values
  for (size_t i = 0; i < numElems; ++i) {
    //vals[i] = min((unsigned int) max((int)normalDist(rng), 0), numBins - 1);
	  unsigned int val = normalDist(rng);
	  val = max(val, 0);
	  val = min(val, numBins - 1);
	  vals[i] = val;
  }

  unsigned int *d_vals, *d_histo;

  GpuTimer timer;
#ifdef ANDY_SOLUTION
  const int q = 10;
  unsigned int *h_studentHisto_10times = new unsigned int[numBins*q];
  checkCudaErrors(cudaMalloc(&d_vals,    sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_histo,   sizeof(unsigned int) * numBins * q));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins * q));
#else
  checkCudaErrors(cudaMalloc(&d_vals, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));
#endif
  checkCudaErrors(cudaMemcpy(d_vals, vals, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

  timer.Start();
  computeHistogram(d_vals, d_histo, numBins, numElems);
  timer.Stop();
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  // copy the student-computed histogram back to the host
#ifdef ANDY_SOLUTION
  checkCudaErrors(cudaMemcpy(h_studentHisto_10times, d_histo, sizeof(unsigned int) * numBins * q, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 1024; i++)
  {
	  int sum = 0;
	  for (int j = 0; j < q; j++)
	  {
		  sum += h_studentHisto_10times[i*q + j];
	  }
	  h_studentHisto[i] = sum;
  }
#else
  checkCudaErrors(cudaMemcpy(h_studentHisto, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
#endif


  //generate reference for the given mean
  timer.Start();
  reference_calculation(vals, h_refHisto, numBins, numElems);
  timer.Stop();


  //Now do the comparison
  checkResultsExact(h_refHisto, h_studentHisto, numBins);

  delete[] h_vals;
  delete[] h_refHisto;
  delete[] h_studentHisto;
#ifdef ANDY_SOLUTION
  delete[] h_studentHisto_10times;
#endif
  cudaFree(d_vals);
  cudaFree(d_histo);

  return 0;
}
