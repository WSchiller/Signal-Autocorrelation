// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE		64     // number of threads per block
#endif

#define	SIZE		32768      // data set size of signals.txt file

#define NUMBLOCKS		( SIZE / BLOCKSIZE )  // Number of thread blocks 

// Global host arrays
float hA[SIZE * 2];
float hSums[SIZE];


// CUDA kernel function
__global__  void shiftMultiply(float* dA, float* dSums)
{
	unsigned int wgNumber = blockIdx.x;
	unsigned int wgDimension = blockDim.x;
	unsigned int threadNum = threadIdx.x;
	unsigned int gid = wgNumber * wgDimension + threadNum;  // global id of thread

	// calculate the shift sums for each signal value
	int shift = gid;
	float sum = 0.;
	for (int i = 0; i < SIZE; i++)
	{
		sum += dA[i] * dA[i + shift];
	}
	dSums[shift] = sum;
	
}


int main() {

	// File pointer to read text file
	FILE* fp = fopen("signal.txt", "r");

	if (fp == NULL)
	{
		fprintf(stderr, "Cannot open file 'signal.txt'\n");
		exit(1);
	}

	int Size;
	fscanf(fp, "%d", &Size);
	for (int i = 0; i < Size; i++)
	{
		fscanf(fp, "%f", &hA[i]);
		hA[i + Size] = hA[i];		// fill 2nd half of array
	}
	fclose(fp);

	// allocate device memory:
	float* dA, * dSums;
	
	dim3 dimsA(SIZE * 2, 1, 1);
	dim3 dimsSums(SIZE, 1, 1);

	cudaError_t status;
	status = cudaMalloc((void**)(&dA), sizeof(hA));
	checkCudaErrors(status);

	status = cudaMalloc((void**)(&dSums), sizeof(hSums));
	checkCudaErrors(status);
	
	// copy host memory to the device:
	status = cudaMemcpy(dA, hA, SIZE*2*sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);

	status = cudaMemcpy(dSums, hSums, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);

	// setup the execution parameters:
	dim3 grid(NUMBLOCKS, 1, 1);
	dim3 threads(BLOCKSIZE, 1, 1);
	
	// create and start timer
	cudaDeviceSynchronize();

	// allocate CUDA events that we'll use for timing:
	cudaEvent_t start, stop;
	status = cudaEventCreate(&start);
	checkCudaErrors(status);
	status = cudaEventCreate(&stop);
	checkCudaErrors(status);

	// record the start event:
	status = cudaEventRecord(start, NULL);
	checkCudaErrors(status);

	// execute the kernel:
	shiftMultiply << < grid, threads >> > (dA, dSums);
	
	// record the stop event:
	status = cudaEventRecord(stop, NULL);
	checkCudaErrors(status);

	// wait for the stop event to complete:
	status = cudaEventSynchronize(stop);
	checkCudaErrors(status);

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime(&msecTotal, start, stop);
	checkCudaErrors(status);

	// compute and print the performance
	double secondsTotal = 0.001 * (double)msecTotal;
	double MultsPerSecond = (float)SIZE * (float)SIZE / secondsTotal;
	double MegaMultsPerSecond = MultsPerSecond / 1000000.;
	fprintf(stderr, "Number of Threads Per Block: %d\n", BLOCKSIZE);
	fprintf(stderr, "Array Size = %10d\tMegaMults/Second = %10.2lf\n", SIZE, MegaMultsPerSecond);

	// copy result from the device to the host:
	status = cudaMemcpy(hSums, dSums, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErrors(status);
	cudaDeviceSynchronize();

	// Print the sums from host device, Sums[1]...Sums[512]
	for(int i = 1; i <= 512 ; i++) {
		printf("Shift Number: %d\tSum: %5.2lf\n", i, hSums[i]);
	}

	// clean up memory:
	status = cudaFree(dA);
	status = cudaFree(dSums);
	checkCudaErrors(status);
	
	return 0;
}