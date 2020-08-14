#include <stdio.h>
#include <iostream>
#include <xmmintrin.h>
#include <omp.h>

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

#define SSE_WIDTH	4

// Function prototype(s)
float SimdMulSum(float* a, float* b, int len);

int main() {

	// OpenMP will be used for timing only
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif

	// File pointer to read text file
	FILE* fp = fopen("signal.txt", "r");

	if (fp == NULL)
	{
		fprintf(stderr, "Cannot open file 'signal.txt'\n");
		exit(1);
	}

	int Size;
	fscanf(fp, "%d", &Size);
	float* A = new float[2 * Size];
	float* Sums = new float[1 * Size];
	for (int i = 0; i < Size; i++)
	{
		fscanf(fp, "%f", &A[i]);
		A[i + Size] = A[i];		// duplicate the array
	}
	fclose(fp);

	// Run SIMD code
	double maxPerformance = 0.;
	for (int j = 0; j < NUMTRIES; j++) {
		double time0 = omp_get_wtime();  // start time
		for (int shift = 0; shift < Size; shift++)
		{
			Sums[shift] = SimdMulSum(&A[0], &A[0 + shift], Size);  // SIMD multiplication

			// Sums[1]...Sums[512]
			if (shift >= 1 && shift <= 512 && j == 0) {
				printf("Shift Number: %d\tSum: %5.2lf\n", shift, Sums[shift]);
			}
		}
		double time1 = omp_get_wtime();  // end time

		double performance = (double)Size * (double)Size / (time1 - time0) / 1000000;
		if (maxPerformance < performance) maxPerformance = performance;

		printf("Trial #%d\tMegaMultsPerSecond: %5.2lf\n", j + 1, performance);
	}

	printf("Max Performance: %5.2lf MegaMultsPerSecond\n", maxPerformance);

	return 0;
}

// SIMD multiplication
float SimdMulSum(float* a, float* b, int len) {
	float sum[4] = { 0., 0., 0., 0. };
	int limit = (len / SSE_WIDTH) * SSE_WIDTH;
	register float* pa = a;
	register float* pb = b;

	__m128 ss = _mm_loadu_ps(&sum[0]);
	for (int i = 0; i < limit; i += SSE_WIDTH)
	{
		ss = _mm_add_ps(ss, _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps(&sum[0], ss);

	for (int i = limit; i < len; i++)
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}

