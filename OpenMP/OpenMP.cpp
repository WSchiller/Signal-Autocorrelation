#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

// setting the number of threads:
#ifndef NUMT
#define NUMT		8
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

int main() {

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


	double maxPerformance = 0.;
	omp_set_num_threads(NUMT);	// set the number of threads to use in the for-loop:
	for (int j = 0; j < NUMTRIES; j++) {
		
		double time0 = omp_get_wtime();  // start timer
		#pragma omp parallel for
		// autocorrelation 
		for (int shift = 0; shift < Size; shift++)
		{
			float sum = 0.;
			for (int i = 0; i < Size; i++)
			{
				sum += A[i] * A[i + shift];
			}
			// Sums[1]...Sums[512]
			if (shift >= 1 && shift <= 512 && j == 0) {
				Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
				printf("Shift Number: %d\tSum: %5.2lf\n", shift, sum);
			}
			
		}
		double time1 = omp_get_wtime();
		double performance = (double)Size * (double)Size / (time1 - time0) / 1000000;
		if (maxPerformance < performance) maxPerformance = performance;
		
		printf("Trial #%d\tMegaMultsPerSecond: %5.2lf\n", j + 1, performance);
	}

	printf("Number of Threads: %d\tMax Performance: %5.2lf MegaMultsPerSecond\n", NUMT, maxPerformance);

	// Write results to CSV file
	ofstream shiftFile;
	shiftFile.open("shiftResults.csv");
	shiftFile << "Shift #" << "," << "Signal Value" << endl;
	for (int i = 1; i <= 512; i++) {
		shiftFile << i << "," << Sums[i] << endl;
	}

	return 0;
}
