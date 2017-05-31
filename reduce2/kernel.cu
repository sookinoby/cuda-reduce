
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;


void check(cudaError_t e)
{
	if (e != cudaSuccess)
	{
		printf(cudaGetErrorString(e));
	}
}



// Kernel function to add the elements of two arrays
__global__
void reduce(int n, float *x, float *y)
{
	int tid = threadIdx.x;
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			x[threadId] += x[threadId+s];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		y[blockIdx.x] = x[threadId];
	}	
}

int main(void)
{
	int N = 1 <<20;
	int reduced_n = N/1024;
	float *x, *y;
	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, reduced_n * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; ++i) {
		x[i] = i+1;
	}

	// Run kernel on 1M elements on the GPU
	reduce<<<reduced_n, reduced_n >>>(N, x,y);

	 //Run on one block
	reduce<<<1, reduced_n>>>(N,y,y);


	

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	cout << "The final sum is" << y[0];
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}


	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	/*for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;
*/
	// Free memory
	cudaFree(x);
	cudaFree(y);
	getchar();
	return 0;
}