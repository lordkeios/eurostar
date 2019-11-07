
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "types.h"

__global__ void tsdb_parse(tsdb_control* control, tsdb_points* points, const void* data, int points_count) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= points_count)
		return;

	
	unsigned long long* item = &((unsigned long long*)data)[i * 2];

	tsdb_points* point = &points[i];
	point->time = item[0];
}

__global__ void tsdb_store(tsdb_control* control, const tsdb_points* points, int points_count) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= points_count)
		return;

	const tsdb_points* point = &points[i];
	unsigned long index = ((long long)point->time) >> 32;

	atomicAdd(&control->sequence, 1);
}

__global__ void tsdb_store_multi(tsdb_control* control, int node_count, const tsdb_points* points, int points_count) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= points_count)
		return;

	const tsdb_points* point = &points[i];
	unsigned long index = ((long long)point->time) >> 32 % node_count;

	atomicAdd(&control->sequence, 1);
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void RMSKernel(float* out, const float* in, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int ptr = blockDim.x * blockIdx.x + 1;

	if (i < numElements) {
		out[i] = in[ptr + threadIdx.x * 3 + 0] * in[ptr + threadIdx.x * 3 + 0]
			+ in[ptr + threadIdx.x * 3 + 1] * in[ptr + threadIdx.x * 3 + 1]
			+ in[ptr + threadIdx.x * 3 + 2] * in[ptr + threadIdx.x * 3 + 2];
	}
}


extern "C"
cudaError_t runTsdbParseKernel(tsdb_control* control, tsdb_points* points, const void* data, int points_count)
{
	cudaError_t cudaStatus;

	const int threads = 256;
	const int blocks = points_count / threads + 1;

	tsdb_parse << <blocks, threads >> > (control, points, data, points_count);
	cudaStatus = cudaDeviceSynchronize();
	return cudaStatus;
}

extern "C"
cudaError_t runTsdbStoreKernel(tsdb_control* control, const tsdb_points* points, int points_count)
{
	cudaError_t cudaStatus;

	const int threads = 256;
	const int blocks = points_count / threads + 1;

	tsdb_store << <blocks, threads >> > (control, points, points_count);
	cudaStatus = cudaDeviceSynchronize();
	return cudaStatus;
}

extern "C"
cudaError_t runTsdbStoreMultiKernel(tsdb_control* control, int node_count, const tsdb_points* points, int points_count)
{
	cudaError_t cudaStatus;

	const int threads = 256;
	const int blocks = points_count / threads + 1;

	tsdb_store_multi << <blocks, threads >> > (control, node_count, points, points_count);
	cudaStatus = cudaDeviceSynchronize();
	return cudaStatus;
}

extern "C"
cudaError_t runTsdbRMSKernel(float* resultSet, float* data, int transactionSize, int transactionCount)
{
	cudaError_t cudaStatus;

	const int blocks = 256;

	while (transactionCount > 0) {
		int transactions = (transactionCount > blocks ? blocks : transactionCount);
		int numElements = transactions * transactionSize;

		RMSKernel << <blocks, transactionSize >> > (resultSet, data, numElements);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			return cudaStatus;
		}

		resultSet += numElements;
		data += transactions * (1 + 3 * transactionSize);
		transactionCount -= blocks;
	}
	return cudaGetLastError();
}


// Helper function for using CUDA to add vectors in parallel.
extern "C"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
