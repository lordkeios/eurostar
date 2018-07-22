#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <sstream>
#include <string>
#include <fstream>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>

#pragma comment(lib, "winmm")


// CUDA/D3D10 kernel
extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern "C" cudaError_t processRMSCuda(float* resultSet, float* data, int transactionSize, int transactionCount);

class TSDB {
public:
	int memSize;
	int count;
	int ptr;

	float* h_data;
	float* d_data;

	float* h_resultset;
	float* d_resultset;

public:
	TSDB() : memSize(0), count(0), ptr(0) {

	}

	virtual ~TSDB() {

	}
};

static TSDB tsdb;

int main()
{
	const int dataSize = 3;
	const int transactionSize = 256;
	const int maxDataCount = 60 * 24 * 365;

	printf("\n**** 1. SYSTEM STARTING ****\n");

	tsdb.memSize = sizeof(float) * ( (1 + dataSize * transactionSize) * maxDataCount );

	cudaError_t cudaStatus;

	cudaEvent_t start, stop;
	DWORD tstart, tstop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	tstart = GetCurrentTime();

	tsdb.h_data = (float*)malloc(tsdb.memSize);
	tsdb.h_resultset = (float*)malloc(tsdb.memSize);

	cudaStatus = cudaMalloc((void**)&tsdb.d_data, tsdb.memSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&tsdb.d_resultset, tsdb.memSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	tstop = GetCurrentTime();

	float allocTimeMs = tstop - tstart;

	printf("alloc storage %d bytes took = %f ms\n", tsdb.memSize, allocTimeMs);

	tstart = GetCurrentTime();

	// data
	printf("\n**** 2. READ DATA ****\n");
	int samplesRead = 0;

	{
		std::ifstream infile("data.csv");
		std::string line;
		while (std::getline(infile, line))
		{
			std::istringstream iss(line);

			float x, y, z;

			iss >> x >> y >> z;

			printf("%f, %f, %f\n", x, y, z);
		}
	}

	for (int i = 0; i < maxDataCount; i++) {
		int ptr = i * (1 + dataSize*transactionSize);
		*(tsdb.h_data + ptr) = i;

		for (int j = 0; j < transactionSize; j++) {
			int ptr_trans = ptr + 1 + (j * dataSize);

			*(tsdb.h_data + ptr_trans + 0) = i + 1;
			samplesRead++;

			*(tsdb.h_data + ptr_trans + 1) = i + 2;
			samplesRead++;

			*(tsdb.h_data + ptr_trans + 2) = i + 3;
			samplesRead++;
		}
	}

	tstop = GetCurrentTime();

	float dataReadTimeMs = tstop - tstart;

	printf("data read %d values took = %f ms\n", samplesRead, dataReadTimeMs);


	// commit
	printf("\n**** 3. WRITE DATA TO DATABASE ****\n");

	for (int test = 0; test < 10; test++) {
		printf("**** data commit test %d ****\n", test);

		int dataCommitted = 0;

		cudaEventRecord(start, 0);

		int dataCount = maxDataCount;

		for (int i = 0; i < dataCount; i++) {
			int ptr = i * (1 + dataSize*transactionSize);

			cudaStatus = cudaMemcpy(tsdb.d_data + ptr, tsdb.h_data + ptr, (1 + dataSize * transactionSize) * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "commit failed!");
				goto Error;
			}
			dataCommitted += dataSize*transactionSize;
		}
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float commitTimeMs;
		cudaEventElapsedTime(&commitTimeMs, start, stop);

		printf("data commit: %d values took = %f ms\n", dataCommitted, commitTimeMs);
	}

	// query
	printf("\n**** 4. QUERY DATA FROM DATABASE ****\n");
	int dataQueried = 0;

	int query_t = 0;
	int query_count = 100;

	tstart = GetCurrentTime();

	int data_ptr = query_t * (1 + dataSize * transactionSize);

	for (int i = 0; i < query_count; i++) {
		cudaStatus = cudaMemcpy(tsdb.h_resultset + i * (1 + dataSize * transactionSize),
			tsdb.d_data + data_ptr + i * (1 + dataSize * transactionSize),
			(1 + dataSize * transactionSize) * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "query failed!");
			goto Error;
		}

		cudaDeviceSynchronize();
		dataQueried += dataSize * transactionSize;
	}

	tstop = GetCurrentTime();
	float queryTimeMs = tstop - tstart;

	printf("data query %d values took = %f ms\n", dataQueried, queryTimeMs);

	for (int i = 0; i < query_count; i++) {
		float* record = tsdb.h_resultset + i * (1 + dataSize * transactionSize);
		printf("[%d] (%f, %f, %f, %f)\n", i, record[0], record[1], record[2], record[3]);
	}

	// process
	printf("\n**** 5. PROCESS DATA ****\n");
	for(int test = 0; test < 10; test++)
	{
		printf("**** data process test %d ****\n", test);

		int dataProcessed = 0;

		int dataCount = maxDataCount;

		cudaEventRecord(start, 0);

		cudaStatus = processRMSCuda(tsdb.d_resultset, tsdb.d_data, transactionSize, dataCount);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "data processing failed!");
			return 1;
		}

		dataProcessed += dataCount * dataSize * transactionSize * sizeof(float);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float processTimeMs;
		cudaEventElapsedTime(&processTimeMs, start, stop);

		printf("data process: %d bytes took = %f ms\n", dataProcessed, processTimeMs);
	}

	// fetch resultset
	printf("\n**** 6. FETCH RESULT ****\n");
	{
		int dataCount = maxDataCount;

		int resultFetched = 0;

		tstart = GetCurrentTime();

		for (int i = 0; i < dataCount; i++) {
			cudaStatus = cudaMemcpy(tsdb.h_resultset + i * transactionSize,
				tsdb.d_resultset + i * transactionSize,
				transactionSize * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "query failed!");
				goto Error;
			}

			cudaDeviceSynchronize();
			resultFetched += transactionSize;
		}

		tstop = GetCurrentTime();
		float queryTimeMs = tstop - tstart;

		printf("data query %d values took = %f ms\n", dataQueried, queryTimeMs);

		for (int i = 0; i < 100; i++) {
			float* record = tsdb.h_resultset + i * (1 + dataSize * transactionSize);
			printf("[%d] (%f, %f, %f, %f)\n", i, record[0], record[1], record[2], record[3]);
		}

	}

Error:
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	printf("\n********************\n");

	return 0;
}
