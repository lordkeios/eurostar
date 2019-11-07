#include "pch.h"
#include "gtsdb.h"
#include "cuda_runtime.h"

#include "System.h"

#include "WebSocketServer.h"
#include "WebSocketClient.h"

extern "C" cudaError_t runTsdbParseKernel(tsdb_control* control, const tsdb_points* points, const void* data, int points_count);
extern "C" cudaError_t runTsdbStoreKernel(tsdb_control* control, const tsdb_points* points, int points_count);
extern "C" cudaError_t runTsdbStoreMultiKernel(tsdb_control* control, int node_count, const tsdb_points* points, int points_count);
extern "C" cudaError_t runTsdbRMSKernel(float* resultSet, float* data, int transactionSize, int transactionCount);

//
void TSDB::init(const options& opts) {
	cudaError_t cudaStatus;

	h_control = (tsdb_control*)malloc(sizeof(tsdb_control));
	h_control->node_count = 2;
	h_control->sequence = 0;

	cudaStatus = cudaMalloc((void**)&d_control, sizeof(tsdb_control));
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMemcpy(d_control, h_control, sizeof(tsdb_control), cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaDeviceSynchronize();
	assert(cudaStatus == cudaSuccess);
}

void TSDB::clear() {
	cudaError_t cudaStatus;

	free(h_control); h_control = nullptr;
	cudaFree(d_control); d_control = nullptr;

	cudaStatus = cudaDeviceReset();
	assert(cudaStatus == cudaSuccess);
}

void TSDB::write(const std::string& msg) {
	const int dataSize = 3;
	const int transactionSize = 256;

	size_t point_count = msg.size() / (sizeof(float) * 4);

	printf("[TSDB] write: received %ld bytes\n", msg.size());

	cudaError_t cudaStatus;

	cudaEvent_t start, stop;
	DWORD tstart, tstop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	tstart = GetCurrentTime();

	printf("[TSDB](ts: @%ld) write: writing %ld metrics\n", tstart, point_count);

	// upload
	char* d_data;
	printf("[TSDB] write: uploading data to device\n");
	cudaEventRecord(start, 0);

	cudaStatus = cudaMalloc((void**)&d_data, msg.size());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	int ptr = 0;

	cudaStatus = cudaMemcpy(d_data + ptr, msg.c_str() + ptr, msg.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "commit failed!");
		goto Error;
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float uploadTimeMs;
	cudaEventElapsedTime(&uploadTimeMs, start, stop);

	printf("[TSDB] write: uploading data to device: done in %f ms\n", uploadTimeMs);

	// parse data
	tsdb_points* h_points;
	tsdb_points* d_points;

	printf("[TSDB] write: parsing data to points in device\n");
	cudaEventRecord(start, 0);

	h_points = (tsdb_points*)malloc(sizeof(tsdb_points) * point_count);

	cudaStatus = cudaMalloc((void**)&d_points, sizeof(tsdb_points) * point_count);
	assert(cudaStatus == cudaSuccess);

	cudaStatus = runTsdbParseKernel((tsdb_control*)d_control, d_points, d_data, point_count);
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMemcpy(h_points, d_points, sizeof(tsdb_points) * point_count, cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaDeviceSynchronize();
	assert(cudaStatus == cudaSuccess);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float parsingTimeMs;
	cudaEventElapsedTime(&parsingTimeMs, start, stop);

	printf("[TSDB] write: parsing data to points in device: done in %f ms\n", parsingTimeMs);

	// index
	printf("[TSDB] write: indexing data\n");
	cudaEventRecord(start, 0);

	cudaStatus = runTsdbStoreKernel((tsdb_control*)d_control, d_points, point_count);
	assert(cudaStatus == cudaSuccess);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float indexTimeMs;
	cudaEventElapsedTime(&indexTimeMs, start, stop);

	printf("[TSDB] write: indexing data: done in %f ms\n", indexTimeMs);

	// sync control
	cudaStatus = cudaDeviceSynchronize();
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMemcpy(h_control, d_control, sizeof(tsdb_control), cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaSuccess);

	// done
	tstop = GetCurrentTime();

	float elapsed = tstop - tstart;

	printf("[TSDB](ts: @%ld) write: done in %f ms\n", tstop, elapsed);

	// cleanup
	cudaFree(d_data);

	free(h_points);
	cudaFree(d_points);

	return;

Error:
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
	printf("\n********************\n");
}

void TSDB::process(size_t data_count) {
	const int dataSize = 3;
	const int transactionSize = 256;

	size_t transaction_count = data_count;
	size_t point_count = data_count * transactionSize;
	size_t mem_size = transaction_count * (1 + dataSize * transactionSize) * sizeof(float);

	printf("[TSDB] [process] block: %" PRId64 " points %" PRId64 " bytes\n", point_count, mem_size);

	cudaError_t cudaStatus;

	cudaEvent_t start, stop;
	DWORD tstart, tstop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	tstart = GetCurrentTime();

	printf("[TSDB](ts: @%ld) [process] starting\n", tstart);

	// alloc
	printf("[TSDB] [process] allocating memory\n");
	cudaEventRecord(start, 0);
	
	float* h_data, *h_resultset;
	h_data = (float*)malloc(mem_size);
	h_resultset = (float*)malloc(mem_size);

	float* d_data, *d_resultset;
	cudaStatus = cudaMalloc((void**)&d_data, mem_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_resultset, mem_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float allocTimeMs;
	cudaEventElapsedTime(&allocTimeMs, start, stop);

	printf("[TSDB] [process] allocating memory: done in %f ms\n", allocTimeMs);

	// read
	size_t samplesRead = 0;
	printf("[TSDB] [process] reading data\n");
	cudaEventRecord(start, 0);


	for (int i = 0; i < transaction_count; i++) {
		int ptr = i * (1 + dataSize * transactionSize);
		*(h_data + ptr) = i;	// t

		for (int j = 0; j < transactionSize; j++) {
			int ptr_trans = ptr + 1 + (j * 3);

			*(h_data + ptr_trans + 0) = i + 1;	// x
			samplesRead++;

			*(h_data + ptr_trans + 1) = i + 2;	// y
			samplesRead++;

			*(h_data + ptr_trans + 2) = i + 3;	// z
			samplesRead++;
		}
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float readTimeMs;
	cudaEventElapsedTime(&readTimeMs, start, stop);

	printf("[TSDB] [process] reading data: done in %f ms\n", readTimeMs);

	// writing
	printf("[TSDB] [process] write data test\n");

	for (int test = 0; test < 10; test++) {
		size_t metrics = 0;

		printf("**** WRITE TEST %d *****\n", test);
		cudaEventRecord(start, 0);

		printf("\tindexing data\n");
		metrics = 0;
		for (int i = 0; i < transaction_count; i++) {
			int ptr = i * (1 + 3 * 256);

			cudaStatus = cudaMemcpy(d_data + ptr, h_data + ptr, sizeof(float), cudaMemcpyHostToDevice);	// t
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "commit failed!");
				goto Error;
			}
			metrics += 256;
		}

		printf("\tstoring data\n");
		metrics = 0;
		for (int i = 0; i < transaction_count; i++) {
			int ptr = i * (1 + 3 * 256);

			cudaStatus = cudaMemcpy(d_data + ptr, h_data + ptr, (1 + dataSize * transactionSize) * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "commit failed!");
				goto Error;
			}
			metrics += 256;
		}

		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float writeTimeMs;
		cudaEventElapsedTime(&writeTimeMs, start, stop);

		printf("**** %" PRId64 " metrics\t%f ms\n", metrics, writeTimeMs);
		printf("******************\n");
	}

	// query
	size_t dataQueried = 0;
	printf("[TSDB] [process] querying data\n");

	int query_t = 0;
	int query_count = 100;

	tstart = GetCurrentTime();

	int data_ptr = query_t * (1 + 3 * 256);

	for (int i = 0; i < query_count; i++) {
		cudaStatus = cudaMemcpy(h_resultset + i * (1 + 3 * 256),
			d_data + data_ptr + i * (1 + 3 * 256),
			(1 + 3 * 256) * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "query failed!");
			goto Error;
		}

		cudaDeviceSynchronize();
		dataQueried += 3 * 256;
	}

	tstop = GetCurrentTime();
	float queryTimeMs = tstop - tstart;

	printf("[TSDB] [process] data query %" PRId64 " values in %f ms\n", dataQueried, queryTimeMs);

	for (int i = 0; i < query_count; i++) {
		float* record = h_resultset + i * (1 + 3 * 256);
		printf("[%d] (%f, %f, %f, %f)\n", i, record[0], record[1], record[2], record[3]);
	}

	// process
	printf("[TSDB] [process] process data test\n");
	for (int test = 0; test < 10; test++)
	{
		size_t dataProcessed = 0;
		const size_t targetDataSize = 220LL * 1024LL * 1024LL * 1024LL;

		printf("**** PROCESS TEST %d *****\n", test);
		cudaEventRecord(start, 0);

		int seq = 0;

		while (dataProcessed < targetDataSize) {
			cudaStatus = runTsdbRMSKernel(d_resultset, d_data, transactionSize, transaction_count);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "data processing failed!");
				goto Error;
			}
			cudaDeviceSynchronize();

			dataProcessed += transaction_count * dataSize * transactionSize * sizeof(float);
			seq++;

			if (seq >= 500) {
				seq = 0;
				printf("\tprocessed %" PRId64 " bytes\n", dataProcessed);
			}
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float processTimeMs;
		cudaEventElapsedTime(&processTimeMs, start, stop);

		printf("**** total %" PRId64 " bytes\t%f ms\n", dataProcessed, processTimeMs);
		printf("******************\n");
	}

	// fetch resultset
	printf("[TSDB] [process] fetching resultset\n");
	{
		int dataCount = transaction_count;

		int resultFetched = 0;

		tstart = GetCurrentTime();

		for (int i = 0; i < dataCount; i++) {
			cudaStatus = cudaMemcpy(h_resultset + i * transactionSize,
				d_resultset + i * transactionSize,
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
			float* record = h_resultset + i * (1 + dataSize * transactionSize);
			printf("[%d] (%f, %f, %f, %f)\n", i, record[0], record[1], record[2], record[3]);
		}

	}
	goto Error;

Error:
	// cleanup
	cudaFree(d_resultset);
	cudaFree(d_data);

	cudaFree(h_data);
	cudaFree(h_resultset);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
	printf("\n********************\n");
}


const std::string
TSDB::store(size_t node_count, size_t transaction_count) {
	std::string message;

	const int dataSize = 4;
	const int transactionSize = 256;

	const int blockSize = 1024;

	size_t point_count = transaction_count * transactionSize;
	size_t mem_size = transaction_count * transactionSize * dataSize * sizeof(float);
	size_t blockCount = ceil((double)point_count / blockSize);

	printf("[TSDB] [store] nodes: %d block: %" PRId64 " points %" PRId64 " bytes\n", node_count, point_count, mem_size);

	cudaError_t cudaStatus;

	cudaEvent_t start, stop;
	DWORD tstart, tstop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	tstart = GetCurrentTime();

	printf("[TSDB](ts: @%ld) [store] starting\n", tstart);

	// alloc
	printf("[TSDB] [store] allocating memory\n");
	cudaEventRecord(start, 0);

	float* h_data, *h_resultset;
	h_data = (float*)malloc(mem_size);
	h_resultset = (float*)malloc(mem_size);

	float* d_data, *d_resultset;
	cudaStatus = cudaMalloc((void**)&d_data, mem_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_resultset, mem_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float allocTimeMs;
	cudaEventElapsedTime(&allocTimeMs, start, stop);

	printf("[TSDB] [store] allocating memory: done in %f ms\n", allocTimeMs);

	// read
	size_t samplesRead = 0;
	printf("[TSDB] [store] reading data\n");
	cudaEventRecord(start, 0);

	for (int i = 0; i < transaction_count; i++) {
		int ptr = i * transactionSize * dataSize;
		*(h_data + ptr) = i;	// t

		for (int j = 0; j < transactionSize; j++) {
			int ptr_trans = ptr + (j * 4);

			*(h_data + ptr_trans + 0) = i;	// t
			samplesRead++;

			*(h_data + ptr_trans + 1) = i + 1;	// x
			samplesRead++;

			*(h_data + ptr_trans + 2) = i + 2;	// y
			samplesRead++;

			*(h_data + ptr_trans + 3) = i + 3;	// z
			samplesRead++;
		}
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float readTimeMs;
	cudaEventElapsedTime(&readTimeMs, start, stop);

	printf("[TSDB] [store] reading data: done in %f ms\n", readTimeMs);

	// indexing
	printf("[TSDB] [store] indexing\n");

	printf("[TSDB] [store] indexing: creating control...");

	cudaEventRecord(start, 0);

	tsdb_control* h_control = (tsdb_control*)malloc(sizeof(tsdb_control));
	tsdb_control* d_control;
	cudaStatus = cudaMalloc((void**)&d_control, sizeof(tsdb_control));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	printf("ok\n");

	printf("[TSDB] [store] indexing: creating blocks...\n");

	size_t blocksCreated = 0;
	tsdb_block* d_blockChunk;
	cudaStatus = cudaMalloc((void**)&d_blockChunk, sizeof(tsdb_block) * blockCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	blocksCreated += blockCount;
	printf("[TSDB] [store] indexing: creating blocks ok: %d blocks\n", blocksCreated);

	printf("[TSDB] [store] indexing: indexing %d blocks for %d nodes...\n", blocksCreated, node_count);
	for (int i = 0; i < transaction_count; i++) {
		int ptr = i * (dataSize * transactionSize);

		cudaStatus = cudaMemcpy(d_data + ptr, h_data + ptr, (dataSize * transactionSize) * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "commit failed!");
			goto Error;
		}
	}
	cudaDeviceSynchronize();

	cudaStatus = runTsdbStoreMultiKernel(d_control, node_count, (const tsdb_points*)d_data, transaction_count * transactionSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "runTsdbStoreKernel failed!");
		goto Error;
	}
	cudaDeviceSynchronize();

	printf("[TSDB] [store] indexing: indexing ok\n");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float indexTimeMs;
	cudaEventElapsedTime(&indexTimeMs, start, stop);

	printf("[TSDB] [store] indexing data: done in %f ms\n", indexTimeMs);

	// done
	tstop = GetCurrentTime();

	float elapsed = tstop - tstart;

	printf("[TSDB](ts: @%ld) [store] done in %f ms\n", tstop, elapsed);


	tstart = GetCurrentTime();
	message.reserve(memSize);
	message = std::string((const char*)h_data, mem_size);

	tstop = GetCurrentTime();
	elapsed = tstop - tstart;

	printf("[TSDB] [store] preparing data done in %f ms\n", elapsed);

Error:
	// cleanup
	cudaFree(d_resultset);
	cudaFree(d_data);

	cudaFree(h_data);
	cudaFree(h_resultset);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
	printf("\n********************\n");

	return message;
}
