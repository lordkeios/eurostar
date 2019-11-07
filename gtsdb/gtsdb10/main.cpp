#include "pch.h"

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

#include "System.h"
#include "Server.h"
#include "gtsdb.h"

// CUDA/D3D10 kernel
extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern "C" cudaError_t processRMSCuda(float* resultSet, float* data, int transactionSize, int transactionCount);

int main()
{
	const int dataSize = 3;
	const int transactionSize = 256;
	const int maxDataCount = 60 * 24 * 365;

	printf("\n**** 1. SYSTEM STARTING ****\n");

	System::start();

	auto& tsdb = TSDB::instance();
	tsdb.init(TSDB::options{ 256 });

	Server server;
	server.start();

	std::string command;

	while (server.isRunning)
	{
		printf("CMD>");

		std::getline(std::cin, command);

		server.onConsoleCommand(command);
	}

	System::stop();

	tsdb.clear();

	return 0;
}
