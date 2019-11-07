#pragma once

#include "types.h"

class TSDB {
public:
	struct options {
		size_t block_size;
	};
public:
	int memSize;
	int count;
	int ptr;

	tsdb_control* h_control;
	void* d_control;

	float* h_data;
	float* d_data;

	float* h_resultset;
	float* d_resultset;

	// singleton
private:
	TSDB() : memSize(0), count(0), ptr(0) {
	}
public:
	TSDB(const TSDB&) = delete;
	TSDB& operator=(const TSDB &) = delete;
	TSDB(TSDB &&) = delete;
	TSDB& operator=(TSDB &&) = delete;

	static TSDB& instance() {
		static TSDB _instance;
		return _instance;
	}

public:
	void init(const options& opts);
	void clear();
	void write(const std::string& msg);
	void process(size_t data_count);
	const std::string store(size_t node_count, size_t transaction_count);
};
