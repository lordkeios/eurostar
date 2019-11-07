#pragma once

struct tsdb_control {
	unsigned int node_count;
	unsigned long long sequence;
};

struct tsdb_points {
	float time;
	float value[3];
};

struct tsdb_block {
	unsigned long index;
	unsigned long long next;
};
