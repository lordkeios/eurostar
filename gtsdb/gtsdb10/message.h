#pragma once

#include "resource_pool.hpp"

struct message
{
	static const int BUFF_SIZE = 7168;

	struct _header
	{
		uint32_t size = 0;
		uint32_t id = 0;
		uint32_t reserved1 = 0;
		uint32_t reserved2 = 0;
	} header;

	char data[BUFF_SIZE] = { 0 };

	//
	bool isValid() {
		if (header.size > BUFF_SIZE) {
			return false;
		}
		if (header.size < 0) {
			return false;
		}
		return true;
	}
};

typedef resource_pool<message> message_pool_type;
