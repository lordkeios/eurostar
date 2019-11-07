#pragma once

#include <functional>

struct websocket_handlers {
	std::function<void(void*)> on_connect;
	std::function<void(void*)> on_disconnect;
	std::function<void(void*, std::string)> on_msg;
};
