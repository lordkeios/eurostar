#pragma once

#include <boost/smart_ptr.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include "websocket_handlers.h"

// Forward declaration
class websocket_session;

// Represents the shared server state
class websocket_shared_state
{
private:
	std::string const doc_root_;

	// This mutex synchronizes all access to sessions_
	std::mutex mutex_;

	// Keep a list of all the connected clients
	std::unordered_set<websocket_session*> sessions_;

public:
	websocket_handlers handlers_;

public:
	explicit websocket_shared_state(std::string doc_root);

	std::string const&
		doc_root() const noexcept
	{
		return doc_root_;
	}

	void join(websocket_session* session);
	void leave(websocket_session* session);
	void send(std::string message);
	void receive(websocket_session* session, std::string message);
	bool hasSession(websocket_session* session);
};
