#include "pch.h"
#include "websocket_shared_state.h"
#include "websocket_session.h"

websocket_shared_state::
websocket_shared_state(std::string doc_root)
	: doc_root_(std::move(doc_root))
{
}

void
websocket_shared_state::
join(websocket_session* session)
{
	std::lock_guard<std::mutex> lock(mutex_);
	sessions_.insert(session);

	if (handlers_.on_connect) {
		handlers_.on_connect(session);
	}
}

void
websocket_shared_state::
leave(websocket_session* session)
{
	std::lock_guard<std::mutex> lock(mutex_);
	
	if (handlers_.on_disconnect) {
		handlers_.on_disconnect(session);
	}
	sessions_.erase(session);
}

// Broadcast a message to all websocket client sessions
void
websocket_shared_state::send(std::string message)
{
	// Put the message in a shared pointer so we can re-use it for each client
	auto const ss = boost::make_shared<std::string const>(std::move(message));

	// Make a local list of all the weak pointers representing
	// the sessions, so we can do the actual sending without
	// holding the mutex:
	std::vector<boost::weak_ptr<websocket_session>> v;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		v.reserve(sessions_.size());
		for (auto p : sessions_)
			v.emplace_back(p->weak_from_this());
	}

	// For each session in our local list, try to acquire a strong
	// pointer. If successful, then send the message on that session.
	for (auto const& wp : v)
		if (auto sp = wp.lock())
			sp->send(ss);
}

void websocket_shared_state::receive(websocket_session* session, std::string message) {
	// Send to all connections
	//send(message);

	if (handlers_.on_msg) {
		handlers_.on_msg(session, message);
	}
}


bool websocket_shared_state::hasSession(websocket_session* session) {
	std::lock_guard<std::mutex> lock(mutex_);
	return sessions_.count(session) != 0;
}
