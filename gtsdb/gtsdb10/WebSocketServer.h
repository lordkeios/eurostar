#pragma once

class websocket_server;
class websocket_shared_state;

class WebSocketServer
{
public:
	typedef struct options {
		std::string address;
		unsigned short port;
		std::string doc_root;
		int num_threads;
	} options;

	typedef struct handlers {
		std::function<void(void*)> on_connect;
		std::function<void(void*)> on_disconnect;
		std::function<void(void*, std::string)> on_msg;
	} handlers;

public:
	WebSocketServer();
	virtual ~WebSocketServer();

public:
	void init(const options& options);
	void setHandlers(const handlers& handlers);

	void start();
	void stop();

	void send(void* session, const std::string& msg);
	void disconnect(void* session);

private:
	options options_;
	handlers handlers_;
	boost::scoped_ptr<boost::asio::io_context> ioc_;
	boost::shared_ptr<websocket_server> ws_;
	boost::shared_ptr<websocket_shared_state> state_;
	std::thread thread_;
};
