#pragma once

#include "ThreadPool.h"

class WebSocketServer;

class Server
{
public:
	Server();
	~Server();

	void start();
	void stop();
	void proc();

	void onConsoleCommand(const std::string& cmd);

public:
	bool isRunning;

private:
	int node = 0;


	boost::shared_ptr<WebSocketServer> webSocketServer_;
	boost::shared_ptr<ThreadPool> ioThread_;
	boost::shared_ptr<ThreadPool> workerThread_;
	boost::asio::io_context io_context_;

	boost::condition_variable cond_started;
	boost::condition_variable cond_stopped;

	boost::mutex mutex_;
};

