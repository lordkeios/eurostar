#include "pch.h"
#include "Server.h"

#include "WebSocketServer.h"
#include "WebSocketClient.h"

#include "System.h"
#include "gtsdb.h"

namespace po = boost::program_options;

Server::Server()
{
	std::vector<std::string> args = po::split_winmain(GetCommandLine());
	node = 0;
	if (args.size() >= 2) {
		try {
			node = boost::lexical_cast<short>(args[1]);
		}
		catch (const std::exception& e) {
			
		}
	}

	printf("[SERVER] [init] node=%d\n", node);

	webSocketServer_ = boost::make_shared<WebSocketServer>();
	ioThread_ = boost::make_shared<ThreadPool>(16);
	workerThread_ = boost::make_shared<ThreadPool>(16);

	WebSocketServer::options wsOptions;
	wsOptions.address = "0.0.0.0";
	wsOptions.port = 4333 + node;
	wsOptions.num_threads = 4;
	webSocketServer_->init(wsOptions);
	WebSocketServer::handlers wsHandlers;
	wsHandlers.on_connect = [=](void* session) {
	};
	wsHandlers.on_msg = [=](void* session, const std::string& msg) {
		printf("[SERVER] on_msg: receive: %ld bytes\n", msg.size());

		webSocketServer_->send(session, "ok");

		System::enqueueTask([=] {
			TSDB::instance().write(msg);
		});
	};
	wsHandlers.on_disconnect = [=](void* session) {
	};
	webSocketServer_->setHandlers(wsHandlers);
}


Server::~Server()
{
	ioThread_.reset();
}

void Server::start() {
	ioThread_->enqueue(boost::bind(&Server::proc, this));
	{
		boost::mutex::scoped_lock lock(mutex_);
		while (!isRunning)
			cond_started.wait(lock);
	}
}

void Server::stop() {
	io_context_.stop();
	{
		boost::mutex::scoped_lock lock(mutex_);
		while (isRunning)
			cond_stopped.wait(lock);
	}
}

void Server::proc() {

	// start
	webSocketServer_->start();

	{
		boost::mutex::scoped_lock lock(mutex_);
		isRunning = true;
		cond_started.notify_all();
	}

	// run
	boost::asio::io_context::work worker(io_context_);
	io_context_.run();


	// stop
	webSocketServer_->stop();

	{
		boost::mutex::scoped_lock lock(mutex_);
		isRunning = false;
		cond_stopped.notify_all();
	}
}

void Server::onConsoleCommand(const std::string& cl)
{
	printf("onConsoleCommand: %s\n", cl.c_str());

	std::vector<std::string> args = po::split_winmain(cl);
	if (args.empty())
		return;

	std::string cmd = args[0];

	if (cmd == "exit") {
		stop();
		return;
	}
	else if (cmd == "process") {
		size_t point_count = args.size() >= 2 ? boost::lexical_cast<int64_t>(args[1]) : 3.5 * 1000 * 1000;
		size_t node_count = args.size() >= 3 ? boost::lexical_cast<int64_t>(args[2]) : 1;
		size_t transaction_count = args.size() >= 2 ? boost::lexical_cast<int64_t>(args[1]) : 13672;

		System::enqueueTask([=] {
			TSDB::instance().process(transaction_count);
		});
		return;


		printf("[CLIENT] load: loading %ld points\n", point_count);

		std::string message;
		message.reserve(point_count * sizeof(float) * 2 * 8);

		for (int32_t i = 0; i < point_count; i++) {
			std::ostringstream ss;
			ss << std::setw(8) << std::setfill('0') << i;
			message += ss.str();

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";
		}

		for (int i = 0; i < node_count; i++) {
			int64_t total_bytes = 0;
			while (total_bytes < 220LL * 1024LL * 1024LL * 1024LL) {
				printf("[CLIENT] send: node: %d websocket: sending %ld points (%ld bytes)\n", i, point_count, message.size());

				bool completed = false;
				auto client = std::make_shared<WebSocketClient>(io_context_);
				client->on_complete = [=, &completed]() {
					completed = true;
				};
				//client->run("localhost", boost::lexical_cast<std::string>(4333 + i).c_str(), message);

				while (!completed) {
					Sleep(0);
				}

				total_bytes += message.size();

				printf("[CLIENT] send: node: %d websocket: sent total %" PRId64 " bytes\n", i, total_bytes);
			}
		}
	}
	else if (cmd == "store") {
		size_t node_count = args.size() >= 2 ? boost::lexical_cast<int64_t>(args[1]) : 2;
		size_t transaction_count = args.size() >= 3 ? boost::lexical_cast<int64_t>(args[2]) : 13672;

		System::enqueueTask([=] {
			DWORD tstart, tstop;
			tstart = GetCurrentTime();

			auto message = boost::make_shared<std::string>(TSDB::instance().store(node_count, transaction_count));

			for (int i = 0; i < node_count; i++) {
				std::make_shared<WebSocketClient>(io_context_)->run("localhost", boost::lexical_cast<std::string>(4333 + i + 1).c_str(), message);
			}

			tstop = GetCurrentTime();
			float elapsed = tstop - tstart;

			printf("******** store done in %f ms\n", elapsed);
		});
	}
	else if(cmd == "test") {
		size_t point_count = 3.5 * 1000 * 1000;

		printf("[CLIENT] load: loading %ld points\n", point_count);

		std::string message;
		message.reserve(point_count * sizeof(float) * 2 * 8);

		for (int32_t i = 0; i < point_count; i++) {
			std::ostringstream ss;
			ss << std::setw(8) << std::setfill('0') << i;
			message += ss.str();

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";

			message += "1234";
			message += "5678";
		}

		printf("[CLIENT] send: websocket: sending %ld points\n", point_count);

		//std::make_shared<WebSocketClient>(io_context_)->run("localhost", "4333", message);
	}
}
