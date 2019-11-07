#include "pch.h"
#include "WebSocketServer.h"

#include "websocket_server.h"
#include "websocket_handlers.h"
#include "websocket_session.h"

WebSocketServer::WebSocketServer() {

}

WebSocketServer::~WebSocketServer() {

}

void WebSocketServer::init(const options& options) {
	options_ = options;
}

void WebSocketServer::setHandlers(const handlers& handlers) {
	handlers_ = handlers;
}

void WebSocketServer::start() {
	websocket_server::options opts;
	opts.address = options_.address;
	opts.port = options_.port;
	opts.doc_root = options_.doc_root;
	opts.num_threads = options_.num_threads;

	websocket_handlers handlers;
	handlers.on_connect = handlers_.on_connect;
	handlers.on_disconnect = handlers_.on_disconnect;
	handlers.on_msg = handlers_.on_msg;

	auto addr = net::ip::make_address(opts.address);
	auto port = static_cast<unsigned short>(opts.port);
	auto doc_root = opts.doc_root;
	auto const threads = std::max<int>(1, opts.num_threads);

	ioc_.reset(new boost::asio::io_context());

	boost::shared_ptr<websocket_shared_state> state = boost::make_shared<websocket_shared_state>(doc_root);
	boost::shared_ptr<websocket_server> ws = boost::make_shared<websocket_server>(
		*ioc_,
		tcp::endpoint{ addr, port },
		state);

	state->handlers_ = handlers;

	ws_ = ws;
	state_ = state;

	thread_ = std::thread([=] {
		std::cout << "* WebSocketServer thread start" << std::endl;

		ws->run();

		// Capture SIGINT and SIGTERM to perform a clean shutdown
		net::signal_set signals(*ioc_, SIGINT, SIGTERM);
		signals.async_wait(
			[=](boost::system::error_code const&, int)
		{
			// Stop the io_context. This will cause run()
			// to return immediately, eventually destroying the
			// io_context and any remaining handlers in it.
			ioc_->stop();
		});

		// Run the I/O service on the requested number of threads
		std::vector<std::thread> v;
		v.reserve(threads - 1);
		for (auto i = threads - 1; i > 0; --i)
			v.emplace_back(
				[=]
		{
			ioc_->run();
		});
		ioc_->run();

		// (If we get here, it means we got a SIGINT or SIGTERM)

		// Block until all the threads exit
		for (auto& t : v)
			t.join();

		std::cout << "* WebSocketServer thread stop" << std::endl;
	});
}

void WebSocketServer::stop() {
	if (ioc_.get())
		ioc_->stop();

	thread_.join();

	ioc_.reset();

	state_.reset();
	ws_.reset();
}

void WebSocketServer::send(void* session, const std::string& msg) {
	websocket_session* ws = static_cast<websocket_session*>(session);

	if (state_.get() && state_->hasSession(ws))
		ws->send(boost::make_shared<std::string const>(std::move(msg)));
}

void WebSocketServer::disconnect(void* session) {
	websocket_session* ws = static_cast<websocket_session*>(session);

	if (state_.get() && state_->hasSession(ws))
		ws->disconnect();
}
