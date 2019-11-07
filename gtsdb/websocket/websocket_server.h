#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/coroutine.hpp>
#include <boost/asio/strand.hpp>
#include <boost/optional.hpp>
#include <boost/smart_ptr.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

// Forward declaration
class websocket_shared_state;
struct websocket_handlers;

// Accepts incoming connections and launches the sessions
class websocket_server : public boost::enable_shared_from_this<websocket_server>
{
public:
	typedef struct options {
		std::string address;
		unsigned short port;
		std::string doc_root;
		int num_threads;
	} options;

private:
	net::io_context& ioc_;
	tcp::acceptor acceptor_;
	boost::shared_ptr<websocket_shared_state> state_;

	void fail(beast::error_code ec, char const* what);
	void on_accept(beast::error_code ec, tcp::socket socket);

public:
	static void run_server(net::io_context& ioc, const options& options, const websocket_handlers& handlers);

public:
	websocket_server(
		net::io_context& ioc,
		tcp::endpoint endpoint,
		boost::shared_ptr<websocket_shared_state> const& state);

	// Start accepting incoming connections
	void run();
};
