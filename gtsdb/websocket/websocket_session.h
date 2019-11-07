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

#include "websocket_shared_state.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
class websocket_shared_state;

/** Represents an active WebSocket connection to the server
*/
class websocket_session : public boost::enable_shared_from_this<websocket_session>
{
	beast::flat_buffer buffer_;
	websocket::stream<beast::tcp_stream> ws_;
	boost::shared_ptr<websocket_shared_state> state_;
	std::vector<boost::shared_ptr<std::string const>> queue_;
	bool destroying_ = false;

	void fail(beast::error_code ec, char const* what);
	void on_accept(beast::error_code ec);
	void on_read(beast::error_code ec, std::size_t bytes_transferred);
	void on_write(beast::error_code ec, std::size_t bytes_transferred);
	void on_close(beast::error_code ec);

public:
	websocket_session(
		tcp::socket&& socket,
		boost::shared_ptr<websocket_shared_state> const& state);

	~websocket_session();

	template<class Body, class Allocator>
	void
		run(http::request<Body, http::basic_fields<Allocator>> req);

	// Send a message
	void send(boost::shared_ptr<std::string const> const& ss);

	void disconnect();

private:
	void on_send(boost::shared_ptr<std::string const> const& ss);
};

template<class Body, class Allocator>
void
websocket_session::
run(http::request<Body, http::basic_fields<Allocator>> req)
{
	// Set suggested timeout settings for the websocket
	ws_.set_option(
		websocket::stream_base::timeout::suggested(
			beast::role_type::server));

	// Set a decorator to change the Server of the handshake
	ws_.set_option(websocket::stream_base::decorator(
		[](websocket::response_type& res)
	{
		res.set(http::field::server,
			std::string(BOOST_BEAST_VERSION_STRING) +
			" websocket-chat-multi");
	}));

	// Accept the websocket handshake
	ws_.async_accept(
		req,
		beast::bind_front_handler(
			&websocket_session::on_accept,
			shared_from_this()));
}
