#pragma once

#include "websocket_shared_state.h"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/optional.hpp>
#include <boost/smart_ptr.hpp>
#include <cstdlib>
#include <memory>

namespace net = boost::asio;                    // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;
namespace beast = boost::beast;                 // from <boost/beast.hpp>
namespace http = beast::http;                   // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;         // from <boost/beast/websocket.hpp>

class websocket_http_session : public boost::enable_shared_from_this<websocket_http_session>
{
	beast::tcp_stream stream_;
	beast::flat_buffer buffer_;
	boost::shared_ptr<websocket_shared_state> state_;

	// The parser is stored in an optional container so we can
	// construct it from scratch it at the beginning of each new message.
	boost::optional<http::request_parser<http::string_body>> parser_;

	struct send_lambda;

	void fail(beast::error_code ec, char const* what);
	void do_read();
	void on_read(beast::error_code ec, std::size_t);
	void on_write(beast::error_code ec, std::size_t, bool close);

public:
	websocket_http_session(
		tcp::socket&& socket,
		boost::shared_ptr<websocket_shared_state> const& state);

	void run();
};
