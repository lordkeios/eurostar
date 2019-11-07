#include "pch.h"
#include "websocket_session.h"
#include <iostream>

websocket_session::
websocket_session(
	tcp::socket&& socket,
	boost::shared_ptr<websocket_shared_state> const& state)
	: ws_(std::move(socket))
	, state_(state)
{
	ws_.read_message_max((std::numeric_limits<std::uint64_t>::max)());
}

websocket_session::
~websocket_session()
{
	destroying_ = true;

	// Remove this session from the list of active sessions
	state_->leave(this);
}

void
websocket_session::
fail(beast::error_code ec, char const* what)
{
	// Don't report these
	if (ec == net::error::operation_aborted ||
		ec == websocket::error::closed)
		return;

	std::cerr << what << ": " << ec.message() << "\n";
}

void
websocket_session::
on_accept(beast::error_code ec)
{
	// Handle the error, if any
	if (ec)
		return fail(ec, "accept");

	// Add this session to the list of active sessions
	state_->join(this);

	// Read a message
	ws_.async_read(
		buffer_,
		beast::bind_front_handler(
			&websocket_session::on_read,
			shared_from_this()));
}

void
websocket_session::
on_read(beast::error_code ec, std::size_t)
{
	// Handle the error, if any
	if (ec)
		return fail(ec, "read");

	// handle receive
	state_->receive(this, beast::buffers_to_string(buffer_.data()));

	// Clear the buffer
	buffer_.consume(buffer_.size());

	// Read another message
	ws_.async_read(
		buffer_,
		beast::bind_front_handler(
			&websocket_session::on_read,
			shared_from_this()));
}

void
websocket_session::on_close(beast::error_code ec)
{
	if (ec)
		return fail(ec, "close");

	// If we get here then the connection is closed gracefully

	// The make_printable() function helps print a ConstBufferSequence
	std::cout << beast::make_printable(buffer_.data()) << std::endl;
}

void
websocket_session::
send(boost::shared_ptr<std::string const> const& ss)
{
	if (destroying_)
		return;

	// Post our work to the strand, this ensures
	// that the members of `this` will not be
	// accessed concurrently.

	net::post(
		ws_.get_executor(),
		beast::bind_front_handler(
			&websocket_session::on_send,
			shared_from_this(),
			ss));
}

void websocket_session::disconnect() {
	if (destroying_)
		return;

	// Close the WebSocket connection
	ws_.async_close(websocket::close_code::normal,
		beast::bind_front_handler(
			&websocket_session::on_close,
			shared_from_this()));
}

void
websocket_session::
on_send(boost::shared_ptr<std::string const> const& ss)
{
	// Always add to queue
	queue_.push_back(ss);

	// Are we already writing?
	if (queue_.size() > 1)
		return;

	// We are not currently writing, so send this immediately
	ws_.async_write(
		net::buffer(*queue_.front()),
		beast::bind_front_handler(
			&websocket_session::on_write,
			shared_from_this()));
}

void
websocket_session::
on_write(beast::error_code ec, std::size_t)
{
	// Handle the error, if any
	if (ec)
		return fail(ec, "write");

	// Remove the string from the queue
	queue_.erase(queue_.begin());

	// Send the next message if any
	if (!queue_.empty())
		ws_.async_write(
			net::buffer(*queue_.front()),
			beast::bind_front_handler(
				&websocket_session::on_write,
				shared_from_this()));
}
