#include "pch.h"
#include "websocket_server.h"
#include "websocket_http_session.h"
#include <iostream>

void websocket_server::run_server(net::io_context& ioc, const options& options, const websocket_handlers& handlers) {
	auto addr = net::ip::make_address(options.address);
	auto port = static_cast<unsigned short>(options.port);
	auto doc_root = options.doc_root;
	auto const threads = std::max<int>(1, options.num_threads);

	boost::shared_ptr<websocket_server> ws = boost::make_shared<websocket_server>(
		ioc,
		tcp::endpoint{ addr, port },
		boost::make_shared<websocket_shared_state>(doc_root));
	
	ws->state_->handlers_ = handlers;

	ws->run();

	// Capture SIGINT and SIGTERM to perform a clean shutdown
	net::signal_set signals(ioc, SIGINT, SIGTERM);
	signals.async_wait(
		[&ioc](boost::system::error_code const&, int)
	{
		// Stop the io_context. This will cause run()
		// to return immediately, eventually destroying the
		// io_context and any remaining handlers in it.
		ioc.stop();
	});

	// Run the I/O service on the requested number of threads
	std::vector<std::thread> v;
	v.reserve(threads - 1);
	for (auto i = threads - 1; i > 0; --i)
		v.emplace_back(
			[&ioc]
	{
		ioc.run();
	});
	ioc.run();

	// (If we get here, it means we got a SIGINT or SIGTERM)

	// Block until all the threads exit
	for (auto& t : v)
		t.join();
}

websocket_server::websocket_server(
	net::io_context& ioc,
	tcp::endpoint endpoint,
	boost::shared_ptr<websocket_shared_state> const& state)
	: ioc_(ioc)
	, acceptor_(ioc)
	, state_(state)
{
	beast::error_code ec;

	// Open the acceptor
	acceptor_.open(endpoint.protocol(), ec);
	if (ec)
	{
		fail(ec, "open");
		return;
	}

	// Allow address reuse
	acceptor_.set_option(net::socket_base::reuse_address(true), ec);
	if (ec)
	{
		fail(ec, "set_option");
		return;
	}

	// Bind to the server address
	acceptor_.bind(endpoint, ec);
	if (ec)
	{
		fail(ec, "bind");
		return;
	}

	// Start listening for connections
	acceptor_.listen(
		net::socket_base::max_listen_connections, ec);
	if (ec)
	{
		fail(ec, "listen");
		return;
	}
}

void
websocket_server::
run()
{
	// The new connection gets its own strand
	acceptor_.async_accept(
		net::make_strand(ioc_),
		beast::bind_front_handler(
			&websocket_server::on_accept,
			shared_from_this()));
}

// Report a failure
void
websocket_server::
fail(beast::error_code ec, char const* what)
{
	// Don't report on canceled operations
	if (ec == net::error::operation_aborted)
		return;
	std::cerr << what << ": " << ec.message() << "\n";
}

// Handle a connection
void
websocket_server::
on_accept(beast::error_code ec, tcp::socket socket)
{
	if (ec)
		return fail(ec, "accept");
	else
		// Launch a new session for this connection
		boost::make_shared<websocket_http_session>(
			std::move(socket),
			state_)->run();

	// The new connection gets its own strand
	acceptor_.async_accept(
		net::make_strand(ioc_),
		beast::bind_front_handler(
			&websocket_server::on_accept,
			shared_from_this()));
}
