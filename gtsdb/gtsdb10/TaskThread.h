#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class TaskThread {
public:
	TaskThread();
	template<class F, class... Args>
	auto enqueue(F&& f, Args&&... args)
		->std::future<typename std::result_of<F(Args...)>::type>;
	~TaskThread();
	void terminate();

	bool isThread();
private:
	// need to keep track of threads so we can join them
	std::thread worker;
	// the task queue
	std::queue< std::function<void()> > tasks;

	// synchronization
	std::mutex queue_mutex;
	std::condition_variable condition;
	bool stop;
};

inline TaskThread::TaskThread()
	: stop(false)
{
	worker = std::thread([this]
	{
		for (;;)
		{
			std::function<void()> task;

			{
				std::unique_lock<std::mutex> lock(this->queue_mutex);
				this->condition.wait(lock,
					[this] { return this->stop || !this->tasks.empty(); });
				if (this->stop && this->tasks.empty())
					return;
				task = std::move(this->tasks.front());
				this->tasks.pop();
			}

			task();
		}
	});
}

// add new work item to the pool
template<class F, class... Args>
auto TaskThread::enqueue(F&& f, Args&&... args)
-> std::future<typename std::result_of<F(Args...)>::type>
{
	using return_type = typename std::result_of<F(Args...)>::type;

	auto task = std::make_shared< std::packaged_task<return_type()> >(
		std::bind(std::forward<F>(f), std::forward<Args>(args)...)
		);

	std::future<return_type> res = task->get_future();
	{
		std::unique_lock<std::mutex> lock(queue_mutex);

		// don't allow enqueueing after stopping the pool
		if (stop)
			throw std::runtime_error("enqueue on stopped TaskThread");

		tasks.emplace([task]() { (*task)(); });
	}
	condition.notify_one();
	return res;
}

// the destructor joins all threads
inline TaskThread::~TaskThread()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();

	if (worker.joinable())
		worker.join();
}

inline
void TaskThread::terminate() {
	enqueue([this] {
		stop = true;
	});

	for (;;) {
		std::unique_lock<std::mutex> lock(this->queue_mutex);
		if (this->tasks.empty())
			break;
	}

	if (worker.joinable())
		worker.join();
}

inline
bool TaskThread::isThread() {
	return std::this_thread::get_id() == worker.get_id();
}
