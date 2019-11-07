#pragma once

#include "TaskThread.h"

class System
{
	boost::shared_ptr<TaskThread> taskThread_;
	// singleton
private:
	System();
public:
	System(const System&) = delete;
	System& operator=(const System &) = delete;
	System(System &&) = delete;
	System& operator=(System &&) = delete;

	static System& instance() {
		static System _instance;
		return _instance;
	}

public:
	static void start();
	static void stop();

public:
	static void enqueueTask(const std::function<void()>& task);
};

