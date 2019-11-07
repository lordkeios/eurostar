#include "pch.h"
#include "System.h"

System::System() {

}

void System::start() {
	System::instance().taskThread_ = boost::make_shared<TaskThread>();
}

void System::stop() {
	System::instance().taskThread_->terminate();
	System::instance().taskThread_.reset();
}

void System::enqueueTask(const std::function<void()>& task) {
	if (System::instance().taskThread_.get() == nullptr) {
		printf("[SYSTEM] enqueueTask: ERROR!! TASK THREAD NULL");
		return;
	}
	System::instance().taskThread_->enqueue(task);
}
