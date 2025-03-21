#include "distributed/host_communicator.h"
#include <cstring>
#include <thread>
#include <chrono>
#include <iostream>

int main(int argc, char *argv[]) {
	distributed::HostCommunicator *hc;
	if (argc > 2) {
		hc = new distributed::HostCommunicator("10.98.6.51:2025", 3, 2);
		char *data = nullptr;
		int nbytes = 0;
		hc->broadcast_data(&data, &nbytes);
		std::cout << "Received: " << data << std::endl;
		delete [] data;
	} else if (argc > 1) {
		hc = new distributed::HostCommunicator("10.98.6.51:2025", 3, 1);
		char *data = nullptr;
		int nbytes = 0;
		hc->broadcast_data(&data, &nbytes);
		std::cout << "Received: " << data << std::endl;
		delete [] data;

	} else {
		hc = new distributed::HostCommunicator("10.98.6.51:2025", 3, 0);
		char *data = "hello body!";
		int nbytes = strlen(data);
		hc->broadcast_data(&data, &nbytes);
		std::cout << "Received: " << data << std::endl;
	}

	std::cout << "sleep ..." << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(60));
	std::cout << "sleep done!" << std::endl;
	delete hc;
	return 0;
}