#include "bmengine/c10d/host_communicator.h"
#include "zmq.hpp"
#include <thread>
#include <chrono>
#include <iostream>

namespace bmengine {
namespace c10d {

HostCommunicator::HostCommunicator(std::string addr, int nnodes, int node_rank) {
    addr_ = "tcp://" + addr; // For example -> tcp://10.91.240.4:2025
    nnodes_ = nnodes;
    node_rank_ = node_rank;

    if (nnodes_ > 1) {
        ctx_ = new zmq::context_t(1);
        if (node_rank == 0) {
            sock_ = new zmq::socket_t(*ctx_, ZMQ_REP);
            sock_->bind(addr_.c_str());
            for (int i = 1; i < nnodes_; ++i) {
                zmq::message_t msg(0);
                sock_->recv(&msg);
                sock_->send(msg);
            }
            std::cout << "Node node_rank=" << node_rank_ << " listening on " << addr_ << std::endl;
        } else {
            sock_ = new zmq::socket_t(*ctx_, ZMQ_REQ);
            sock_->connect(addr_.c_str());
            zmq::message_t msg(0);
            sock_->send(msg);
            sock_->recv(&msg);
            std::cout << "Node node_rank=" << node_rank_ << " connected to " << addr_ << std::endl;
        }
    }
}

HostCommunicator::~HostCommunicator() {
    if (nnodes_ > 1) {
        sock_->close();
        delete sock_;
        delete ctx_;
    }
}

void HostCommunicator::broadcast_data(char **data, int *nbytes) {
    if (node_rank_ == 0) {
        for (int i = 1; i < nnodes_; ++i) {
            zmq::message_t msg0(0);
            sock_->recv(&msg0);
            zmq::message_t msg(*nbytes);
            memcpy(msg.data(), *data, *nbytes);
            sock_->send(msg);
        }
    } else {
        zmq::message_t msg0(0);
        sock_->send(msg0);
        zmq::message_t msg;
        sock_->recv(&msg);
        //*data = new char[msg.size()];
        *nbytes = msg.size();
        memcpy(*data, msg.data(), *nbytes);
    }
}

} // namespace c10d
} // namespace bmengine