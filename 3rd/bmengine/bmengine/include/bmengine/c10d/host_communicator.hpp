#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

#ifdef ENABLE_DIST_INFER
#include "zmq.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#endif

namespace bmengine {
namespace c10d {

class HostCommunicator {
public:
    explicit HostCommunicator(
        const std::string addr,
        int nnodes,
        int node_rank = 0) {
            addr_ = "tcp://" + addr; // For example -> tcp://10.91.240.4:2025
            nnodes_ = nnodes;
            node_rank_ = node_rank;
        
            buffer_.resize(4 * 1024 * 1024); // 4M
        
            if (nnodes_ > 1) {
#ifdef ENABLE_DIST_INFER
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
#else
                throw std::runtime_error("ENABLE_DIST_INFER is not on for multi nodes.");
#endif
            }
        }

    ~HostCommunicator() {
        if (nnodes_ > 1) {
#ifdef ENABLE_DIST_INFER
            sock_->close();
            delete sock_;
            delete ctx_;
#endif
        }
    }

    template<typename T>
    void broadcast_data(T &obj_or_buf, int nbytes = 0) {
#ifdef ENABLE_DIST_INFER
        if (nnodes_ == 1) {
            return;
        }
        char *data;
        if (node_rank_ == 0) { // master node, send
            if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
                data = reinterpret_cast<char *>(obj_or_buf);
            } else {
                std::ostringstream oss(std::ios::binary);
                boost::archive::binary_oarchive oa(oss);
                oa << obj_or_buf;
                const std::string &str = oss.str();
                buffer_.assign(str.begin(), str.end());
                data = buffer_.data();
                nbytes = static_cast<int>(buffer_.size());
            }

            for (int i = 1; i < nnodes_; ++i) {
                zmq::message_t msg0(0);
                sock_->recv(&msg0);
                zmq::message_t msg(nbytes);
                // maybe no need to copy
                memcpy(msg.data(), data, nbytes);
                sock_->send(msg);
            }
        } else { // slave node, recv
            zmq::message_t msg0(0);
            sock_->send(msg0);
            zmq::message_t msg;
            sock_->recv(&msg);
            if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
                //assert(nbytes == static_cast<int>(msg.size()));
                data = reinterpret_cast<char *>(obj_or_buf);
                memcpy(data, msg.data(), msg.size());
            } else {
                data = reinterpret_cast<char *>(msg.data());
                std::istringstream iss(std::string(data, data + msg.size()), std::ios::binary);
                boost::archive::binary_iarchive ia(iss);
                ia >> obj_or_buf;
            }
        }
#endif
    }

    inline int get_nnodes() const {
        return nnodes_;
    }

    inline int get_node_rank() const {
        return node_rank_;
    }

private:
    std::string addr_;
    int nnodes_;
    int node_rank_;

#ifdef ENABLE_DIST_INFER
    zmq::context_t *ctx_;
    zmq::socket_t *sock_;
#endif

    std::vector<char> buffer_;
};

} // namespace c10d
} // namespace bmengine