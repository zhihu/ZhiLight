#pragma once

#include "zmq.hpp"
#include <string>
#include <sstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace bmengine {
namespace c10d {

class HostCommunicator {
public:
    explicit HostCommunicator(
        const std::string addr,
        int nnodes,
        int node_rank = 0);

    ~HostCommunicator();

    template<typename T>
    void broadcast_data(T &obj_or_buf, int nbytes = 0) {
        if (nnodes_ == 1) {
            return;
        }
        char *data;
        if (node_rank_ == 0) { // master node, send
            if (nbytes == 0) {
                std::ostringstream oss(std::ios::binary);
                boost::archive::binary_oarchive oa(oss);
                oa << obj_or_buf;
                const std::string &str = oss.str();
                buffer_.assign(str.begin(), str.end());
                data = buffer_.data();
                nbytes = static_cast<int>(buffer_.size());
            } else {
                data = reinterpret_cast<char *>(obj_or_buf);
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
            if (nbytes == 0) {
                data = reinterpret_cast<char *>(msg.data());
                std::istringstream iss(std::string(data, data + msg.size()), std::ios::binary);
                boost::archive::binary_iarchive ia(iss);
                ia >> obj_or_buf;
            } else {
                //assert(nbytes == static_cast<int>(msg.size()));
                data = reinterpret_cast<char *>(obj_or_buf);
                memcpy(data, msg.data(), msg.size());
            }
        }
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

    zmq::context_t *ctx_;
    zmq::socket_t *sock_;

    std::vector<char> buffer_;
};

} // namespace c10d
} // namespace bmengine