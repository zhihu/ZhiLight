#pragma once

#include <string>

namespace zmq {
class context_t;
class socket_t;
}

namespace c10d {

class HostCommunicator {
public:
    explicit HostCommunicator(const std::string addr, int nnodes, int node_rank = 0);

    ~HostCommunicator();

    void broadcast_data(char **data, int *nbytes);

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
};

} // namespace c10d