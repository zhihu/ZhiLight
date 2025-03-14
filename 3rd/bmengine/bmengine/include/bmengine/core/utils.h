#pragma once
#include <utility>
#include <type_traits>
#include <vector>
#include <iostream>

#define BM_PIMPL                                                                                   \
private:                                                                                           \
    class impl;                                                                                    \
    std::unique_ptr<impl> pimpl;

#define BM_LAYER_DEF(name)                                                                         \
    BM_PIMPL                                                                                       \
public:                                                                                            \
    ~name();                                                                                       \
    name(const name&) = delete;                                                                    \
    name(name&&) = delete;                                                                         \
    template<typename... Params>                                                                   \
    inline auto operator()(Params&&... params) {                                                   \
        pthread_testcancel();                                                                      \
        return forward(std::forward<Params>(params)...);                                           \
    }                                                                                              \
    const char* layer_type() const override { return #name; }

#define BM_LAYER_DEF_PUBLIC(name)                                                                  \
public:                                                                                            \
    name(const name&) = delete;                                                                    \
    name(name&&) = delete;                                                                         \
    template<typename... Params>                                                                   \
    inline auto operator()(Params&&... params) {                                                   \
        pthread_testcancel();                                                                      \
        return forward(std::forward<Params>(params)...);                                           \
    }                                                                                              \
    const char* layer_type() const override { return #name; }

#define BM_KERNEL(name) BMEngine_KERNEL_##name

template<typename T, typename Tb>
inline T round_up(T m, Tb d) {
    return ((m + T(d) - 1) / T(d)) * T(d);
}
template<typename T, typename Tb>
inline T ceil_div(T m, Tb d) {
    return (m + T(d) - 1) / T(d);
}

#define MAX_NUM_THREADS 1024

template<typename T>
inline T round_up_thread(T m) {
    T d = 32, limit = MAX_NUM_THREADS;
    T x = m > limit ? limit : m;
    return ((x + d - 1) / d) * d;
}

template<typename T>
inline bool vector_equal(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] == b[i])
            ;
        else
            return false;
    return true;
}

template<typename Ta, typename Tb>
inline bool vector_equal_2(const std::vector<Ta>& a, const std::vector<Tb>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] == b[i])
            ;
        else
            return false;
    return true;
}
