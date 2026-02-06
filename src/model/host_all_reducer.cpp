#include "model/model_context.h"
#include "model/allocate_util.hpp"
#include "model/dyn_batch_context.h"
#include "model/model.h"
#include "utils/env.h"
#include <bmengine/core/thread_pool.h>
#include <bmengine/functions/element.h>
#include "bmengine/logger/kernel_time_trace.hpp"
#include <bmengine/logger/std_log_op.hpp>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#if defined(__x86_64__)
#include <immintrin.h>
#include <emmintrin.h>
#endif

namespace model {

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::functions::BinaryElementwiseOp;
using std::vector;
typedef std::unique_lock<std::mutex> Lock;
// std::experimental::simd (P0214)

#if defined(__x86_64__)
#pragma GCC push_options
#pragma GCC optimize ("O3")
#pragma GCC target ("avx", "f16c", "avx512f")
void sum_inplace_avx(float* out, const float* in, int len) {
    for (int i = 0; i < len; i += 8) {
        __m256 s = _mm256_load_ps(&out[i]);
        __m256 v = _mm256_load_ps(&in[i]);
        s = _mm256_add_ps(s, v);
        _mm256_store_ps(&out[i], s);
    }
}
//
//void sum_inplace_avx(float* out, float* a, float* b, float* c, int len) {
//    for (int i = 0; i < len; i += 8) {
//        __m256 s = _mm256_load_ps(&out[i]);
//        __m256 v = _mm256_load_ps(&a[i]);
//        s = _mm256_add_ps(s, v);
//        v = _mm256_load_ps(&b[i]);
//        s = _mm256_add_ps(s, v);
//        v = _mm256_load_ps(&c[i]);
//        s = _mm256_add_ps(s, v);
//        _mm256_store_ps(&out[i], s);
//    }
//}
//
void sum_inplace(float* out, const float* a, const float* b, int len) {
    for (int i = 0; i < len; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sum_inplace(float* out, float* a, float* b, float* c, float* d, int len) {
    for (int i = 0; i < len; ++i) {
        out[i] = a[i] + b[i] + c[i] + d[i];
    }
}
void sum_inplace(float* out, float* a, float* b, float* c, float* d, float* e, float* f, float* g, float* h, int len) {
    for (int i = 0; i < len; ++i) {
        out[i] = a[i] + b[i] + c[i] + d[i] + e[i] + f[i] + g[i] + h[i];
    }
}
void sum_inplace(half* out, half* a, half* b, int len) {
    // _mm_load_si128: Load 8 x half
    // _mm256_cvtph_ps: Convert 8 x half to 8 x float
    // _mm256_add_ps: Add (8 x float, 8 x float)
    // _mm256_cvtps_ph: Convert 8 x float to 8 x half
    // _mm_store_si128: Store 8 x half
    for (int i = 0; i < len; i += 8) {
        __m256 s = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &a[i]));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &b[i])));
        _mm_store_si128((__m128i*) &out[i], _mm256_cvtps_ph(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
}
void sum_inplace(half* out, half* a, half* b, half* c, half* d, int len) {
    for (int i = 0; i < len; i += 8) {
        __m256 s = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &a[i]));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &b[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &c[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &d[i])));
        _mm_store_si128((__m128i*) &out[i], _mm256_cvtps_ph(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    // avx512f is NOT faster
//    for (int i = 0; i < len; i += 16) {
//        __m512 s = _mm512_cvtph_ps(_mm256_load_si256((const __m256i*) &a[i]));
//        s = _mm512_add_ps(s, _mm512_cvtph_ps(_mm256_load_si256((const __m256i*) &b[i])));
//        s = _mm512_add_ps(s, _mm512_cvtph_ps(_mm256_load_si256((const __m256i*) &c[i])));
//        s = _mm512_add_ps(s, _mm512_cvtph_ps(_mm256_load_si256((const __m256i*) &d[i])));
//        _mm256_store_si256((__m256i*) &out[i], _mm512_cvtps_ph(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
//    }
}
//void sum_inplace(short* out, short* a, short* b, short* c, short* d, int len) {
//    for (int i = 0; i < len; i += 8) {
//        __m256 s = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &a[i]));
//        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &b[i])));
//        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &c[i])));
//        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &d[i])));
//        _mm_store_si128((__m128i*) &out[i], _mm256_cvtneps_pbh(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
//    }
//}
void sum_inplace(half* out, half* a, half* b, half* c, half* d, half* e, half* f, half* g, half* h, int len) {
    for (int i = 0; i < len; i += 8) {
        __m256 s = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &a[i]));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &b[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &c[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &d[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &e[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &f[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &g[i])));
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_load_si128((const __m128i*) &h[i])));
        _mm_store_si128((__m128i*) &out[i], _mm256_cvtps_ph(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
}
#pragma GCC pop_options
#endif

class Synchronizer1 {
    const int num;
    std::mutex mutex;
    volatile int count;
    std::condition_variable cv;
public:
    explicit Synchronizer1(int num) : num(num), count(0) {}
    void syn() {
        Lock lock(mutex);
        if (++count < num) {
            cv.wait(lock);
        } else {
            lock.unlock();
            cv.notify_all();
        }
    }
    void reset() {
        Lock lock(mutex);
        count = 0;
    }
};

class HostAllReducerImpl : public HostAllReducer {
    static constexpr int MAX_PART=20;
    const int PART_SIZE;

    int world_size;
    size_t num_threads;
    core::TaskThreadPool thread_pool;

    std::vector<void*> host_buf;
    void* result;
    size_t nbytes;
    std::atomic<int> buf_idx;

    Synchronizer1 gather_synchronizer;

    std::mutex reduce_mutex;
    volatile bool reduced0 { false };
    volatile int reduced_count;
    std::condition_variable reduce_cond;

    std::mutex sum_mutex;
    volatile bool sum_done { false };
    std::condition_variable sum_cond;

    std::mutex scatter_mutex;
    volatile int scatter_count;
    std::condition_variable scatter_done_cond;
    std::vector<long> times;

    // int part_size;
    char _padding1[64];
    std::atomic<int> arrived[MAX_PART][16];
    cudaEvent_t arrived_ev[8][MAX_PART];

    char _padding2[64];
    std::atomic<int> reduced[MAX_PART][16];
    volatile bool reduce_done[MAX_PART];
    cudaEvent_t scatter_ev[8];
    cudaEvent_t scatter_event;

    char _padding3[64];
    std::atomic<int> begin_count[16];
    std::atomic<int> end_count[16];
    char _padding4[64];
    bool log { false };

    std::mutex copy_mutex;
    std::condition_variable copy_cond;

public:
    HostAllReducerImpl() = delete;
    HostAllReducerImpl(int world_size, int num_threads, cudaStream_t stream, size_t buffer_size = 10000 * 8192)
        : PART_SIZE(utils::get_int_env("HOST_REDUCE_PART_SIZE", 128 * 1024)),
          world_size(world_size),
          num_threads(num_threads),
          thread_pool(num_threads),
          gather_synchronizer(world_size) {
        end_count[0] = world_size;
        nbytes = buffer_size * sizeof(float);
        for (int i = 0; i < world_size + 1; ++i) {  // +1 for result
            float* ptr;
            BM_CUDART_ASSERT(cudaHostAlloc(&ptr, nbytes + 40960, 0));
            BM_ASSERT(reinterpret_cast<std::uintptr_t>(ptr) % 32 == 0, "not aligned");
            host_buf.push_back(ptr);
        }
        result = host_buf[world_size];
        buf_idx = 0;
        reduced_count = 0;
        scatter_count = world_size;
        for (int i = 0; i < world_size; ++i) {
            arrived_ev[i][0] = nullptr;
        }
        reset_arrived(MAX_PART);
        reset_reduce(MAX_PART);
        for (int a = 0; a < 8; ++a) {
            BM_CUDART_ASSERT(cudaEventCreateWithFlags(&scatter_ev[a], cudaEventDisableTiming));
        }
        BM_CUDART_ASSERT(cudaEventRecord(scatter_ev[0], stream));
        scatter_event = scatter_ev[0];
    }
    HostAllReducerImpl(const HostAllReducerImpl& other) = delete;
    virtual ~HostAllReducerImpl() {
        for (auto t : host_buf) {
            if (cudaSuccess != cudaFreeHost(t)) {
                std::cout << "cudaFreeHost error\n";
            }
        }
    }

    void wait_scattered() {
        Lock lock(scatter_mutex);
        while (scatter_count < world_size) {
            scatter_done_cond.wait(lock);
        }
    }
    void notify_scattered() {
        bool done;
        {
            Lock lock(scatter_mutex);
            done = ++scatter_count == world_size;
        }
        if (done) {
            gather_synchronizer.reset();
            sum_done = false;
            scatter_done_cond.notify_all();
        }
    }
    void chain_scatter_ev(int rank, cudaStream_t stream) {
        Lock lock(scatter_mutex);
        if (scatter_event != nullptr) {
            BM_CUDART_ASSERT(cudaStreamWaitEvent(stream, scatter_event));
        }
        BM_CUDART_ASSERT(cudaEventRecord(scatter_ev[rank], stream));
        scatter_event = scatter_ev[rank];
    }

    template<class T>
    void sum_up_part(const vector<T*>& buf, int len) {
#if defined(__x86_64__)
        if (world_size == 2) {
            sum_inplace(buf[2], buf[0], buf[1], len);
        } else if (world_size == 4) {
            sum_inplace(buf[4], buf[0], buf[1], buf[2], buf[3], len);
        } else if (world_size == 8) {
            sum_inplace(buf[8], buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], len);
        } else {
            throw std::runtime_error("Invalid world size.");
        }
#endif
    }

    template<class T> vector<T*> get_buf(int offset) {
        vector<T*> buf(world_size + 1);  // +1 for result
        for (int r = 0; r < world_size + 1; ++r) {
            buf[r] = reinterpret_cast<T*>(host_buf[r]) + offset;
        }
        return buf;
    }

    template<class T>
    void do_sum_up(int offset, int len, int id=-1) {
        if (len <= 81920 && id < 0) {
            sum_up_part(get_buf<T>(offset), len);
            return;
        }
        int part = std::min(int(num_threads), (len  + 81920 - 1)/ 81920);
        int part_len = round_up((len + part - 1) / part, 16);
//        if (log)
//        std::cout << "part=" << part << ", part_len=" << part_len << ", *=" << part * part_len << ", LEN=" << len << "\n";
        if (id >= 0) reduced[id][0] = 0;
        for (size_t j = 0; j < part; ++j) {
            thread_pool.run([=]() {
                vector<T*> buf = get_buf<T>(offset + j * part_len);
                sum_up_part(buf, part_len);
                if (id >= 0) {
                    if (++reduced[id][0] == part) {
                        reduce_done[id] = true;
                        BM_ASSERT_EQ(arrived[id][0], world_size, "Wrong arrived status after sum up");
                        arrived[id][0] = 0;
                    }
                }
            });
        }
    }

    void sum_up_async(int part, int offset, int len, DataType dtype) {
        if (dtype == DataType::kHalf) {
            do_sum_up<half>(offset, len, part);
        } else if (dtype == DataType::kFloat) {
            do_sum_up<float>(offset, len, part);
        } else {
            throw std::runtime_error("Unsupported type.");
        }
    }

    void sum_up(int idx, int offset, int len, DataType dtype) {
        if (idx == 0) {
            buf_idx = 0;
            scatter_count = 0;
            if (dtype == DataType::kHalf) {
                do_sum_up<half>(offset, len);
            } else if (dtype == DataType::kFloat) {
                do_sum_up<float>(offset, len);
            } else {
                throw std::runtime_error("Unsupported type.");
            }
            thread_pool.wait();
            {
                Lock lock(sum_mutex);
                sum_done = true;
            }
            sum_cond.notify_all();
        } else {
            Lock lock(sum_mutex);
            if (!sum_done)
                sum_cond.wait(lock);
        }
    }

    Tensor reduce_sum(int rank, int layer, Tensor& data) override {
        log = layer == 0 && data.numel() > 81920;
        BM_ASSERT(data.dtype() == DataType::kFloat || data.dtype() == DataType::kHalf, "Only float is supported.");
        BM_ASSERT_LE(data.nbytes(), nbytes, "buffer too small.");
        BM_ASSERT(data.numel() % num_threads == 0, "data not aligned");
        wait_scattered();

        // Gather concurrently
        data.to_buffer(host_buf[rank]);
        gather_synchronizer.syn();

        // Do reduce sum
        long ts0 = logger::get_time_us();
        sum_up(rank, 0, data.numel(), data.dtype());
        long elapsed = logger::get_time_us() - ts0;
        if (log && rank == 0 && layer == 0)
            std::cout << "CPU SumUp " << data.shape() << " takes: " << elapsed << "us\n";
//        if (log) std::cout << "Done sum_up " << idx << "\n";

        // Scatter concurrently
        data.from_buffer(result);
        notify_scattered();
//        std::cout << "Done notify_scattered " << idx << "\n";
        return data;
    }

    void nop(){
        asm(".rept 3000 ; nop ; .endr");
        __asm__("nop\n\t");
    }

    void reset_begin() {
        BM_ASSERT_EQ(begin_count[0].load(), world_size, "Wrong status");
        begin_count[0] -= world_size;
    }
    void reset_end() {
        BM_ASSERT_EQ(end_count[0].load(), world_size, "Wrong status");
        end_count[0] = 0;
    }
    void reset_arrived(int num_part) {
        for (int i = 0; i < num_part; ++i) {
            arrived[i][0] = 0;
        }
    }
    void reset_reduce(int num_part) {
        for (int i = 0; i < num_part; ++i) {
            reduce_done[i] = false;
        }
    }
    void on_begin() {
        // SPIN synchronize
        while (end_count[0].load() != world_size) nop();
        int c = ++begin_count[0];
        BM_ASSERT_LE(c, world_size, "Wrong begin status");
        if (c == world_size) {
            reset_end();
        }
    }
    bool on_end(int num_part, bool copy_only) {
        if (copy_only) {
            Lock lock(sum_mutex);
            int ec = end_count[0]++;
            if (ec == 0) {
                gather_synchronizer.reset();
                reset_begin();
            }
            return true;
        }
        int ec = ++end_count[0];
        BM_ASSERT_LE(ec, world_size, "Wrong end status");
        if (ec == world_size) {
            // reset_arrived(num_part);
            reset_reduce(num_part);
//            reset_end();
            return true;
        }
        return false;
    }

    Tensor reduce_sum_async(int rank, int layer, Tensor& data, Tensor& out, cudaStream_t is, cudaStream_t os, bool copy_only) override {
        log = rank == 0 && layer == 0 && data.numel() > 81920;
        if (!copy_only)
            BM_ASSERT(data.dtype() == DataType::kFloat || data.dtype() == DataType::kHalf, "Only float is supported.");
        BM_ASSERT_LE(data.nbytes(), nbytes, "buffer too small.");
        BM_ASSERT(data.numel() % num_threads == 0, "data not aligned");
        BM_ASSERT(data.nbytes() < nbytes, "data too big");
        if (copy_only)
            BM_ASSERT(data.data() != out.data(), "data and out is same tensor");

//        gather_synchronizer.syn();
//        std::cout << "layer=" << layer << ", rank=" << rank << " BEGIN" << endl;
        on_begin();
        // BM_CUDART_ASSERT(cudaEventSynchronize(scatter_event));

        if (arrived_ev[rank][0] == nullptr) {
            for (int a = 0; a < MAX_PART; ++a) {
                 BM_CUDART_ASSERT(cudaEventCreateWithFlags(&arrived_ev[rank][a], cudaEventDisableTiming));
            }
        }
        // get part size, num
        int numel = data.numel();
        int num_part = std::min(MAX_PART, round_up(numel, PART_SIZE) / PART_SIZE);
        int part_size = round_up(round_up(numel, num_part) / num_part, 32);
        num_part = round_up(numel, part_size) / part_size;
//        if (log) std::cout << "num_part=" << num_part << ", part_size=" << part_size << endl;
        // Gather: async copy
        char* buf = (char*)(host_buf[rank]);
        int ele_size = core::get_elem_size(data.dtype());
        int part_bytes = part_size * ele_size;
        for (int i = 0; i < num_part; ++i) {
            char* dst = buf + i * part_bytes;
            char* src = data.data<char>() + i * part_bytes;
            int copy_bytes = std::min(part_bytes, int(data.nbytes()) - i * part_bytes);
            BM_CUDART_ASSERT(cudaMemcpyAsync(dst, src, copy_bytes, cudaMemcpyDeviceToHost, is));
            BM_CUDART_ASSERT(cudaEventRecord(arrived_ev[rank][i], is));
        }
        int reduced_idx = 0;
        int peer = (rank + 1) % 2;
        char* res_buf = copy_only ? (char*)(host_buf[peer]) : (char*)(result);
        auto scatter_fn = [=, &out](int i) {
            char* dst = out.data<char>() + i * part_bytes;
            char* src = res_buf + i * part_bytes;
            int copy_bytes = std::min(part_bytes, int(data.nbytes()) - i * part_bytes);
            BM_CUDART_ASSERT(cudaMemcpyAsync(dst, src, copy_bytes, cudaMemcpyHostToDevice, os));
        };
        if (copy_only) {
            gather_synchronizer.syn(); // WAIT all arrived_ev are recorded
            for (int i = 0; i < num_part; ++i) {
                BM_CUDART_ASSERT(cudaEventSynchronize(arrived_ev[peer][i]));
                scatter_fn(i);
            }
            // syn all copy; then host_buf is writable for next round
            BM_CUDART_ASSERT(cudaStreamSynchronize(os));
            on_end(num_part, copy_only);
            return out;
        }
        // WAIT copy and trigger sum_up
        for (int i = 0; i < num_part; ++i) {
            BM_CUDART_ASSERT(cudaEventSynchronize(arrived_ev[rank][i]));
//            if (copy_only) {
//                ++arrived[i][rank];
//                continue;
//            }
            int ac = ++arrived[i][0];
//            if (i == 0)
//                std::cout << "layer=" << layer << ", rank=" << rank << " ac=" << ac << endl;
            BM_ASSERT_LE(ac, world_size, "Wrong arrived status");
            if (world_size == ac) {
                if (i == 0) {
                    //##############################################################################
                    // Implicit synchronizing point: All worker arrived here, and no one get any result
                    //##############################################################################
                    reset_begin();
//                    std::cout << "layer=" << layer << " Reset begin" << endl;
                }
                // sum up part "i"
                reduce_done[i] = false;
                int copy_bytes = std::min(part_bytes, int(data.nbytes()) - i * part_bytes);
                sum_up_async(i, i * part_size, copy_bytes / ele_size, data.dtype());
            }
            while(reduced_idx < num_part) {
                if (!reduce_done[reduced_idx]) {
                    break;
                }
                scatter_fn(reduced_idx);
                reduced_idx++;
            }
        }

        // Scatter left parts
        while(reduced_idx < num_part) {
            // SPIN synchronize
            while (!reduce_done[reduced_idx]) {
                nop();
            }
            scatter_fn(reduced_idx);
            reduced_idx++;
        }
        on_end(num_part, copy_only);
//        std::cout << "layer=" << layer << ", rank=" << rank << " END" << endl;
        return out;
    }
};

HostAllReducer* ModelContext::create_host_reducer() {
    int num_thread = utils::get_int_env("HOST_REDUCE_THREAD", world_size() == 2 ? 16 : 32);
    auto stream = current_stream()->ptr;
    return new HostAllReducerImpl(world_size(), num_thread, stream);
}

}