#include "generator/batch_generator.h"
#include "generator/generator.h"
#include "generator/beam_buffer_manager.hpp"
#include "generator/beam_result_manager.hpp"
#include "generator/beam_util.h"
#include "generator/prefix_cache.hpp"
#include "generator/random_util.h"
#include "model/allocate_util.hpp"
#include "model/llama.h"
#include "model/rag_buffer_context.h"
#include "utils/env.h"
#include "utils/exception.h"
#include "utils/matrix.hpp"

#include <bmengine/core/thread_pool.h>
#include <bmengine/functions/all.h>
#include <bmengine/functions/topk.h>
#include <bmengine/functions/typecast.h>
#include <bmengine/functions/index_select.h>
#include <bmengine/logger/std_log_op.hpp>
#include <bmengine/logger/kernel_time_trace.hpp>
#include "private/allocator.h"

#include <algorithm>
#include <limits>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <sys/syscall.h>

#define NO_GCC_OPT __attribute__((optimize("O0"))) /* for gdb */

// clang-format off
namespace batch_generator {

using namespace bmengine;
//using namespace beam_utility;

using bmengine::core::DataType;
using bmengine::core::Tensor;
using bmengine::core::TaskThreadPool;
using generator::BeamHypothesis;
using generator::SearchResult;
using generator::SearchResults;
using model::is_power_of_2;
using std::unique_ptr;
using std::vector;
using utils::Matrix2D;

using utils::Matrix3D;
typedef std::unique_lock<std::mutex> Lock;
typedef utils::Matrix2D<int32_t> Mat2DInt;
typedef unsigned int len_t;

template <class T>
using RagVector = vector<vector<T>>; // sub-vector has different size.

#ifdef __APPLE__
int pthread_setname_np(pthread_t thread, const char *name) {}
#endif

len_t round_up_len(len_t len, len_t d = 32) {
    return (len + d - 1) / d * d;
}

void SearchTask_::finish(generator::SearchResults&& results) {
    BM_ASSERT(!results.results.empty(), "finish without result!");
    callback(results);
    res_queue.emplace(std::move(results));
}

void SearchTask_::update_stream(const generator::SearchResults& results) {
    res_queue.push(results, true);
}

TaskQueue::TaskQueue(int max_size) : max_size_(max_size) {}

bool TaskQueue::push(SearchTask task, bool wait, bool notify) {
    Lock lock(mutex_);
    if (!wait && queue_.size() >= max_size_) {
        return false;
    }
    while (queue_.size() >= max_size_) {
        can_push_cond_.wait(lock);
    }
    queue_.push(task);
    if (notify)
        can_pop_cond_.notify_one();
    return true;
}

SearchTask TaskQueue::front() {
    Lock lock(mutex_);
    return queue_.empty() ? SearchTask() : queue_.front();
}

bool TaskQueue::empty() {
    Lock lock(mutex_);
    return queue_.empty();
}

SearchTask TaskQueue::pop(bool wait) {
    Lock lock(mutex_);
    while (wait && !stopping_ && queue_.empty()) {
        can_pop_cond_.wait(lock);
    }
    if (queue_.empty())
        return SearchTask();
    auto task = queue_.front();
    queue_.pop();
    can_push_cond_.notify_one();
    return task;
}

vector<SearchTask> TaskQueue::pop_multi(int limit, bool wait, int require, int max_token, bool pre_alloc) {
    vector<SearchTask> tasks;
    int total_token_len = 0;
    {
        Lock lock(mutex_);
        while (wait && !stopping_ && queue_.size() < require) {
            can_pop_cond_.wait(lock);
            if (!queue_.empty() && queue_.front()->canceled) {
                queue_.pop(); // Drop canceled task
                continue;
            }
        }
        while (!queue_.empty() && tasks.size() < limit) {
            total_token_len += pre_alloc
                ? round_up_len(queue_.front()->full_length() + 2, 32)
                : queue_.front()->input_length();
            if (total_token_len > max_token) {
//                std::cout << "#### max_token: " << max_token
//                    << ", input_length=" << queue_.front()->input_length()
//                    << ", beam_size=" << queue_.front()->beam_size
//                    << ", max_len=" << queue_.front()->max_length
//                    << ", full_length=" << queue_.front()->full_length()
//                    << endl;
                break;
            }
            tasks.push_back(queue_.front());
            queue_.pop();
        }
    }
    if (!tasks.empty())
        can_push_cond_.notify_one();
    return tasks;
}

void TaskQueue::stop() {
    Lock lock(mutex_);
    stopping_ = true;
    can_pop_cond_.notify_one();
}
size_t TaskQueue::size() {
    Lock lock(mutex_);
    return queue_.size();
}

class TopKWrapper {
    model::ModelContext& ctx;
    unique_ptr<functions::TopK> topk;
    bool diverse;
    curandGenerator_t gen { nullptr };

    void create_gen() {
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
        CURAND_CHECK(curandSetStream(gen, ctx.current_stream()->ptr));
        CURAND_CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST));
    }

public:
    explicit TopKWrapper(model::ModelContext& ctx)
        : ctx(ctx), topk(std::make_unique<functions::TopK>(ctx)), diverse(false) {
    }

    TopKWrapper(TopKWrapper&& o)
        : ctx(o.ctx), topk(std::move(o.topk)), diverse(o.diverse), gen(o.gen) {
        o.gen = nullptr;
    }

    ~TopKWrapper() {
        if (gen != nullptr) {
            curandDestroyGenerator(gen);
            gen = nullptr;
        }
    }

    curandGenerator_t& generator() { return gen; }

    void set_seed(bool diverse, int seed) {
        this->diverse = diverse;
        if (diverse) {
            if (gen == nullptr) {
                create_gen();
            }
            CURAND_CHECK(curandSetGeneratorOffset(gen, 0));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        }
    }

    static void check_shape(const std::pair<Tensor, Tensor>& top_scores, int top) {
        //        std::cerr << top_scores.first.info() << std::endl;
        BM_ASSERT_EQ(top_scores.first.size(-1), top, "Invalid topk size");
        BM_ASSERT_EQ(top_scores.first.numel(), top_scores.second.numel(), "TopK error");
    }

    void get_top_k(
        Tensor& logits_all, int top, float* top_probs, int* top_ids) {
        if (diverse) {
            // apply gumbel softmax
            auto gumbel_logits = beam_utility::apply_gumbel_softmax(ctx, gen, logits_all);
            auto top_scores = topk->forward(ctx, gumbel_logits, top);
            check_shape(top_scores, top);

            // 还是用 logits_all 里面的分数
            beam_utility::gather_logits(ctx, top_scores.second, logits_all).to_buffer(top_probs);
            top_scores.second.to_buffer(top_ids);
        } else {
            auto top_scores = topk->forward(ctx, logits_all, top);
            check_shape(top_scores, top);

            functions::typecast(ctx, top_scores.first, DataType::kFloat).to_buffer(top_probs);
            top_scores.second.to_buffer(top_ids);
        }
    }
    void get_top_k(
        Tensor& logits_all, int top, Matrix2D<float>* top_probs, Matrix2D<int>* top_ids) {
        BM_ASSERT_EQ(logits_all.size(0) * top, top_ids->size(), "size mismatch");
        get_top_k(logits_all, top, top_probs->mutable_data(), top_ids->mutable_data());
    }

    void get_top_k(
        Tensor& logits_all, int top, vector<float>* top_probs, vector<int>* top_ids) {
        BM_ASSERT_EQ(logits_all.size(0) * top, top_ids->size(), "size mismatch");
        get_top_k(logits_all, top, top_probs->data(), top_ids->data());
    }
};

class RepetitionPenalty {
public:
    float repetition_penalty;
    float ngram_penalty;

    RepetitionPenalty(float repetition_penalty, float ngram_penalty)
        : repetition_penalty(repetition_penalty), ngram_penalty(ngram_penalty) {}

    virtual ~RepetitionPenalty() = default;

    virtual void apply_penalty(
        ModelContext& ctx,
        const std::vector<int>& hypo_last_poses,
        Tensor* logits_all) = 0;
};

class LlamaRepetitionPenalty : public RepetitionPenalty {
    beam_utility::BeamBufferManager<int>& bm;
public:
    LlamaRepetitionPenalty(
        float repetition_penalty,
        float ngram_penalty,
        beam_utility::BeamBufferManager<int>& bm)
        : RepetitionPenalty(repetition_penalty, ngram_penalty), bm(bm) {}

    void apply_penalty(
        ModelContext& ctx,
        const std::vector<int>& hypo_last_poses,
        Tensor* logits_all) override {
        apply_beam_repetition_penalty(
            ctx,
            bm,
            hypo_last_poses,
            ngram_penalty,
            repetition_penalty,
            logits_all);
    }
};

void resize_h(len_t batch, len_t beam, Mat2DInt& a, Mat2DInt& b, Mat2DInt& c, Matrix2D<float>& d) {
    a.resize(batch, beam);
    b.resize(batch, beam);
    c.resize(batch, beam);
    d.resize(batch, beam);
}

len_t calc_max_input_len(const vector<SearchTask>& tasks) {
    len_t max_len = 0;
    for (const SearchTask& task: tasks) {
        max_len = std::max(max_len, len_t(task->input_length()));
    }
    return max_len;
}

len_t calc_max_beam_size(vector<SearchTask>& tasks, len_t beam_size) {
    for (SearchTask& task: tasks) {
        if (task) {
            beam_size = std::max(beam_size, len_t(task->beam_size));
        }
    }
    return beam_size;
}

static int _get_tid() {
    return syscall(SYS_gettid);
}

struct SwapBuf {
    char* ptr{ nullptr };
    size_t len{ 0 };

    SwapBuf() = default;

    explicit SwapBuf(size_t len) : len(len) {
        BM_CUDART_ASSERT(cudaMallocHost(&ptr, len));
    }
    SwapBuf(const SwapBuf&) = delete;
    SwapBuf(SwapBuf&& other) {
        ptr = other.ptr;
        len = other.len;
        other.ptr = nullptr;
        other.len = 0;
    }
    SwapBuf& operator=(const SwapBuf&) = delete;
    SwapBuf& operator=(SwapBuf&& other) {
        ptr = other.ptr;
        len = other.len;
        other.ptr = nullptr;
        other.len = 0;
        return *this;
    }

    ~SwapBuf() {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};

static void print_model_dim(const ModelBase& m) {
    std::cout << "num_layers=" << m.num_layers
              << ", dim_model=" << m.dim_model
              << ", dim_ff=" << m.dim_ff
              << ", num_heads=" << m.num_heads
              << ", num_kv_heads=" << m.num_kv_heads
              << ", dim_head=" << m.dim_head
              << ", num_heads*dim_head=" << m.num_heads * m.dim_head
              << std::endl;
}

static void print_config(const DynBatchConfig& config, bool parallel) {
    std::cout << "DynBatch (";
    std::cout << "max_batch=" << config.max_batch;
    std::cout << ", beam_size=" << config.max_beam_size;
    std::cout << ", queue_size=" << config.task_queue_size;
    std::cout << ", bos=" << config.bos_id;
    std::cout << ", eos=" << config.eos_id;
    if (parallel) {
        std::cout << ", parallel=true, reduce_sum=" << (config.nccl ? "NCCL" : "manual");
    }
    std::cout << (config.flash_attention ? ", flash_attention" : "");
    if (config.rag_buffer) {
        std::cout << ", rag_buffer=true, reserved_work_mem=" << config.reserved_work_mem_mb << "MB";
    }
    std::cout << ")\n";
}

static size_t get_workspace_bytes(const ModelBase& m, int len, int part=1, bool debug=false) {
    int ws_emb = m.dim_model;

    int ws_ff = 2 * m.dim_model + 2 * m.dim_ff / part; // dim_ff will be freed after return from ff

    int dim_q = m.num_heads * m.dim_head / part;  // transpose?
    int dim_kv = m.num_kv_heads * m.dim_head / part;
    int ws_attn_w = 2 * m.dim_model + (dim_q + 2 * dim_kv) / part;
    int n_rep = m.num_heads / m.num_kv_heads;
//    int ws_attn_score = m.num_heads * len / rep;  // main cost，can reduce
    int ws_attn_score = n_rep * len;  // reduced
    int ws_attn_v = m.num_heads * m.dim_head / part;  // transpose?
    int ws_attn = ws_attn_w + ws_attn_score + ws_attn_v;
    size_t ws = sizeof(half) * (ws_emb + std::max(ws_ff, ws_attn));
    if (debug) {
        std::cout << "[len=" << len << ", part=" << part
                  << "] ws=" << (ws / 1024) << "KB len*ws=" << (len * ws / 1024 / 1024) << "MB "
                  << "(ws_emb=" << ws_emb << ", ws_ff=" << ws_ff<< ", ws_attn=" << ws_attn << ")\n";
    }
    return size_t(len) * ws;
}

static size_t get_kv_buf_bytes(const ModelBase& m, int len, int rep=1) {
    int dim_kv = m.num_kv_heads * m.dim_head;
    int buf_size_pt = 2 * m.num_layers * dim_kv;
    return size_t(len) * sizeof(half) * buf_size_pt;
}

static size_t get_weight_bytes(const ModelBase& m, size_t world_size) {
    return sizeof(half) * (3UL * m.dim_model * m.dim_ff
        + size_t(m.dim_model) * (2 * m.num_kv_heads + 2 * m.num_heads) * m.dim_head) / world_size;
}

template<class T>
Tensor rag_tensor(ModelContext& ctx, const RagVector<T>& v2d) {
    vector<T> flat;
    for (const vector<T>& v : v2d) {
        flat.insert(flat.end(), v.begin(), v.end());
    }
    return ctx.tensor_of<T>(flat);
}

template <typename TokenT, typename ResultT>
class SearcherImplV1 {
    typedef beam_utility::BeamBufferInfo<TokenT> BeamBufferInfo;
    typedef beam_utility::BeamBufferManager<TokenT> BeamBufferManager;
    typedef generator::BeamSearchResultManager<ResultT> BeamSearchResultManager;

    len_t max_batch;
    len_t max_beam_size;
    ModelContext& ctx;
    vector<ModelContext*> peer_ctx;
    DynBatchConfig config;
    BatchGenerator* searcher;

    vector<SearchTask> tasks;  // batch tasks
    TopKWrapper topk_all;
    vector<unique_ptr<TopKWrapper>> topk;
    vector<BeamSearchResultManager> result_mgr;
    vector<SearchResults> stream_res;

    vector<BeamBufferManager> bm;

    vector<int> steps;
    vector<vector<BeamHypothesis>> hypotheses;
    vector<vector<BeamBufferInfo>> next_tokens; // 多机共享
    vector<SearchTask> new_tasks;
    len_t beam_size;
    len_t max_batch_active { 0 };
    len_t len_buf { 0 };
    
    int debug_batch { -1 };
    int debug_batch_idx { -1 };
    int batch_idx { 0 };
    int debug_level { 0 };
    len_t num_new_tasks { 0 };
    int dual_stream {false };
    bool pre_alloc { false };
    int chunking_b { -1 }; // index of chunking task, usually the last
    len_t chunked { 0 }; // processed chunked tokens
    bool enabled_chunk_prefill;
    len_t chunk_size;

    vector<shared_ptr<PrefixCache>> prefix_cache;

    // 这里怎么又有一处threads ???
    std::vector<TaskThreadPool*> device_threads;

    vector<vector<SwapBuf>> swapped_buffers;
    len_t swap_count { 0 };
    len_t kv_buf_bytes;
    len_t total_len_buf { 0 };
    len_t max_buf_token_num;

    len_t calc_new_len_buf(len_t new_beam_size);
    void resize_rag_buf();
    void resize(len_t new_beam_size, len_t new_len_buf);

    len_t sum_len_buf() {
        len_t sum = 0;
        if (config.rag_buffer) {
            for (len_t b = 0; b < max_batch_active; ++b) {
                sum += bm[b].len_buf;
            }
        }
        return sum;
    }

    bool in_chunking() { return chunking_b != -1; }

public:
    NO_GCC_OPT
    SearcherImplV1(ModelContext& ctx, DynBatchConfig config, BatchGenerator* searcher)
        : max_batch(config.max_batch),
          max_beam_size(config.max_beam_size),
          ctx(ctx),
          config(config),
          searcher(searcher),
          tasks(max_batch),
          topk_all(ctx),
          topk(max_batch),
          result_mgr(max_batch, BeamSearchResultManager(0)),
          stream_res(max_batch),
          bm(max_batch, BeamBufferManager(config.chunk_size)),
          steps(max_batch),
          hypotheses(max_batch),
          next_tokens(max_batch),
          swapped_buffers(max_batch),
          beam_size(max_beam_size) {
        debug_level = utils::get_int_env("DYN_BATCH_DEBUG", utils::get_int_env("BM_DEBUG_LEVEL"));
        if (debug_level)
            std::cout << "LWP " << _get_tid() << " DynBatch-Main\n";
        topk_all.set_seed(config.seed != 0, config.seed);
        debug_batch_idx = utils::get_int_env("DEBUG_BATCH", -1);
        bool need_dequant_weight = utils::get_int_env("NEED_DEQUANT_WEIGHT", 0) > 0;
        auto dev = ctx.with_device(0);
        dual_stream = utils::get_int_env("DUAL_STREAM", 0) && ctx.get_compute_capability() > 80;
        bool host_reduce = utils::get_int_env("HOST_REDUCE", 0) > 0;
        enabled_chunk_prefill = utils::get_int_env("CHUNKED_PREFILL", 0) > 0;
        chunk_size = utils::get_int_env("CHUNKED_PREFILL_SIZE", 512);
        pre_alloc = utils::get_int_env("PRE_ALLOC_ALL_TOKEN", 1) > 0
            && config.flash_attention
            && dual_stream > 0;
        if (pre_alloc)
            std::cout << ">>> Use pre_alloc: " << pre_alloc << endl;
        std::cout << ">>> CHUNKED_PREFILL:" << enabled_chunk_prefill << ", SIZE: " << chunk_size << endl;

        if (config.enable_prompt_caching) {
            BM_ASSERT_EQ(ctx.devices().size(), 1, "Only support TP");
        }

        // set peer context
        for (int i = 1; i < searcher->par_models_.size(); ++i) {
            device_threads.push_back(new TaskThreadPool(1, i));
        }
        peer_ctx.resize(device_threads.size() + 1);
        peer_ctx[0] = &ctx;
        peer_run([this, searcher](int i) {
            if (i == 0) return;
            std::string name = "DynBatch-TP" + std::to_string(i);
            if (debug_level > 1)
                std::cout << "LWP " << _get_tid() << " " << name << "\n";
            pthread_setname_np(pthread_self(), name.c_str());
            // others model contexts
            peer_ctx[i] = new ModelContext(
                ModelContext::create(*searcher->engine_, *searcher->par_models_[i], this->config, i, true));
        }, true, false);

        // set peer info for reduce sum
        if (config.nccl == -1) {
            config.nccl = 1;
            if (debug_level) std::cout << "Auto set config.nccl to " << config.nccl << "\n";
        }
        std::shared_ptr<model::HostAllReducer> host_reducer;
        if (host_reduce && searcher->par_models_.size() > 1) {
            host_reducer.reset(this->ctx.create_host_reducer());
            auto fn = [&](int i) { peer_ctx[i]->set_host_reducer(host_reducer); };
            peer_run(fn, true);
        }
        if (!config.nccl) {
            auto peer_ctx2 = peer_ctx;
            for (auto& peer: peer_ctx) {
                BM_ASSERT(peer, "thread pool bug.");
                ctx.reducer()->peers_.push_back(peer);
                peer->reducer()->peers_.push_back(&ctx);
                for (auto& peer2: peer_ctx2) {
                    if (peer != peer2)
                        peer->reducer()->peers_.push_back(peer2);
                }
            }
        }

        // calc max_buf_token_num
        auto& model = *searcher->model_;
        auto allocator = ctx.get_allocator();
        size_t free_mem = allocator->get_free_memory();
        kv_buf_bytes = get_kv_buf_bytes(model, 1) / ctx.world_size();
        if (ctx.rag_buffer()->is_cache_quant())
            kv_buf_bytes /= 2;
        if (ctx.latent_cache() && ctx.cfg.kv_lora_rank > 0) {
            kv_buf_bytes = model.num_layers * (ctx.cfg.kv_lora_rank + ctx.cfg.qk_rope_head_dim) * sizeof(half);
        }
        size_t reserve_mem = size_t(config.reserved_work_mem_mb) * 1024U * 1024U;
        if (need_dequant_weight) {
            size_t dequant_ws = get_weight_bytes(model, ctx.world_size());
            reserve_mem += dequant_ws;
            std::cout << ">>> ADD dequant workspace=" << (dequant_ws / 1024 / 1024) << "MB\n";
        }
        if (dual_stream) {
            int max_total_token = enabled_chunk_prefill ? std::max(chunk_size, 4096U) : config.max_total_token;
            size_t ws = size_t(model.dim_model * max_total_token * 2 * sizeof(half));
            reserve_mem += ws;
            std::cout << ">>> ADD dual stream reduce workspace=" << (ws / 1024 / 1024) << "MB\n";
        }
        max_buf_token_num = (free_mem - reserve_mem) / kv_buf_bytes;
        static bool logged = false;
        if (debug_level > 0 && !logged) {
            logged = true;
            print_model_dim(model);
            print_config(config, !peer_ctx.empty());
            std::cout << "free_mem=" << (free_mem / 1024000) << "MB, ";
            std::cout << "reserve_mem=" << (reserve_mem / 1024000) << "MB, ";
            std::cout << "kv_per_token=" << (kv_buf_bytes / 1024) << "KB, ";
            std::cout << "max_buf_token_num=" << max_buf_token_num << "\n";
        }
        if (reserve_mem > free_mem)
            throw std::runtime_error("Not enough memory for workspace");
        if (config.max_total_token > max_buf_token_num)
            throw std::runtime_error("Not enough memory for max_total_token > " + std::to_string(max_buf_token_num));

        prefix_cache.resize(device_threads.size() + 1);
        if (config.enable_prompt_caching) {
            // TODO: better config
            max_buf_token_num /= 2; // use half memory for cache
            int block_size = utils::get_int_env("CACHE_BLOCK_SIZE", 128);
            int block_num = max_buf_token_num / block_size;
            int max_prefix = utils::get_int_env("CACHE_MAX_PREFIX", 4096) / block_size;
            std::cout << "cache block_num: " << block_num << endl;
            auto fn = [&](int i) {
                int num_kv_heads = model.num_kv_heads / peer_ctx[i]->world_size();
                prefix_cache[i] = std::make_shared<PrefixCache>(
                    *peer_ctx[i],
                    block_num,
                    block_size,
                    max_prefix,
                    model.num_layers,
                    num_kv_heads,
                    model.dim_head,
                    model.dtype
                );
            };
            peer_run(fn, true);  // create prefix cache
        }
        if (dual_stream) {
            BM_ASSERT(!config.enable_prompt_caching, "");
            auto fn = [&](int i) {
                peer_ctx[i]->reserve_cache_alloc(free_mem - reserve_mem);
                if (i == 0) {
                    peer_ctx[i]->use_cache_alloc(true);
                    size_t free_mem = peer_ctx[i]->get_allocator()->get_free_memory();
                    peer_ctx[i]->use_cache_alloc(false);
                    std::cout << ">>> KV cache_allocator: free_mem=" << (free_mem / 1024000) << "MB\n";
                }
            };
            peer_run(fn, true, true);  // reserve_cache_alloc
        }
    }

    ~SearcherImplV1() {
        try {
            peer_run([this](int i) {
                if (i > 0)
                    delete peer_ctx[i];
            }, true, false);
        } catch (const std::exception& e) {
            std::cerr << "Stop error: " << e.what() << std::endl;
        }
        for (auto p : device_threads) {
            delete p;
        }
        device_threads.clear();
    }

    void peer_wait() {
        for (auto & device_thread : device_threads) {
            device_thread->wait();
        }
    }
    void peer_run(std::function<void(int)> fn, bool wait = false, bool with_dev = true) {
        for (int i = 1; i <= device_threads.size(); ++i) {
            if (with_dev) {
                auto fn1 = [=]() {
                    auto dev = peer_ctx[i]->with_device(0);
                    fn(i);
                };
                device_threads[i-1]->run(fn1);
            } else {
                device_threads[i-1]->run(std::bind(fn, i));
            }
        }
        fn(0);
        if (wait) {
            peer_wait();
        }
    }

    void batch_search();

    void set_debug_batch(int i, int b) {
        if (batch_idx++ == debug_batch_idx) {
            std::cout << "b=" << b << ", i=" << i
                << ", batch_idx=" << batch_idx - 1
                << ", debug_batch_idx=" << debug_batch_idx << "\n";
            debug_batch = b;
            ctx.dyn_batch()->debug_batch = i;
        }
    }

    void fill_encode_input(vector<SearchTask>& new_tasks);
    
    void fill_search_tokens(Mat2DInt& h_placement, Matrix2D<float>& h_prob_prev);

    void save_prompt_cache() {}

    Tensor join_forward(Tensor* hidden);

    void pick_top_k(Tensor logits_all, Tensor hidden, Mat2DInt& h_placement, Matrix2D<float>& h_prob_prev);

    void beam_sample(len_t b, Tensor penalised_logits, Tensor prob_prev, vector<int>* top_ids, vector<float>* probs);
    void random_sample(len_t b, Tensor penalised_logits, vector<int>* top_ids, vector<float>* probs);

    void update_stream(len_t b, int sent_id, int word_id, int last_hypo_pos, float score);

    len_t assign_free_slot(SearchTask task) {
        for (len_t b = 0; b < tasks.size(); ++b) {
            if (!tasks[b]) {
                return b;
            }
        }
        return -1;
    }

    void resize_task_buf(len_t b, len_t new_len_buf, bool init=false) {
        if (pre_alloc && !init) {
            len_t full_len = ctx.rag_buffer()->get_buf_len(b);
            BM_ASSERT_LE(new_len_buf, full_len, "");
            return;
        }
        auto fn = [=](int i) {
            if (dual_stream)
                peer_ctx[i]->use_cache_alloc(true);
            peer_ctx[i]->resize_task_buf(b, new_len_buf);
            if (dual_stream)
                peer_ctx[i]->use_cache_alloc(false);
        };
        peer_run(fn, false);  // resize_task_buf
    }

    // dump task buffer from GPU to Host; free GPU memory
    void dump_task_buf(len_t e) {
        for (size_t i = 0; i < peer_ctx.size() + 1; ++i) {
            // allocate host memory
            swapped_buffers[e].emplace_back(bm[e].len_buf * kv_buf_bytes);
        }
        int num_layers = searcher->model_->num_layers;
        auto fn = [=](int i) {
            auto rag_buffer = peer_ctx[i]->rag_buffer().get();
            rag_buffer->dump_task_buf(
                e, num_layers, swapped_buffers[e][i].ptr, swapped_buffers[e][i].len);
            rag_buffer->free_task_buf(e);
        };
        peer_run(fn, true);  // dump_task_buf
    }

    // allocate GPU memory; load from Host swap buffer.
    void load_task_buf(len_t e) {
        int num_layers = searcher->model_->num_layers;
        auto fn = [=](int i) {
            if (dual_stream)
                peer_ctx[i]->use_cache_alloc(true);
            peer_ctx[i]->resize_task_buf(e, bm[e].len_buf); // allocate GPU memory
            peer_ctx[i]->rag_buffer()->load_task_buf(
                e, num_layers, swapped_buffers[e][i].ptr, swapped_buffers[e][i].len);
            if (dual_stream)
                peer_ctx[i]->use_cache_alloc(false);
        };
        peer_run(fn, true);  // load_task_buf
        swapped_buffers[e].clear();  // free host memory
    }

    void init_slot(len_t b, SearchTask task) {
        max_batch_active = std::max(b + 1, max_batch_active); // 这里传入的max_batch_active会不会有问题，会不会有多个空洞？？？

        tasks[b] = task;
        // std::cout << "task: b=" << b  << ", random=" << task->is_random() << ", seed=" << task->seed << "\n";
        if (!topk[b]) // 按理说这里应该一直是null才合理, 不然应该delete
            topk[b].reset(new TopKWrapper(ctx));
        topk[b]->set_seed(task->diverse || task->is_random(), task->seed);
        result_mgr[b].reset(std::max(task->beam_size, task->num_results));
        if (!config.rag_buffer) {
            bm[b].reset(len_buf); // global len_buf
        } else {
            // individual len_buf
            len_t new_len_buf = round_up_len(task->input_length() + 2, 32);
            bm[b].reset(new_len_buf);
            if (pre_alloc) {
                len_t full_len_buf = round_up_len(task->full_length() + 2, 32);
                resize_task_buf(b, full_len_buf, true);
            } else {
                resize_task_buf(b, new_len_buf); // alloc new
            }
        }

        steps[b] = 0;
        hypotheses[b].clear();
        hypotheses[b].resize(1);
        next_tokens[b].clear();
        stream_res[b].stream.clear();
    }

    len_t get_batch_active() {
        return max_batch_active;
    }

    void apply_repetition_penalty(Tensor& logits_all, Matrix2D<int32_t>& h_placement);

    void add_logit_bias(Tensor& logits_all);

    Tensor log_softmax(const Tensor& logits_all, Matrix2D<float>& h_prob_prev);

    void erase_task(size_t b) {
        tasks.erase(tasks.begin() + b);
        topk.erase(topk.begin() + b);
        result_mgr.erase(result_mgr.begin() + b);
        stream_res.erase(stream_res.begin() + b);
        bm.erase(bm.begin() + b);
        steps.erase(steps.begin() + b);
        hypotheses.erase(hypotheses.begin() + b);
        next_tokens.erase(next_tokens.begin() + b);
        swapped_buffers.erase(swapped_buffers.begin() + b);
        auto fn = [this, b](int i) {
            peer_ctx[i]->free_task_buf(b);
        };
        peer_run(fn, false);  // free_task_buf
    }
    void pad_task() {
        // padding to max_batch
        while (tasks.size() < max_batch) {
            tasks.emplace_back();
            topk.emplace_back();
            result_mgr.emplace_back(0);
            stream_res.emplace_back();
            bm.emplace_back(0);
            steps.emplace_back();
            hypotheses.emplace_back();
            next_tokens.emplace_back();
            swapped_buffers.emplace_back();
        }
    }
    void move_task(size_t b, size_t e) {
        tasks[b] = std::move(tasks[e]);
        topk[b] = std::move(topk[e]);
        result_mgr[b] = std::move(result_mgr[e]);
        stream_res[b] = std::move(stream_res[e]);
        bm[b] = std::move(bm[e]);
        steps[b] = std::move(steps[e]);
        hypotheses[b] = std::move(hypotheses[e]);
        next_tokens[b] = std::move(next_tokens[e]);
        swapped_buffers[b] = std::move(swapped_buffers[e]);
    }

    void record_top_logprobs(size_t b, const Tensor& logits) {
        Tensor zero_bias = ctx.tensor_of(std::vector<float>(max_beam_size));
        // Tensor log_probs = logits;
        Tensor log_probs = ctx.tensor(logits.shape(), logits.dtype());
        beam_utility::log_softmax_bias(ctx, logits, zero_bias, tasks[b]->temperature, &log_probs);
        functions::TopK tk(ctx);
        auto log_probs_ch = log_probs.chunk();
        BM_ASSERT_EQ(log_probs.size(0), size_t(max_beam_size), "Wrong size");
        BM_ASSERT(!next_tokens[b].empty(), "");
        BM_ASSERT_LE(next_tokens[b].size(), hypotheses[b].size(), "");
        log_probs_ch.resize(next_tokens[b].size());

        int top_num = tasks[b]->top_logprobs;
        for (len_t i = 0; i < next_tokens[b].size(); i++) {
            auto[d_probs, d_ids] = tk.forward(ctx, log_probs.slice_dim0_len(i, 1), top_num);
            vector<int> h_ids(top_num);
            vector<float> h_probs(top_num);
            d_ids.to_buffer(h_ids.data());
            functions::typecast(ctx, d_probs, DataType::kFloat).to_buffer(h_probs.data());
            BM_ASSERT_EQ(hypotheses[b][i].top_logprobs.size(), steps[b] * top_num, "");

            for (int k = 0; k < top_num; ++k) {
                hypotheses[b][i].top_logprobs.emplace_back(h_ids[k], h_probs[k]);
            }
        }
    }
};

template <typename TokenT, typename ResultT>
void SearcherImplV1<TokenT, ResultT>::resize_rag_buf() {
    total_len_buf = sum_len_buf(); // TODO: 调整了beamBuffer之后调整kvcache
    // std::cout << "max_batch_active=" << max_batch_active << ", total_len_buf=" << total_len_buf << ", max_buf_token_num=" << max_buf_token_num << "\n";
    // resize buf if necessary to hold next step tokens
    for (len_t b = 0; b < max_batch_active; ++b) {
        if (bm[b].unused_buffer_pos.size() < next_tokens[b].size()) {
            len_t ext_len = round_up_len(next_tokens[b].size());
            total_len_buf += ext_len;
            if (total_len_buf > max_buf_token_num && !pre_alloc) { // 多机这块不是很好实现，TODO
                len_t e = max_batch_active - 1;
                if (b < e) {
                    // std::cout << "Swap out task " << e << "\n";
                    dump_task_buf(e);
                    total_len_buf -= bm[e].len_buf;
                    max_batch_active--;
                    swap_count++;
                }
            }
            bm[b].extend_buffer(ext_len);
            resize_task_buf(b, bm[b].len_buf);
        }
    }
    while (swap_count > 0) {
        len_t e = max_batch_active;
        if (total_len_buf + bm[e].len_buf + 32 < max_buf_token_num) {
            // std::cout << "Swap in task " << e << "\n";
            load_task_buf(e);
            len_t b = e;
            if (bm[b].unused_buffer_pos.size() < next_tokens[b].size()) {
                len_t ext_len = round_up_len(next_tokens[b].size());
                bm[b].extend_buffer(ext_len);
                resize_task_buf(b, bm[b].len_buf);
            }
            total_len_buf += bm[b].len_buf;
            max_batch_active++;
            swap_count--;
        } else {
            break;
        }
    }
}

template <typename TokenT, typename ResultT>
len_t SearcherImplV1<TokenT, ResultT>::calc_new_len_buf(len_t new_beam_size) {
    len_t max_new_input_len = calc_max_input_len(new_tasks);
    len_t round_size = 32;
    len_t new_len_buf = round_up_len(max_new_input_len + 2, round_size);
    BM_ASSERT(config.max_beam_size < round_size, "max_beam_size to big");

    // resize buf if necessary to hold next step tokens
    for (len_t b = 0; b < max_batch_active; ++b) {
        if (bm[b].unused_buffer_pos.size() < next_tokens[b].size()) {
            bm[b].extend_buffer(round_up_len(next_tokens[b].size()));
            if (config.rag_buffer) {
                // resize task buffer
                len_t len_b = bm[b].len_buf;
                resize_task_buf(b, len_b);
            }
        }
        new_len_buf = std::max(len_t(bm[b].len_buf), new_len_buf);
    }
    return new_len_buf;
}

template <typename TokenT, typename ResultT>
void SearcherImplV1<TokenT, ResultT>::resize(len_t new_beam_size, len_t new_len_buf) {
    BM_ASSERT(!config.rag_buffer, "");
    auto fn = [this](int i) {
        peer_ctx[i]->resize_transformer_buf(len_buf);
    };
    if (new_len_buf > len_buf) {
        // std::cout << "########## Resize to " << new_len_buf << std::endl;
        len_buf = new_len_buf;
        peer_run(fn, true);  // resize_transformer_buf
    } else if (new_len_buf < len_buf * 0.75) {
        // std::cout << "########## Shrink from " << len_buf << " to " << new_len_buf << std::endl;
        len_buf = new_len_buf;
        peer_run(fn, true);  // resize_transformer_buf
    }
    for (len_t b = 0; b < max_batch_active; ++b) {
        bm[b].mask_stride = len_buf;
    }
}

template <>
void SearcherImplV1<int, int>::fill_encode_input(vector<SearchTask>& new_tasks) {
    bool chunking = in_chunking();
    if (chunking) {
        BM_ASSERT(new_tasks.empty(), "");
        // BM_ASSERT_EQ(chunking_b, max_batch_active, "");
        // Fake chunking task as new task
        new_tasks.push_back(tasks[chunking_b]); // chunking时，只有一条处于chunking的prefill请求
    }

    vector<int> v_batch(new_tasks.size()); // 每个请求在batch中的第几个位置, batch_indices
    vector<int> input_lens(new_tasks.size()); // 每个请求这次处理的tokens数，不一定是全部，比如prefix_cache和chunked时, 这里乘beam_size吗？？？
    vector<int> full_input_lens(new_tasks.size()); // flash attention need real length, 已处理的tokens+当前要处理的tokens
    vector<int> buf_lens(new_tasks.size());
    // fill matrices of input tokens
    RagVector<int32_t> h_token;   // 要处理的tokens
    RagVector<int32_t> h_batch;   // batch_indices, batch in buffer
    RagVector<int32_t> h_placement; // beambuffermanager中的位置, pos in buffer
    RagVector<int32_t> h_position;  // seq len pos
    RagVector<int8_t> h_mask;       // beambuffermanager mask

    for (size_t i = 0; i < new_tasks.size(); ++i) {
        auto& task = new_tasks[i];
        // 这里是截止当前的总长度
        auto& tokens = task->input_tokens; // 每次迭代，这里是所有tokens，还是新的tokens？beam情况怎么算？还是只有prompt？？
        len_t b;
        if (chunking) {
            b = chunking_b;
        } else {
            b = assign_free_slot(task);
            BM_ASSERT(b != len_t(-1), "No free slot");
            set_debug_batch(i, b);
            // B3: resize kv buffer
            init_slot(b, task);  // update max_batch_active
        }
        v_batch[i] = b;

        len_t cached_len = chunked;
        if (config.enable_prompt_caching && tokens.size() > 10 && chunked == 0) {
            auto fn = [=, &cached_len](int i) {
                auto rag_buf = peer_ctx[i]->rag_buffer().get();
                int len = prefix_cache[i]->get(
                    *peer_ctx[i], task->input_tokens, {rag_buf->buf_k_[b].get(), rag_buf->buf_v_[b].get()});
                if (i == 0) cached_len = len;
            };
            peer_run(fn, true); // prefix_cache[i]->get
            // task->cached_len_ = cached_len;
            // std::cout << "cached_len: " << cached_len << endl;
        }
        len_t token_num = tokens.size() - 1;  // Reserve last token for search, 如果是chunking中,这里始终获得的是所有的tokens
        BM_ASSERT_LT(cached_len, token_num, "");
        len_t encode_len = token_num - cached_len;

        bool first_chunk = false;
        if (!chunking && enabled_chunk_prefill && encode_len > chunk_size) {
            first_chunk = true;
            chunking = true;
            chunking_b = b; // start chunk_prefill
            chunked = cached_len;
            // std::cout << "Start chunked prefill b=" << b << endl;
        }

        if (!chunking || first_chunk) {
            // std::cout << "Init bm[" << b << "] len=" << token_num << endl;
            bm[b].init(task->input_tokens, int(token_num));
        }
        if (chunking) {
            len_t beam_size1 = calc_max_beam_size(tasks, 0);
            BM_ASSERT(beam_size1 > 0, "");
            BM_ASSERT(max_batch_active > 0, "");
            len_t chunk_size_adj = chunk_size - chunking_b * beam_size1; // 减去chunking_b * beam_size1个decode的tokens???
            if (encode_len <= chunk_size_adj) {
                // std::cout << "Done chunking b=" << b << ", len=" << (chunked + encode_len) << endl;
                max_batch_active = chunking_b + 1; // 怎么感觉这里没有啥必然联系???
                chunking_b = -1; // end chunk_prefill
                chunked = 0;
                chunking = false;
            } else {
                // std::cout << "Prefill size=" << chunk_size_adj << " chunk=["
                //    << chunked << ", " << (chunked + chunk_size_adj) << "] of " << tokens.size() << endl;
                encode_len = chunk_size_adj;
                chunked += chunk_size_adj;
            }
        }

        input_lens[i] = encode_len;      // 本次要计算的长度
        full_input_lens[i] = token_num;  // 输入总长度
        if (chunking)
            full_input_lens[i] = chunked; // 当前总长度
        len_t len1 = round_up_len(encode_len, 32);
        buf_lens[i] = (len1 <= len_buf / 2 || (len1 + 256) <= len_buf) ? len1 : len_buf;
        buf_lens[i] = config.rag_buffer ? bm[b].len_buf : buf_lens[i];

        h_token.emplace_back(encode_len);
        h_batch.emplace_back(encode_len);
        h_placement.emplace_back(encode_len);
        h_position.emplace_back(encode_len);

        for (len_t pos = 0; pos < encode_len; pos++) {
            h_token[i][pos] = tokens[cached_len + pos];
            h_batch[i][pos] = b;
            h_placement[i][pos] = cached_len + pos;
            h_position[i][pos] = cached_len + pos;
        }
        h_mask.emplace_back(encode_len * buf_lens[i]);
        bm[b].mask_input(&h_mask[i][0], encode_len, buf_lens[i], cached_len);
        // set last token to search
        if (!chunking)
            next_tokens[b].emplace_back(tokens[token_num], bm[b].last_input_buf_pos, 0.0, 1);
    }

    auto set_fn = [&](ModelContext& ctx) {
        // convert matrices to tensors
        Tensor e_token = rag_tensor(ctx, h_token);
        Tensor e_placement = rag_tensor(ctx, h_placement);
        Tensor e_mask = rag_tensor(ctx, h_mask);
        Tensor e_position = rag_tensor(ctx, h_position);
        if (!new_tasks[0]->position_ids.empty()) { // 这里干什么用？？？
            const len_t num_ids = new_tasks[0]->position_ids.size();
            BM_ASSERT_EQ(new_tasks.size(), 1, "multi-task is not supported");
            BM_ASSERT_EQ(num_ids % new_tasks[0]->input_length(), 0, "position_ids size mismatch");
            len_t m = num_ids / new_tasks[0]->input_length();
            e_position = ctx.tensor_of(new_tasks[0]->position_ids);
            // TODO: support prefix cache
            BM_ASSERT(!config.enable_prompt_caching, "");
            e_position = e_position.slice_dim0(0, m * h_position[0].size());
        }

        if (debug_level > 1 && ctx.rank() == 0) {
            std::cout << "e_token: " << e_token << std::endl;
            std::cout << "e_placement: " << e_placement << std::endl;
            std::cout << "e_position: " << e_position << std::endl;
            std::cout << "e_mask: " << e_mask << std::endl;
        }

        ctx.dyn_batch()->set_encode(e_token, Tensor(), e_placement, e_position, e_mask);
        ctx.dyn_batch()->set_encode_batch(v_batch, rag_tensor(ctx, h_batch));
        ctx.dyn_batch()->set_encode_len(input_lens, full_input_lens, ctx.tensor_of(input_lens));
        ctx.dyn_batch()->ev_len_buf = buf_lens;
    };
    peer_run([&](int i) { set_fn(*peer_ctx[i]); }, true);  // prepare dyn_batch encode
}

template <>
void SearcherImplV1<int, int>::save_prompt_cache() {
    for (size_t b = 0; b < max_batch_active; ++b) {
        auto fn = [=](int i) {
            auto rag_buf = peer_ctx[i]->rag_buffer().get();
            prefix_cache[i]->put(*peer_ctx[i], tasks[b]->input_tokens, {rag_buf->buf_k_[b].get(), rag_buf->buf_v_[b].get()});
        };
        if (steps[b] == 0 && config.enable_prompt_caching) {
            // std::cout << "prefix_cache[i]->put\n";
            peer_run(fn, true); // prefix_cache[i]->put
        }
    }
}

template <>
void SearcherImplV1<int, int>::fill_search_tokens(
    Matrix2D<int32_t>& h_placement, Matrix2D<float>& h_prob_prev) {
    len_t batch_active = get_batch_active();
    if (batch_active == 0) {
        BM_ASSERT(in_chunking(), "");
        auto set_fn = [&](ModelContext& ctx) {
            ctx.dyn_batch()->set_search(Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
            ctx.dyn_batch()->sv_len_buf = {};
            ctx.dyn_batch()->s_len_buf = Tensor();
        };
        peer_run([&](int i) { set_fn(*peer_ctx[i]); }, true); // prepare dyn_batch search
        return;
    }
    BM_ASSERT(batch_active > 0, "");
    Matrix2D<int32_t> h_token(batch_active, max_beam_size);
    Matrix2D<int32_t> h_position(batch_active, max_beam_size); // pos in sentence
    Matrix3D<int8_t> h_mask(batch_active, max_beam_size, len_buf);
    vector<int> rag_buf_lens;
    RagVector<int8_t> rag_mask;
    vector<int> h_cu_q_seqlens;
    vector<int> h_cu_k_seqlens;
    h_cu_q_seqlens.push_back(0);
    h_cu_k_seqlens.push_back(0);
    int sum_q = 0;
    int sum_k = 0;
    int max_hyp_num = 0;
    int max_q_seqlen = 0;
    int max_k_seqlen = 0;

    // fill matrices of searching tokens
    for (size_t b = 0; b < batch_active; ++b) {
        BM_ASSERT(!config.rag_buffer || tasks[b].get(), "rag_buffer with null task.");
        int hyp_num = next_tokens[b].size(); // ~=beam_size; =0 if b is unused; =1 1th round.
        if (hyp_num > max_hyp_num) {
            max_hyp_num = hyp_num;
        }
        int position = tasks[b] ? int(tasks[b]->input_tokens.size()) + steps[b] : 0;
        bool is_random = tasks[b] && tasks[b]->is_random();
        // int cached_len = tasks[b] ? int(tasks[b]->cached_len_) : 0;
        for (int i = 0; i < hyp_num; i++) {
            BeamBufferInfo& hypo_i = next_tokens[b][i];
            h_token(b, i) = hypo_i.token;
            h_placement(b, i) = bm[b].place_token(hypo_i);  // place in buffer
            h_position(b, i) = position - 1;  // position in sentence
            h_prob_prev(b, i) = is_random ? 0 : hypo_i.log_prob;
        }
        if (!config.rag_buffer) {
            bm[b].mask_hypotheses(&h_placement(b, 0), hyp_num, &h_mask(b, 0, 0));
        } else {
            size_t len_q = max_beam_size;
            size_t len_buffer = bm[b].len_buf;
            rag_buf_lens.push_back(len_buffer);
            rag_mask.emplace_back(len_q * len_buffer);
            bm[b].mask_hypotheses(&h_placement(b, 0), hyp_num, &rag_mask[b][0]);
            if (ctx.is_BSHD() && hyp_num > 0) {
                int input_len = int(tasks[b]->input_tokens.size()) + steps[b];
                sum_q += hyp_num;
                sum_k += input_len;
                h_cu_q_seqlens.push_back(sum_q);
                h_cu_k_seqlens.push_back(sum_k);
                if (hyp_num > max_q_seqlen) {
                    max_q_seqlen = hyp_num;
                }
                if (input_len > max_k_seqlen) {
                    max_k_seqlen = input_len;
                }
            }
        }
    }
    auto set_fn = [&](ModelContext& ctx) {
        // convert matrices to tensors
        Tensor d_token = h_token.to_tensor(ctx).view({h_token.size()});
        Tensor d_placement = h_placement.to_tensor(ctx);
        Tensor d_position = h_position.to_tensor(ctx).view({h_token.size()});
        Tensor d_mask = config.rag_buffer ? rag_tensor(ctx, rag_mask) : h_mask.to_tensor(ctx);

        ctx.dyn_batch()->set_search(d_token, Tensor(), d_placement, d_position, d_mask);
        ctx.dyn_batch()->sv_len_buf = rag_buf_lens;
        ctx.dyn_batch()->s_len_buf = ctx.tensor_of(rag_buf_lens);
        // TODO: max_hyp_num > 1
        if (max_hyp_num == 1 && h_cu_q_seqlens.size() > 0) {
            ctx.dyn_batch()->cu_q_seqlens = ctx.tensor_of(h_cu_q_seqlens);
            ctx.dyn_batch()->cu_k_seqlens = ctx.tensor_of(h_cu_k_seqlens);
            ctx.dyn_batch()->max_q_seqlen = max_q_seqlen;
            ctx.dyn_batch()->max_k_seqlen = max_k_seqlen;
            ctx.dyn_batch()->total_k = sum_k;
        }
    };
    peer_run([&](int i) { set_fn(*peer_ctx[i]); }, true); // prepare dyn_batch search
}

using functions::concat_tensor;

template <>
Tensor SearcherImplV1<int, int>::join_forward(Tensor* hidden) {
    long ts0 = logger::get_time_us();

    Tensor ret_logits;
    auto peer_fn = [&](int i) {
        auto& ctx1 = *peer_ctx[i];
        model::DynBatchContext* dyn_ctx = ctx1.dyn_batch().get();
        // join encode and search input together to call model->encode()
        Tensor group_token = concat_tensor(ctx1, dyn_ctx->e_token, dyn_ctx->s_token, 0);
        Tensor group_position = concat_tensor(ctx1, dyn_ctx->e_position, dyn_ctx->s_position, 0);

        ctx1.rag_buffer()->skip_last = in_chunking();
        ctx1.rag_buffer()->set_buffer_addr(ctx1);

        Tensor input_embeddings;
        Tensor& task0_emb = tasks[0]->input_embeddings;
        if (!task0_emb.empty() && !dyn_ctx->e_token.empty()) {
            BM_ASSERT_EQ(1, dyn_ctx->s_token.numel(), "Feed embedding decode tasks[0] only");
            BM_ASSERT_EQ(group_token.size(0), task0_emb.size(0), "Feed embedding decode tasks[0] only");
            input_embeddings = ctx1.tensor(task0_emb.shape(), task0_emb.dtype());
            ctx1.assign_or_copy(&input_embeddings, &task0_emb);
        }

        auto md = dynamic_cast<model::LLaMALike*>(searcher->par_models_[i]);
        Tensor hidden_g = md->encode(
            ctx1,
            group_token,
            group_position,
            Tensor(),
            Tensor(),
            ctx1.dyn_batch()->s_mask,
            ctx1.dyn_batch()->s_placement,
            input_embeddings,
            false);
        BM_ASSERT_EQ(hidden_g.size(0), group_token.size(0), "encode result dim mismatch");

        if (dyn_ctx->e_token.numel() == group_token.size(0)) {
            // Only chunking, no search tokens. Keep logits_all as empty Tensor()
            BM_ASSERT(in_chunking(), "");
        } else {
            // cut out encoding
            Tensor hidden_search = hidden_g.slice_dim0(dyn_ctx->e_token.numel(), group_token.size(0));

            Tensor logits = md->get_logits(ctx1, hidden_search, true);

            // assign result in rank 0
            if (i == 0) ret_logits = logits;
        }

        ctx1.clear_identity_cache();
        BM_CUDART_ASSERT(cudaStreamSynchronize(ctx1.current_stream()->ptr));
    };
    peer_run(peer_fn, true); // join_forward

    return ret_logits;
}

template <typename TokenT, typename ResultT>
void SearcherImplV1<TokenT, ResultT>::batch_search() {
    int active_count = 0;

    while (true) {
        if (in_chunking()) {
            max_batch_active = chunking_b + 1; // chunked_prefix仅有一条且是batch最后一条
        }
        if (config.rag_buffer) {
            auto dev = ctx.with_device(0);
            // B1: 这里需要每张卡都做
            // swap_out/swap_in kv_cache & resize token buffer & kv_buffer
            // TODO: 这里可能更改max_batch_active和swap_count
            resize_rag_buf(); // 仅针对decode tokens分配kv cache，resize kv_cache buf 按需copy旧的kv_cache, 按需swap out/in
            active_count = max_batch_active; // 内部可能改，所以这里重置一下
        }
        searcher->active_size_ = active_count;
        if (active_count == 0 && searcher->queue_.size() == 0) { // 这里slave需要跳过
            searcher->done_cond_.notify_all();
        }
        int max_total_token;
        if (swap_count == 0 && active_count < max_batch && !in_chunking()) { // 取一条新的prefill
            int free_token_num = int(max_buf_token_num - total_len_buf - 2);
            if (dual_stream)
                free_token_num -= config.max_total_token - 5000;
            max_total_token = std::min(config.max_total_token, free_token_num);
            if (pre_alloc && dual_stream) {
                auto dev = ctx.with_device(0);
                ctx.use_cache_alloc(true);
                max_total_token = ctx.get_allocator()->get_free_memory() / kv_buf_bytes;
                ctx.use_cache_alloc(false);
            }
            int limit = 1; // dual_stream ? 1 : max_batch - active_count;
            //vector<SearchTask> new_tasks;
            if (searcher->engine_->node_rank() == 0) {
                new_tasks = searcher->queue_.pop_multi(
                    limit, active_count == 0, 1, max_total_token, pre_alloc);
                for (auto task: new_tasks) {
                    task->begin_ts = logger::get_time_us();
                }
                if (searcher->engine_->nnodes() > 1) {
                    std::vector<char> buffer;
                    serialize_to_buffer(new_tasks, buffer);
                    char *data = buffer.data();
                    int nbytes = buffer.size();
                    searcher->engine_->broadcast_data(&data, &nbytes);
                    std::cout << "broadcast send!" << std::endl;
                }
            } else {
                // TODO:
                char *data = new char[128*1024*1024];
                int nbytes = 0;
                searcher->engine_->broadcast_data(&data, &nbytes);
                vector<char> new_buffer(data, data + nbytes);
                deserialize_from_buffer(new_buffer, new_tasks);
                std::cout << "broadcast recv!" << std::endl;
                delete [] data;
            }
            std::cout << "New tasks: " << new_tasks.size() << "IDs: " << std::endl;
            for (int i = 0; new_tasks.size() > 0 && i < new_tasks[0]->input_tokens.size(); ++i) {
                std::cout << new_tasks[0]->input_tokens[i] << " ";
            }
            std::cout << std::endl;
        }
        // TODO: broadcast new_tasks

        if (searcher->stopping_) { // stop & cancel 都需要通知slave, 这里可以先不管
            // B2: 需要通知结束
            break;
        } else if (max_batch_active == 0) {
            BM_ASSERT(!new_tasks.empty(), "pop_multi() return 0 tasks.");
        }
        auto dev = ctx.with_device(0);
        // ctx.alloc_device(0);

        // resize fields
        if (!config.rag_buffer) {
            len_t new_len_buf = calc_new_len_buf(max_beam_size); // max_beam_size可能也需要broadcast
            this->resize(max_beam_size, new_len_buf);
        }
        num_new_tasks = new_tasks.size();
        if (debug_level && num_new_tasks > 0) {
            std::cout << "new_tasks=" << new_tasks.size()
                << ", active_count=" << active_count << endl;
        }
        bool feed_input_embedding = !new_tasks.empty() && !new_tasks[0]->input_embeddings.empty();
        if (feed_input_embedding && tasks[0]) { // 为什么要这么做？？？
            int b = assign_free_slot(tasks[0]);
            if (debug_level) std::cout << "Move task 0 to " << b << endl;
            move_task(b, 0); // move tasks[0] to b
            BM_ASSERT(!tasks[0], "Feed input_embeddings need put task at index=0");
        }

        /** -------------------------- Fill Encode Input -------------------------- **/
        if (!new_tasks.empty() || in_chunking()) {
            fill_encode_input(new_tasks); // update max_batch_active in init_slot()
            new_tasks.clear();
        } else {
            peer_run([&](int i) { peer_ctx[i]->dyn_batch()->clear_encode(); });
        }
        max_beam_size = calc_max_beam_size(tasks, 1);

        /** -------------------------- Fill Next Search --------------------------- **/
        if (feed_input_embedding) max_batch_active = 1;
        if (in_chunking()) max_batch_active = chunking_b; // chunking之前的所有batch请求都是decode
        Matrix2D<int32_t> h_placement(max_batch_active, max_beam_size, -1); // pos in buffer
        Matrix2D<float> h_prob_prev(max_batch_active, max_beam_size, -50000);
        fill_search_tokens(h_placement, h_prob_prev);

        /** -------------------------- Get Search Logits --------------------------- **/
        Tensor hidden;
        Tensor logits_all = join_forward(&hidden);
        if (config.enable_prompt_caching) {
            save_prompt_cache();
        }

        // 没有decode输出，只能是chunked prefill的一部分
        if (logits_all.numel() == 0) {
            BM_ASSERT(in_chunking(), "");
            max_batch_active = 1;
            continue; // no other tasks, goto next chunk directly
        }

        size_t logits_dim = searcher->model_->vocab_size;
        // chunked prefill仅有最后一个，在这之前剔除了这一个，修改了max_batch_active
        logits_all = logits_all.view({max_batch_active, max_beam_size, logits_dim });
        if (hidden.numel())
            hidden = hidden.view({max_batch_active, max_beam_size, hidden.size(-1) });

        /** ------------------------------ Pick top k ------------------------------- **/
        pick_top_k(logits_all, hidden, h_placement, h_prob_prev);
        // ctx.release_device();

        // 清理分数肯定不达标的句子
        for (len_t b = 0; b < max_batch_active; ++b) {
            if (!next_tokens[b].empty()) {
                float best_score = next_tokens[b][0].log_prob / float(steps[b] + 1);
                if (ctx.debug() > 1 && b == debug_batch) {
                    std::cout << "b=" << b << ", t=" << steps[b] << ", score=" << best_score << "\n";
                }
                if (!result_mgr[b].accept_score(best_score)) {
                    // stop search, because current result's score is too small
                    next_tokens[b].clear(); // 这里可能产生空洞batch，下边会重排序
                }
            }
            steps[b]++;
            if (steps[b] == 1 && tasks[b])
                tasks[b]->first_token_delay_ms = (logger::get_time_us() - tasks[b]->begin_ts) / 1000;
        }

        // TODO1: broadcast max_batch_active & tasks & next_tokens & chunking_b

        /** ------------------------------ Fill result ------------------------------- **/
        active_count = 0;
        len_t max_batch_active1 = max_batch_active;
        max_batch_active = 0; // 重新计算这个值
        for (len_t b = 0; b < max_batch_active1; ++b) {
            if (tasks[b] && next_tokens[b].empty()) {
                if (result_mgr[b].get_current_results() == 0) {
                    std::cerr << "No Result and no next_tokens!\n";
                }
                int num = std::min(tasks[b]->num_results, result_mgr[b].get_current_results());
                SearchResults results{};
                results.results = result_mgr[b].get_search_results(num);
                results.first_token_delay_ms = tasks[b]->first_token_delay_ms;
                tasks[b]->finish(std::move(results));
                tasks[b].reset();
                if (in_chunking()) {
                    BM_ASSERT(chunking_b > 0, "");
                    BM_ASSERT_LE(b, chunking_b - 1, "");
                    chunking_b--;
                }
                // std::cout << "Finish task " << b << "\n";
            }
            if (tasks[b] && tasks[b]->canceled) { // TODO: need broadcast
                tasks[b].reset();
                std::cout << "Cancel search task " << b << "\n";
            }
            if (tasks[b]) {
                active_count++;
                max_batch_active = b + 1; 
            } else if (config.rag_buffer) { // 这里释放了空洞部分
                erase_task(b);
                b--;
                max_batch_active1--;
            }
        }
        if (config.rag_buffer) {
            pad_task();
        } else if (searcher->queue_.empty() && active_count < max_batch_active) {
            for (len_t b = 0, e = max_batch_active - 1; b < e && active_count < max_batch_active;) {
                while (tasks[b]) b++;
                move_task(b, e);
                do {
                    e--;
                    max_batch_active--;
                } while (!tasks[e]);
            }
        }
    }
}

template <>
void SearcherImplV1<int, int>::apply_repetition_penalty(
    Tensor& logits_all, // (batch, beam_size, vocab_size)
    Matrix2D<int32_t>& h_placement) {
    {
        // implement: logits[:, self.tokenizer.bos_token_id] = -float("inf")
        vector<int> batch_hypos;  // (batch, beam_size)
        for (int b = 0; b < max_batch_active; ++b) {
            if (tasks[b] && steps[b] == 0) {
                float penalty_factor = tasks[b]->repetition_penalty;
                for (len_t h = 0; h < max_beam_size; ++h) {
                    batch_hypos.push_back(b * max_beam_size + h);
                }
            }
        }
        if (!batch_hypos.empty()) {
            vector<int> bos_ids(batch_hypos.size(), config.bos_id);
            vector<int> eos_ids(batch_hypos.size(), config.eos_id);
            vector<float> neg_inf(batch_hypos.size(), -50000);
            beam_utility::scatter_update(ctx, neg_inf, bos_ids, batch_hypos, logits_all);
            beam_utility::scatter_update(ctx, neg_inf, eos_ids, batch_hypos, logits_all);
        }
    }

    bool need_ngram_penalty = false;
    for (int b = 0; b < max_batch_active; ++b) {
        if (tasks[b] && tasks[b]->ngram_penalty > 1.0) {
            need_ngram_penalty = true;
            break;
        }
    }
    if (!need_ngram_penalty) {
        vector<int> batch_hypos;  // (batch, beam_size)
        vector<int> tokens;
        vector<float> repeat_penalty;
        vector<float> presence_penalties;

        for (int b = 0; b < max_batch_active; ++b) {
            if (tasks[b] && (tasks[b]->repetition_penalty != 1.0 || tasks[b]->presence_penalty != 0.)) {
                float penalty_factor = tasks[b]->repetition_penalty;
                float presence_penalty = tasks[b]->presence_penalty;
                for (size_t h = 0; h < hypotheses[b].size(); ++h) {
                    for (int token_id: hypotheses[b][h].token_id_set) {
                        batch_hypos.push_back(b * max_beam_size + h);
                        tokens.push_back(token_id);
                        repeat_penalty.push_back(penalty_factor);
                        presence_penalties.push_back(presence_penalty);
                    }
                }
            }
        }
        if (!batch_hypos.empty()) {
            // sparsely update
            beam_utility::beam_repetition_penalty(ctx, repeat_penalty, tokens, batch_hypos, logits_all, presence_penalties);
        }
        return;
    }

    vector<Tensor> logits_chunks = logits_all.chunk();
    for (int b = 0; b < max_batch_active; ++b) {
        if (!tasks[b])
            continue;
        size_t hyp_num = next_tokens[b].size();
        BM_ASSERT(hyp_num > 0, "hyp_num > 0");
        std::vector<int> b_placement(&h_placement(b, 0), &h_placement(b, 0) + hyp_num);
        Tensor chunk = logits_chunks[b].slice_dim0(0, hyp_num);
        LlamaRepetitionPenalty(tasks[b]->repetition_penalty, tasks[b]->ngram_penalty, bm[b])
            .apply_penalty(ctx, b_placement, &chunk);
    }
}

template <>
void SearcherImplV1<int, int>::add_logit_bias(
    Tensor& logits_all // (batch, beam_size, vocab_size)
) {
    vector<int> batch_hypos;  // (batch, beam_size)
    vector<int> token_ids;
    vector<float> bias;
    for (int b = 0; b < max_batch_active; ++b) {
        if (tasks[b] && !tasks[b]->logit_bias.empty()) {
            for (len_t h = 0; h < max_beam_size; ++h) {
                for (auto& it: tasks[b]->logit_bias) {
                    batch_hypos.push_back(b * max_beam_size + h);
                    token_ids.push_back(it.first);
                    bias.push_back(it.second);
                }
            }
        }
    }
    if (!batch_hypos.empty()) {
        beam_utility::scatter_update(ctx, bias, token_ids, batch_hypos, logits_all, true);
    }
}

template <typename TokenT, typename ResultT>
Tensor SearcherImplV1<TokenT, ResultT>::log_softmax(const Tensor& logits_all, Matrix2D<float>& h_prob_prev) {
    Tensor d_prob_prev = h_prob_prev.to_tensor(ctx);
    Tensor score_all = beam_utility::log_softmax_bias(ctx, logits_all, d_prob_prev);
    // TODO different temperature of every task
    return score_all;
}

template <>
void SearcherImplV1<int, int>::update_stream(
    len_t b, int sent_id, int word_id, int last_hypo_pos, float score) {
    len_t t = steps[b];
    stream_res[b].stream.score = score;
    if (sent_id == 0 && t == stream_res[b].stream.step + 1) {
        // increasingly update
        if (word_id != config.eos_id) {
            stream_res[b].stream.append(word_id);
            tasks[b]->update_stream(stream_res[b]);
        }
    } else {
        // full update
        bool is_eos = word_id == config.eos_id && !config.keep_eos;
        auto tmp_res = bm[b].get_hypo_tokens(word_id, is_eos, last_hypo_pos);
        BM_ASSERT(tmp_res.size() > 0, "No results");
        stream_res[b].stream.update(std::move(tmp_res), t);
        tasks[b]->update_stream(stream_res[b]);
    }
}

template <>
void SearcherImplV1<int, int>::pick_top_k(
    Tensor logits_all, Tensor hidden, Mat2DInt& h_placement, Matrix2D<float>& h_prob_prev) {
    size_t logits_dim = searcher->model_->vocab_size;
    this->apply_repetition_penalty(logits_all, h_placement);
    add_logit_bias(logits_all);
    auto penalised_logits = logits_all.chunk();

    // shape (max_batch_active, beam_size, logits_dim)
    Tensor score_all = log_softmax(logits_all, h_prob_prev);

    // calculate top k
    len_t num_top = std::min(max_beam_size * 2, len_t(32)); // * 2 because may have eos_id
    Matrix2D<float> top_probs_all(max_batch_active, num_top);
    Matrix2D<int> top_ids_all(max_batch_active, num_top);
    // 拍平，后面的 top k 对应的是所有 this_step_size 个 hypothesis 的所有 vocab 的打分
    score_all = score_all.view({max_batch_active, max_beam_size * logits_dim });
    topk_all.get_top_k(score_all, num_top, &top_probs_all, &top_ids_all);
    // TODO diverse for every task

    for (size_t b = 0; b < max_batch_active; ++b) {
        BM_ASSERT_EQ(hypotheses[b].size(), next_tokens[b].size(), "size mismatch");
        size_t hyp_num = next_tokens[b].size();
        if (tasks[b] && tasks[b]->top_logprobs > 0)
            record_top_logprobs(b, penalised_logits[b]);
        vector<BeamHypothesis> old_hypotheses = std::move(hypotheses[b]);
        hypotheses[b].clear();
        next_tokens[b].clear();
        if (hyp_num == 0)
            continue;
        len_t t = steps[b];
        vector<int> top_ids = top_ids_all.vec(b);
        vector<float> top_probs = top_probs_all.vec(b);
        if (tasks[b]->is_random()) {
            random_sample(b, penalised_logits[b], &top_ids, &top_probs);
        }
        // put top k into next_tokens
        for (len_t i = 0; i < top_ids.size(); i++) {
            int word_id = top_ids[i] % logits_dim; // token id
            int sent_id = top_ids[i] / logits_dim; // hypo id
            if (sent_id >= int(hyp_num)) {
                std::cerr << "[b=" << steps[b] << ", i=" << i << "] send_id out of range!\n";
                continue;
            }
            int last_hypo_pos = h_placement(b, sent_id);
            float score = top_probs[i] / float(t + 1);

            if (i == 0 && tasks[b]->stream) {  //  && score > stream_res[b].stream.score
                update_stream(b, sent_id, word_id, last_hypo_pos, score);
            }

            bool is_eos = word_id == config.eos_id;
            if (is_eos || t + 1 == tasks[b]->max_length) {
                bool ignore_eos = config.ignore_eos && (t + 1) < tasks[b]->max_length;
                if (!ignore_eos && result_mgr[b].accept_score(score)) {
                    // add_result if end of beam
                    auto res = bm[b].get_hypo_tokens(word_id, is_eos && !config.keep_eos, last_hypo_pos);
                    int res_idx = result_mgr[b].add_result(res, {}, top_probs[i], score);
                    if (res_idx >= 0 && tasks[b]->top_logprobs > 0) {
                        BM_ASSERT_LE(size_t(sent_id) + 1, old_hypotheses.size(), "");
                        result_mgr[b].set_top_logprobs(
                            res_idx, old_hypotheses[sent_id].get_top_logprobs(tasks[b]->top_logprobs));
                    }
                    // std::cout << "Add result:" << result_mgr[b].get_current_results() << " at step " << steps[b]
                    //     << ", b=" << b << ", beam_size=" << tasks[b]->beam_size << "\n";
                    if (tasks[b]->is_random()) {
                        if (0 == --tasks[b]->beam_size) {
                            // std::cout << "Stop random search\n";
                            next_tokens[b].clear();
                            break;
                        }
                    }
                }
            } else {
                bm[b].increase_buf_ref(last_hypo_pos);
                if (hyp_num == 1 && tasks[b]->beam_size == 1) {
                    hypotheses[b].emplace_back(std::move(old_hypotheses[sent_id]));
                } else {
                    hypotheses[b].push_back(old_hypotheses[sent_id]);
                }
                hypotheses[b].back().add_token(word_id, top_probs[i]);
                next_tokens[b].emplace_back(word_id, last_hypo_pos, top_probs[i], 0);
            }

            if (next_tokens[b].size() >= tasks[b]->beam_size) {
                break;
            }
        }

        bm[b].release_buffer(&h_placement(b, 0), hyp_num); // 这里release上一轮输入的引用，下一轮输入的引用已经inc了
    }
}

template <typename TokenT, typename ResultT>
void SearcherImplV1<TokenT, ResultT>::random_sample(
    len_t b, Tensor penalised_logits, vector<int>* top_ids, vector<float>* fake_probs
) {
    SearchTask task = tasks[b];
    bool first_step = steps[b] == 0;
    penalised_logits = penalised_logits.slice_dim0(0, first_step ? 1 : task->beam_size);
    Tensor probs = ctx.tensor(penalised_logits.size(), penalised_logits.dtype());
    functions::softmax(ctx, penalised_logits, probs, tasks[b]->temperature);

    core::Tensor selection;

    // sample many results in the first step
    // sample one for each instance after the first step
    int num_samples = first_step ? task->num_results : 1;
    int num_selections = task->beam_size;
//    std::cout << "b:" << b << ", step=" << steps[b] << ", num_samples=" << num_samples << ", num_selections=" << num_selections << "\n";
    selection = ctx.tensor({ (size_t) num_selections }, core::DataType::kInt32);
    beam_utility::random_sampler_gpu(ctx, topk[b]->generator(), probs, selection, task->top_p, task->top_k, num_samples);

    top_ids->resize(num_selections);
    selection.to_buffer(top_ids->data());

    fake_probs->clear();
    int logits_dim = probs.size(-1);
    for (int i = 0; i < num_selections; i++) {
        if (!first_step)
            (*top_ids)[i] += logits_dim * i;  // set sent_id to id
        fake_probs->push_back(steps[b] * (steps[b] + 1));
    }
}

BatchGenerator::BatchGenerator(
    DynBatchConfig config,
    std::vector<model::ModelBase *> par_models,
    bmengine::core::Engine* engine)
    : config(config),
      par_models_(par_models),
      engine_(engine),
      queue_(config.task_queue_size) {
    // CHECK_IS_POWER_OF_2(config.max_beam_size);
    BM_ASSERT(config.max_total_token % 64 == 0, "max_total_token should align to 64");
    BM_ASSERT(!par_models_.empty(), "No model");
    model_ = par_models_[0];
}

BatchGenerator::~BatchGenerator() {
    if (!stopping_)
        stop();
}

bool BatchGenerator::submit(SearchTask task, bool wait, bool notify)  {
    if (task->input_tokens.size() <= 1) {
        throw std::invalid_argument("Empty input");
    }
    int max_input_token = config.max_total_token - task->beam_size;
    if (task->input_length() > max_input_token) {
        throw std::invalid_argument(
            "Input tokens are too long: " + std::to_string(task->input_length()));
    }
    if (task->beam_size > config.max_beam_size || task->beam_size < 1) {
        throw std::invalid_argument("Invalid beam size: " + std::to_string(task->beam_size));
    }
    if (task->is_random()) {
        task->beam_size = task->num_results;
    }
    if (task->is_random() && task->seed == 0) {
        task->seed = rand();
    }
    return queue_.push(task, wait, notify);
}

void BatchGenerator::start() {
    thread_ = std::make_shared<std::thread>(&BatchGenerator::run, this);
}

using namespace std::chrono_literals;
void BatchGenerator::stop() {
    stopping_ = true;
    queue_.stop();
    if (thread_) {
        int max_retry = 20;
        for (int i = 0; i < 20; ++i) {
            if (stopped_) {
                break;
            }
            Lock lock(mutex_);
            if (stop_cond_.wait_for(lock, 100ms) == std::cv_status::timeout && i > 0) {
                std::cout << "stop_cond_.wait timeout\n";
            }
            queue_.stop();
        }
        if (stopped_) {
            thread_->join();
        } else {
            std::cerr << "Search thread can not stop!!!\n";
        }
        thread_.reset();
    }
}

void BatchGenerator::run() {
    {
        pthread_setname_np(pthread_self(), "DynBatch");
        // context must create and destroy in the same thread
        ModelContext ctx = ModelContext::create(
            *engine_, *model_, config, par_models_.empty() ? -1 : 0, !par_models_.empty());
        if (llama_model()) {
            SearcherImplV1<int, int>(ctx, config, this).batch_search();
        } else if (!model_) {
            throw std::invalid_argument("No model");
        } else {
            throw std::invalid_argument(std::string("Unknown model:") + model_->layer_type());
        }
        Lock lock(mutex_);
        stopped_ = true;
        stop_cond_.notify_one();
        // std::cerr << "Exit BeamSearcher::run()" << std::endl;
    }
}

void BatchGenerator::wait_all_done() {
    Lock lock(mutex_);
    done_cond_.wait(lock);
}

}  // namespace batch_generator
