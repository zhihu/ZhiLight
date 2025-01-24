#pragma once
#include "generator/generator.h"
#include "model/model.h"
#include "model/dyn_batch_context.h"
#include "model/model_context.h"
#include "model/llama.h"
#include "utils/ts_queue.hpp"
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <vector>

namespace batch_generator {

using model::DynBatchConfig;
using model::ModelBase;
using model::ModelContext;
using std::shared_ptr;
using std::vector;

struct SearchTask_ {
    vector<int32_t> input_tokens;
    int beam_size;
    int max_length;
    float presence_penalty;
    float repetition_penalty;
    float ngram_penalty;
    bool diverse;
    int seed;
    float temperature;
    int num_results;
    float top_p;
    int top_k;
    int top_logprobs;
    int stream { 0 };  // 0: non stream; 1: single stream result; 2: multiple stream result
    utils::TSQueue<generator::SearchResults> res_queue {INT_MAX};
    std::function<void(const generator::SearchResults& results)> callback;
    volatile bool canceled { false };
    long begin_ts { 0 };
    long first_token_delay_ms { 0 };

    std::vector<int> position_ids;  // passed-in position ids of 'PROMPT'
    bmengine::core::Tensor input_embeddings;  // passed-in embeddings of 'PROMPT', device=CPU
    int position_delta { 0 };  // for multi-modal model

    std::map<int, float> logit_bias;

public:
    void finish(generator::SearchResults&& results);
    void update_stream(const generator::SearchResults& results);
    size_t input_length() const {
        return input_tokens.size();
    }
    size_t full_length() const {
        return input_tokens.size() + size_t(beam_size * max_length);
    }
    bool is_random() const { return top_p < 1. or top_k > 0; }
};

typedef shared_ptr<SearchTask_> SearchTask;

class TaskQueue {
    int max_size_;
    std::mutex mutex_;
    std::condition_variable can_push_cond_;
    std::condition_variable can_pop_cond_;
    std::queue<SearchTask> queue_;
    volatile bool stopping_ { false };
public:
    explicit TaskQueue(int max_size);

    bool push(SearchTask task, bool wait, bool notify=true);
    SearchTask front();
    bool empty();
    SearchTask pop(bool wait);
    vector<SearchTask> pop_multi(int limit, bool wait, int require, int max_token, bool pre_alloc);
    void stop();
    size_t size();
};

template <class, class> class SearcherImplV1;

class BatchGenerator {
    friend class SearcherImplV1<int, int>;

    DynBatchConfig config;
    ModelBase* model_;
    std::vector<model::ModelBase*> par_models_;
    bmengine::core::Engine* engine_;

    TaskQueue queue_;
    int active_size_;

    shared_ptr<std::thread> thread_;
    volatile bool stopping_ { false };

    std::mutex mutex_;
    std::condition_variable done_cond_;
    std::condition_variable stop_cond_;
    volatile bool stopped_ { false };

public:
    BatchGenerator(
        DynBatchConfig config,
        std::vector<model::ModelBase *> par_models,
        bmengine::core::Engine *engine);
    ~BatchGenerator();

    const DynBatchConfig& get_config() const { return config; }

    model::LLaMALike* llama_model() {
        return dynamic_cast<model::LLaMALike*>(model_);
    }

    bool submit(SearchTask task, bool wait, bool notify = true);
    void wait_all_done();
    int queue_size() { return queue_.size(); }
    int active_size() { return active_size_; }

    void start();
    void stop();
    void run();
};

}  // namespace batch_generator