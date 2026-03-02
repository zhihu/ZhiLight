#include "py_export/bind.h"
#include "py_export/py_model_base.h"
#include "generator/batch_generator.h"
#include "model/llama.h"
#include "model/dyn_batch_context.h"
#include "model/model_context.h"
#include "utils/env.h"
#include <bmengine/logger/kernel_time_trace.hpp>
#include <bmengine/logger/std_log_op.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <ctime>
#include <queue>
#include <vector>

// clang-format off
namespace bind {

using namespace batch_generator;
using namespace model;
using namespace std::chrono_literals;
using bmengine::logger::get_time_us;
using std::shared_ptr;
using std::vector;
typedef std::unique_lock<std::mutex> Lock;

class __attribute__ ((visibility("hidden"))) PySearchTask {
public:
    SearchTask task_;
    generator::SearchResults results;

    vector<int> output_tokens_nums_;
    long enqueue_ts;
    long finish_ts;

    static shared_ptr<PySearchTask> create(
        py::object input_tokens_or_str,
        int beam_size,
        int max_length,
        float presence_penalty,
        float repetition_penalty,
        float ngram_penalty,
        bool diverse,
        int seed,
        float temperature,
        int num_results,
        float top_p,
        int top_k,
        bool bee_answer_multi_span,
        int top_logprobs,
        int stream,
        int output_hidden_states
        ) {
        SearchTask t = std::make_shared<SearchTask_>();
        t->input_tokens = py::cast<std::vector<int32_t>>(input_tokens_or_str);
        t->beam_size = beam_size;
        t->max_length = max_length;
        t->presence_penalty = presence_penalty;
        t->repetition_penalty = repetition_penalty;
        t->ngram_penalty = ngram_penalty;
        t->diverse = diverse;
        t->seed = seed;
        t->temperature = temperature;
        t->num_results = num_results;
        t->top_p = top_p;
        t->top_k = top_k;
        t->top_logprobs = top_logprobs;
        t->stream = stream;
        t->output_hidden_states = output_hidden_states;
        t->callback = [=](auto& r) {};
        shared_ptr<PySearchTask> py_task = std::make_shared<PySearchTask>();
        py_task->task_ = t;
        return py_task;
    }

    ~PySearchTask() {
        task_->canceled = true;
    }

    // pass in extra fields for some special models
    void set_position_ids(py::list ids) {
        task_->position_ids = bind::to_int_vector(ids);
    }
    void set_position_delta(int delta) {
        task_->position_delta = delta;
    }
    void set_input_embeddings(py::array data) {
        auto t = bind::numpy_to_tensor("input_embeddings", data, true);
        BM_ASSERT(t.dtype() == core::DataType::kFloat || t.dtype() == core::DataType::kHalf, "Invalid input_embeds dtype");
        task_->input_embeddings = t;
    }
    void set_logit_bias(std::map<int, float> &bias) {
        task_->logit_bias = bias;
    }

    void set_session_info(const std::string& session_id, bool session_continue, int sess_chunk_pos) {
        // std::cout << "Session:" << session_id << ", continue=" << session_continue << ", chunkPos=" << sess_chunk_pos << std::endl;
        task_->session_id = session_id;
        task_->session_continue = session_continue;
        task_->sess_chunk_pos = sess_chunk_pos;
    }

    generator::SearchResults pop_res(float timeout) {
        return task_->res_queue.pop_timeout(timeout);
    }

    bool has_result() {
        return task_->res_queue.size() > 0;
    }

    py::object get_result(float timeout);

    long get_first_schedule_ts() {
        return task_->begin_ts;
    }
    
    long get_time_in_queue() {
        return task_->begin_ts - enqueue_ts;
    }

    void cancel() {
        task_->canceled = true;
    }

    int input_tokens_num() {
        return int(task_->input_length());
    }
    vector<int> output_tokens_nums() {
        return output_tokens_nums_;
    }

    py::list convert_hidden_states(const vector<vector<vector<short>>>& hiddens) {
        py::list token_list;
        for (int i = 0; i < hiddens.size(); ++i) {
            py::list layer_list;
            for (int j = 0; j < hiddens[i].size(); ++j) {
                BM_ASSERT_EQ(hiddens[i][j].size() % task_->dim_model, 0, "Wrong hiddens[i][j].size()");
                py::array_t<short> arr({1UL, hiddens[i][j].size() / task_->dim_model, task_->dim_model}, hiddens[i][j].data());
                layer_list.append(std::move(arr));
            }
            token_list.append(std::move(layer_list));
        }
        return token_list;
    }

    // LLaMA: tuple(tokens, score, time_ms)
    py::object convert_output_to_py(const generator::SearchResult& result,
                                    float time_ms,
                                    float first_token_delay_ms) {
        {
            // LLaMA output
            auto py_tokens = py::cast(result.tokens);
            auto top_logprobs = py::cast(result.top_logprobs);
            return py::make_tuple(py_tokens, result.score, time_ms, first_token_delay_ms, top_logprobs);
        }
    }

    py::list convert_outputs_to_py(const generator::SearchResults& results,
                                   float time_ms) {
        py::list py_list;
        for (auto& result: results.results) {
            py_list.append(convert_output_to_py(result, time_ms, results.first_token_delay_ms));
        }
        return py_list;
    }
};

class __attribute__ ((visibility("hidden"))) PyBatchGenerator {
public:
    shared_ptr<BatchGenerator> searcher_;

    static shared_ptr<PyBatchGenerator> create(DynBatchConfig& config, PyModelBase* py_model) {
        if (!py_model->is_loaded()) throw std::runtime_error("Model is not loaded!!!");

        shared_ptr<PyBatchGenerator> self = std::make_shared<PyBatchGenerator>();
        self->searcher_ = std::make_shared<BatchGenerator>(
            config, py_model->par_models(), py_model->engine());

        return self;
    }

    ~PyBatchGenerator() {
        if (searcher_) {
            // std::cerr << "~PyBatchGenerator\n";
            searcher_->stop();
            searcher_.reset();
        }
    }

    void run() {
        py::gil_scoped_release release;
        searcher_->run();
    }

    void stop() {
        searcher_->stop();
    }

    int queue_size() {
        return searcher_->queue_size();
    }
    int active_size() {
        return searcher_->active_size();
    }

    // return list of list; i.e. [batch, num_results] of convert_result()
    py::list batch_search(py::list py_tasks) {
        vector<PySearchTask*> tasks;
        for (auto& it : py_tasks) {
            tasks.push_back(py::cast<PySearchTask *>(it));
        }
        {
            py::gil_scoped_release release;
            auto notify_batch = std::min(searcher_->get_config().first_batch, int(tasks.size()));
            notify_batch = std::min(searcher_->get_config().task_queue_size, notify_batch);
            int i = 0;
            int show_progress = utils::get_int_env("SHOW_PROGRESS", 0);
            long begin_ts = get_time_us();
            long finished = 0;
            int total = tasks.size();
            std::string time_str;

            std::mutex batch_mutex;
            std::condition_variable batch_cond;
            int remaining_tasks = tasks.size();

            for (auto task : tasks) {
                task->task_->callback = [task, begin_ts, &finished, total, show_progress, &batch_mutex, &batch_cond, &remaining_tasks](const generator::SearchResults& results) {
                    task->results = results;
                    task->finish_ts = get_time_us();
                    if (show_progress) {
                        ++finished;
                        float qps = float(finished * 1e9 / (get_time_us() - begin_ts)) / 1000;
                        std::cout << "\rFinish " << finished << "/" << total
                            << ", QPS=" << qps;
                    }

                    {
                        std::lock_guard<std::mutex> lock(batch_mutex);
                        --remaining_tasks;
                        if (remaining_tasks == 0) {
                            batch_cond.notify_all();
                        }
                    }
                };
                BM_ASSERT(searcher_->submit(task->task_, true, ++i >= notify_batch), "submit task failed");
                task->enqueue_ts = get_time_us();
            }
            
            std::unique_lock<std::mutex> lock(batch_mutex);
            batch_cond.wait(lock, [&remaining_tasks]() { return remaining_tasks == 0; });
            if (show_progress)
                std::cout << "\r";
        }
        py::list py_batch_list;
        for (auto task : tasks) {
            BM_ASSERT(!task->results.results.empty(), "results is empty");
            float time_ms = float((task->finish_ts - task->enqueue_ts) / 1000);
            py::object result_list = task->convert_outputs_to_py(task->results, time_ms);
            py_batch_list.append(result_list);
        }
        return py_batch_list;
    }

#pragma GCC push_options
#pragma GCC optimize ("O0")
    bool submit(py::object py_task, bool wait) {
        PySearchTask& task = py::cast<PySearchTask&>(py_task);
        auto t = task.task_;
        {
            py::gil_scoped_release release;
            if (!searcher_->submit(t, wait))
                return false;
        }
        task.enqueue_ts = get_time_us();
        return true;
    }

    void close_session(const std::string& session_id) {
        searcher_->close_session(session_id);
    }
};

// for Stream API
// update_flag: 1: Incremental; 2: update all; 3: final result
// return py::tuple(update_flag, out_tokens, score)
// if update_flag==3 (final), then
//   out_tokens is list. i.e. len(out_tokens) = num_results
// else
//  out_tokens: CpmBee: str; LLaMA: list[int]
py::object PySearchTask::get_result(float timeout) {
    generator::SearchResults results;
    {
        py::gil_scoped_release release;
        results = std::move(pop_res(timeout));
    }
    int update_flag;
    py::object out_tokens = py::none();
    py::list final_results;
    float score;
    if (!results.results.empty()) {
        const generator::SearchResult& result0 = results.results[0];
        update_flag = 3;
        score = result0.score;
        output_tokens_nums_.clear();
        for (auto& result: results.results) {
            output_tokens_nums_.push_back(int(result.tokens_num()));
        }
        float time_ms = float((get_time_us() - enqueue_ts) / 1000);
        final_results = convert_outputs_to_py(results, time_ms);
    } else if (!results.stream.tokens.empty()) {
        const generator::StreamResult& result = results.stream;
        update_flag = result.update_flag;
        score = result.score;
        if (update_flag == 1) {
            vector<int> single_token{ result.tokens.back() };
            out_tokens = py::cast(single_token);
        } else if (update_flag == 2) {
            out_tokens = py::cast(result.tokens);
        } else {
            std::cerr << "Wrong update_flag: " << result.update_flag << "\n";
        }
    } else if (task_->canceled) {
        throw std::runtime_error("Canceled");
    } else {
        std::cerr << "results: No Results\n";
    }
    py::list hs;
    if (this->task_->stream == 0) {
        // Return hidden_states for non-stream result only.
        hs = this->convert_hidden_states(task_->hidden_states);
    }
    py::tuple tuple = py::make_tuple(update_flag, out_tokens, score, final_results, hs);
    return tuple;
}

void define_dynamic_batch(py::module_& m) {
    py::class_<DynBatchConfig>(m, "DynBatchConfig")
        .def_readwrite("max_batch", &DynBatchConfig::max_batch)
        .def_readwrite("max_beam_size", &DynBatchConfig::max_beam_size)
        .def_readwrite("task_queue_size", &DynBatchConfig::task_queue_size)
        .def_readwrite("max_total_token", &DynBatchConfig::max_total_token)
        .def_readwrite("seed", &DynBatchConfig::seed)
        .def_readwrite("eos_id", &DynBatchConfig::eos_id)
        .def_readwrite("bos_id", &DynBatchConfig::bos_id)
        .def_readwrite("unk_id", &DynBatchConfig::unk_id)
        .def_readwrite("first_batch", &DynBatchConfig::first_batch)
        .def_readwrite("nccl", &DynBatchConfig::nccl)
        .def_readwrite("rag_buffer", &DynBatchConfig::rag_buffer)
        .def_readwrite("ignore_eos", &DynBatchConfig::ignore_eos)
        .def_readwrite("keep_eos", &DynBatchConfig::keep_eos)
        .def_readwrite("reserved_work_mem_mb", &DynBatchConfig::reserved_work_mem_mb)
        .def_readwrite("high_precision", &DynBatchConfig::high_precision)
        .def_readwrite("flash_attention", &DynBatchConfig::flash_attention)
        .def_readwrite("enable_prompt_caching", &DynBatchConfig::enable_prompt_caching)
        .def(py::init());

    py::class_<PySearchTask, shared_ptr<PySearchTask>>(m, "SearchTask")
        .def("has_result", &PySearchTask::has_result)
        .def("get_result", &PySearchTask::get_result)
        .def("get_time_in_queue", &PySearchTask::get_time_in_queue)
        .def("get_first_schedule_ts", &PySearchTask::get_first_schedule_ts)
        .def("input_tokens_num", &PySearchTask::input_tokens_num)
        .def("output_tokens_nums", &PySearchTask::output_tokens_nums)
        .def("cancel", &PySearchTask::cancel)
        .def("set_position_ids", &PySearchTask::set_position_ids)
        .def("set_input_embeddings", &PySearchTask::set_input_embeddings)
        .def("set_position_delta", &PySearchTask::set_position_delta)
        .def("set_logit_bias", &PySearchTask::set_logit_bias)
        .def("set_session_info", &PySearchTask::set_session_info)
        .def(py::init(&PySearchTask::create));

    py::class_<PyBatchGenerator, shared_ptr<PyBatchGenerator>>(m, "BatchGenerator")
        .def(py::init(&PyBatchGenerator::create))
        .def("run", &PyBatchGenerator::run)
        .def("stop", &PyBatchGenerator::stop)
        .def("queue_size", &PyBatchGenerator::queue_size)
        .def("active_size", &PyBatchGenerator::active_size)
        .def("submit", &PyBatchGenerator::submit)
        .def("close_session", &PyBatchGenerator::close_session)
        .def("batch_search", &PyBatchGenerator::batch_search);

}

}