#pragma once
#include "generator/generator.h"
#include <algorithm>
#include <deque>
#include <stack>
#include <vector>

namespace beam_utility {

template<typename TokenT>
class BeamBufferInfo {
public:
    TokenT token;
    int prev;
    float log_prob;
    int ref_count;
    int hyp_id;

    BeamBufferInfo(TokenT token, int prev, float log_prob = 0, int ref_count = 0, int hyp_id = 0)
        : token(token), prev(prev), log_prob(log_prob), ref_count(ref_count), hyp_id(hyp_id) { }

    BeamBufferInfo() : token(), prev(-1), log_prob(0), ref_count(-1) { }
    ~BeamBufferInfo() = default;
};

// clang-format off
/**
 * 为了实现在单个 buffer 上做 beam search，需要管理 beam_size 个 beam hypotheses sentences 的 pos 分配
 * [示例]
 *   pos:        0, 1, 2, 3, 4, 5, 6, 7
 *   ref_count:  1, 1, 1, 1, 2, 2, 1, 1
 *   inputs:     X, A, B, C
 *   hyp1:                   D, E, F
 *   hyp2:                   D, E,    G
 * hyp1 当前使用了 pos 4,5,6
 * hyp2 当前使用了 pos 4,5,7
 */
// clang-format on
template<typename TokenT>
class BeamBufferManager {
public:
    std::vector<BeamBufferInfo<TokenT>> buf_local;
    int len_buf;
    std::deque<int> unused_buffer_pos;
    int last_input_buf_pos { -1 };
    int mask_stride { -1 };

    explicit BeamBufferManager(int len_buf) : buf_local(len_buf), len_buf(len_buf) { }

    BeamBufferManager(const BeamBufferManager& other) = default;

    BeamBufferManager& operator=(BeamBufferManager&& other) {
        // use swap and reset() to reserve other's vector's capacity
        buf_local.swap(other.buf_local);
        len_buf = other.len_buf;
        unused_buffer_pos.swap(other.unused_buffer_pos);
        last_input_buf_pos = other.last_input_buf_pos;
        other.reset(0);
        return *this;
    }

    void init(const std::vector<TokenT>& input, int len_input, int start_pos=0) {
        // fill inputs
        for (int i = 0; i < len_input; i++) {
            buf_local[i] = BeamBufferInfo<TokenT>(input[i], i - 1, 0.0, 1);
        }
        last_input_buf_pos = len_input - 1;
        // unused pos: len_input, len_input+1 ... len_buf-1 in reverse order
        for (int i = 0; i < len_buf - len_input; i++) {
            unused_buffer_pos.push_back(len_buf - i - 1);
        }
    }

    void init_chunk(const std::vector<TokenT>& input, int len_input, int start_pos) {
        BM_ASSERT_LE(start_pos + len_input, len_buf, "len_buf isn't big enough, forget call extend_buffer?");
        BM_ASSERT_LE(len_input, unused_buffer_pos.size(), "unused_buffer_pos.size() is too small");
        // fill inputs
        for (int i = 0; i < len_input; i++) {
            int pos = unused_buffer_pos.back();
            unused_buffer_pos.pop_back();
            BM_ASSERT_EQ(start_pos + i, pos, "start_pos not match");
            buf_local[pos] = BeamBufferInfo<TokenT>(input[i], pos - 1, 0.0, 1);
        }
        last_input_buf_pos = start_pos + len_input - 1;
    }

    void reset(int new_len_buf) {
        len_buf = new_len_buf;
        buf_local.clear();
        buf_local.resize(len_buf);
        unused_buffer_pos.clear();
        last_input_buf_pos = -1;
    }

    void increase_buf_ref(int hypo_end_pos) { buf_local[hypo_end_pos].ref_count++; }
    void decrease_buf_ref(int hypo_end_pos) { buf_local[hypo_end_pos].ref_count--; }

    bool full() const { return unused_buffer_pos.empty(); };

    int pop_unused_pos() {
        BM_ASSERT(!unused_buffer_pos.empty(), "extend_buffer should be called before pop");
        int pos = unused_buffer_pos.back();
        unused_buffer_pos.pop_back();
        return pos;
    }
    int total_used_pos() {
        return unused_buffer_pos.empty() ? len_buf : unused_buffer_pos.back();
    }

    int place_token(const BeamBufferInfo<TokenT>& token) {
        int place = pop_unused_pos();
        buf_local[place] = token;
        return place;
    }

    int extend_buffer(int size) {
        for (int j = 0; j < size; j++) {
            unused_buffer_pos.push_front(len_buf + j);
        }
        len_buf += size;
        buf_local.resize(len_buf);
        return len_buf;
    }

    static int extend_buffer(std::vector<BeamBufferManager>& all_bm, int size) {
        for (auto& bm : all_bm) {
            bm.extend_buffer(size);
        }
        return all_bm[0].len_buf;
    }

    void print_buffer(int full = false) {
	int start = full? 0 : last_input_buf_pos;
	std::cout << "buf(" << len_buf << "): ";
	//for (int i = start; i < buf_local.size() - (full? 0 : unused_buffer_pos.size()); ++i) {
	for (int i = start; i < buf_local.size() - unused_buffer_pos.size(); ++i) {
	    std::cout << i << "(" << buf_local[i].prev << ") ";
	}
	std::cout << " unsued: ";
	for (int i = 0; i < unused_buffer_pos.size(); ++i) {
	    std::cout << unused_buffer_pos[i] << "(-1) ";
	}
	std::cout << std::endl;
    }

    void release_buffer(int last_pos) {
        int pos = last_pos;
        // 当前 hypothesis 分数低，被淘汰
        if (buf_local[pos].ref_count == 0) {
            unused_buffer_pos.push_back(pos);
            pos = buf_local[pos].prev;
            // 往回遍历, 释放该 hypothesis 所占用的全部 pos
            while (pos != last_input_buf_pos) {
                buf_local[pos].ref_count--;
                if (buf_local[pos].ref_count > 0)
                    break;
                unused_buffer_pos.push_back(pos);
                pos = buf_local[pos].prev;
            }
        }
    }

    void release_buffer(int* hypotheses_last_pos, int num) {
        for (int i = 0; i < num; ++i) {
            release_buffer(hypotheses_last_pos[i]);
        }
    }

    void release_buffer(const std::vector<int>& hypotheses_last_pos) {
        for (int pos : hypotheses_last_pos) {
            release_buffer(pos);
        }
    }

    void get_hypothesis_tokens(
        int hypo_last_pos, std::vector<TokenT>* tokens, bool reverse = false) const {
        // 往回遍历该 hypothesis 所有的 token；不包含 input
        int pos = hypo_last_pos;
        while (pos > last_input_buf_pos) {
            int token_id = buf_local[pos].token;
            tokens->push_back(token_id);
            pos = buf_local[pos].prev;
        }
	BM_ASSERT(pos >= 0, "postion overflow!");
        if (!reverse) {
            // 转成正序
            std::reverse(tokens->begin(), tokens->end());
        }
    }

    std::vector<TokenT> get_hypo_tokens(TokenT t, bool eos, int hypo_last_pos) const {
        std::vector<TokenT> v;
        if (!eos) {
            v.emplace_back(t);
        }
        get_hypothesis_tokens(hypo_last_pos, &v, true);
        v.pop_back(); // pop the last token of input
        std::reverse(v.begin(), v.end());
        return v;
    }

    void get_hypothesis_tokens(
        int hypo_last_pos,
        std::vector<int>* tokens,
        int bos_id,
        const std::map<std::pair<int, int>, int>& real_token_id_map) const;

    // mask: matrix: [x, len_buf]
    void mask_hypothesis(int8_t* mask, size_t hyp_i, int last_pos) const {
        // 往回遍历该 hypothesis 所有的 token 对应的 pos，包含 input
        int pos = last_pos;
        size_t stride = mask_stride == -1 ? len_buf : size_t(mask_stride);
        while (pos != -1) {
            mask[hyp_i * stride + pos] = 1; // mask[i][pos]
            pos = buf_local[pos].prev;
        }
    }

    void mask_hypothesis(std::vector<int8_t>& mask, size_t hyp_i, int last_pos) const {
        mask_hypothesis(mask.data(), hyp_i, last_pos);
    }

    void mask_hypotheses(
        const std::vector<int>& hypotheses_end_pos, std::vector<int8_t>& mask) const {
        size_t stride = mask_stride == -1 ? len_buf : size_t(mask_stride);
        mask.resize(hypotheses_end_pos.size() * stride);
        std::fill(mask.begin(), mask.end(), 0);
        for (size_t i = 0; i < hypotheses_end_pos.size(); i++) {
            mask_hypothesis(mask, i, hypotheses_end_pos[i]);
        }
    }

    void mask_hypotheses(const int* hypotheses_last_pos, size_t hyp_num, int8_t* mask) {
        for (size_t i = 0; i < hyp_num; i++) {
            mask_hypothesis(mask, i, hypotheses_last_pos[i]);
        }
    }

    void mask_input(int8_t* mask, int len_input, int stride = -1, int pos = 0) const {
        stride = stride == -1 ? len_buf : stride;
        for (size_t i = 0; i < len_input; i++) {
            for (size_t j = 0; j < stride; j++) {
                mask[i * stride + j] = (i + pos) < j ? 0 : 1;
            }
        }
    }

    void mask_input(std::vector<int8_t>& mask, int len_input) const {
        mask.resize(len_input * len_buf);
        mask_input(mask.data(), len_input);
    }

};

} // namespace beam_utility
