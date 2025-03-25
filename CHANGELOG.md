# zhilight Changelog
## 0.4.5~0.4.17
- Major Features
  - Integrated DeepSeekV3 FusedMLA & DeepGEMM
  - Support multi-nodes tensor parallel inference

## 0.4.3~4
- Major Features
  - INT8 kvcache
  - Varlen flash-decoding

## 0.4.2
- Major Features
  - Fix MiniCPM loader
  - FlashDecoding depends on standard flash-attn so.

## 0.4.1
- Major Features
  - Refactor and delete unused code.
  - Migrate OpenAI compatible server to zhilight.

## 0.3.83~84
- Major Features
  - Set HOST_REDUCE=1 for qwen2 14b.

## 0.3.81~82
- Major Features
  - Fix linear.cpp with bias for qwen2 14b.
  - Use TensorRT for Int8Linear.

## 0.3.80
- Major Features
  - GPTQ support marlin kernel.
  - Chunked prefill.

## 0.3.77~79
- Major Features
  - Dynamic batch support PRE_ALLOC_ALL_TOKEN.(Fix OOM)
  - Reserve more memory for DUAL_STREAM=1(controled by NEED_DEQUANT_WEIGHT)
  - Enlarge BLOCK_M for gemm_warp_reduce(faster W4A16 decode on A800)

## 0.3.76
- Major Features
  - Support CommandR+ model.
- Bug Fix
  - Fix desc_act for TP.

## 0.3.68-75
- Major Features
  - Support DeepSeekV2 model.

## 0.3.67
- Bug Fix
  - Fix ChatMLTokenizer adaption.

## 0.3.63~65
- Major Features
  - W4A16 support TensorRT kernel: GPTQ_KERNEL_ALGO=2
  - W4A8 INT8 support TensorRT kernel: W4_INT8_ALGO=2
  - Optimize CPM_FUSE_FF_IN=2

## 0.3.62
- Bug Fix
  - Fix stream decode.

## 0.3.59~61
- Bug Fix
  - Fix crash when py_task is destroyed.

## 0.3.48~58
- Major Features
  - Dual stream encode (speedup 20%+ on 4090pro).
- Bug Fix
  - Fix flash_attention when real batch > 1.

## 0.3.47
- Major Features
  - Fuse gate activation with multiplication in FeedForward.
- Bug Fix
  - Fix Gemm bf16 when HIGH_PRECISION=0.
  - Fix silu and gelu from double to single (double is very slow).

## 0.3.46
- Bug Fix for 0.3.45
  - Add 89 to CMAKE_CUDA_ARCHITECTURES to build fp8 kernel.

## 0.3.45
- Major Features
  - W8A8 FP8.
  - Update ModelContext::reduce_tp_int8.

## 0.3.44
- Major Features
  - W4A8 fp8 v1.
  - Optimize W4A8 int8 v1.

## 0.3.43
- Major Features
  - W4A8 int8 v1.

## 0.3.41-42
- Major Features
  - Group-wise int8 quant reduce_sum for tensor parallel.

## 0.3.39
- Bug Fix
  - Fix GPTQ new kernel for dynamic batch.

## 0.3.38
- Bug Fix
  - Fix loading CPMBee.

## 0.3.37
- Major Features
  - Fuse GPTQ MOE when batch_size=1.
  - Update MOE: change router's out_type to float; use cudaMemcpy.
  - GPTQ kernel support HIGH_PRECISION env.
  - Optimize gemm_fuse_gate_in: preload matrix A to shared memory; use lop3 instruct.

## 0.3.35
- Bug Fix
  - Remove auto set work memeory.

## 0.3.34
- Major Features
  - Support Qwen models.
  - Optimize GPTQ new kernel with symmetric quant.

## 0.3.33
- Major Features
  - Update GPTQ new kernel: 
    - support desc_act; 
    - support CPM_FUSE_FF_IN=2 CPM_FUSE_QKV=2 : fuse only for small m.

## 0.3.32
- Features
  - Dynamic batch return top_logprobs.

## 0.3.30~31
- Major Features
  - Add new GPTQ kernel.
- Other Changes
  - Disable fuse weight for GPTQ model.
  - Add block index into prefix cache key.
- Bug Fix
  - Fix parsing config when rope_scaling=None.

## 0.3.29
 - Bug Fix
  - Fix chat model eos_id.

## 0.3.28
 - Major Features
  - Support minicpm_moe.

## 0.3.27
 - Major Features
  - support Dynamic NTK-aware context length extention.

## 0.3.26

- Bug Fix
  - cancel fix: invalid ref when python task obj destructed

## 0.3.25
  - Other Changes
    - loader supports AWQ models.
  - Bug Fix
    - fix long context crash up to 128k(no rope scaling yet).

## 0.3.25
  - Other Changes
    - loader supports AWQ models.
  - Bug Fix
    - fix long context crash up to 128k(no rope scaling yet).

## 0.3.24

- Bug Fix
  - Fix Dragonfly residual scaling depth apply.

## 0.3.23

- Major Features
  - Add AWQ kernel.

## 0.3.21,22

- Bug Fix
  - Fix attention for long input.

## 0.3.20

- Bug Fix
  - fix typecast of pass-in hidden_states.

## 0.3.19

- Other Changes
  - Improved Paged KVCache page reuse and separate physical/logical paged lifecycle management.

## 0.3.18

- Fix softmax kernel for long input.

## 0.3.17

- Major Features
  - Copy-on-Write for paged attention, identical outputs of parallel sampling are stored in the same physical blocks.

## 0.3.16

- Major Features
  - Support Paged Attention.
  - Support parallel sampling(num_results > 1) with shared prefix kv cache.

## 0.3.14

- Major Features
  - Support AWQ model with exllama kernel.
- Other Changes
  - exllama kernel uses float accumulator.
- Bug fix
  - Dynamic batch with flash attention: Set len_k to full_input_len during encode.

## 0.3.14

- Support presence_penalty.
- Fix prompt cache with flash attention.

## 0.3.13

- Fix stream decode for HF tokenizer.

## 0.3.12

- Major Features
  - Support prompt cache.
  - flash decoding layer supports paged attention.
- Config Changes
  - GeneratorArg: beam_size default to 1; random search default seed to 42.
  - Add QuantType GPTQ

## 0.3.11

- Speed Optimizations
  - Using FlashAttention to copy buffer, first token latency reduces ~= 10%.

## 0.3.10

- Speed Optimizations
  - Split KV Attention during decoding.
- New Features
  - Dynamic batch support chat-format input.

## 0.3.09

## 0.3.08

- Major Features
  - Support GPTQ int4 model.
  - Support CUDA12.

## 0.3.07

- Major Features
  - Optimize MOE; MOE support INT8.
  - Optimize loading distributed parameters.

## 0.3.06

- Bugfix for chatml, use CHATML_COMPATIBLE=0 to disable prepend \<s>.

## 0.3.05

- Other Changes
  - FlashDecoding now compiles against FlashAttention 2.5(with Paged Attention support).
  - Reduce libflash_decoding.a size to 255MB.

## 0.3.04

- Major Features
  - Support MOE model: mixtral and cpmd_moe.
  - Add ModelLoader.convert_pt_to_safetensors() util method.

## 0.3.03

- Major Features
  - Support mistral model.
  - Support load model safetensors.
  - Support load model multiple threads for multiple files.

## 0.3.02

- Major Features
  - FlashDecoding ops now supports batching(enabled in static batch);

## 0.3.01

- Major Features
  - Dynamic batch dragonfly model support.
  - rotary embedding support rope_theta config.

## 0.2.99

- Other Changes
  - Loader adapts CPMLive checkpoints weights file naming.

## 0.2.98

- Major Features
  - Dragonfly model support.
- Other Changes
  - Loader interface changes, is_chatml option moved to config.json.
  - added an explicit model_type options to avoid ambiguous parameters caused misbehaviour.

## 0.2.97

- Important Bug Fixes
  - Fix attention kernel bug introduced in 0.2.92.

## 0.2.96

- Major Features
  - Dynamic batch support flash attention.
- Major Optimizations
  - Optimize attention kernel for bfloat16.
- Deprecated Features
  - Dynamic batch disable seed for beam search.

## 0.2.95

- Major Features
  - Support flash attention.

## 0.2.94

- Add first token delay to dynamic batch API.

## 0.2.93

- Bug Fixes
  - Fix random search when num_results > 1.
  - Fix loader with chatml and support init tokenizer with tokens.

## 0.2.92

- Major Optimizations
  - Optimize attention kernel.
    - Turn on custom kernel for single request.
    - Use non-mma version kernel.
- Other changes
  - Change RawEmbedding parallel mode to row.
- Bug Fixes
  - Fix random search when num_results > 1.
  - Fix config bug.

## 0.2.91

- Speed Optimizations
  - Turn on CPM_FUSE_QKV and CPM_FUSE_FF_IN by default.
  - Optimize batch_generator.cpp: remove copy context.

## 0.2.90

- Speed Optimizations
  - Use NCCL by default; remove cudaDeviceSynchronize before/after reduce sum.

## 0.2.89

- Major Optimizations
  - Dynamic batch support timeout and cancel.
  - Fuse feedforward w_in with w_gated.

## 0.2.88

- Bug Fixes
  - fix bf16 in speculative sampling

## 0.2.87

- Major Features
  - Support Int4

## 0.2.86

- Major Optimizations
  - Fuse int8 linear for QKV
- Code Refactoring
  - Fuse NormalLinear with ParallelLinear
- Bug Fixes
  - Fix random_search generate <s>
