## Performance Notes

We conducted performance reviews on various mainstream NVIDIA GPUs with different model sizes and precisions. For dense models ranging from 2B to 110B parameters on PCIe devices, ZhiLight demonstrates significant performance advantages compared to mainstream open-source inference engines.

To quickly start a benchmark task, refer to the [guide](README.md).

Test Description:
- Test purpose is to demonstrate performance, applicable scenarios and limitations
- Test metrics include:
  - QPS: Queries Per Second
  - TTFT (Time To First Token): First token generation latency
  - TPOT (Time Per Output Token): Generation latency per output token
- Test environments include:
  - AD102 PCIe : Consumer-grade GPU for experimental research
  - A800: Data center GPU for production deployment
  - H20: Data center GPU for production deployment
- Test models include:
  - Large-scale models: DeepSeek-R1, Qwen1.5-110B, Qwen2-72B, LLama-3.1-70B
  - Medium-scale models: Qwen2.5-14B, Llama-3.1-8B, Minicpm-2B
- Compared inference engines include:
  - vLLM
  - SGLang
  - ZhiLight

### DeepSeek-R1 AWQ
- Prompt length: 2.8k
- Output token length: 750
>  **NVIDIA A800*8**
- vLLM args: `python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8080 --max-model-len 24000 --trust-remote-code --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --model /mnt/models --quantization moe_wna16 --max-num-seqs 64 --reasoning-parser deepseek_r1 --enable-reasoning`
- ZhiLight args: `
zhilight --env "REDUCE_TP_INT8_THRES=100;ATTN_DATA_PARALLEL=1" --dyn-max-batch-size 32 `

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
|    vLLM           | 0.08    | 1817.7    | 	2556.42 |	109.86    |  129.98  | 
|  **ZhiLight**       | **0.16**|**1590.96** | **2214.34**|**115.97**  |**139.99**|

> **NVIDIA H20*8**
- ZhiLight args: `zhilight --env "RESERVE_MEM_MB=4000;ATTN_DATA_PARALLEL=1" --dyn-max-batch-size 64`

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
|  **ZhiLight**       | **0.23**|**2025.15** | **3278.09**|**128.28**  |**152.64**|

### MiniCPM-2B-sft-bf16
- NVIDIA AD102 PCIe  * 1
- Prompt length: 3.7k
- vLLM args: 
`python -m vllm.entrypoints.openai.api_server --model /mnt/models --host 127.0.0.1 --port 8080 --max-num-seqs 100 --gpu-memory-utilization 0.9 --trust-remote-code`
- SGLang args: `python -m sglang.launch_server --port 8080 --model-path /mnt/models --log-requests --chunked-prefill-size 256 --enable-metrics --mem-fraction-static 0.8 --trust-remote-code --disable-radix-cache`
- ZhiLight args:
`python -m zhilight.server.openai.entrypoints.api_server --model-path /mnt/models`

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
|    vLLM           | 1.67    | 527.55    | 	1062.96 |	16.71    |  31.95  |
|    SGLang         |1.67	    | 466.19    |   1181.5	| 33.96    |	59.44  |
|  **ZhiLight**       | **1.67**|**434.64** | **989.03**|**26.1**  |**61.14**|
### Llama-3.1-8B
- NVIDIA AD102 PCIe  * 2
- Prompt length: 3.7k
- vLLM args: `python -m vllm.entrypoints.openai.api_server --model /mnt/models --host 127.0.0.1 --port 8080 --max-num-seqs 64 --gpu-memory-utilization 0.9 --max-model-len 25000 -tp 2 --enable-chunked-prefill`
- SGlang args: `python -m sglang.launch_server --model-path /mnt/models --port 8080 --enable-mixed-chunk --disable-radix-cache --tp 2 --enable-p2p-check --context-length 25000 --mem-fraction-static 0.8 --enable-torch-compile --max-num-reqs 64`
- ZhiLight args: `python -m zhilight.server.openai.entrypoints.api_server --model-path /mnt/models --env "HIGH_PRECISION=0;CPM_FUSE_QKV=1;CPM_FUSE_FF_IN=2;CHUNKED_PREFILL=1;CHUNKED_PREFILL_SIZE=256" --dyn-max-batch-size 64`

| Inference Engine | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:            |   :---: | :---:     | :----:    |    :---: |   :---: |  
| vLLM             |  0.46   | 915.58    |  1472.96  |    18.68 |    23.42|
| **SGLang**       |**0.84** | **599.13**|**1148.46**|**28.99** |**37.95**|
|ZhiLight            |0.4      | 1091.12   |2123.93    |66.24     |88.5     |
### Llama-3.1-70B-Instruct-GPTQ-INT4
- NVIDIA AD102 PCIe  * 4
- Prompt length: 3.7k
- vLLM args: `python -m vllm.entrypoints.openai.api_server --model /mnt/models --port 8080 --max-num-seqs 32 --gpu-memory-utilization 0.9 --max-model-len 32000 -tp 4 --enable-chunked-prefill`
- SGLang args: `python -m sglang.launch_server --port 8080 --model-path /mnt/models --tp 4 --log-requests --chunked-prefill-size 256 --enable-metrics --mem-fraction-static 0.8 --disable-radix-cache`
- ZhiLight args: `python -m zhilight.server.openai.entrypoints.api_server --model-path /mnt/models --env "HIGH_PRECISION=0;CPM_FUSE_QKV=1;CPM_FUSE_FF_IN=2;REDUCE_TP_INT8_THRES=100;DUAL_STREAM=1" --dyn-max-batch-size 32`

| Inference Engine | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:            |   :---: | :---:     | :----:    |    :---: |   :---: |  
| vLLM             | 0.18	   | 4796.32   | 9149.49	 | 41.54	  | 90.12   |
| SGLang           | 0.18	   | 3962.99	 | 8886.13	 | 63.22	  | 134.48  |
| **ZhiLight**   | **0.18**|**1419.74**|**2295.92**|**30.97** |**56.08**|

### Qwen2.5-14B-Instruct-GPTQ-Int4
- NVIDIA AD102 PCIe  * 2
- Prompt length: 3.7k
- vLLM args: `python -m vllm.entrypoints.openai.api_server --model /mnt/models --port 8080 --max-num-seqs 32 --gpu-memory-utilization 0.9 --max-model-len 25000 -tp 2 --enable-chunked-prefill -q fp8`
- SGLang args: `python -m sglang.launch_server --host 0.0.0.0 --port 8080 --model-path /mnt/models --tp 2 --disable-radix-cache --chunked-prefill-size 2048 --disable-custom-all-reduce --max-running-requests 32 --mem-fraction-static 0.8 --context-length 25000`
- ZhiLight args: `python -m zhilight.server.openai.entrypoints.api_server --model-path /mnt/models --dyn-max-batch-size 32`

| Inference Engine | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:            |   :---: | :---:     | :----:    |    :---: |   :---: |  
| vLLM             |  0.35   | 1113.4    |  1770.56  |    21.69 |    28.96|
| **SGLang**       |**0.56** |**613.89** | **957.38**| **28.09**|**35.57**|
| ZhiLight           |  0.57   | 795.33    |  1475.42  |  31.98   | 40.01   |

### Qwen2-72B-Instruct-GPTQ-Int4
- Prompt length: 3.7k
- vLLM args: `python -m vllm.entrypoints.openai.api_server --model /mnt/models --port 8080 --max-num-seqs 40 --gpu-memory-utilization 0.9 --max-model-len 32000 --enable-chunked-prefill --max-num-batched-tokens 512 -tp 4 --distributed-executor-backend mp --disable-custom-all-reduce`
- SGLang args: `python -m sglang.launch_server --port 8080 --model-path /mnt/models --disable-radix-cache --tp 4 --chunked-prefill-size 2048 --disable-custom-all-reduce --max-num-reqs 40`
- ZhiLight args: `python -m zhilight.server.openai.entrypoints.api_server --model-path /mnt/models --env "HIGH_PRECISION=0;CPM_FUSE_QKV=1;CPM_FUSE_FF_IN=2;REDUCE_TP_INT8_THRES=100;DUAL_STREAM=1" --dyn-max-batch-size 16`

> **NVIDIA AD102 PCIe  * 4**

| Inference Engine | QPS     | TTFT Mean | TTFT P95 | TPOT Mean| TPOT P95|
| :---:            |   :---: | :---:     | :----:   |    :---: |   :---: | 
| vLLM             |  0.18   | 3493.97   |  6852.07 |    35.47 |    61.74|
| SGLang           |  0.18   | 2276.1    |  3820.7  |    38.12 |    65.16|
| **ZhiLight**       |**0.18** | **1111.8**|**1882.5**| **26.75**|**41.81**|
> **NVIDIA A800 * 4**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean | TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---:  |   :---: | 
| vLLM              |  0.18   | 1457.65   |  2136.5   |    22.14  |    28.96|
| **SGLang**        |**0.36** |**1113.06**|**1850.57**|  **30.41**|**43.65**|
| ZhiLight            |    0.18 | 1227.37   | 1968.95   | 31.95     | 48.53   |
### Qwen1.5-110B-Chat-GPTQ-Int4
- Prompt length: 3.7k
- vLLM args: `python -m vllm.entrypoints.openai.api_server --model /mnt/models --host 127.0.0.1 --port 8080 --max-num-seqs 100 --gpu-memory-utilization 0.95 --enable-chunked-prefill --max-num-batched-tokens 256 --max-model-len 30000 -tp 4 --disable-custom-all-reduce`
- SGLang args: `python -m sglang.launch_server --port 8080 --model-path /mnt/models --disable-radix-cache --tp 4 --chunked-prefill-size 2048`
- ZhiLight args: `python -m zhilight.server.openai.entrypoints.api_sever --model-path /mnt/models --env "HIGH_PRECISION=0;CPM_FUSE_QKV=1;CPM_FUSE_FF_IN=2;REDUCE_TP_INT8_THRES=100;DUAL_STREAM=1" --dyn-max-batch-size 16`
> **NVIDIA AD102 PCIe  * 4**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
| vLLM              |  0.09   | 3085.74   |  4274.03  |    30.34 |    44.08|
| SGLang            |  0.09   |2418.56    |  3187.73  |    31.39 |    53.1 |
| **ZhiLight**        |**0.18** |**1671.38**|**2669.82**|  **39.68**|**64.35**|

> **NVIDIA A800 * 4**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
| vLLM              |  0.09   | 1899.07   |  2719.59  |    23.8  |    33.02|
| **SGLang**        |**0.18** |**1514.49**|**2135.75**|  **28.5**|**47.28**|
| ZhiLight            |     0.1 | 1574.85   | 2086.8    | 27.07    | 38.82   |
