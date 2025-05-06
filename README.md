# ZhiLight

✨ __ZhiLight__ ✨is a highly optimized LLM inference engine developed by Zhihu and ModelBest Inc. The "Zhi" in its name stands for **Z**hihu. ZhiLight can accelerate the inference of models like Llama and its variants, especially on PCIe-based GPUs. Compared to mainstream open-source inference engines, for example, vllm, it has significant performance advantages.

## News 🔥
* [2025/05] Support DeepSeek-V3/DeepSeek-R1 FP8 block-quantized model.
* [2025/02] Support DeepSeek-V3/DeepSeek-R1 AWQ/GPTQ quantized model.
* [2025/01] Support [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2) multimodal models: deepseek-vl2-tiny, deepseek-vl2-small, deepseek-vl2.
* [2025/01] Support LLaMA3 models: Llama-3.1-8B, Llama-3.1-70B, Llama-3.2-1B, Llama-3.2-3B, Llama-3.3-70B.

## Roadmap
see [Roadmap](https://github.com/zhihu/ZhiLight/wiki/Roadmap)

## 🎉🎉 Main Features

* Asynchronous OpenAI compatible interface adapted from vllm
* Custom defined tensor and unified global memory management
* 🔥 Encode and all-reduce overlap, we named "dual streams"
   * Support Int8-quantized all-reduce to further reduce all-reduce cost.   
* Host all-reduce based on SIMD instructions
* Optimized fused kernels, qkv, residual & layernorm etc.
* 🔥 Fused batch attention for decoding based on tensor core instructions
* Support TP and PP on one node, TP is recommended
* Support dynamic batch
* Support flash attention prefill 
* Support chunked prefill
* Support prefix cache
* Support Native INT8/SmoothQuant/FP8/AWQ/GPTQ quantization
* Support Marlin kernel for GPTQ
* Support MoE, DeepseekV2 MoE and DeepseekV2 MLA
* Support Llama/Llama2, Mixtral, Qwen2 series and similar models
## 🔧 Basic Usage
```bash
# Concurrently compile the wheel package, and turn off the unit test
CMAKE_BUILD_PARALLEL_LEVEL=32 TESTING=0 python setup.py bdist_wheel

# Compile with ninja backend
CMAKE_GENERATER="Ninja" python setup.py bdist_wheel

# Install directly
cd ./ZhiLight && pip install -e .

# Start OpenAI compatible server
python -m zhilight.server.openai.entrypoints.api_server [options]
```
## ✈️ Docker Image
ZhiLight only depends on the CUDA runtime, cuBLAS, NCCL, and a few Python packages in requirements.txt. You can use the image below for running or building it. You can also directly refer to docker/Dockerfile.
```bash
docker pull ghcr.io/zhihu/zhilight/zhilight:0.4.8-cu124
```

## 📈 Performance Notes

We conducted performance reviews on various mainstream NVIDIA GPUs with different model sizes and precisions. For dense models ranging from 2B to 110B parameters on PCIe devices, ZhiLight demonstrates significant performance advantages compared to mainstream open-source inference engines.

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
- **NVIDIA A800*8**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
|    vLLM           | 0.08    | 1817.7    | 	2556.42 |	109.86    |  129.98  |
|  **ZhiLight**       | **0.16**|**1590.96** | **2214.34**|**115.97**  |**139.99**|

- **NVIDIA H20*8**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
|  **ZhiLight**       | **0.23**|**2025.15** | **3278.09**|**128.28**  |**152.64**|

### MiniCPM-2B-sft-bf16

- **NVIDIA AD102 PCIe  * 1**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
|    vLLM           | 1.67    | 527.55    | 	1062.96 |	16.71    |  31.95  |
|    SGLang         |1.67	    | 466.19    |   1181.5	| 33.96    |	59.44  |
|  **ZhiLight**       | **1.67**|**434.64** | **989.03**|**26.1**  |**61.14**|

### Qwen2-72B-Instruct-GPTQ-Int4

- **NVIDIA AD102 PCIe  * 4**

| Inference Engine | QPS     | TTFT Mean | TTFT P95 | TPOT Mean| TPOT P95|
| :---:            |   :---: | :---:     | :----:   |    :---: |   :---: | 
| vLLM             |  0.18   | 3493.97   |  6852.07 |    35.47 |    61.74|
| SGLang           |  0.18   | 2276.1    |  3820.7  |    38.12 |    65.16|
| **ZhiLight**       |**0.18** | **1111.8**|**1882.5**| **26.75**|**41.81**|

- **NVIDIA A800 * 4**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean | TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---:  |   :---: | 
| vLLM              |  0.18   | 1457.65   |  2136.5   |    22.14  |    28.96|
| **SGLang**        |**0.36** |**1113.06**|**1850.57**|  **30.41**|**43.65**|
| ZhiLight            |    0.18 | 1227.37   | 1968.95   | 31.95     | 48.53   |
### Qwen1.5-110B-Chat-GPTQ-Int4

- **NVIDIA AD102 PCIe  * 4**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
| vLLM              |  0.09   | 3085.74   |  4274.03  |    30.34 |    44.08|
| SGLang            |  0.09   |2418.56    |  3187.73  |    31.39 |    53.1 |
| **ZhiLight**        |**0.18** |**1671.38**|**2669.82**|  **39.68**|**64.35**|

- **NVIDIA A800 * 4**

| Inference Engine  | QPS     | TTFT Mean | TTFT P95  | TPOT Mean| TPOT P95|
| :---:             |   :---: | :---:     | :----:    |    :---: |   :---: | 
| vLLM              |  0.09   | 1899.07   |  2719.59  |    23.8  |    33.02|
| **SGLang**        |**0.18** |**1514.49**|**2135.75**|  **28.5**|**47.28**|
| ZhiLight            |     0.1 | 1574.85   | 2086.8    | 27.07    | 38.82   |

more benchmarks can be found in [benchmarks](docs/benchmarks/benchmarks.md)

## License
Apache License 2.0

## Contributors
- [@a710128](https://github.com/a710128)
- [@spetrel](https://github.com/spetrel) 
- [@unix1986](https://github.com/unix1986)
- [@gnap](https://github.com/gnap)
