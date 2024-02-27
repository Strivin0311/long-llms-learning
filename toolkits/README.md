# Toolkits
*Here're some resources about Toolkits for enhancing LLMs efficiency and effectiveness*
  
### Table of Contents

* [Toolkit List](#toolkit-list)
* [Meta List Info](#meta-list-info)

### Toolkit List

*Note: We sort the toolkit list below in the alphabetical order.*


#### Accelerate

* Type: `library`
* Link: [github](https://github.com/huggingface/accelerate)
* Utilities for Pretraining & Finetuning: 
  * Integrate TorchRun, FSDP, DeepSpeed, Megatron-LM
  * Local SGD

#### AutoGen

* Type: `framework`
* Link: [github](https://github.com/microsoft/autogen)
* Utilities for Inference: 
  * Multi-Config Inference 
* Utilities for Application: 
  * Multi-Agent Conversation


#### BitsandBytes

* Type: `library`
* Link: [github](https://github.com/TimDettmers/bitsandbytes)
* Utilities for Pretraining & FineTuning: 
  * 8bit Optimizers
  * 8bit Matrix Multiplication
  * QLoRA
* Utilities for Inference:
  * 8bit/4bit quantization
  * double quantization


#### Colossal-AI

* Type: `library`
* Link: [github](https://github.com/hpcaitech/ColossalAI)
* Utilities for Pretraining: 
  * Integrate DP, PP, 1D/2D/2.5D/3D TP, ZERO
  * Auto Parallelism


#### DeepSpeed

* Type: `framework`
* Link: [github](https://github.com/microsoft/DeepSpeed)
* Utilities for Pretraining & Finetuning & Inference: 
  * DP, PP, TP, ZERO, Offload
  * Sparse Attention Kernel

#### DeepSpeed-MII

* Type: `framework`
* Link: [github](https://github.com/microsoft/DeepSpeed-MII)
* Utilities for Inference: 
  * Dynamic SplitFuse


#### FlashAttention

* Type: `library`
* Link: [github](https://github.com/Dao-AILab/flash-attention)
* Utilities for Pretraining & Finetuning & Inference: 
  * Kernel-Fused Flash-Attention


#### HuggingFace TGI

* Type: `system`
* Link: [github](https://github.com/huggingface/text-generation-inference)
* Utilities for Inference: 
  * TP
  * Optimized Architectures
  * Continuous Batching
  * Quantization


#### LangChain

* Type: `framework`
* Link: [github](https://github.com/langchain-ai/langchain)
* Utilities for Application: 
  * Prompt Management
  * Memory Management
  * Agent Management


#### LangChain-Chatchat

* Type: `framework`
* Link: [github](hhttps://github.com/chatchat-space/Langchain-Chatchat)
* Utilities for Application: 
  * Integrate Langchain
  * RAG


#### Llama-Factory

* Type: `framework`
* Link: [github](https://github.com/hiyouga/LLaMA-Factory)
* Utilities for Finetuning: 
  * Integrate LoRA, QLoRA, PPO, DPO
  * Reward Modeling


#### Megatron-LM

* Type: `framework`
* Link: [github](https://github.com/NVIDIA/Megatron-LM)
* Utilities for Pretraining: 
  * DP, PP, SP, ZERO
  * Activation Checkpointing


#### Orca

* Type: `system`
* Link: [paper](https://www.usenix.org/system/files/osdi22-yu.pdf)
* Utilities for Inference: 
  * Iteration-level scheduling
  * Selective Batching


#### PEFT

* Type: `framwork`
* Link: [github](https://github.com/huggingface/peft)
* Utilities for Pretraining & Finetuning: 
  * LoRA
  * Prefix-Tuning
  * Prompt-Tuning


#### Petals

* Type: `framework`
* Link: [github](https://github.com/bigscience-workshop/petals)
* Utilities for Finetuning: 
  * Multi-Party Distributed Collaboration


#### PrivateGPT

* Type: `framework`
* Link: [github](https://github.com/imartinez/privateGPT)
* Utilities for Application: 
  * RAG for private documents


#### Pytorch FSDP

* Type: `library`
* Link: [blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
* Utilities for Pretraining & Finetuning: 
  * Fully Sharded Data Parallel


#### Pytorch SDPA

* Type: `function`
* Link: [doc](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
* Utilities for Pretraining & Finetuning & Inference: 
  * Integrate Flash-Attention, Memory-Efficient Attention


#### TensorRT-LLM

* Type: `library`
* Link: [github](https://github.com/NVIDIA/TensorRT-LLM)
* Utilities for Inference: 
  * Python API for TensorRT Engines
  * In-flight Batching


#### Triton

* Type: `compiler`
* Link: [github](https://github.com/openai/triton)
* Utilities for Pretraining & Finetuning & Inference: 
  * Python API for GPU Kernels

#### vLLM

* Type: `library`
* Link: [github](https://github.com/vllm-project/vllm)
* Utilities for Inference: 
  * Paged Attention


#### xFormers

* Type: `library`
* Link: [github](https://github.com/facebookresearch/xformers)
* Utilities for Pretraining: 
  * Memory-Efficient Attention



### Meta List Info

We offer a detailed explanation of the [list](#toolkit-list) as follows:

* **Type**: This specifies the usage type of each toolkit, including: 

  * Library: Typically found as GitHub projects, these toolkits offer functional implementations of specific tasks or algorithms.
  * Framework: Usually encompass a whole systematic pipeline, consisting of multiple interconnected modules designed to support various aspects of LLMs.
  * System: Offer a complete environment that comes pre-configured with all the necessary components and settings to facilitate the deployment of LLMs.
  * Compiler: Fuse operations and compile them into optimized GPU kernels with specific programming languages to accelerate the execution of LLMs.
  

* **Stages**: We categorize the whole LLM lifecycle simply into four stages as follows:
  * Pretraining: LLMs undergo unsupervised training on large-scale datasets to learn basic language modeling.
  * Finetuning: LLMs are further trained in a supervised manner on full/partial parameters to adapt them to specific tasks or align them with human values.
  * Inference: Involves feeding prompts into LLMs and generating outputs iteratively using various control strategies.
  * Application: Off-the-shelf and even black-box LLMs are utilized for context-aware tasks, often involving domain-specific local documents.

* **Utilities**: For each toolkit, we highlight diverse utilities with concise keywords to indicate core techniques w.r.t the corresponding stages. Readers can refer to the toolkit links for more detailed information on these utilities if lost on any keyword.