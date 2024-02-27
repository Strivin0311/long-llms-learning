# Baselines
*Here're some resources about Baseline models used for long-context compatibilities*

### Table of Contents

* [Baseline Table](#baseline-table)
* [Meta Table Info](#meta-table-info)

### Baseline Table


| Model | Open Source | Base | Main Usage | Main Lang | $\bm{L_{max}}$ (k) | Param Size (B) | Mem Occ (GB) | Disk Occ (GB) | Links |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Anima-7B-100k | ✅ | Llama2 | chat | zh | 100 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/lyogavin/Anima-7B-100K) \| [github](https://github.com/lyogavin/Anima/tree/main/anima_100k) |
| ChatGLM2-6B-32k | ✅ | GLM | chat | zh | 32 | 6.2 | 11.7 | 11.6 | [hf](https://huggingface.co/THUDM/chatglm2-6b-32k) \| [github](https://github.com/THUDM/ChatGLM2-6B) |
| ChatGLM3-6B-32k | ✅ | GLM | chat | zh | 32 | 6.2 | 11.7 | 11.6 | [hf](https://huggingface.co/THUDM/chatglm3-6b) \| [github](https://github.com/THUDM/ChatGLM3) |
| Chinese-Alpaca2-7B-16k | ✅ | Llama2 | instruct | zh | 16 | 6.9 | 25.9 | 12.9 | [hf](https://huggingface.co/hfl/chinese-alpaca-2-7b-16k) \| [github](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/) |
| Chinese-Llama2-7B-16k | ✅ | Llama2 | chat | zh | 16 | 6.9 | 26.3 | 12.9 | [hf](https://huggingface.co/hfl/chinese-llama-2-7b-16k) \| [github](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/) |
| Chinese-Mixtral | ✅ | Mixtral | chat | zh | 32 | 46.7 | 175.0 | 87.0 | [hf](https://huggingface.co/hfl/chinese-mixtral) \| [github](https://github.com/ymcui/Chinese-Mixtral) |
| Chinese-Mixtral-Instruct | ✅ | Mixtral | instruct | zh | 32 | 46.7 | 175.0 | 87.0 | [hf](https://huggingface.co/hfl/chinese-mixtral-instruct) \| [github](https://github.com/ymcui/Chinese-Mixtral) |
| Claude2 | ❌ | Claude | chat | en | 100 | ? | ? | ? | [acc](https://claude.ai/onboarding) \| [home](https://www.anthropic.com/news/claude-2) |
| CodeLlama-7B | ✅ | Llama2 | code | py | 16 | 6.7 | 25.6 | 12.6 | [hf](https://huggingface.co/codellama/CodeLlama-7b-hf) \| [home](https://huggingface.co/codellama) \| [paper](https://arxiv.org/pdf/2308.12950.pdf) |
| CodeLlama-13B | ✅ | Llama2 | code | py | 16 | 13.0 | 49.1 | 24.2 | [hf](https://huggingface.co/codellama/CodeLlama-13b-hf) \| [home](https://huggingface.co/codellama) \| [paper](https://arxiv.org/pdf/2308.12950.pdf) |
| CodeLlama-34B | ✅ | Llama2 | code | py | 16 | 33.7 | 126.5 | 62.9 | [hf](https://huggingface.co/codellama/CodeLlama-34b-hf) \| [home](https://huggingface.co/codellama) \| [paper](https://arxiv.org/pdf/2308.12950.pdf) |
| Giraffe-13B-32k-v3 | ✅ | Llama2 | instruct | en | 32 | 13.0 | 48.6 | 24.2 | [hf](https://huggingface.co/abacusai/Giraffe-13b-32k-v3) \| [github](https://github.com/abacusai/long-context) \| [paper](https://arxiv.org/pdf/2308.10882.pdf) |
| Giraffe-v2-70B-32k | ✅ | Llama2 | instruct | en | 32 | 69.0 | 227.4 | 128.5 | [hf](https://huggingface.co/abacusai/Giraffe-v2-70b-32k) \| [github](https://github.com/abacusai/long-context) \| [paper](https://arxiv.org/pdf/2308.10882.pdf) |
| GPT3.5-Turbo-16k | ❌ | GPT3 | chat | en | 16 | ? | ? | ? | [acc](https://chat.openai.com/auth/login) \| [home](https://openai.com/chatgpt) \| [doc](https://platform.openai.com/docs/models/gpt-3-5-turbo) |
| GPT4 | ❌ | GPT4 | chat | en | 8 | ? | ? | ? | [acc](https://chat.openai.com/auth/login) \| [home](https://openai.com/gpt-4) \| [doc](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) |
| GPT4-32k | ❌ | GPT4 | chat | en | 32 | ? | ? | ? | [acc](https://chat.openai.com/auth/login) \| [home](https://openai.com/gpt-4) \| [doc](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) |
| GPT4-Turbo | ❌ | GPT4 | chat | en | 128 | ? | ? | ? | [acc](https://chat.openai.com/auth/login) \| [home](https://openai.com/gpt-4) \| [doc](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) |
| InternLM-Chat-7B | ✅ | Llama2 | chat | en | 200 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/internlm/internlm2-chat-7b) \| [github](https://github.com/InternLM/InternLM) |
| Llama2-7B-32k | ✅ | Llama2 | chat | en | 32 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K) \| [home](https://www.together.ai/) |
| Llama2-7B-Instruct-32k | ✅ | Llama2 | instruct | en | 32 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/togethercomputer/Llama-2-7B-32K-Instruct) \| [home](https://www.together.ai/) |
| LLongMA2-7B-16k-flash | ✅ | Llama2 | chat | en | 16 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/emozilla/LLongMA-2-7b-16k-flash) \| [paper](https://arxiv.org/pdf/2309.00071.pdf) |
| LongChat-v1.5-7B-32k | ✅ | Llama2 | chat | en | 32 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/lmsys/longchat-7b-v1.5-32k) \| [github](https://github.com/DachengLi1/LongChat) \| [blog](https://lmsys.org/blog/2023-06-29-longchat/) |
| Mistral-7B-v0.1 | ✅ | Mistral | chat | en | 32 | 7.2 | 28.0 | 13.5 | [hf](https://huggingface.co/mistralai/Mistral-7B-v0.1) \| [paper](https://arxiv.org/pdf/2310.06825.pdf) |
| Mistral-7B-Instruct-v0.2 | ✅ | Mistral | instruct | en | 32 | 7.2 | 28.0 | 13.5 | [hf](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) \| [paper](https://arxiv.org/pdf/2310.06825.pdf) |
| Mixtral-8x7B-v0.1 | ✅ | Mixtral | chat | en | 32 | 46.7 | 175.0 | 87.0 | [hf](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) \| [blog](https://mistral.ai/news/mixtral-of-experts/) |
| Mixtral-8x7B-Instruct-v0.1 | ✅ | Mixtral | instruct | en | 32 | 46.7 | 175.0 | 87.0 | [hf](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) \| [blog](https://mistral.ai/news/mixtral-of-experts/) |
| MPT-7B-Storywriter | ✅ | MPT | gen | en | 65 | 6.6 | 12.4 | 12.4 | [hf](https://huggingface.co/mosaicml/mpt-7b-storywriter) \| [blog](https://www.mosaicml.com/blog/mpt-7b) |
| NeuralChat-7B-v3.1 | ✅ | Mistral | chat | en | 32 | 7.2 | 28.0 | 13.5 | [hf](https://huggingface.co/Intel/neural-chat-7b-v3-1) \| [blog](https://medium.com/intel-analytics-software/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3) |
| OpenHermes2.5-7B | ✅ | Mistral | chat | en | 32 | 7.2 | 28.0 | 13.5 | [hf](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) \| [github](https://github.com/sponsors/teknium1) |
| QWen-7B | ✅ | QWen | chat | zh | 32 | 7.7 | 14.4 | 14.4 | [hf](https://huggingface.co/Qwen/Qwen-7B) \| [paper](https://arxiv.org/pdf/2309.16609.pdf) |
| Vicuna-v1.5-7B-16k | ✅ | Llama2 | chat | en | 16 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k) \| [github](https://github.com/lm-sys/FastChat) \| [blog](https://lmsys.org/blog/2023-03-30-vicuna/) |
| WizardCoder-Python-7B-v1.0 | ✅ | Llama2 | code | py | 16 | 6.7 | 12.8 | 12.6 | [hf](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0) \| [github](https://github.com/nlpxucan/WizardLM) |
| WizardMath-7B-v1.1 | ✅ | Mistral | math | en | 32 | 7.2 | 14.0 | 13.5 | [hf](https://huggingface.co/WizardLM/WizardMath-7B-V1.1) \| [github](https://github.com/nlpxucan/WizardLM) |
| XGen-7B-Instruct-8k | ✅ | Llama2 | instruct | en | 8 | 6.7 | 12.6 | 12.6 | [hf](https://huggingface.co/Salesforce/xgen-7b-8k-inst) \| [paper](https://arxiv.org/pdf/2309.03450.pdf) |

*Note: The rows are basically sort by the model names in the alphabetical order, and we use question mark "?" to indicate unknown information for any cell.*


### Meta Table Info

Some meta information about the [table](#baseline-table) is interpreted as follows:

* Open Source: Indicates whether the model is open-sourced (✅) or closed-sourced that can be accessible only through official remote API (❌).
* Base: Specifies the base modeling structure upon which the long-context model is built.
* Main Usage: Highlights the primary usage and capability of the model, categorized as `instruct` for instruction-following, `code/math` for code/math-related tasks, `gen` for text generation, and `chat` for general-purpose tasks through chat-like interaction.
* Main Lang: Indicates the primary language the model can understand\footnote{Models are typically pretrained on multi-language corpora and may be finetuned for specific languages as needed. So we choose the most suitable one corresponding to its application objectives.}, considering natural language, programming language, etc.
* **${L_{max}}$**: Represents the maximum context length handled by the model, measured in tokens (one unit "k" equals 1024 tokens).
* Statistics: Provides statistics about the model, including the number of parameters, memory footprint, and disk storage. All models are loaded with precision to float16 onto Nvidia GPU(s) without any quantization.
* Links: Includes publication links for accessing and learning more about the model, with `hf` indicating the Hugging Face hub for open-sourced models and `acc` representing official access for closed-sourced ones.

