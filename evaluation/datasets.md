# Datasets
*Here're some resources about Datasets specific for various NLP tasks with long-text inputs*

### Table of Contents

* [Dataset Table](#dataset-table)
* [Meta Table Info](#meta-table-info)


### Dataset Table


| Dataset | Language | Task  Amount | Task Types |  |  |  |  |  |  |  |  |  | Lengths (kilo words) |  |  | Quality |  | Splits | Count | Format |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  |  |  | LM | MCQA | ExtQA | Summ | Class | Match | Math | Code | OpenW | MT | Avg | Min | Max | Human Labeled | Model Assisted |  |  |  |
| [ArXiv + PubMed](https://github.com/armancohan/long-summarization) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 5.2 | 0 | 157.3 | ✅ | ❌ | train/test/val | 322K/13.1K/13.1K | jsonl |
| [BigPatent](https://github.com/evasharma/bigpatent) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 3.2 | 0.2 | 83.2 | ✅ | ❌ | train/test/val | 1.2M/67.1K/67.1K | json |
| [BookSum](https://huggingface.co/datasets/kmfoda/booksum) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 4.5 | 0.04 | 115.8 | ✅ | ❌ | train/test/val | 9.6K/1.4K/1.5K | csv |
| [CAIL2019-SCM](https://github.com/china-ai-law-challenge/CAIL2019/tree/master/scm) | zh | 1 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | 2.0 | 1.8 | 2.6 | ✅ | ❌ | train/test/val | 5.1K/1.5K/1.5K | jsonl |
| [ChapterBreak](https://github.com/simengsun/chapterbreak) | en  | 1 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 25.4 | 2.3 | 405.8 | ✅ | ❌ | train | 9.6K | json |
| [CNN/DailyMail](https://github.com/theamrzaki/text_summurization_abstractive_methods) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.8 | 0 | 2.9 | ✅ | ❌ | test | 312K | txt |
| [ContractNLI](https://github.com/stanfordnlp/contract-nli-bert) | en | 1 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 2.0 | 0.5 | 8.7 | ✅ | ❌ | train/test/dev | 423/123/61 | json |
| [DuLeMon](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon) | zh | 1 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.6 | 0.3 | 1.4 | ✅ | ❌ | train/test/dev | 25.4K/1.1K/1.1K | jsonl |
| [ECtHR](https://huggingface.co/datasets/huynguyendayrui/ecthr) | en | 1 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 2.2 | 0.01 | 51.3 | ✅ | ❌ | train/test/dev | 7.3K/3K/1.3K | jsonl |
| [GovReport](https://github.com/luyang-huang96/LongDocSum) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 43.5 | 0.2 | 1386.2 | ✅ | ❌ | test | 19.4K | json |
| [HotpotQA](https://huggingface.co/datasets/hotpot_qa) | en | 1 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.9 | 0.01 | 2.0 | ✅ | ❌ | train/dev | 90K/14.8K | json |
| [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) | en/zh | 12 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | 71.1 | 0.1 | 560.3 | ✅ | ❌ | test | 3.9K | jsonl |
| [LCC-Python](https://huggingface.co/datasets/microsoft/LCC_python) | py | 1 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 1.4 | 0.2 | 23.3 | ✅ | ❌ | train/test/val | 100K/10K/10K | parquet |
| [LEval](https://github.com/OpenLMLab/LEval) | en | 20 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 9.2 | 2.0 | 137.5 | ✅ | ✅ | test | 537 | jsonl |
| [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k) | en | 1 | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 6.7 | 0 | 32.7 | ✅ | ❌ | train | 12K | json |
| [LongBench](https://github.com/THUDM/LongBench) | en/zh | 21 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | 7.2 | 0.1 | 44.2 | ✅ | ✅ | test | 8.4K | jsonl |
| [LongChat-Lines](https://huggingface.co/datasets/abacusai/LongChat-Lines) | en | 1 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 2.6 | 0.6 | 5.6 | ✅ | ❌ | test | 700 | parquet |
| [LOT](https://github.com/thu-coai/LOT-LongLM) | zh | 4 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | 0.2 | 0.06 | 0.5 | ✅ | ❌ | train/test/dev | 35.2K/2.4K/1.8K | jsonl |
| [LRA - AAN](https://github.com/google-research/long-range-arena) | en  | 1 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | 4.7 | 0.02 | 55.5 | ✅ | ❌ | train/test/dev | 147K/17.4K/18K | tsv |
| [LRA - ListOps](https://github.com/google-research/long-range-arena) | en | 1 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 3 | 0.01 | 5.9 | ✅ | ❌ | train/test/dev | 96K/2K/2K | tsv |
| [MuLD](https://github.com/ghomashudson/muld) | en | 6 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | 27.7 | 0 | 359.1 | ✅ | ❌ | train/test/val | 155.9K/14.4K/11.6K | jsonl |
| [MultiNews](https://github.com/Alex-Fabbri/Multi-News) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2.1 | 0.1 | 464.2 | ✅ | ❌ | train/test/val | 45.0K/5.6K/5.6K | txt |
| [Multi-Session Chat](https://huggingface.co/datasets/nayohan/multi_session_chat) | en | 1 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 0.3 | 0.1 | 1.2 | ✅ | ❌ | train/test/val | 17.9K/2.5K/3K | parquet |
| [Nature Questions](https://huggingface.co/datasets/natural_questions) | en | 1 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 9.8 | 0.2 | 169.3 | ✅ | ❌ | train/dev | 307K/7.8K | json |
| [NewsGroups](http://qwone.com/~jason/20Newsgroups/) | en | 1 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.3 | 0 | 11.8 | ✅ | ❌ | test | 20K | txt |
| [NewsRoom](https://github.com/lil-lab/newsroom) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.7 | 0 | 178.5 | ✅ | ❌ | train/test/dev | 995.0K/108.9K/108.8K | jsonl |
| [OpenChat-ShareGPT4-Clean](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset) | en | 1 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 1.6 | 0 | 152.8 | ✅ | ✅ | train | 80.2K | json |
| [ProofNet](https://huggingface.co/datasets/hoskinson-center/proofnet) | en | 1 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 0.2 | 0.05 | 0.7 | ✅ | ❌ | test/val | 186/185 | jsonl |
| [QMSum](https://github.com/Yale-LILY/QMSum) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 10.8 | 1.7 | 26.8 | ✅ | ❌ | train/test/val | 162/35/35 | jsonl |
| [SCROLLS](https://github.com/tau-nlp/scrolls) | en | 7 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 33.0 | 0.2 | 356.1 | ✅ | ❌ | train/test/val | 89.7K/17.5K/12.3K | jsonl |
| [SQuAD](https://huggingface.co/datasets/squad) | en | 1 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.1 | 0.02 | 0.7 | ✅ | ❌ | train/val | 87.6K/10.6K | parquet |
| [SummScreen](https://github.com/mingdachen/SummScreen) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 7.3 | 1.6 | 24.0 | ✅ | ❌ | train/test/dev | 22.6K/2.1K/2.1K | jsonl |
| [Synthetic-Persona-Chat](https://huggingface.co/datasets/google/Synthetic-Persona-Chat) | en | 1 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 0.4 | 0.05 | 0.8 | ✅ | ✅ | train/test/val | 8.9K/968/1K | csv |
| [THUCnews](https://github.com/ShannonAI/ChineseBert/tree/main/tasks/THUCNew) | zh | 1 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.9 | 0 | 79.5 | ✅ | ❌ | test | 836K | txt |
| [UltraChat](https://huggingface.co/datasets/stingning/ultrachat) | en | 1 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 1.0 | 0.03 | 3.6 | ✅ | ✅ | train | 1.4M | jsonl |
| [WikiQA-AlteredNumericQA](https://huggingface.co/datasets/abacusai/WikiQA-Altered_Numeric_QA) | en | 1 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 4.0 | 0.8 | 11.2 | ✅ | ❌ | test | 1.8K | parquet |
| [WikiQA-FreeFormQA](https://huggingface.co/datasets/abacusai/WikiQA-Free_Form_QA) | en | 1 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 3.8 | 0.6 | 11.5 | ✅ | ❌ | test | 2.4K | parquet |
| [WMT14 EN-CS](https://huggingface.co/datasets/wmt14) | en/cs | 1 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 0.04 | 0 | 3.6 | ✅ | ❌ | train/test/cal | 1M/3K/3K | sgm |
| [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) | en | 1 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0.4 | 0 | 29.2 | ✅ | ❌ | train/test/val | 204K/11.3K/11.3K | summary |

*Note 1: We sort datasets at each row in the alphabetical order and use slash "/" to separate the multiple contents in any single cell.*

*Note 2: The presence of common dirty data may result in extremely short samples, thus many datasets in the table containing samples with a minimum length approaching zero.*




### Meta Table Info

The meta information regarding the columns in [table](#dataset-table) is as follows:


* Language: Language information is represented using abbreviations such as `en` for English, `zh` for Chinese, and `py` for Python.

* Task Amount and Types: We categorize the common NLP tasks into ten types, including language modeling (`LM`), multi-choice question-answering (`MCQA`), extractive question-answering with information retrieval (`ExtQA`), document summarization (`Summ`), text classification (`Class`), text-pair matching (`Match`), math problem solving and reasoning (`Math`), code tasks (`Code`), open-ended writing (`OpenW`), and machine translation (`MT`).

* Lengths: Average (`avg`), minimum (`min`), and maximum (`max`) sample lengths are provided in kilo "words" for each dataset (In the context of our study, "words" are approximately considered to be separated by spaces in English and code, while individual Chinese characters are treated as words, to avoid the inconsistency by different tokenizers.), where "words" is defined based on sample content (For example, if one typical sample has the prompt template like: "Read this `{context}`, and answer the question below: `{question}`, we will calculate the number of words in both context and question part, ignoring the fixed remaining part in the template).

* Quality: Quality assessment is simply based on two dimensions: `Human Labeled` (labels generated by humans) and `Model Assisted` (prompts or labels generated by off-the-shelf LLMs), since the lack of quantitative oracles. 

* Splits: This indicates dataset partitioning, including conventional triple-split formats like `train/test/val`, a single `test` split for evaluation, a single `train` split for training/finetuning, etc.

* Count: Provides statistics on the number of samples for each split (one unit "K"/"M" equals 1,000/1,000,000 samples).

* Format: Tags the file format of samples, including `jsonl`, `json`, `csv`, `txt`, `tsv`, `parquet`, and more.



