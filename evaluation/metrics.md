# Metrics
*Here're some resources about Metrics commonly adopted for evaluation on specific NLP tasks*



### Table of Contents

* [Metric Table](#metric-table)
* [Meta Table Info](#meta-table-info)


### Metric Table

| Task Types | Metric Types |  |  |  |  |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  | CE/PPL | BPC/BPW | Acc/F1 | EM | ROUGE-1/-2/-L | BLEU/METEOR/TER | EntMent | Pass@k | Human/Model Judge |
| LM | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| MCQA | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| ExtQA | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Summ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Class | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Match | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Math | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Code | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenW | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MT | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |

*Note: The ❌ in the table does not imply that a specific metric cannot be applied to a task. Rather, it suggests that the metric might be less commonly used or that there could be more suitable alternatives.*


### Meta Table Info

We provide a concise introduction to the metrics in the [table](#metric-table):

* CE/PPL (Cross-Entropy/Perplexity): CE quantifies the $K\!L$ divergence between predicted distributions and the true distribution from the training corpus. PPL measures how well a language model predicts a sequence, simply formalized as $\exp (loss)$, where $loss$ denotes the cross-entropy loss for the test set.
    
* BPC/BPW (Bits per Character/Bits per Word): They measure the average number of bits required to encode characters or words, i.e. assess the efficiency of a model to compress text, simply calculated by $\mathrm{avg}_{\mathrm{T}}(loss)$, where $T$ is the number of characters or words respectively.

* Acc/F1 (Accuracy/F1 Score): Accuracy measures correct predictions in tasks with objective answers like classification and MCQA, while F1 balances precision and recall, as a more robust accuracy score.

* EM (Exact Matching): Evaluates exact sequence matches, crucial for tasks like code completion.

* [ROUGE-1/-2/-L](https://arxiv.org/pdf/1803.01937): Assess text similarity using n-grams overlapping, typically setting n=1 (unigram), 2 (bigram), and L (longest). They are widely used in tasks that EM may fail, such as summarization.

* [BLEU/METEOR/TER](https://aclanthology.org/W08-0312.pdf): These metrics are specific in machine translation tasks. BLEU measures the overlap of generated and reference translations based on n-grams. METEOR evaluates translation quality by considering various linguistic factors. TER quantifies the edit distance between the generated and reference translations.

* [Entity Mention (EntMent)](https://www.sciencedirect.com/science/article/pii/S0020025517310952): Evaluates coverage and correctness of important entities mentioned in the generated output text, especially for summarization.

* Pass@k: Evaluates if the generated answer ranks within the top-k provided answers, commonly used in code generation and some math tasks with multiple possible solutions.

* Human/Model Judge: Involves human or power models like GPT-4 to score text quality based on fluency, coherence, and other subjective criteria suitable for tasks like story generation.
