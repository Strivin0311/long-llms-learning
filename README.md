# long-llms-learning

<p align="left">
<a href="https://arxiv.org/abs/2311.12351v2">
<img alt="survey" src="https://img.shields.io/badge/survey-arxiv:2311.12351v2-blue">
</a>
</p>

A repository sharing the panorama of the methodology literature on Transformer **architecture** upgrades in Large Language Models for handling **extensive context windows**, with real-time updating the newest published works.


## Overview of Survey

For a clear taxonomy and more insights about the methodology, you can refer to our **survey**: [Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2311.12351v2) with a overview shown below

(*Note: this is currently the draft version, with a few more work to do about some writing and the appendix*)


![Overview of the survey](./imgs/overview_with_caption_v2.png)


## Latest Works

* [2024.02.15] [Data Engineering for Scaling Language Models to 128K Context](https://arxiv.org/abs/2402.10171), located [here](./methodology/miscellaneous.md#long-contenxt-training) in this repo.


## More to Learn

* We've also released a building repo [long-llms-evals](https://github.com/Strivin0311/long-llms-evals) as a pipeline to evaluate various methods designed for general / specific LLMs to enhance their long-context capabilities on well-known long-context benchmarks.

* This repo is also a sub-track for my [llms-learning](https://github.com/Strivin0311/llms-learning) repo, where you can learn more technologies and applicated tasks about the full-stack of Large Language Models.


## Table of Contents

* [Methodology](./methodology/)
  * [Efficient Attention](./methodology/efficient_attn.md)
  * [Long-Term Memory](./methodology/long-term_memory.md)
  * [Extrapolative PEs](./methodology/extrapolative_pes.md)
  * [Context Processing](./methodology/context_process.md)
* [Evaluation](./evaluation/)
  * [Datasets](./evaluation/datasets.md)
  * [Metrics](./evaluation/metrics.md)
  * [Baselines](./evaluation/baselines.md)
* [Tookits](./toolkits/README.md)
* [Empirical Study & Survey](./empirical.md)



## Contribution

If you want to make contribution to this repo, you can just make a pr / email us with the link to the paper(s) or use the format as below:

* (un)read paper format:
```
#### <paper title> [(UN)READ]

paper link: [here](<link address>)

xxx link: [here](<link address>)

citation:
<bibtex citation>
```


## Citation

If you find the survey or this repo helpful in your research or work, you can cite our paper as below:

```bibtex
@misc{huang2024advancing,
      title={Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey}, 
      author={Yunpeng Huang and Jingwei Xu and Junyu Lai and Zixu Jiang and Taolue Chen and Zenan Li and Yuan Yao and Xiaoxing Ma and Lijuan Yang and Hao Chen and Shupeng Li and Penghao Zhao},
      year={2024},
      eprint={2311.12351},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
