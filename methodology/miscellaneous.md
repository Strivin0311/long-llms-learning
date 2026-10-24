# Miscellaneous Methodology for long LLMs
*Here're some resources about Miscellaneous Methodology for long LLMs*


### Intro

This section provides a concise overview of miscellaneous solutions that extend the previously discussed four categories, offering a broader perspective on enhancing the effective context window of LLMs or optimizing the efficiency when using off-the-shelf LLMs. It is worth noting that the literature covered here may not be exhaustive or specific to Transformer-based models. Many of these techniques are applicable universally to any model equipped with deep neural networks, albeit particularly crucial for large-scale LLMs.


### Table of Contents
* [Intro](#intro)
* [Specific Objectives](./miscellaneous_sec/spec_objective.md)
* [Specific Activation Functions](./miscellaneous_sec/spec_activation.md)
* [Mixture of Experts](./miscellaneous_sec/moe.md)
* [Parallelism](./miscellaneous_sec/parallel.md)
* [Weight Compression](./miscellaneous_sec/weight_compress.md)
* [Long-contenxt Training](#long-contenxt-training)



### Long-contenxt Training


#### Data Engineering for Scaling Language Models to 128K Context

paper link: [here](https://arxiv.org/pdf/2402.10171.pdf)

citation:

```bibtex
@misc{fu2024data,
    title={Data Engineering for Scaling Language Models to 128K Context}, 
    author={Yao Fu and Rameswar Panda and Xinyao Niu and Xiang Yue and Hannaneh Hajishirzi and Yoon Kim and Hao Peng},
    year={2024},
    eprint={2402.10171},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


#### Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization (LongMab-PO)

paper link: [here](https://arxiv.org/pdf/2508.13993)

github link: [here](https://github.com/NEUIR/LongMab-PO)

citation:

```bibtex
@inproceedings{duan-etal-2026-chunks,
    title={Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context {LLM} Preference Optimization},
    author={Shaohua Duan and Pengcheng Huang and Xinze Li and Zhenghao Liu and Xiaoyuan Yi and Yukun Yan and Shuo Wang and Yu Gu and Ge Yu and Maosong Sun},
    booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    year={2026},
    url={https://aclanthology.org/2026.acl-long.161/}
}
```



    