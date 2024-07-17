# IO-Aware Attention
*Here're some resources about IO-Aware Attention*

### Intro

All of the methods above in pursuit of efficient attention can be considered as trading off high attention quality for low computation complexity, based on some theoretical or empirical properties of attention matrix and NLP tasks, including locality, sparsity, low-rankness, and other heuristic or mathematical tricks. 

In comparison, these IO-aware attention mechanisms below collectively represent efforts to optimize attention computations by considering the memory bottleneck while preserving the exactness of attention kernel calculations.


### Table of Contents
* [Intro](#intro)
* [Memory-Efficient Attention](#memory-efficient-attention)
* [Flash Attention](#flash-attention)
* [Paged Attention](#paged-attention)
* [Lightning Attention](#lightning-attention)


### Memory-Efficient Attention


#### DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training (LightSeq)

paper link: [here](https://arxiv.org/pdf/2310.03294)

github link: [here](https://github.com/RulinShao/LightSeq)

citation:

```bibtex
@misc{li2024distflashattn,
      title={DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training}, 
      author={Dacheng Li and Rulin Shao and Anze Xie and Eric P. Xing and Xuezhe Ma and Ion Stoica and Joseph E. Gonzalez and Hao Zhang},
      year={2024},
      eprint={2310.03294},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Self-attention Does Not Need Memory

paper link: [here](https://arxiv.org/pdf/2112.05682)

citation: 
```bibtex
@article{rabe2021self,
  title={Self-attention Does Not Need $ O (n\^{} 2) $ Memory},
  author={Rabe, Markus N and Staats, Charles},
  journal={arXiv preprint arXiv:2112.05682},
  year={2021}
}
```

### Flash Attention


#### FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

paper link: [here](https://arxiv.org/pdf/2407.08608)

blog link: [here](https://tridao.me/blog/2024/flash3/)

github link: [here](https://github.com/Dao-AILab/flash-attention)

citation:

```bibtex
@misc{shah2024flashattention3fastaccurateattention,
      title={FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision}, 
      author={Jay Shah and Ganesh Bikshandi and Ying Zhang and Vijay Thakkar and Pradeep Ramani and Tri Dao},
      year={2024},
      eprint={2407.08608},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.08608}, 
}
```

#### Is Flash Attention Stable?

paper link: [here](https://arxiv.org/pdf/2405.02803)

citation:

```bibtex
@misc{golden2024flashattentionstable,
      title={Is Flash Attention Stable?}, 
      author={Alicia Golden and Samuel Hsia and Fei Sun and Bilge Acun and Basil Hosmer and Yejin Lee and Zachary DeVito and Jeff Johnson and Gu-Yeon Wei and David Brooks and Carole-Jean Wu},
      year={2024},
      eprint={2405.02803},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      url={https://arxiv.org/abs/2405.02803}, 
}
```

#### Faster Causal Attention Over Large Sequences Through Sparse Flash Attention (SCFA)

paper link: [here](https://arxiv.org/pdf/2306.01160)

citation: 
```bibtex
@article{pagliardini2023faster,
  title={Faster Causal Attention Over Large Sequences Through Sparse Flash Attention},
  author={Pagliardini, Matteo and Paliotta, Daniele and Jaggi, Martin and Fleuret, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2306.01160},
  year={2023}
}
```
    

#### Flashattention-2: Faster attention with better parallelism and work partitioning


paper link: [here](https://arxiv.org/pdf/2307.08691.pdf)

github link: [here](https://github.com/Dao-AILab/flash-attention)

tutorial link: [here](../../notebooks/tutorial_triton.ipynb)

derivation manualscript link: [here](./flash_attn2.md)

citation: 
```bibtex
@article{dao2023flashattention,
  title={Flashattention-2: Faster attention with better parallelism and work partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}
```
    

#### Flashattention: Fast and memory-efficient exact attention with io-awareness

$$
\begin{align}
  O &:= \mathrm{softmax}\left( \left[\begin{matrix} P^{(1)} & P^{(2)} \end{matrix} \right]  \right) \left[\begin{matrix} V^{(1)} \\ V^{(2)} \end{matrix} \right]\\
  &= \alpha^{(1)} \mathrm{softmax}(P^{(1)}) V^{(1)} + \alpha^{(2)} \mathrm{softmax}(P^{(2)}) V^{(2)}
\end{align}
$$

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)

github link: [here](https://github.com/Dao-AILab/flash-attention)

citation: 
```bibtex
@article{dao2022flashattention,
  title={Flashattention: Fast and memory-efficient exact attention with io-awareness},
  author={Dao, Tri and Fu, Dan and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={16344--16359},
  year={2022}
}
```

### Paged Attention


#### Efficient memory management for large language model serving with pagedattention

paper link: [here](https://arxiv.org/pdf/2309.06180)

citation: 
```bibtex
@article{kwon2023efficient,
  title={Efficient memory management for large language model serving with pagedattention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E and Zhang, Hao and Stoica, Ion},
  journal={arXiv preprint arXiv:2309.06180},
  year={2023}
}
```

### Lightning Attention


#### Linear Attention Sequence Parallelism (LASP)

paper link: [here](https://arxiv.org/pdf/2404.02882)

github link: [here](https://github.com/OpenNLPLab/LASP)

citation:

```bibtex
@misc{sun2024linearattentionsequenceparallelism,
      title={Linear Attention Sequence Parallelism}, 
      author={Weigao Sun and Zhen Qin and Dong Li and Xuyang Shen and Yu Qiao and Yiran Zhong},
      year={2024},
      eprint={2404.02882},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      url={https://arxiv.org/abs/2404.02882}, 
}
```


#### Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models

paper link: [here](https://arxiv.org/pdf/2401.04658.pdf)

github link: [here](https://github.com/OpenNLPLab/lightning-attention)

citation:
```bibtex
@misc{qin2024lightning,
      title={Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models}, 
      author={Zhen Qin and Weigao Sun and Dong Li and Xuyang Shen and Weixuan Sun and Yiran Zhong},
      year={2024},
      eprint={2401.04658},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

#### TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer

paper link: [here](https://arxiv.org/pdf/2307.14995.pdf)

github link: [here](https://github.com/OpenNLPLab/lightning-attention)

citation:
```bibtex
@misc{qin2024transnormerllm,
      title={TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer}, 
      author={Zhen Qin and Dong Li and Weigao Sun and Weixuan Sun and Xuyang Shen and Xiaodong Han and Yunshen Wei and Baohong Lv and Xiao Luo and Yu Qiao and Yiran Zhong},
      year={2024},
      eprint={2307.14995},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```