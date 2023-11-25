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


### Memory-Efficient Attention

#### Self-attention Does Not Need Memory [`READ`]

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

#### Faster Causal Attention Over Large Sequences Through Sparse Flash Attention (SCFA) [`READ`]

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
    

#### Flashattention-2: Faster attention with better parallelism and work partitioning [`READ`]

paper link: [here](https://arxiv.org/pdf/2307.08691.pdf?trk=public_post_comment-text)

citation: 
```bibtex
@article{dao2023flashattention,
  title={Flashattention-2: Faster attention with better parallelism and work partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}
```
    

#### Flashattention: Fast and memory-efficient exact attention with io-awareness [`READ`]

$$
\begin{align}
  O &:= \mathrm{softmax}\left( \left[\begin{matrix} P^{(1)} & P^{(2)} \end{matrix} \right]  \right) \left[\begin{matrix} V^{(1)} \\ V^{(2)} \end{matrix} \right]\\
  &= \alpha^{(1)} \mathrm{softmax}(P^{(1)}) V^{(1)} + \alpha^{(2)} \mathrm{softmax}(P^{(2)}) V^{(2)}
\end{align}
$$

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)

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


#### Efficient memory management for large language model serving with pagedattention [`READ`]

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