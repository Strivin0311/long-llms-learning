# Attention Bias
*Here're some resources about Attention Bias*

### Intro

As alternative mechanisms to explicitly encoding positional information, *attention bias* have been explored to capture the sequentiality and temporality of natural language incorporated into attention kernel. 


### Table of Contents
* [Intro](#intro)
* [Definition](#definition)
* [Super Baseline](#super-baseline)
* [More Elaborate Designs](#more-elaborate-designs)

### Definition

As shown in equation below, the attention bias is depicted as a matrix, denoted as $B$, which is added to the unnormalized attention weights matrix $P$ before applying the softmax operation. Each element of this matrix, indexed by $(i,j)$, carries positional information encoded by a function $\mathcal{B}$. Thus, it is reasonable to regard the attention bias as a form of relative PEs.

$$
\begin{align}
    & \widetilde{P} := P + B, \quad B \in \mathbb{R}^{L\times L},\quad where\quad B_{ij} := \mathcal{B}(i,j), \quad\forall i,j \in \{0,1,..,L-1\}
\end{align}
$$


### Super Baseline

#### Transformer Upgrade Roadmap: 7. Length Extrapolation and Local Attention [`READ`]

$$
  \begin{align}
      & \mathcal{B}_{super\text{-}baseline}(i,j) := \begin{cases}
          0, & |i-j| \in [0, max\text{-}length]\\
          -\infty, & otherwise\\
      \end{cases}
  \end{align}
$$

blog link: [here](https://spaces.ac.cn/archives/9431)

citation: 
```bibtex
@misc{transformer-upgrade-7,
    author = "Su, Jianlin",
    title = "Transformer Upgrade Roadmap: 7. Length Extrapolation and Local Attention",
    year = "2023",
    month = "Jan",
    howpublished = "\url{https://spaces.ac.cn/archives/9431}"
}
```

illustration: 
Su introduced a simple method in his blog where he utilizes a *super-baseline* approach during inference, as illustrated in the equation. This method relies on a local causal attention mask, where each query attends to keys whose distances have not exceeded $L_{max}$ while still applying RoPE. According to Su's experiments, this approach proves to be simple, low-cost and performs sufficiently well compared to the more elaborate designs below, thus referred as a *super-baseline*.
  



### More Elaborate Designs

#### Dissecting transformer length extrapolation via the lens of receptive field analysis (Sandwitch) [`READ`]

$$
  \begin{align}
      & \mathcal{B}_{Sandwitch}(i,j) := \lambda\cdot \langle \mathrm{SinPE}(i), \mathrm{SinPE}(j) \rangle
  \end{align}
$$

paper link: [here](https://aclanthology.org/2023.acl-long.756.pdf)

citation: 
```bibtex
@inproceedings{chi2023dissecting,
  title={Dissecting transformer length extrapolation via the lens of receptive field analysis},
  author={Chi, Ta-Chung and Fan, Ting-Han and Rudnicky, Alexander and Ramadge, Peter},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={13522--13537},
  year={2023}
}
```

#### Kerple: Kernelized relative positional embedding for length extrapolation [`READ`]

$$
  \begin{align}
      & \mathcal{B}_{\textit{KERPLE}}(i,j) := \begin{cases}
          -r_1\log(1 + r_2|i-j|), \quad r_1, r_2 > 0\\
          -r_1 |i-j|^{r_2}, \quad r_1 > 0, r_2 \in (0,2]\\
      \end{cases}
  \end{align}
$$

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/37a413841a614b5414b333585e7613b8-Paper-Conference.pdf)

citation: 
```bibtex
@article{chi2022kerple,
  title={Kerple: Kernelized relative positional embedding for length extrapolation},
  author={Chi, Ta-Chung and Fan, Ting-Han and Ramadge, Peter J and Rudnicky, Alexander},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={8386--8399},
  year={2022}
}
```


#### Train short, test long: Attention with linear biases enables input length extrapolation (Alibi) [`READ`]

$$
  \begin{align}
      & \mathcal{B}_{ALiBi}^{(h)}(i,j) := -\lambda^{(h)}\cdot |i-j|, \quad\lambda^{(h)} := \cfrac{1}{2^h} \space or\space \cfrac{1}{2^{h/2}}
  \end{align}
$$

paper link: [here](https://arxiv.org/pdf/2108.12409.pdf%5C)

citation: 
```bibtex
@article{press2021train,
  title={Train short, test long: Attention with linear biases enables input length extrapolation},
  author={Press, Ofir and Smith, Noah A and Lewis, Mike},
  journal={arXiv preprint arXiv:2108.12409},
  year={2021}
}
```
    
