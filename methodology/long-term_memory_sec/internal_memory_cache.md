# Internal MemoryCache
*Here's some resources about Internal MemoryCache*


### Intro

Recalling the temporality of natural language representations instead of the success of full parallelism in Transformer, we introduce the concept of *Internal MemoryCache* based on recurrence mechanisms. It divides long text into a stream of fixed-length segments and enhances the query $Q_t^n$ of the current $t$-th segment in the $n$-th layer with more contextual information $\widetilde{K}_t^n, \widetilde{V}_t^n$. This contextual information is obtained from cached or distilled information from previous segments, stored in a memory cache denoted as $Mem$, as shown in the equation below.

$$
\begin{align}
    & Q_{t}^{n}, \widetilde K_{t}^{n}, \widetilde V_{t}^{n} := X_t^{n}W_q, \widetilde X_t^{n}W_k,\widetilde X_{t}^{n}W_v,\\
    &where\quad X_{t}^{n} := O_{t}^{n-1},  \widetilde X_{t}^{n} := \left[ \mathrm{Mem}(n,t,..) \circ O_{t}^{n-1}  \right]
\end{align}
$$


To facilitate later equations, we assume that each segment has the same length $l$, and the models consist of $N$ layers of transformer blocks. The notation $[\circ]$ represents the concatenation operation along the length dimension. It’s worth noting that the variables in the memory cache $Mem$ are usually detached from the computation graph, eliminating the need for gradient computation, which we denote with a hat accent, such as $\widehat{X}$.


### Table of Contents

* [Segment-Level Recurrence](#segment-level-recurrence)
* [Retrospective Recurrence](#retrospective-recurrence)
* [Continuous-Signal Memory](#continuous-signal-memory)
* [Alternate Cache Designs](#alternate-cache-designs)



### Segment-Level Recurrence


#### Segatron: Segment-aware transformer for language modeling and understanding [`READ`]

paper link: [here](https://ojs.aaai.org/index.php/AAAI/article/download/17485/17292)

citation: 
```bibtex
@inproceedings{bai2021segatron,
  title={Segatron: Segment-aware transformer for language modeling and understanding},
  author={Bai, He and Shi, Peng and Lin, Jimmy and Xie, Yuqing and Tan, Luchen and Xiong, Kun and Gao, Wen and Li, Ming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={14},
  pages={12526--12534},
  year={2021}
}
```
    


#### Compressive transformers for long-range sequence modelling [`READ`]

$$
\begin{align}
    &\mathrm{Mem_{Comp}}(n,t,m_1,m_2,c) := \left[\mathrm{Mem_{f_c}} \circ  \mathrm{Mem_{XL}}(n,t,m_1)\right]\\
    &where\quad \mathrm{Mem_{f_c}} := \left[ f_c(\widehat O_{t-m_1-m_2}^{n-1}) \circ..\circ  f_c(\widehat O_{t-m_1-1}^{n-1})\right]
\end{align}
$$


paper link: [here](https://arxiv.org/pdf/1911.05507)

citation: 
```bibtex
@article{rae2019compressive,
  title={Compressive transformers for long-range sequence modelling},
  author={Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and Lillicrap, Timothy P},
  journal={arXiv preprint arXiv:1911.05507},
  year={2019}
}
```


#### Transformer-xl: Attentive language models beyond a fixed-length context [`READ`]

$$
\begin{align}
    \mathrm{Mem_{XL}}(n,t,m) := \left[\widehat{O_{t-m}^{n-1}} \circ..\circ \widehat{O_{t-1}^{n-1}}\right]
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/1901.02860.pdf%3Ffbclid%3DIwAR3nwzQA7VyD36J6u8nEOatG0CeW4FwEU_upvvrgXSES1f0Kd-)

citation: 
```bibtex
@article{dai2019transformer,
  title={Transformer-xl: Attentive language models beyond a fixed-length context},
  author={Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:1901.02860},
  year={2019}
}
```



### Retrospective Recurrence


#### Readtwice: Reading very large documents with memories [`READ`]

paper link: [here](https://arxiv.org/pdf/2105.04241)

citation: 
```bibtex
@article{zemlyanskiy2021readtwice,
  title={Readtwice: Reading very large documents with memories},
  author={Zemlyanskiy, Yury and Ainslie, Joshua and de Jong, Michiel and Pham, Philip and Eckstein, Ilya and Sha, Fei},
  journal={arXiv preprint arXiv:2105.04241},
  year={2021}
}
```


#### Addressing some limitations of transformers with feedback memory [`READ`]

paper link: [here](https://arxiv.org/pdf/2002.09402)

citation: 
```bibtex
@article{fan2020addressing,
  title={Addressing some limitations of transformers with feedback memory},
  author={Fan, Angela and Lavril, Thibaut and Grave, Edouard and Joulin, Armand and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:2002.09402},
  year={2020}
}
```
    

#### ERNIE-Doc: A retrospective long-document modeling transformer [`READ`]

$$
 \begin{align}
    & \mathrm{Mem_{Ernie}}(n,t) := \widehat O^{n}_{t-1}
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2012.15688)

citation: 
```bibtex
@article{ding2020ernie,
  title={ERNIE-Doc: A retrospective long-document modeling transformer},
  author={Ding, Siyu and Shang, Junyuan and Wang, Shuohuan and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2012.15688},
  year={2020}
}
```


### Continuous-Signal Memory


#### ∞-former: Infinite Memory Transformer [`READ`]

$$
 \begin{align}
    & \mathrm{Mem}_{\infty}:= \widetilde{X}(s) = B^{\mathrm{T}} \Phi(s), \\
    &s.t.\quad \widetilde X(s_i) \approx X_i, \quad s_i := i / L, \quad \forall i \in [1,..,L]
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2109.00301)

citation: 
```bibtex
@article{martins2021infty,
  title={$$\backslash$infty $-former: Infinite Memory Transformer},
  author={Martins, Pedro Henrique and Marinho, Zita and Martins, Andr{\'e} FT},
  journal={arXiv preprint arXiv:2109.00301},
  year={2021}
}
```


### Alternate Cache Designs


#### Scaling Transformer to 1M tokens and beyond with RMT [`READ`]

$$
\begin{align}
    & \widetilde O_{t}^N := \mathrm{Transformer}(\widetilde X_{t}^0),\quad\widetilde X_{t}^0 := \left[ X_{t}^{mem} \circ X_{t}^0 \circ X_{t}^{mem} \right]\\
    &where\quad X_{t}^{mem} := O_{t-1}^{write},\quad \left[O_{t-1}^{read}\circ O_{t-1}^{N}  \circ O_{t-1}^{write}\right] := \widetilde  O_{t-1}^{N}
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2304.11062.pdf??ref=eiai.info)

citation: 
```bibtex
@article{bulatov2023scaling,
  title={Scaling Transformer to 1M tokens and beyond with RMT},
  author={Bulatov, Aydar and Kuratov, Yuri and Burtsev, Mikhail S},
  journal={arXiv preprint arXiv:2304.11062},
  year={2023}
}
```


#### Memorizing transformers [`READ`]

$$
\begin{align}
    & (\widetilde K_t^{N},\widetilde V_t^{N}) := \left[ \mathrm{retr}(Q_t^{N},m,k) \circ (K_t^{N},V_t^{N}) \right],\\
    & where\quad \mathrm{retr}(Q_t^{N},m,k) := \mathrm{kNN}\left[Q_t^{N},\{(K_{t-\tau}^{N},V_{t-\tau}^{N})\}_{\tau=1}^{m}\right]
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2203.08913)

citation: 
```bibtex
@article{wu2022memorizing,
  title={Memorizing transformers},
  author={Wu, Yuhuai and Rabe, Markus N and Hutchins, DeLesley and Szegedy, Christian},
  journal={arXiv preprint arXiv:2203.08913},
  year={2022}
}
```


#### Memformer: A memory-augmented transformer for sequence modeling [`READ`]

paper link: [here](https://arxiv.org/pdf/2010.06891)

citation: 
```bibtex
@article{wu2020memformer,
  title={Memformer: A memory-augmented transformer for sequence modeling},
  author={Wu, Qingyang and Lan, Zhenzhong and Qian, Kun and Gu, Jing and Geramifard, Alborz and Yu, Zhou},
  journal={arXiv preprint arXiv:2010.06891},
  year={2020}
}
```





    