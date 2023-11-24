# Approximated Attention
*Here're some resources about Approximated Attention*



### Intro

In addition to heuristic approaches aimed at restricting full attention computation, some research explores the mathematical essence behind attention kernel computations. These studies use estimation methods based on the sparsity or low-rank properties of attention matrices to approximate attention with linear complexity, albeit at the cost of precision. We introduce several of these approximation techniques below.


## Definitions

Prior to delving into the literature, we introduce the following definitions to establish a unified representantion of mathematical symbols within the equations below.

* *Definition 1 (Attention Kernel):* We define a simplified version of the attention kernel computation below (*omitted the mask $M$, and other projection operations*), where $P$ denotes the unnormalized relevance matrix for each pair of $\bold q, \bold k$, $A$ is the row-wise normalized and scaled attention matrix and $O$ is the output hidden states.

$$
\begin{align}
P := Q\times K^{\mathrm{T}}, \;\;
A := \mathrm{softmax}[\cfrac{P}{\sqrt{d_k}}], \;\;
    O := A\times V
\end{align}
$$

* *Definition 2 (Causal Mask Function):* Note that we have not distinguished whether the methods are employed for BERT-like encoder-only LLMs or GPT-like decoder-only LLMs in previous sections since most of them can be trivially transferred from the BERT setting to the GPT setting with a causal attention mask. However, the casual mask is often non-trivial for many approximation strategies. 
  

  So to facilitate later discussions, we first define a general weighted causal function $\xi_{\bold w}$ in Eq.$(2)$, where $\bold w \in \mathbb{R}^L$ represents a weights vector for each row. This function will substitute the causal attention mask operation thus, we 

$$
\begin{align}
\xi_{\bold w}(Q,K,V) := \left[w_i\cdot \bold  q_i^{\mathrm{T}} \sum\limits_{j=1}^i \bold k_j \bold{v}_j^{\mathrm{T}} \right]_{i=1}^L
\end{align}
$$

* *Definition 3 (Generalized Kernelizable Attention):* The standard attention kernel computation can be generalized to kernelizable as below, where the kernel function $\mathcal{K}(\cdot,\cdot): \mathbb{R}^{d}\times \mathbb{R}^{d}\rightarrow R_+$ is applied row-wise to each pair of $\bold q_i, \bold k_j$ in $Q,K$, and $D$ is the normalization factor. From this view, the vanilla softmax attention just implements a specific kernel function as $\mathcal{K}(Q,K) = \exp(\frac{QK^{\mathrm{T}}}{\sqrt{d_k}})$, which explicitly derives a $L\times L$ attention matrix. But if we carefully choose another kernel function to be factorizable as the condition in Eq.$(4),(7)$ below, then simply applying the associative property, we can compute matrix multiplication of $K,V$ and $K, \bold 1_L$ ahead with lower complexity $O(Ld^2)$.

$$
\begin{align}
O &:= \left(D^{-1}\times\mathcal{K}(Q, K)\right)\times V\\
    &\xlongequal[associative]{\mathcal{K}(Q, K)=\widetilde Q\times \widetilde K^\mathrm{T}} D^{-1}\times\widetilde Q \times \left(\widetilde K^\mathrm{T} \times V\right) \\
    &\xlongequal[]{causal} D^{-1}\times \xi_{\bold 1_L}(\widetilde Q, \widetilde K, V)
\end{align}
$$
$$
\begin{align}
D &:=\mathrm{diag}\left[\mathcal{K}(Q, K) \times \bold 1_L\right]\\
&\xlongequal[associative]{\mathcal{K}(Q, K)=\widetilde Q\times \widetilde K^\mathrm{T}} \mathrm{diag}\left[\widetilde Q \times\left(\widetilde K\times \bold 1_L\right)\right]\\
&\xlongequal[]{causal} \mathrm{diag}\left[ \xi_{\bold 1_L}(\widetilde Q, \widetilde K, \bold 1_L)\right]
\end{align}
$$


### Table of Contents

* [Low-Rank Approximation](#low-rank-approximation)
* [Nested Attention](#nested-attention)
* [Kernelized Approximation](#kernelized-approximation)
* [Sparse-Kernelized Hybrid](#sparse-kernelized-hybrid)


### Low-Rank Approximation



#### Linformer: Self-attention with linear complexity [`READ`]

$$
\begin{align}
\widetilde O:= \mathrm{softmax}\left( \frac{Q\times\widetilde K^{\mathrm{T}}}{\sqrt{d_k}} \right)\times\widetilde V, \quad\widetilde K = E^{\mathrm{T}} K, \widetilde V = F^{\mathrm{T}} V
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2006.04768)

citation: 
```bibtex
@article{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and Li, Belinda Z and Khabsa, Madian and Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}
```


### Nested Attention

#### Luna: Linear unified nested attention [`READ`]

$$
\begin{align}
&A_{s} := \mathrm{elu}\left( \frac{Q_s \times K^{\mathrm{T}}}{\sqrt{d_k}} \right), \quad\widetilde S := A_{s}\times V, \;\;where\;\; Q_s := S\times W_q\\
        &A_{u} := \mathrm{softmax}\left( \xi_{\bold w_{inv}}(Q, V, A_{s}^{\mathrm{T}}) \right),\;
        \widetilde O := \xi_{\bold w_{inv}}(A_{u}, A_{s}^{\mathrm{T}}, V) ,\;\;where\;\; \bold w_{inv} := \left[ i^{-1} \right]_{i=1}^L\
\end{align}
$$

paper link: [here](https://proceedings.neurips.cc/paper/2021/file/14319d9cfc6123106878dc20b94fbaf3-Paper.pdf)

citation: 
```bibtex
@article{ma2021luna,
  title={Luna: Linear unified nested attention},
  author={Ma, Xuezhe and Kong, Xiang and Wang, Sinong and Zhou, Chunting and May, Jonathan and Ma, Hao and Zettlemoyer, Luke},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2441--2453},
  year={2021}
}
```

### Kernelized Approximation


#### Random feature attention [`READ`]

paper link: [here](https://arxiv.org/pdf/2103.02143)

citation: 
```bibtex
@article{peng2021random,
  title={Random feature attention},
  author={Peng, Hao and Pappas, Nikolaos and Yogatama, Dani and Schwartz, Roy and Smith, Noah A and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2103.02143},
  year={2021}
}
```


#### Rethinking attention with performers [`READ`]

$$
\begin{align}
&\mathcal{K}_{Pe}(\bold q,\bold k) := \mathbb{E}_{\omega}\left[\varphi_{Pe}(\bold q)\times \varphi_{Pe}(\bold k)^\mathrm{T}\right], \\
&where\;\;\; \varphi_{Pe}(\bold x) = \frac{h(\bold x)}{\sqrt{m}} \left[b_1(\omega_1^{\mathrm{T}}\bold x),..,b_1(\omega_m^{\mathrm{T}}\bold x),.., b_l(\omega_1^{\mathrm{T}}\bold x),..,b_l(\omega_m^{\mathrm{T}}\bold x) \right]
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2009.14794)

citation: 
```bibtex
@article{choromanski2020rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and Dohan, David and Song, Xingyou and Gane, Andreea and Sarlos, Tamas and Hawkins, Peter and Davis, Jared and Mohiuddin, Afroz and Kaiser, Lukasz and others},
  journal={arXiv preprint arXiv:2009.14794},
  year={2020}
}
```


#### Transformers are rnns: Fast autoregressive transformers with linear attention [`READ`]

$$
\begin{align}
\mathcal{K}_{Li}(\bold q,\bold k) := \varphi_{Li}(\bold q)\times \varphi_{Li}(\bold k)^\mathrm{T}, \;\;where\;\; \varphi_{Li}(\bold x) = \mathrm{elu}(\bold x) + 1
\end{align}
$$

paper link: [here](http://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)

citation: 
```bibtex
@inproceedings{katharopoulos2020transformers,
  title={Transformers are rnns: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois},
  booktitle={International conference on machine learning},
  pages={5156--5165},
  year={2020},
  organization={PMLR}
}
```



### Sparse-Kernelized Hybrid


#### Scatterbrain: Unifying sparse and low-rank attention approximation [`READ`]

$$
\begin{align}
\widetilde O := \left( \widetilde Q \times \widetilde K^{\mathrm{T}} + S \right)\times V = \widetilde Q \times (\widetilde K^{\mathrm{T}}\times V) + S\times V 
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2110.15343)

citation: 
```bibtex
@article{chen2021scatterbrain,
  title={Scatterbrain: Unifying sparse and low-rank attention approximation},
  author={Chen, Beidi and Dao, Tri and Winsor, Eric and Song, Zhao and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2110.15343},
  year={2021}
}
```
    