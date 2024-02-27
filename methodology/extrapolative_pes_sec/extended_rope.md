# Extended RoPE
*Here're some resources about Extended RoPE*


### Intro

[RoPE](../extrapolative_pes.md#roformer-enhanced-transformer-with-rotary-position-embedding-read), is a widely-used positional encoding scheme utilized in popular LLMs such as Llama, GLM, PaLM. It offers advantages such as relative distance decay, training stability, compatibility with linear attention, and better length extrapolation capabilities compared to the traditional SinPE, as demonstrated in various experiments, albeit not that satisfactory. Therefore, several research works have aimed to extend RoPE using various strategies to enhance its length extrapolation capabilities. 




### Table of Contents
* [Intro](#intro)
* [Scaling Strategies](#scaling-strategies)
* [Truncation Strategies](#truncation-strategies)
* [Rearrangement Strategies](#rearrangement-strategies)



### Scaling Strategies


#### Effective Long-Context Scaling of Foundation Models [`READ`]

paper link: [here](https://arxiv.org/pdf/2309.16039.pdf)

citation:
```bibtex
@misc{xiong2023effective,
      title={Effective Long-Context Scaling of Foundation Models}, 
      author={Wenhan Xiong and Jingyu Liu and Igor Molybog and Hejia Zhang and Prajjwal Bhargava and Rui Hou and Louis Martin and Rashi Rungta and Karthik Abinav Sankararaman and Barlas Oguz and Madian Khabsa and Han Fang and Yashar Mehdad and Sharan Narang and Kshitiz Malik and Angela Fan and Shruti Bhosale and Sergey Edunov and Mike Lewis and Sinong Wang and Hao Ma},
      year={2023},
      eprint={2309.16039},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Giraffe: Adventures in expanding context lengths in llms (Power-Scaling) [`READ`]

$$
\begin{align}
  &\quad \widetilde\beta^{i} := \beta^{i} / (1-2i/d)^{\kappa}
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2308.10882)

citation: 
```bibtex
@article{pal2023giraffe,
  title={Giraffe: Adventures in expanding context lengths in llms},
  author={Pal, Arka and Karkhanis, Deep and Roberts, Manley and Dooley, Samuel and Sundararajan, Arvind and Naidu, Siddartha},
  journal={arXiv preprint arXiv:2308.10882},
  year={2023}
}
```


#### Yarn: Efficient context window extension of large language models [`READ`]

paper link: [here](https://arxiv.org/pdf/2309.00071)

citation: 
```bibtex
@article{peng2023yarn,
  title={Yarn: Efficient context window extension of large language models},
  author={Peng, Bowen and Quesnelle, Jeffrey and Fan, Honglu and Shippole, Enrico},
  journal={arXiv preprint arXiv:2309.00071},
  year={2023}
}
```
    


#### Add NTK-Aware interpolation "by parts" correction (NTK-by-parts) [`READ`]

github pr link: [here](https://github.com/jquesnelle/yarn/pull/1)

citation:
```bibtex
@misc{ntk-by-parts,
    author = "bloc97",
    title = {Add NTK-Aware interpolation "by parts" correction, 2023},
    year = "2023",
    month = "Jul",
    howpublished = "\url{https://github.com/jquesnelle/yarn/pull/1}"
}
```


#### Transformer Upgrade Roadmap: 11. Taking beta-base Encoding to the Limit (NTK-mixed RoPE) [`READ`]

blog link: [here](https://spaces.ac.cn/archives/9706)

citation:
```bibtex
@misc{transformer-upgrade-11,
    author = "Su, Jianlin",
    title = "Transformer Upgrade Roadmap: 11. Taking beta-base Encoding to the Limit",
    year = "2023",
    month = "Jul",
    howpublished = "\url{https://spaces.ac.cn/archives/9706}"
}
```


#### Dynamically Scaled RoPE further increases performance of long context LLaMA with zero fine-tuning (Dynamic-NTK) [`READ`]

blog link: [here](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)

citation:
```bibtex
@misc{dynamic-ntk,
    author = "emozilla",
    title = "Dynamically Scaled RoPE further increases performance of long context LLaMA with zero fine-tuning",
    year = "2023",
    month = "Jun",
    howpublished = "\url{https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically\\_scaled\\_rope\\_further\\_increases/}"
}
```


#### NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation (NTK-RoPE) [`READ`]

$$
\begin{align}
  &\quad  \widetilde\beta := c_{\kappa}\cdot\beta, \\
  &s.t.\quad\cfrac{n}{\widetilde\beta^{d/2-1}} = \cfrac{n/\kappa}{\beta^{d/2-1}} \Rightarrow c_{\kappa} = \kappa^{2/(d-2)}
\end{align}
$$

blog link: [here](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)

citation: 
```bibtex
@misc{ntk-aware-rope,
    author = "bloc97",
    title = "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation",
    year = "2023",
    month = "Jun",
    howpublished = "\url{https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware\\_scaled\\_rope\\_allows\\_llama\\_models\\_to\\_have/}"
}
```


#### Extending context window of large language models via positional interpolation (PI) [`READ`]

$$
\begin{align}
  &\quad  \widetilde P_{i,j} := \langle R_{i/\kappa}\mathbf q, R_{j/\kappa}\mathbf k\rangle = \mathbf q^{\mathrm{T}} R_{\frac{j-i}{\kappa}} \mathbf k 
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2306.15595)

citation: 
```bibtex
@article{chen2023extending,
  title={Extending context window of large language models via positional interpolation},
  author={Chen, Shouyuan and Wong, Sherman and Chen, Liangjian and Tian, Yuandong},
  journal={arXiv preprint arXiv:2306.15595},
  year={2023}
}
```


#### A length-extrapolatable transformer (LEX / XPOS) [`READ`]

$$
  \begin{align}
    & P_{i,j} := \langle\widetilde{\mathbf q_i}, \widetilde{\mathbf k_j} \rangle = \gamma^{i-j}(\mathbf q^{\mathrm{T}} R_{j-i} \mathbf k),\\
    &\quad where\quad \widetilde{\mathbf q_i} := \gamma^i(R_i \mathbf q), \quad \widetilde{\mathbf k_j} := \gamma^{-j} (R_j \mathbf k),\quad i \ge j
  \end{align}
$$

paper link: [here](https://arxiv.org/pdf/2212.10554)

citation: 
```bibtex
@article{sun2022length,
  title={A length-extrapolatable transformer},
  author={Sun, Yutao and Dong, Li and Patra, Barun and Ma, Shuming and Huang, Shaohan and Benhaim, Alon and Chaudhary, Vishrav and Song, Xia and Wei, Furu},
  journal={arXiv preprint arXiv:2212.10554},
  year={2022}
}
```

#### Permuteformer: Efficient relative position encoding for long sequences [`READ`]

paper link: [here](https://arxiv.org/pdf/2109.02377)

citation: 
```bibtex
@article{chen2021permuteformer,
  title={Permuteformer: Efficient relative position encoding for long sequences},
  author={Chen, Peng},
  journal={arXiv preprint arXiv:2109.02377},
  year={2021}
}
```


### Truncation Strategies


#### Giraffe: Adventures in expanding context lengths in llms (Basis Truncation) [`READ`]

$$
\begin{align}
    &\widetilde\theta^{i} := \begin{cases}
    \theta^{i}, & \theta^{i} \ge b\\
    \rho, & \theta^{i} \in (a,b) \\
    0, & \theta^{i} \le a
    \end{cases}
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2308.10882)

citation: 
```bibtex
@article{pal2023giraffe,
  title={Giraffe: Adventures in expanding context lengths in llms},
  author={Pal, Arka and Karkhanis, Deep and Roberts, Manley and Dooley, Samuel and Sundararajan, Arvind and Naidu, Siddartha},
  journal={arXiv preprint arXiv:2308.10882},
  year={2023}
}
```


#### Transformer Upgrade Roadmap: 12. ReRoPE for Infinite Extrapolation? [`READ`]

$$
\begin{align}
    &\quad \widetilde P_{i,j} := \langle R_{\alpha(i,j,w,\kappa)} \mathbf q,\space \mathbf k\rangle, \\
    &\quad where\quad \alpha(i,j,w,\kappa) := \begin{cases}
      \min\lbrace  i-j, w+\frac{i-j-w}{\kappa}\rbrace, & 0<\kappa<\infty\space  (\mathrm{Leaky\space  ReRoPE})\\
      \min\lbrace i-j,w\rbrace & \kappa \rightarrow \infty\space  (\mathrm{ReRoPE})
    \end{cases}
\end{align}
$$

blog link: [here](https://spaces.ac.cn/archives/9708)

citation:
```bibtex
@misc{transformer-upgrade-12,
    author = "Su, Jianlin",
    title = "Transformer Upgrade Roadmap: 12. ReRoPE for Infinite Extrapolation?",
    year = "2023",
    month = "Aug",
    howpublished = "\url{https://spaces.ac.cn/archives/9708}"
}
```




### Rearrangement Strategies


#### A Frustratingly Easy Improvement for Position Embeddings via Random Padding [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.04859)

citation: 
```bibtex
@article{tao2023frustratingly,
  title={A Frustratingly Easy Improvement for Position Embeddings via Random Padding},
  author={Tao, Mingxu and Feng, Yansong and Zhao, Dongyan},
  journal={arXiv preprint arXiv:2305.04859},
  year={2023}
}
```
    
#### Randomized Positional Encodings Boost Length Generalization of Transformers [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.16843)

citation: 
```bibtex
@article{ruoss2023randomized,
  title={Randomized Positional Encodings Boost Length Generalization of Transformers},
  author={Ruoss, Anian and Del{\'e}tang, Gr{\'e}goire and Genewein, Tim and Grau-Moya, Jordi and Csord{\'a}s, R{\'o}bert and Bennani, Mehdi and Legg, Shane and Veness, Joel},
  journal={arXiv preprint arXiv:2305.16843},
  year={2023}
}
```

#### SHAPE: Shifted absolute position embedding for transformers [`READ`]

paper link: [here](https://arxiv.org/pdf/2109.05644)

citation: 
```bibtex
@article{kiyono2021shape,
  title={SHAPE: Shifted absolute position embedding for transformers},
  author={Kiyono, Shun and Kobayashi, Sosuke and Suzuki, Jun and Inui, Kentaro},
  journal={arXiv preprint arXiv:2109.05644},
  year={2021}
}
```
