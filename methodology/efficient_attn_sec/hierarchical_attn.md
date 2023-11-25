# Hierarchical Attention
*Here're some resources about Hierarchical Attention*


### Intro

To further think of either the global token techniques, or the inter-block attention mentioned in [Local Attention](./local_attn.md/), we can regard them as introducing some **hierarchical features** to self-attention to compensate with more
global information from the higher-level attention while keeping the low computation cost from the low-level local
attention at the same time. From this view, many works have explored various hierarchical mechanisms that
introduce a structured hierarchy into self-attention, leveraging both higher-level global information and lower-level
local attention for multi-scaled contextual receptive fields.


### Table of Contents
* [Intro](#intro)
* [Two-Level Hierarchy](#two-level-hierarchy)
* [Multi-Level Hierarchy](#multi-level-hierarchy)

### Two-Level Hierarchy


#### Hegel: Hypergraph transformer for long document summarization [`READ`]

paper link: [here](https://arxiv.org/pdf/2210.04126)

citation: 
```bibtex
@article{zhang2022hegel,
  title={Hegel: Hypergraph transformer for long document summarization},
  author={Zhang, Haopeng and Liu, Xiao and Zhang, Jiawei},
  journal={arXiv preprint arXiv:2210.04126},
  year={2022}
}
```


#### Hierarchical learning for generation with long source sequences [`READ`]

paper link: [here](https://arxiv.org/pdf/2104.07545)

citation: 
```bibtex
@article{rohde2021hierarchical,
  title={Hierarchical learning for generation with long source sequences},
  author={Rohde, Tobias and Wu, Xiaoxia and Liu, Yinhan},
  journal={arXiv preprint arXiv:2104.07545},
  year={2021}
}
```

#### Lite transformer with long-short range attention [`READ`]

paper link: [here](https://arxiv.org/pdf/2004.11886)

citation: 
```bibtex
@article{wu2020lite,
  title={Lite transformer with long-short range attention},
  author={Wu, Zhanghao and Liu, Zhijian and Lin, Ji and Lin, Yujun and Han, Song},
  journal={arXiv preprint arXiv:2004.11886},
  year={2020}
}
```


#### Hierarchical transformers for long document classification (HAN) [`READ`]

paper link: [here](https://arxiv.org/pdf/1910.10781)

citation: 
```bibtex
@inproceedings{pappagari2019hierarchical,
  title={Hierarchical transformers for long document classification},
  author={Pappagari, Raghavendra and Zelasko, Piotr and Villalba, Jes{\'u}s and Carmiel, Yishay and Dehak, Najim},
  booktitle={2019 IEEE automatic speech recognition and understanding workshop (ASRU)},
  pages={838--844},
  year={2019},
  organization={IEEE}
}
```

#### HIBERT: Document level pre-training of hierarchical bidirectional transformers for document summarization [`READ`]

paper link: [here](https://arxiv.org/pdf/1905.06566)

citation: 
```bibtex
@article{zhang2019hibert,
  title={HIBERT: Document level pre-training of hierarchical bidirectional transformers for document summarization},
  author={Zhang, Xingxing and Wei, Furu and Zhou, Ming},
  journal={arXiv preprint arXiv:1905.06566},
  year={2019}
}
```

#### Document-level neural machine translation with hierarchical attention networks [`READ`]

paper link: [here](https://arxiv.org/pdf/1809.01576)

citation: 
```bibtex
@article{miculicich2018document,
  title={Document-level neural machine translation with hierarchical attention networks},
  author={Miculicich, Lesly and Ram, Dhananjay and Pappas, Nikolaos and Henderson, James},
  journal={arXiv preprint arXiv:1809.01576},
  year={2018}
}
```

#### A discourse-aware attention model for abstractive summarization of long documents [`READ`]

paper link: [here](https://arxiv.org/pdf/1804.05685)

citation: 
```bibtex
@article{cohan2018discourse,
  title={A discourse-aware attention model for abstractive summarization of long documents},
  author={Cohan, Arman and Dernoncourt, Franck and Kim, Doo Soon and Bui, Trung and Kim, Seokhwan and Chang, Walter and Goharian, Nazli},
  journal={arXiv preprint arXiv:1804.05685},
  year={2018}
}
```
    



### Multi-Level Hierarchy

#### Combiner: Full attention transformer with sparse computation cost [`READ`]

paper link: [here](https://proceedings.neurips.cc/paper/2021/file/bd4a6d0563e0604510989eb8f9ff71f5-Paper.pdf)

citation: 
```bibtex
@article{ren2021combiner,
  title={Combiner: Full attention transformer with sparse computation cost},
  author={Ren, Hongyu and Dai, Hanjun and Dai, Zihang and Yang, Mengjiao and Leskovec, Jure and Schuurmans, Dale and Dai, Bo},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={22470--22482},
  year={2021}
}
```


#### H-transformer-1d: Fast one-dimensional hierarchical attention for sequences [`READ`]

paper link: [here](https://arxiv.org/pdf/2107.11906)

citation: 
```bibtex
@article{zhu2021h,
  title={H-transformer-1d: Fast one-dimensional hierarchical attention for sequences},
  author={Zhu, Zhenhai and Soricut, Radu},
  journal={arXiv preprint arXiv:2107.11906},
  year={2021}
}
```

#### Bp-transformer: Modelling long-range context via binary partitioning (BPT) [`READ`]

paper link: [here](https://arxiv.org/pdf/1911.04070)

citation: 
```bibtex
@article{ye2019bp,
  title={Bp-transformer: Modelling long-range context via binary partitioning},
  author={Ye, Zihao and Guo, Qipeng and Gan, Quan and Qiu, Xipeng and Zhang, Zheng},
  journal={arXiv preprint arXiv:1911.04070},
  year={2019}
}
```


#### Adaptive attention span in transformers [`READ`]

paper link: [here](https://arxiv.org/pdf/1905.07799)

citation: 
```bibtex
@article{sukhbaatar2019adaptive,
  title={Adaptive attention span in transformers},
  author={Sukhbaatar, Sainbayar and Grave, Edouard and Bojanowski, Piotr and Joulin, Armand},
  journal={arXiv preprint arXiv:1905.07799},
  year={2019}
}
```
    
    
