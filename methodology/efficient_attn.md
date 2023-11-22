# Efficient Attention

This section shares the literatures which propose the methods to make original scaled dot-product self-attention mechanism with $O(n^2)$ time/space comlexity more efficient to train on longer texts


## Taxonomy

### Local Attention

#### Efficient streaming language models with attention sinks [`READ`]

paper link: [here](https://arxiv.org/pdf/2309.17453)

citation: 
```bibtex
@article{xiao2023efficient,
  title={Efficient streaming language models with attention sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  journal={arXiv preprint arXiv:2309.17453},
  year={2023}
}
```

#### Longlora: Efficient fine-tuning of long-context large language models [`READ`]

paper link: [here](https://arxiv.org/pdf/2309.12307.pdf?trk=public_post_comment-text)

citation: 
```bibtex
@article{chen2023longlora,
  title={Longlora: Efficient fine-tuning of long-context large language models},
  author={Chen, Yukang and Qian, Shengju and Tang, Haotian and Lai, Xin and Liu, Zhijian and Han, Song and Jia, Jiaya},
  journal={arXiv preprint arXiv:2309.12307},
  year={2023}
}
```
    


#### Landmark Attention: Random-Access Infinite Context Length for Transformers [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.16300)

citation: 
```bibtex
@article{mohtashami2023landmark,
  title={Landmark Attention: Random-Access Infinite Context Length for Transformers},
  author={Mohtashami, Amirkeivan and Jaggi, Martin},
  journal={arXiv preprint arXiv:2305.16300},
  year={2023}
}
```
    


#### Efficient long sequence modeling via state space augmented transformer (SPADE) [`READ`]

paper link: [here](https://arxiv.org/pdf/2212.08136)

citation: 
```bibtex
@article{zuo2022efficient,
  title={Efficient long sequence modeling via state space augmented transformer},
  author={Zuo, Simiao and Liu, Xiaodong and Jiao, Jian and Charles, Denis and Manavoglu, Eren and Zhao, Tuo and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2212.08136},
  year={2022}
}
```
    

#### LongT5: Efficient text-to-text transformer for long sequences [`READ`]

paper link: [here](https://arxiv.org/pdf/2112.07916)

citation: 
```bibtex
@article{guo2021longt5,
  title={LongT5: Efficient text-to-text transformer for long sequences},
  author={Guo, Mandy and Ainslie, Joshua and Uthus, David and Ontanon, Santiago and Ni, Jianmo and Sung, Yun-Hsuan and Yang, Yinfei},
  journal={arXiv preprint arXiv:2112.07916},
  year={2021}
}
```
    

#### Longformer: The long-document transformer [`READ`]

paper link: [here](https://arxiv.org/pdf/2004.05150.pdf?forcedefault=true)

citation: 
```bibtex
@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}
```

#### ETC: Encoding long and structured inputs in transformers [`READ`]

paper link: [here](https://arxiv.org/pdf/2004.08483)

citation: 
```bibtex
@article{ainslie2020etc,
  title={ETC: Encoding long and structured inputs in transformers},
  author={Ainslie, Joshua and Ontanon, Santiago and Alberti, Chris and Cvicek, Vaclav and Fisher, Zachary and Pham, Philip and Ravula, Anirudh and Sanghai, Sumit and Wang, Qifan and Yang, Li},
  journal={arXiv preprint arXiv:2004.08483},
  year={2020}
}
```
    
    

#### Big bird: Transformers for longer sequences [`READ`]

paper link: [here](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)

citation: 
```bibtex
@article{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={17283--17297},
  year={2020}
}
```


#### Reformer: The efficient transformer [`READ`]

paper link: [here](https://arxiv.org/pdf/2001.04451)

citation: 
```bibtex
@article{kitaev2020reformer,
  title={Reformer: The efficient transformer},
  author={Kitaev, Nikita and Kaiser, {\L}ukasz and Levskaya, Anselm},
  journal={arXiv preprint arXiv:2001.04451},
  year={2020}
}
```

#### Sparse sinkhorn attention [`READ`]

paper link: [here](http://proceedings.mlr.press/v119/tay20a/tay20a.pdf)

citation: 
```bibtex
@inproceedings{tay2020sparse,
  title={Sparse sinkhorn attention},
  author={Tay, Yi and Bahri, Dara and Yang, Liu and Metzler, Donald and Juan, Da-Cheng},
  booktitle={International Conference on Machine Learning},
  pages={9438--9447},
  year={2020},
  organization={PMLR}
}
```
    

#### Blockwise self-attention for long document understanding [`READ`]

paper link: [here](https://arxiv.org/pdf/1911.02972)

citation: 
```bibtex
@article{qiu2019blockwise,
  title={Blockwise self-attention for long document understanding},
  author={Qiu, Jiezhong and Ma, Hao and Levy, Omer and Yih, Scott Wen-tau and Wang, Sinong and Tang, Jie},
  journal={arXiv preprint arXiv:1911.02972},
  year={2019}
}
```


### Hierarchical Attention

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
    
#### Hierarchical transformers for long document classification [`READ`]

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
    
    


### Sparse Attention


#### Longnet: Scaling transformers to 1,000,000,000 tokens [`READ`]

paper link: [here](https://arxiv.org/pdf/2307.02486.pdf?trk=public_post_comment-text)

citation: 
```bibtex
@article{ding2023longnet,
  title={Longnet: Scaling transformers to 1,000,000,000 tokens},
  author={Ding, Jiayu and Ma, Shuming and Dong, Li and Zhang, Xingxing and Huang, Shaohan and Wang, Wenhui and Wei, Furu},
  journal={arXiv preprint arXiv:2307.02486},
  year={2023}
}
```
    


#### Sparse is enough in scaling transformers [`READ`]

paper link: [here](https://proceedings.neurips.cc/paper/2021/file/51f15efdd170e6043fa02a74882f0470-Paper.pdf)

citation: 
```bibtex
@article{jaszczur2021sparse,
  title={Sparse is enough in scaling transformers},
  author={Jaszczur, Sebastian and Chowdhery, Aakanksha and Mohiuddin, Afroz and Kaiser, Lukasz and Gajewski, Wojciech and Michalewski, Henryk and Kanerva, Jonni},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={9895--9907},
  year={2021}
}
```

#### Efficient content-based sparse attention with routing transformers [`READ`]

paper link: [here](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00353/1923932/tacl_a_00353.pdf)

citation: 
```bibtex
@article{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={Transactions of the Association for Computational Linguistics},
  volume={9},
  pages={53--68},
  year={2021},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}
```
    


#### Generating long sequences with sparse transformers [`READ`]

paper link: [here](https://arxiv.org/pdf/1904.10509)

citation: 
```bibtex
@article{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1904.10509},
  year={2019}
}
```
    
### Approximated Attention


#### Luna: Linear unified nested attention [`READ`]

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
    

#### Linformer: Self-attention with linear complexity [`READ`]

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



### IO-Aware Attention


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
    


#### Faster Causal Attention Over Large Sequences Through Sparse Flash Attention [`READ`]

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



## Survey


#### Efficient Transformers: A Survey [`READ`]

paper link: [here](https://arxiv.org/pdf/2009.06732.pdf)

citation: 
```bibtex
@misc{tay2022efficient,
      title={Efficient Transformers: A Survey}, 
      author={Yi Tay and Mostafa Dehghani and Dara Bahri and Donald Metzler},
      year={2022},
      eprint={2009.06732},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Efficient attentions for long document summarization [`READ`]

paper link: [here](https://arxiv.org/pdf/2104.02112)

citation: 
```bibtex
@article{huang2021efficient,
  title={Efficient attentions for long document summarization},
  author={Huang, Luyang and Cao, Shuyang and Parulian, Nikolaus and Ji, Heng and Wang, Lu},
  journal={arXiv preprint arXiv:2104.02112},
  year={2021}
}
```
    



