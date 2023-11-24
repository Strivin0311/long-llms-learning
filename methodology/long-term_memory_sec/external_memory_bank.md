# External MemoryBank
*Here's some resources about External MemoryBank*


### Intro

The series of [Internal MemoryCache](./internal_memory_cache.md) mechanisms enhance the vanilla stateless Transformer model with sequential recurrence by prepending extra hidden states of previous inputs from an internal memory cache. However, these mechanisms have certain drawbacks. Firstly, a slight change in the memory mechanism may necessitate retraining the model from scratch, not fully utilizing pre-trained LLMs that already possess a good representation of dependency across the context window, albeit not long enough. Secondly, as highlighted in [Memorizing Transformer](./internal_memory_cache.md#memorizing-transformers-read), they often encounter the problem of memory staleness, where older hidden states in memory cache may exhibit *distributional shifts* from the latest ones during training, limiting the effectiveness of memory augmentation. 

As a solution, another retrieval-augmented mechanisms decouples the model itself from its long-term memory storage. They leverage the part before the language head as a well-performing contextual information encoder to store long sequences as an **external memory bank** in the form of embeddings. And during queries, the model **retrieves** information from this memory bank based on certain **criteria** and concatenates it to comprise in-context working memory in real-time.



### Table of Contents

* [Cosine-Based Retrieval Criteria](#cosine-based-retrieval-criteria)
* [Heuristic Retrieval Criteria](#heuristic-retrieval-criteria)
* [Leanable Retrieval Criteria](#leanable-retrieval-criteria)


### Cosine-Based Retrieval Criteria


#### Creating large language model applications utilizing langchain: A primer on developing llm apps fast [`READ`]

paper link: [here](https://www.researchgate.net/profile/Oguzhan-Topsakal/publication/372669736_Creating_Large_Language_Model_Applications_Utilizing_LangChain_A_Primer_on_Developing_LLM_Apps_Fast/links/64d114a840a524707ba4a419/Creating-Large-Language-Model-Applications-Utilizing-LangChain-A-Primer-on-Developing-LLM-Apps-Fast.pdf)

citation: 
```bibtex
@inproceedings{topsakal2023creating,
  title={Creating large language model applications utilizing langchain: A primer on developing llm apps fast},
  author={Topsakal, Oguzhan and Akinci, Tahir Cetin},
  booktitle={Proceedings of the International Conference on Applied Engineering and Natural Sciences, Konya, Turkey},
  pages={10--12},
  year={2023}
}
```


####  Langchain

github link: [here](https://github.com/langchain-ai/langchain)

citation:
```bibtex
@misc{langchain2022,
    author = "Chase, Harrison",
    month = "10",
    title = "LangChain",
    howpublished = "\url{https://github.com/langchain-ai/langchain}",
    year = "2022"
}
```


### Heuristic Retrieval Criteria


#### Unlimiformer: Long-range transformers with unlimited length input [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.01625.pdf)

citation: 
```bibtex
@article{bertsch2023unlimiformer,
  title={Unlimiformer: Long-range transformers with unlimited length input},
  author={Bertsch, Amanda and Alon, Uri and Neubig, Graham and Gormley, Matthew R},
  journal={arXiv preprint arXiv:2305.01625},
  year={2023}
}
```
    

#### RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.13304)

citation: 
```bibtex
@article{zhou2023recurrentgpt,
  title={RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text},
  author={Zhou, Wangchunshu and Jiang, Yuchen Eleanor and Cui, Peng and Wang, Tiannan and Xiao, Zhenxin and Hou, Yifan and Cotterell, Ryan and Sachan, Mrinmaya},
  journal={arXiv preprint arXiv:2305.13304},
  year={2023}
}
```

#### RecallM: An Architecture for Temporal Context Understanding and Question Answering [`READ`]

paper link: [here](https://arxiv.org/pdf/2307.02738)

citation: 
```bibtex
@article{kynoch2023recallm,
  title={RecallM: An Architecture for Temporal Context Understanding and Question Answering},
  author={Kynoch, Brandon and Latapie, Hugo},
  journal={arXiv preprint arXiv:2307.02738},
  year={2023}
}
```


#### RET-LLM: Towards a General Read-Write Memory for Large Language Models [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.14322)

citation: 
```bibtex
@article{modarressi2023ret,
  title={RET-LLM: Towards a General Read-Write Memory for Large Language Models},
  author={Modarressi, Ali and Imani, Ayyoob and Fayyaz, Mohsen and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2305.14322},
  year={2023}
}
```


#### MemoryBank: Enhancing Large Language Models with Long-Term Memory (siliconFriend) [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.10250)

citation: 
```bibtex
@article{zhong2023memorybank,
  title={MemoryBank: Enhancing Large Language Models with Long-Term Memory},
  author={Zhong, Wanjun and Guo, Lianghong and Gao, Qiqi and Wang, Yanlin},
  journal={arXiv preprint arXiv:2305.10250},
  year={2023}
}
```


#### Improving language models by retrieving from trillions of tokens (RETRO) [`READ`]

paper link: [here](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf)

citation: 
```bibtex
@inproceedings{borgeaud2022improving,
  title={Improving language models by retrieving from trillions of tokens},
  author={Borgeaud, Sebastian and Mensch, Arthur and Hoffmann, Jordan and Cai, Trevor and Rutherford, Eliza and Millican, Katie and Van Den Driessche, George Bm and Lespiau, Jean-Baptiste and Damoc, Bogdan and Clark, Aidan and others},
  booktitle={International conference on machine learning},
  pages={2206--2240},
  year={2022},
  organization={PMLR}
}
```



### Leanable Retrieval Criteria 
    

#### Augmenting Language Models with Long-Term Memory (LongMem) [`READ`]

paper link: [here](https://arxiv.org/pdf/2306.07174)

citation: 
```bibtex
@article{wang2023augmenting,
  title={Augmenting Language Models with Long-Term Memory},
  author={Wang, Weizhi and Dong, Li and Cheng, Hao and Liu, Xiaodong and Yan, Xifeng and Gao, Jianfeng and Wei, Furu},
  journal={arXiv preprint arXiv:2306.07174},
  year={2023}
}
```

#### REALM: Retrieval augmented language model pre-training [`READ`]

paper link: [here](http://proceedings.mlr.press/v119/guu20a/guu20a.pdf)

citation: 
```bibtex
@inproceedings{guu2020retrieval,
  title={Retrieval augmented language model pre-training},
  author={Guu, Kelvin and Lee, Kenton and Tung, Zora and Pasupat, Panupong and Chang, Mingwei},
  booktitle={International conference on machine learning},
  pages={3929--3938},
  year={2020},
  organization={PMLR}
}
```
    