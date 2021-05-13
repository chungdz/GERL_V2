# GERL

## Preprocess

构建dict：
- news_id => news index
- user_id => user index
- word => word index
- news_index => title seq

构建两个1-hop关系表
- news index => user index list, 可以提前做好sample
- user index => news index list, 可以提前做好sample

构建两个 2-hop关系表
- news index => 2-hop news index list, 可以提前做好sample
- user index => 2-hop user index list, 可以提前做好sample

## 构建数据集
每个训练数据的格式为: 
```json
{
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: [7, 8, 9, 10, 11],
    neighbor_news: [
        [27, 28, 29],
        [30, 31, 32],
        [33, 34, 35],
        [36, 37, 38],
        [39, 40, 41]
    ]
}
```

每个测试数据的格式为: 
```json
{
    imp_id: 000,
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: 7,
    neighbor_news: [27, 28, 29],
    y: 1
}
```

## How to run

1. export GERL=/path/to/GERL
2. 按照下面的结构把 data 下的子目录建好
```shell
data
├── glove.840B.300d.txt
├── L
│   ├── dev
│   │   ├── behaviors.tsv
│   │   └── news.tsv
│   ├── news_title.npy
│   ├── test
│   │   ├── behaviors.tsv
│   │   ├── news.tsv
│   │   └── __placeholder__
│   └── train
│       ├── behaviors.tsv
│       ├── news.tsv
│       ├── samples.tsv
│       └── samples.zip
└── vocabs
    ├── newsid_vocab.bin
    ├── test-news_one_hops.txt
    ├── test-news_two_hops.txt
    ├── test-user_one_hops.txt
    ├── test-user_two_hops.txt
    ├── train-news_one_hops.txt
    ├── train-news_two_hops.txt
    ├── train-user_one_hops.txt
    ├── train-user_two_hops.txt
    ├── userid_vocab.bin
    ├── word_embeddings.bin.npy
    └── word_vocab.bin
```
3. 跑scripts
- build_train_samples.py
- build_vocabs.py
- build_neighbors.py
- build_train_examples.py
- build_eval_examples.py
- build_test_examples.py
- split_data.py 
- split_data.py -i L/examples/test_examples.tsv -o L/examples/test_examples
- CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py 必须4张卡
- CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py training.validate_epoch=6  必须4张卡
 