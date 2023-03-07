# AST-clone-detection
基于AST和注意力机制的代码克隆检测


## Dataset

主要使用的数据集为[OJClone](https://arxiv.org/pdf/1409.5718.pdf)。

### 下载与预处理数据集

1. 从 [google drive](https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/view?usp=sharing) 下载数据集

```shell
cd dataset/OJClone
pip install gdown
gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
tar -xvf programs.tar.gz
```

2. 处理数据

```shell
python preprocess.py
```

会得到三个文件`dataset/OJClone/train.jsonl`, `dataset/OJClone/test.jsonl`, `dataset/OJClone/valid.jsonl` 
