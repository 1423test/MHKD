# MHKD
This repo is for source code of paper "Meta-Path and  Hypergraph Fused Distillation Framework for Heterogeneous Information Networks Embedding". 

## Environment Settings
> python==3.8.13 \
> scipy==1.7.3 \
> torch==1.11.0 \
> numpy==1.21.2 \
> scikit_learn==1.0.2 \
> dgl ==0.8.2

GPU: GeForce RTX 3090 Ti

## Usage
1. go into `cd ./teacher`, and then you can follow the corresponding “readme.md” to train teacher model. And the publicly source codes of teacher model can be available at the following urls:

[HAN] https://github.com/dmlc/dgl/tree/master/examples/pytorch/han

[MAGNN] https://github.com/cynricfu/MAGNN

[HGT] https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt

[Heco] https://github.com/BUPT-GAMMA/HeCo

2. use `python hy_dblp.py`  to run student model evaluated in DBLP dataset.
Here, "acm" can be replaced by "dblp", "aminer", "freebase"  or "imdb".

## Some tips in parameters
1. We provide our hyperparameters setting as reported in our Appendix. 
2. The hypermeters *“Lg”* and *“Lg”* (existed in ./Utils/stuparams.py) are suggested to carefully select to ensure the performance of distillation framework. This is very important to final results. 

## Results

There are some results on Heco teacher model, with DBLP dataset and student model in Mi-F1 metric. More results can be seen in our paper.


| Split | Heco(teacher) | Student | Improvement |
|-------|---------------|---------|-------------|
| 20    | 0.9103        | 0.9274  | 1.71%       |
| 40    | 0.9019        | 0.9246  | 2.27%       |
| 60    | 0.9041        | 0.9260  | 2.19%       |



