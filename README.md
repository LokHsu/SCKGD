# SCKGD

This repository contains source codes and datasets for the paper:

- Spurious Correlation Knowledge Graph Disentanglement for Multi-Behavior Recommendation

## Usage
### Train & Test

First, run the kg_generate.py file in data/xxx to construct the spurious correlation knowledge graph.

- Training SCKGD on IJCAI15:
```shell
python main.py --dataset=IJCAI_15
```

- Training SCKGD on Tmall:
```shell
python main.py --dataset=Tmall
```

- Training SCKGD on Retail:
```shell
python main.py --dataset=retailrocket
```

- Testing SCKGD using a saved model file:
```shell
ipython evaluation.ipynb
```
