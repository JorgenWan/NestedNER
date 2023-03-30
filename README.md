# Nested Named Entity Recognition with Span-level Graphs


## Requirements
- cuda 11.5
- python 3.9
- pytorch 1.11
- dgl 0.9.1
- transformers 4.24.0


## Usage
1. Create virtual environment. For DGL, according to https://www.dgl.ai/pages/start.html, we choose python 3.9 and pytorch 1.11 as our cuda version is 11.5.
   1. Create virtual environment with conda: `conda create --yes --quiet --name ner python=3.9`.
   2. Install pytorch: `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch`
   3. Install DGL，`conda install -c dglteam dgl-cuda11.5`. We get DGL=0.9.1post1.
   4. Install transformers， `conda install transformers`. We get transformers=4.24.0.
2. Prepare data. We use three datasets, ACE 2004, ACE 2005, and GENIA. Put these datasets into a directory `dataset` with json mode.
3. Preprocess data. Before training and testing, we need to build span-level sub-graph for each raw span offline.
   1. Run `python build_nodes.py`. This step will number the entity nodes in the training set and creates `Graph/nodes.json`.
   2. Run `python build_edges.py`. This step will calculate the edge weights between entity nodes (in file `Graph/nodes.json`) and create `edges.json`.
   3. Run `python build_candidates.py`. This step will find the entity candidates for all spans of the train/valid/test set and create candidate files in the directory `Graph/candidates`.
4. Run `cd ./shells` and `bash train_ace04.sh`.

You can configure the architecture in `models/architecture.py` or by command line explicitly. The arguments have priority:
```
Command Line > Architecture Config > Default Value
```

## Authors
```bibtex
@inproceedings{wan2022nested,
  title={Nested Named Entity Recognition with Span-level Graphs},
  author={Wan, Juncheng and Ru, Dongyu and Zhang, Weinan and Yu, Yong},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={892--903},
  year={2022}
}
```

- Juncheng Wan, jorgenwan@gmail.com, 2023.1.24

