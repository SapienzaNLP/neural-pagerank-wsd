# Integrating Personalized PageRank into Neural Word Sense Disambiguation


This repo hosts the code necessary to reproduce the results of our EMNLP 2021 paper, *Integrating Personalized PageRank into Neural Word Sense Disambiguation*, by Ahmed ElSheikh, Michele Bevilacqua and Roberto Navigli, which you can read on [ACL Anthology]().

This repository heavily relies on Simone's [code](https://github.com/sapienzaNLP/multilabel-wsd).

## How to Cite
```
@inproceedings{elsheikh-etal-2021-integrating,
    title = "Integrating Personalized PageRank into Neural Word Sense Disambiguation",
    author = "ElSheikh, Ahmed and Bevilacqua, Michele  and Navigli, Roberto",
    year = "2021",
    address = "Online",
    publisher = "Emperical Method for Natural Language Processing",
}
```

## Installation

- make sure to have miniconda installed. if not, [install it](https://docs.conda.io/en/latest/miniconda.html)
- It is recommended to create a fresh `conda` env to use the repo
  
  ```bash
  - conda create -n wsd_ppr python=3.6.9 pip
  - conda activate wsd_ppr
  - git clone git@github.com:mbevila/neural-pagerank-wsd.git
  - pip install -r requirements.txt
  - pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  - pip install torch-sparse torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
  ```

- if it needs `APEX` to be installed

  ```bash
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

---

### Datasets

- [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval): contains the SemCor training corpus, along with the evaluation datasets from Senseval and SemEval.

### Sense Embeddings

Pre-preprocessed SensEmBERT+LMMS <!--  OR ARES  --> embeddings is needed to train your model:

- [SensEmBERT + LMMS Embeddings](https://drive.google.com/file/d/11v4FUMyHdpFBrkRJt8cGyy6xkM9a_Emp/view?usp=sharing)

<!-- - [ARES Embeddings](https://drive.google.com/file/d/11riHw5BLay9ORAbLC-2Cl6dYXnd9ZJnx/view?usp=sharing) -->

---

## Train

Run `train.sh`.

## Evaluate

Check out `predict_eval_script.sh`. 

---

## License
This project is released under the CC-BY-NC 4.0 license (see `LICENSE.txt`). If you use this project, please put a link to this repo.

## Acknowledgements
The authors gratefully acknowledge the support of the <a href="http://mousse-project.org">ERC Consolidator Grant MOUSSE</a> No. 726487 under the European
Union's Horizon 2020 research and innovation programme.
