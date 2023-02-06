# Sarcasm Detection using Bidirectional Encoder Representations from Transformers and Graph Convolutional Network

This repository contains the code used in our paper:

<h3><b><a href = "https://www.sciencedirect.com/science/article/pii/S1877050922024991">Sarcasm Detection using Bidirectional Encoder Representations from Transformers and Graph Convolutional Network</a></b></h3>
<b>International Conference on Machine Learning and Data Mining (ICMLDE), 2022</b>

Anuraj Mohan, Abhilash M Nair, Bhadra Jayakumar, Sanjay Muraleedharan</br>
<i>Department of Computer Science and Engineering, NSS College of Engineering, Palakkad, Kerala, India</i>

---

## Requirements
- numpy
- spacy
- torch
- scikit-learn
- matplotlib
- pytorch-pretrained-bert

---

## Usage
- Install the dependencies
```bash
pip3 install -r requirements.txt
```

- Download spaCy language model
```bash
python3 -m spacy download en
```

- Generate adjacency and affective dependency graphs
```bash
python3 graph.py
```

- Train the model. Optional arguments can be found in `train.py`
```bash
python3 train.py
```
---

# CREDITS
- The affective knowledge used in this work is from [SenticNet](https://sentic.net/).
- The code in this repository partially relies on [ADGCN](https://github.com/BinLiang-NLP/ADGCN-Sarcasm) and [SenticGCN](https://github.com/BinLiang-NLP/Sentic-GCN).

---

# LICENCE
This repository is licensed under MIT License. See [LICENSE](https://github.com/abhilashmnair/Sarcasm-Detection-with-BERT-and-GCN/blob/main/LICENSE) for full licensing text.
