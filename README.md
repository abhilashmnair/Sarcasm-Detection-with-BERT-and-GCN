# Sarcasm Detection using BERT and GCN

This repository contains the code used in our paper:

**Sarcasm Detection using Bidirectional Encorder Representations for Transformers and Graph Convolutional Network** submitted at ICMLDE '22, UPES, Dehradun, India

Abhilash M Nair, Bhadra Jayakumar, Sanjay Muraleedharan, Dr. Anuraj Mohan

## Requirements
- numpy
- spacy
- torch
- scikit-learn
- matplotlib
- pytorch-pretrained-bert

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

# LICENCE
This repository is licensed under MIT License. See [LICENSE](https://github.com/abhilashmnair/Sarcasm-Detection-with-BERT-and-GCN/blob/main/LICENSE) for full licensing text.