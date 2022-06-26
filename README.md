# Sarcasm Detection using BERT and GCN

This repository contains the code used in our paper:

**Sarcasm Detection using Bidirectional Encorder Representations for Transformers and Graph Convolutional Network**
Abhilash M Nair, Bhadra Jayakumar, Sanjay Muraleedharan

## Requirements
- numpy
- spacy
- torch
- scikit-learn
- matplotlib
- pytorch-pretrained-bert

## Usage
- Install the dependencies
`pip3 install -r requirements.txt`

- Download spaCy language model
`python3 -m spacy download en`

- Generate adjacency and affective dependency graphs
`python3 graph.py`

- Train the model. Optional arguments can be found in `train.py`
`python3 train.py`