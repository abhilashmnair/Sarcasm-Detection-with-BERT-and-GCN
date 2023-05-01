import numpy as np
import spacy
import pickle
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm')

def load_sentic_word():
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet

def sentic_dependency_adj_matrix(text, senticNet):
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1
        else:
            sentic = 0

        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic
    return matrix

def process_sdat(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph_sdat', 'wb')
    print('Generating sentic dependancy graph...')
    for i in tqdm((range(0, len(lines), 2))):
        text = lines[i].lower().strip()
        adj_matrix = sentic_dependency_adj_matrix(text, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('Done!')
    fout.close()
