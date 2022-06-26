import numpy as np
import spacy
import pickle
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def gen_adj_matrix(text):
    document = nlp(text)
    seq_len = len(text.split())
    
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    return matrix

def process_adj_matrix(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    print('Generating adjacency matrix...')
    for i in tqdm(range(0, len(lines), 2)):
        text = lines[i].lower().strip()
        adj_matrix = gen_adj_matrix(text)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('Done!')        
    fout.close()