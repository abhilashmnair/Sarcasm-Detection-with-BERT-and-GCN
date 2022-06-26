import numpy as np
import pickle
from tqdm import tqdm


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


def dependency_adj_matrix(text, senticNet):
    word_list = text.split()
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for i in range(seq_len):
        word = word_list[i]
        if word in senticNet:
            sentic = float(senticNet[word]) + 1.0
        else:
            sentic = 0

        for j in range(seq_len):
            matrix[i][j] += sentic
            matrix[j][i] += sentic
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1

    return matrix

def process_sentic_graph(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.sentic', 'wb')
    print('Generating sentic graph...')
    for i in tqdm(range(0, len(lines), 2)):
        text = lines[i].lower().strip()
        adj_matrix = dependency_adj_matrix(text, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('Done!')
    fout.close() 
