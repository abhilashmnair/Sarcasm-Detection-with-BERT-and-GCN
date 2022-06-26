import pickle
import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class DatasetReader(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        fin = open(fname+'.graph_sdat', 'rb')
        idx2graph_sdat = pickle.load(fin)
        fin.close()

        #graph_id = 0
        all_data = []
        for i in range(0, len(lines), 2):
            graph_id = i
            context = lines[i].lower().strip()
            label = int(lines[i + 1].strip())

            text_indices = tokenizer.text_to_sequence(context)
            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + context + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)


            dependency_graph = idx2graph[graph_id]
            sdat_graph = idx2graph_sdat[graph_id]

            dependency_graph = np.pad(idx2graph[graph_id], \
                ((0,tokenizer.max_seq_len-idx2graph[graph_id].shape[0]),(0,tokenizer.max_seq_len-idx2graph[graph_id].shape[0])), 'constant')

            sdat_graph = np.pad(idx2graph_sdat[graph_id], \
                ((0,tokenizer.max_seq_len-idx2graph_sdat[graph_id].shape[0]),(0,tokenizer.max_seq_len-idx2graph_sdat[graph_id].shape[0])), 'constant')
           
            data = {
                'text_bert_indices': concat_bert_indices,
                'bert_segments_indices': concat_segments_indices,
                'text_indices': text_indices,
                'dependency_graph': dependency_graph,
                'affective_graph': sdat_graph,
                'label': label,
            }
            graph_id += 1
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
